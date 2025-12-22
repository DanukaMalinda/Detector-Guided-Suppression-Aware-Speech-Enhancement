#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
from transformers import AutoModel, AutoFeatureExtractor

# project imports
from modules import Model                 # your SSFCVAE
from bigvgan_dummy import get_vocoder     # BigVGAN 22k vocoder wrapper from your project
from utils import spectrogram_torch       # same function used in preprocess.py

# ---------------------- helpers ----------------------

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, map_location="cpu"):
    """Flexible checkpoint loader: supports several common dict keys."""
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "generator", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
        if state is None:
            # maybe the ckpt itself is a state_dict
            state = ckpt if all(isinstance(v, torch.Tensor) for v in ckpt.values()) else None
    if state is None:
        raise RuntimeError(f"Could not find a state_dict in checkpoint: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing} unexpected={unexpected}")
    return model

def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: [C,T] or [T] -> [1,T]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def trim_to_multiple(x: torch.Tensor, hop: int) -> torch.Tensor:
    """Trim 1D waveform to be a multiple of `hop` (like your preprocess)."""
    T = x.shape[-1]
    r = T % hop
    return x[..., :T-r] if r != 0 else x

def standardize_w2v_per_file(w2v: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Per-utterance standardization over time (matches sklearn StandardScaler(channel-wise) behavior
    but computed on the fly). w2v: [B, D, T].
    """
    mean = w2v.mean(dim=-1, keepdim=True)  # [B,D,1]
    std  = w2v.std(dim=-1, keepdim=True).clamp_min(eps)
    return (w2v - mean) / std

def resample_tensor(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """Resample mono [1,T] (or [B,T]) using torchaudio on the same device as `wav`."""
    if sr_in == sr_out:
        return wav
    resampler = torchaudio.transforms.Resample(sr_in, sr_out)
    try:
        resampler = resampler.to(wav.device)
    except Exception:
        # fallback: CPU resample then send back
        return torchaudio.transforms.Resample(sr_in, sr_out)(wav.cpu().float()).to(wav.device)
    return resampler(wav.float())


def save_audio(path: Path, wav: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach()
    # Force to [channels, time]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim > 2:
        wav = wav.squeeze()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            # Collapse leading dims into channel dim
            T = wav.shape[-1]
            C = int(wav.numel() // T)
            wav = wav.reshape(C, T)
    # safety: float32 and clipped
    wav = wav.to(torch.float32).clamp_(-1.0, 1.0).cpu()
    torchaudio.save(str(path), wav, sample_rate=sr)


def wav2vec_features_16k(
    wav_16k: torch.Tensor,  # [1,T] on CPU or GPU ok
    feat_extractor: AutoFeatureExtractor,
    w2v_model: AutoModel,
    device: torch.device,
    layer_index: int = 15
) -> torch.Tensor:
    """Compute wav2vec2 features like preprocess.py (layer 15, [B, D, T])."""
    # librosa in preprocess used CPU; we stay consistent and use HF processor
    wav_np = wav_16k.squeeze(0).cpu().numpy()
    inputs = feat_extractor(wav_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = w2v_model(**inputs, output_hidden_states=True).hidden_states[layer_index]  # [B, T', D]
    feats = out.transpose(1, 2).contiguous()  # [B, D, T']
    return feats

# ---------------------- core inference ----------------------

def enhance_folder(
    in_dir: Path,
    out_dir: Path,
    ckpt_path: Path,
    config_path: Path,
    device: torch.device,
    out_sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    win: int = 1024,
    w2v_name: str = "facebook/wav2vec2-large-lv60",
    w2v_layer: int = 15,
    w2v_standardize: bool = True
):
    # Load config
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = Model(
        cfg['n_mel_channels'], cfg['n_spec_channels'], cfg['n_w2v_channels'],
        cfg['hdim'], cfg['latent_dim'], cfg['n_head'], cfg['d_inner'],
        cfg['n_layers'], cfg['n_flows']
    ).to(device).eval()
    load_checkpoint_into_model(model, str(ckpt_path), map_location="cpu")
    model.to(device).eval()

    import types
    def _safe_resample(self, x, spec_lengths, w2v_lengths, target_T=None):
        """
        x: [B, C, T_spec]
        spec_lengths: [B]
        w2v_lengths:  [B]  (target per-sample length)
        target_T: optional int; if given it should equal w2v_lengths[i]
        returns: x resampled to per-sample target length, [B, C, max(target_lengths)]
        """
        B, C, Tspec = x.shape
        device, dtype = x.device, x.dtype

        # per-sample targets
        if w2v_lengths is not None:
            tgt = w2v_lengths.to(device)
        elif target_T is not None:
            tgt = torch.full((B,), int(target_T), device=device, dtype=torch.long)
        else:
            tgt = spec_lengths.to(device)

        maxT = int(tgt.max().item())
        out = x.new_zeros(B, C, maxT)

        for i in range(B):
            Ti = int(spec_lengths[i].item())
            To = int(tgt[i].item())
            xi = x[i:i+1, :, :Ti]  # [1,C,Ti]

            if To == Ti:
                yi = xi                                    # equal length → passthrough
            else:
                yi = torch.nn.functional.interpolate(      # up/downsample along time
                    xi, size=To, mode="linear", align_corners=False
                )  # [1,C,To]

            out[i, :, :To] = yi[0]
        return out

    # attach to model
    model.resample = types.MethodType(_safe_resample, model)

    # Vocoder (22.05 kHz)
    vocoder = get_vocoder("bigvgan_22khz_80band").to(device).eval()
    voc_sr = 22050

    # HF wav2vec2 (16 kHz)
    print(f"Loading wav2vec2: {w2v_name}")
    w2v_model = AutoModel.from_pretrained(w2v_name).to(device).eval()
    feat_extractor = AutoFeatureExtractor.from_pretrained(w2v_name)

    # Hann window is internal to spectrogram_torch; we just feed 22.05k audio
    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    if not files:
        print(f"[warn] no audio files found under: {in_dir}")
        return

    print(f"[info] found {len(files)} files")
    for i, in_path in enumerate(files, 1):
        try:
            # ---- load & mono ----
            wav, sr_in = torchaudio.load(str(in_path))
            wav = ensure_mono(wav)

            # ---- build 22.05k for spectrogram; trim to multiple of hop=256 ----
            wav_22k = resample_tensor(wav, sr_in, 22050)
            wav_22k = trim_to_multiple(wav_22k, hop)
            if wav_22k.numel() == 0:
                print(f"[{i}/{len(files)}] skip (too short): {in_path.relative_to(in_dir)}")
                continue

            # Run spectrogram_torch on CPU to match its internal kernels, then move to device
            with torch.inference_mode():
                spec_cpu = spectrogram_torch(wav_22k.cpu(), n_fft, 22050, hop, win)  # [1, C_spec, T_spec] on CPU
            spec_noisy = spec_cpu.to(device=device, dtype=torch.float32)
            T_spec = spec_noisy.shape[-1]
            if T_spec == 0:
                print(f"[{i}/{len(files)}] skip (empty spec): {in_path.relative_to(in_dir)}")
                continue
            spec_lengths = torch.tensor([T_spec], device=device, dtype=torch.long)

            # ---- build wav2vec2 at 16k ----
            wav_16k = resample_tensor(wav, sr_in, 16000)
            feats = wav2vec_features_16k(wav_16k, feat_extractor, w2v_model, device, layer_index=w2v_layer)  # [1,D,Tw]
            # per-file standardization (similar to StandardScaler used in training)
            if w2v_standardize:
                feats = standardize_w2v_per_file(feats)  # [1,D,Tw]

            # match feature channel count from config
            n_w2v_cfg = int(cfg['n_w2v_channels'])
            if feats.size(1) > n_w2v_cfg:
                feats = feats[:, :n_w2v_cfg, :]
            elif feats.size(1) < n_w2v_cfg:
                pad_c = n_w2v_cfg - feats.size(1)
                feats = torch.cat([feats, feats.new_zeros((1, pad_c, feats.size(2)))], dim=1)

            # ---- align wav2vec time axis to spectrogram time axis BEFORE inference ----
            Tw = feats.shape[-1]
            if Tw != T_spec:
                feats = torch.nn.functional.interpolate(
                    feats, size=T_spec, mode="linear", align_corners=False
                )  # [1,D,T_spec]
            w2v_feats_noisy = feats
            w2v_lengths = spec_lengths.clone()  # now both lengths are T_spec

            # ---- run model.inference ----
            with torch.inference_mode():
                mel_pred = model.inference(
                    spec_noisy, spec_lengths,
                    w2v_feats_noisy, w2v_lengths
                )  # [1, n_mels, T_spec]

                mel_pred = mel_pred[:, :, :T_spec].float()
                y = vocoder(mel_pred[:, :, :T_spec].float())
                # Handle tuple/list returns
                if isinstance(y, (list, tuple)):
                    y = y[0]
                # Ensure 2D [C,T]
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                elif y.ndim > 2:
                    y = y.squeeze()
                    if y.ndim == 1:
                        y = y.unsqueeze(0)
                # Now y is [1, T]
                wav_22k_out = y


                # choose output SR
                if out_sr < 0:
                    wav_out, sr_out = wav_22k_out, voc_sr
                elif out_sr == 0:
                    wav_out, sr_out = resample_tensor(wav_22k_out, voc_sr, sr_in), sr_in
                else:
                    wav_out, sr_out = resample_tensor(wav_22k_out, voc_sr, out_sr), out_sr

            # ---- write with same filename under out_dir ----
            rel = in_path.relative_to(in_dir)
            out_path = out_dir / rel
            save_audio(out_path, wav_out, sr_out)
            print(f"[{i}/{len(files)}] ✓ {rel}")

        except Exception as e:
            print(f"[{i}/{len(files)}] ✗ {in_path.relative_to(in_dir)}  ({type(e).__name__}: {e})")

    print("[done] inference complete.")


# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser("SSFCVAE On-the-fly Inference")
    p.add_argument("--in_dir", required=True, type=str, help="Folder with noisy wav/flac (recurses).")
    p.add_argument("--out_dir", required=True, type=str, help="Output folder (same filenames & tree).")
    p.add_argument("--ckpt", required=True, type=str, help="Trained SSFCVAE checkpoint.")
    p.add_argument("--config", required=True, type=str, help="config.yaml used for training (dims).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Output SR: <0 keep 22.05 kHz; 0 match input SR; >0 set explicit SR
    p.add_argument("--out_sr", type=int, default=0)
    # Feature params (must mirror training/preprocess)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--w2v_name", type=str, default="facebook/wav2vec2-large-lv60")
    p.add_argument("--w2v_layer", type=int, default=15)
    p.add_argument("--no_w2v_standardize", action="store_true",
                   help="Disable per-file (channel-wise) standardization of wav2vec features.")
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enhance_folder(
        in_dir=in_dir,
        out_dir=out_dir,
        ckpt_path=Path(args.ckpt),
        config_path=Path(args.config),
        device=device,
        out_sr=args.out_sr,
        n_fft=args.n_fft,
        hop=args.hop,
        win=args.win,
        w2v_name=args.w2v_name,
        w2v_layer=args.w2v_layer,
        w2v_standardize=not args.no_w2v_standardize
    )

if __name__ == "__main__":
    main()
