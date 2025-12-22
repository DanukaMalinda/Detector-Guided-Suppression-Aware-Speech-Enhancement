#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from torchaudio.transforms import Resample

# Import your DeepVQE class
from deepvqe import DeepVQE  # ensure this is importable (same as training)

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # Be generous: handle different checkpoint formats
    state = None
    for key in ["model_state_dict", "model", "state_dict"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None and isinstance(ckpt, dict):
        # maybe it IS the state dict
        state = ckpt
    if state is None:
        raise RuntimeError(f"Could not find a state_dict in checkpoint: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing} unexpected={unexpected}")
    return model

def ensure_mono(x: torch.Tensor) -> torch.Tensor:
    # x: [C, T] or [T]
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, T]
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)  # [1, T]
    return x

def enhance_file(model, in_path: Path, out_path: Path, device: torch.device,
                 target_sr=16000, n_fft=512, hop=256, win=512):
    # Read audio
    wav, sr = torchaudio.load(str(in_path))  # wav: [C, T]
    wav = ensure_mono(wav)                   # [1, T]
    if sr != target_sr:
        wav = Resample(sr, target_sr)(wav)
        sr = target_sr
    wav = wav.to(device)
    orig_len = wav.shape[-1]

    # STFT params
    window = torch.hann_window(win, device=device)

    with torch.inference_mode():
        Xc = torch.stft(
            wav, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True, center=True
        )                               # [1, F, T]
        Xri = torch.view_as_real(Xc)    # [1, F, T, 2]
        Yri = model(Xri)                # [1, F, T, 2]
        Yc = torch.complex(Yri[..., 0], Yri[..., 1])
        enh = torch.istft(
            Yc, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=True, length=orig_len
        )                               # [1, T]
        enh = enh.clamp_(-1.0, 1.0)
    enh = enh.cpu()
    

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), enh, sample_rate=sr)

def is_audio_file(p: Path):
    return p.suffix.lower() in {".wav", ".flac", ".ogg"}

def main():
    parser = argparse.ArgumentParser("DeepVQE folder inference")
    
    ##### REQUIRED ######
    parser.add_argument("--in_dir", required=True, type=str, help="Folder with noisy audio (recurses).")
    parser.add_argument("--out_dir", required=True, type=str, help="Output folder (mirrors tree, same filenames).")
    parser.add_argument("--ckpt", required=True, type=str, help="Model checkpoint path.")
    #####################


    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pattern", type=str, default="**/*", help="Glob under in_dir (default: all).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output files.")
    # STFT/Resample options (must match training)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--win", type=int, default=512)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        print(f"[err] in_dir does not exist: {in_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[info] using device: {device}")

    # Build and load model
    model = DeepVQE().to(device).eval()
    load_checkpoint_into_model(model, args.ckpt, map_location="cpu")
    model.to(device).eval()

    # Walk and enhance
    files = [p for p in in_dir.glob(args.pattern) if p.is_file() and is_audio_file(p)]
    if not files:
        print("[warn] no audio files found.")
        return

    print(f"[info] found {len(files)} files")
    for i, in_path in enumerate(files, 1):
        rel = in_path.relative_to(in_dir)
        out_path = out_dir / rel  # same name in mirrored tree
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue
        try:
            enhance_file(
                model, in_path, out_path, device,
                target_sr=args.sr, n_fft=args.n_fft, hop=args.hop, win=args.win
            )
            print(f"[{i}/{len(files)}] ✓ {rel}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ✗ {rel}  ({type(e).__name__}: {e})")

if __name__ == "__main__":
    main()
