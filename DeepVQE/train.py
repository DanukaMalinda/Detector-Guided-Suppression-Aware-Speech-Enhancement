# DeepVQE training from scratch with your existing pipeline pieces
# - Uses STFT->DeepVQE->iSTFT, trains in time-domain (SI-SNR) + optional MR-STFT aux loss
# - Keeps your DDP setup, CSV loader, speaker-split, detector, and teacher loss option

import os
import random
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from glob import glob
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# ---- NEW: DeepVQE import ----
from deepvqe import DeepVQE 

# OPTIONAL: your teacher loss; keep API the same as before
from teacher_loss import teacher_loss as teacher_loss_fn

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEAKER_EMB_PATH = "train_speaker_embeddings.pt"

# ===== Loss bits =====
def si_snr_loss(est, ref, eps=1e-8):
    """Scale-Invariant SNR loss (negative SI-SNR, averaged over batch)."""
    # [B, T]
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)
    proj = (torch.sum(est_zm * ref_zm, dim=-1, keepdim=True) / (torch.sum(ref_zm ** 2, dim=-1, keepdim=True) + eps)) * ref_zm
    noise = est_zm - proj
    ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_snr = 10 * torch.log10(ratio + eps)
    return -si_snr.mean()

class MRSTFTLoss(nn.Module):
    """Simple multi-resolution STFT loss: magnitude L1 across a few resolutions."""
    def __init__(self, fft_sizes=(256, 512, 1024), hops=(64, 128, 256), win_lengths=(256, 512, 1024)):
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.win_lengths = win_lengths

    def forward(self, x, y):
        loss = 0.0
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hops, self.win_lengths):
            w = torch.hann_window(win_len, device=x.device)
            X = torch.view_as_real(torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_len,
                                              window=w, return_complex=True, center=True))
            Y = torch.view_as_real(torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_len,
                                              window=w, return_complex=True, center=True))
            # magnitude
            magX = torch.linalg.norm(X, dim=-1)  # (...,2) -> magnitude
            magY = torch.linalg.norm(Y, dim=-1)
            loss = loss + (magX - magY).abs().mean()
        return loss / len(self.fft_sizes)

# === UTILITY FUNCTIONS (unchanged where possible) ===
def load_embeds(path: str, nested_key: str | None = None) -> dict:
    obj = torch.load(path, map_location="cpu")
    if nested_key:
        if not isinstance(obj, dict) or nested_key not in obj or not isinstance(obj[nested_key], dict):
            raise TypeError(f"File does not contain a dict at key '{nested_key}'.")
        return obj[nested_key]
    if isinstance(obj, dict) and "speaker_embeds" in obj and isinstance(obj["speaker_embeds"], dict):
        return obj["speaker_embeds"]
    if not isinstance(obj, dict):
        raise TypeError("Loaded object is not a dict of embeddings.")
    return obj

def get_speaker_key(example):
    for k in ("speaker_id",):
        if k in example:
            return k
    raise KeyError("No speaker id key found in samples.")

def split_by_speaker(
    file_list,
    val_ratio=0.1,
    seed=42,
    speaker_index=2,
    max_files_per_speaker=None,
):
    if not file_list:
        raise ValueError("file_list is empty.")
    if max_files_per_speaker is not None:
        if not isinstance(max_files_per_speaker, int) or max_files_per_speaker <= 0:
            raise ValueError("max_files_per_speaker must be a positive int or None.")

    def get_spk(ex):
        if isinstance(ex, (list, tuple)):
            if len(ex) <= speaker_index:
                raise IndexError(f"Item too short for speaker_index={speaker_index}: {ex}")
            return ex[speaker_index]
        elif isinstance(ex, dict):
            for k in ("speaker_id", "spaker_id", "spk_id", "spk"):
                if k in ex:
                    return ex[k]
            raise KeyError(f"No speaker key in dict item. Keys: {list(ex.keys())[:10]}")
        else:
            raise TypeError(f"Unsupported item type: {type(ex).__name__}")

    speakers = sorted({get_spk(ex) for ex in file_list})
    if len(speakers) < 2:
        raise ValueError("Need at least 2 unique speakers to create a train/val split.")

    rng = np.random.default_rng(seed)
    speakers = speakers.copy()
    rng.shuffle(speakers)

    n_val_spk = max(1, int(round(len(speakers) * val_ratio)))
    n_val_spk = min(n_val_spk, len(speakers) - 1)
    val_speakers = set(speakers[:n_val_spk])
    train_speakers = set(speakers[n_val_spk:])

    train_files = [ex for ex in file_list if get_spk(ex) in train_speakers]
    val_files   = [ex for ex in file_list if get_spk(ex) in val_speakers]

    def cap_per_speaker(files):
        if max_files_per_speaker is None:
            return files
        groups = defaultdict(list)
        for ex in files:
            groups[get_spk(ex)].append(ex)
        capped = []
        for spk, items in groups.items():
            if len(items) > max_files_per_speaker:
                idx = rng.choice(len(items), size=max_files_per_speaker, replace=False)
                selected = [items[i] for i in sorted(idx)]
                capped.extend(selected)
            else:
                capped.extend(items)
        return capped

    train_files = cap_per_speaker(train_files)
    val_files   = cap_per_speaker(val_files)

    # 1) Speakers must be disjoint
    assert train_speakers.isdisjoint(val_speakers), "Speaker leakage into both splits."

    # 2) (Optional) also ensure no file overlap by (noisy_path, clean_path)
    def _keys(files):
        out = set()
        for ex in files:
            if isinstance(ex, dict) and "noisy_path" in ex and "clean_path" in ex:
                out.add((ex["noisy_path"], ex["clean_path"]))
            elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
                out.add((ex[0], ex[1]))
        return out

    assert _keys(train_files).isdisjoint(_keys(val_files)), "File overlap between train and val."


    print(
        f"Speakers: total={len(speakers)} | train={len(train_speakers)} | val={len(val_speakers)}"
        + (f" | cap={max_files_per_speaker}/speaker" if max_files_per_speaker else "")
    )
    return train_files, val_files

def list_flac_files(root_dir):
    return sorted(glob(os.path.join(root_dir, "**", "*.flac"), recursive=True))

def list_wav_files(root_dir):
    return sorted(glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))

def load_audio(file, target_sr):
    wav, sr = torchaudio.load(file)
    if sr != target_sr:
        wav = Resample(sr, target_sr)(wav)
    return wav.mean(0, keepdim=True)  # Convert to mono

def mix_audio(clean, noise, snr_db):
    clean = clean[:, :min(clean.shape[1], noise.shape[1])]
    noise = noise[:, :clean.shape[1]]
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    factor = (clean_power / (10 ** (snr_db / 10)) / (noise_power + 1e-8)).sqrt()
    scaled_noise = noise * factor
    return (clean + scaled_noise).clamp(-1, 1)

def load_pt_files(root_dir):
    data_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt"):
                file_path = os.path.join(dirpath, filename)
                try:
                    data = torch.load(file_path, map_location="cpu")
                    if all(k in data for k in ("noisy_path", "clean_path", "speaker_id")):
                        data_list.append({
                            "path": file_path,
                            "noisy_path": data["noisy_path"],
                            "clean_path": data["clean_path"],
                            "speaker_id": data["speaker_id"]
                        })
                    else:
                        print(f"Skipping {file_path}: Missing expected keys")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return data_list

# === GENERATE NOISY DATASET === (unchanged)
def generate_noisy_dataset():
    clean_files = list_flac_files(CLEAN_DIR)
    noise_files = list_wav_files(NOISE_DIR) + list_flac_files(SPEAKER_NOISE_DIR)
    metadata = []
    for i, clean_fp in enumerate(clean_files):
        clean_wav = load_audio(clean_fp, TARGET_SR)
        duration_samples = SAMPLE_DURATION * TARGET_SR
        if clean_wav.shape[1] < duration_samples:
            continue
        file_name = os.path.basename(clean_fp)
        name_without_ext = os.path.splitext(file_name)[0]
        clean_wav = clean_wav[:, :duration_samples]
        noise_fp = random.choice(noise_files)
        noise_wav = load_audio(noise_fp, TARGET_SR)
        start = random.randint(0, max(0, noise_wav.shape[1] - duration_samples))
        noise_crop = noise_wav[:, start:start + duration_samples]
        snr = random.choice(SNR_RANGE)
        noisy_wav = mix_audio(clean_wav, noise_crop, snr)
        speaker_id_str = name_without_ext.split('-')[0]
        out_name = f"{name_without_ext}.pt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        torchaudio.save(f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac", noisy_wav, TARGET_SR)
        torch.save({"noisy_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac",
                    "clean_path": clean_fp, "speaker_id": speaker_id_str}, out_path)
        metadata.append({
            "noisy_path": out_path,
            "noisy_flac_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac",
            "clean_flac_path": clean_fp,
            "clean_path": clean_fp,
            "noise_path": noise_fp,
            "snr_db": snr,
            "offset": start
        })
    pd.DataFrame(metadata).to_csv(CSV_PATH, index=False)
    print(f"Saved metadata to {CSV_PATH}")


# === GENERATE NOISY DATASET === (new with more silence)
def generate_noisy_dataset_with_silence():
    clean_files = list_flac_files(CLEAN_DIR)
    noise_files = list_wav_files(NOISE_DIR) + list_flac_files(SPEAKER_NOISE_DIR)
    metadata = []
    for i, clean_fp in enumerate(clean_files):
        clean_wav = load_audio(clean_fp, TARGET_SR)

        # add 1s silence
        silence_samples = TARGET_SR  # 1 second
        silence = torch.zeros(clean_wav.shape[0], silence_samples)
        clean_wav = torch.cat([clean_wav, silence], dim=1)

        duration_samples = SAMPLE_DURATION * TARGET_SR
        if clean_wav.shape[1] < duration_samples:
            continue
        file_name = os.path.basename(clean_fp)
        name_without_ext = os.path.splitext(file_name)[0]
        clean_wav = clean_wav[:, :duration_samples]
        noise_fp = random.choice(noise_files)
        noise_wav = load_audio(noise_fp, TARGET_SR)
        start = random.randint(0, max(0, noise_wav.shape[1] - duration_samples))
        noise_crop = noise_wav[:, start:start + duration_samples]
        snr = random.choice(SNR_RANGE)
        noisy_wav = mix_audio(clean_wav, noise_crop, snr)
        speaker_id_str = name_without_ext.split('-')[0]
        out_name = f"{name_without_ext}.pt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        torchaudio.save(f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac", noisy_wav, TARGET_SR)
        torch.save({"noisy_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac",
                    "clean_path": clean_fp, "speaker_id": speaker_id_str}, out_path)
        metadata.append({
            "noisy_path": out_path,
            "noisy_flac_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac",
            "clean_flac_path": clean_fp,
            "clean_path": clean_fp,
            "noise_path": noise_fp,
            "snr_db": snr,
            "offset": start
        })
    pd.DataFrame(metadata).to_csv(CSV_PATH, index=False)
    print(f"Saved metadata to {CSV_PATH}")

def read_file_list_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    file_list = list(zip(df["noisy_flac_path"], df["clean_flac_path"]))
    return file_list

# === DATASET === (unchanged signatures; loads wav chunks; training does STFT)
class AudioChunkDataset(Dataset):
    def __init__(self, file_list, chunk_size=16000*4, hop_size=None, sample_rate=16000, teaching=False):
        self.file_list = file_list  # dicts with noisy_path, clean_path, speaker_id
        self.chunk_size = chunk_size
        self.hop_size = hop_size or chunk_size
        self.sample_rate = sample_rate
        self.teaching = teaching
        

        # Cache embeddings ONCE (minor optimization)
        self.spk_embeds = load_embeds(SPEAKER_EMB_PATH)

        self.chunk_metadata = []  # (noisy_path, clean_path, start, speaker_id)
        for file in file_list:
            noisy_path = file['noisy_path']
            clean_path = file['clean_path']
            speaker_id = file['speaker_id']
            length = min(self._get_length(noisy_path), self._get_length(clean_path)+sample_rate)  # added 1s of silence
            for start in range(0, max(0, length - chunk_size + 1), self.hop_size):
                self.chunk_metadata.append((noisy_path, clean_path, start, speaker_id))

    def _get_length(self, path):
        # info = torchaudio.info(path)
        # return info.num_frames
        waveform, _ = torchaudio.load(path)
        return waveform.shape[1]

    def __len__(self):
        return len(self.chunk_metadata)

    def __getitem__(self, idx):
        noisy_path, clean_path, start, speaker_id = self.chunk_metadata[idx]
        noisy, _ = torchaudio.load(noisy_path, frame_offset=start, num_frames=self.chunk_size)

        clean_len = self._get_length(clean_path)
        effective_len = clean_len + self.sample_rate

        if start >= effective_len:
            # chunk completely in silence
            clean = torch.zeros(noisy.shape[0], self.chunk_size)

        elif start + self.chunk_size > effective_len:
            # chunk partially overlaps real clean audio
            valid = max(0, clean_len - start)

            if valid > 0:
                part1, _ = torchaudio.load(
                    clean_path,
                    frame_offset=start,
                    num_frames=valid
                )
            else:
                part1 = torch.zeros(noisy.shape[0], 0)

            part2 = torch.zeros(noisy.shape[0], self.chunk_size - part1.shape[1])
            clean = torch.cat([part1, part2], dim=1)

        else:
            # fully inside real clean audio
            clean, _ = torchaudio.load(
                clean_path,
                frame_offset=start,
                num_frames=self.chunk_size
            )

        clean = clean[:, :self.chunk_size]
        if clean.shape[1] < self.chunk_size:
            pad = torch.zeros(clean.shape[0], self.chunk_size - clean.shape[1])
            clean = torch.cat([clean, pad], dim=1)

        # make storage safe for DataLoader
        clean = clean.contiguous().clone()

        # mono
        noisy = noisy[:1, :]
        clean = clean[:1, :]
        # speaker embedding
        if isinstance(self.spk_embeds, dict):
            key = speaker_id if speaker_id in self.spk_embeds else int(speaker_id) if str(int(speaker_id)) in map(str, self.spk_embeds.keys()) else None
            if key is None:
                raise KeyError(f"No speaker embedding for {speaker_id}")
            embedding = torch.as_tensor(self.spk_embeds[int(speaker_id)], dtype=torch.float32).view(-1)
        else:
            embedding = torch.as_tensor(self.spk_embeds[int(speaker_id)], dtype=torch.float32).view(-1)

        if self.teaching:
            return noisy.squeeze(0), clean.squeeze(0), embedding
        return noisy.squeeze(0), clean.squeeze(0)

# === DDP helpers ===
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9003'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# === TRAIN ===
def train(rank, world_size, dataset_dir, TARGET_SR=16000):
    print(f"[Rank {rank}] DeepVQE training.")
    setup(rank, world_size)

    file_list = load_pt_files(dataset_dir)
    train_files, val_files = split_by_speaker(file_list, val_ratio=0.1, seed=42, max_files_per_speaker=10)

    chunk_size = TARGET_SR * 4
    hop_size   = TARGET_SR // 2

    train_ds = AudioChunkDataset(train_files, chunk_size=chunk_size, hop_size=hop_size, teaching=True,  sample_rate=TARGET_SR)
    val_ds   = AudioChunkDataset(val_files,   chunk_size=chunk_size, hop_size=hop_size, teaching=False, sample_rate=TARGET_SR)

    is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = torch.utils.data.DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        train_loader  = torch.utils.data.DataLoader(train_ds, batch_size=16, sampler=train_sampler,
                                                    num_workers=4, pin_memory=True, drop_last=True)
        val_loader    = torch.utils.data.DataLoader(val_ds,   batch_size=16, sampler=val_sampler,
                                                    num_workers=0, pin_memory=True)
    else:
        train_loader  = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True,
                                                    num_workers=4, pin_memory=True, drop_last=True)
        val_loader    = torch.utils.data.DataLoader(val_ds,   batch_size=16, shuffle=False,
                                                    num_workers=0, pin_memory=True)

    # ---- DeepVQE model ----
    model = DeepVQE()  # default config matches the public repo
    model = model.to(rank)

    # STFT params from DeepVQE example / README
    N_FFT  = 512
    HOP    = 256
    WINLEN = 512
    hann = torch.hann_window(WINLEN, device=rank)

    # Loss setup
    use_teacher = False  # flip to False to train purely with SI-SNR + MR-STFT
    mrstft = MRSTFTLoss().to(rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    grad_clip = 5.0

    # OPTIONAL: your suppression detector
    from detector.models import SpeakerSuppressionDetector
    detector = SpeakerSuppressionDetector(input_dim=64).to(rank)
    ckpt = torch.load("/home/local1/workspace/ilias/speaker_suppression_detection/src/transformer_based_detection/checkpoints/best_model.pt", map_location=device)
    detector.load_state_dict(ckpt["model_state_dict"])
    detector.eval()
    
    # from detector.models import FNN_Autoencoder
    # detector = FNN_Autoencoder(input_dim=2048, hidden_dim=512, latent_dim=40)
    # ckpt = torch.load("/home/local1/workspace/ilias/speaker_suppression_detection/src/checkpoints/fnn_ae_model_big_rho1.0_cw2.0_ew1.0_m0.4.pt", map_location=device)
    # detector.load_state_dict(ckpt)
    # detector.to(rank)
    # detector.eval()

    num_epochs = 150
    best_score = float('inf')
    save_dir = "."
    best_path = f"{save_dir}/deepvqe_best_no_teacher_transformer_new_best_with_silence_10.pt"
    last_path = f"{save_dir}/deepvqe_last_no_teacher_transformer_new_last_with_silence_10.pt"

    start_epoch = 0

    if os.path.exists(last_path):
        print(f"[Rank {rank}] Loading checkpoint from {last_path}")
        ckpt = torch.load(last_path, map_location=f"cuda:{rank}")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"]
        best_score = ckpt["best_score"]

        print(f"[Rank {rank}] Resumed from epoch {start_epoch}, best_score={best_score:.4f}")

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            if use_teacher:
                noisy, clean, speaker_emb = batch
                speaker_emb = speaker_emb.to(rank)
            else:
                noisy, clean, _ = batch

            noisy = noisy.to(rank, non_blocking=True)  # [B, T]
            clean = clean.to(rank, non_blocking=True)  # [B, T]
            orig_len = noisy.shape[-1]

            # --- STFT -> DeepVQE -> iSTFT ---
            Xc = torch.stft(noisy, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                            window=hann, return_complex=True, center=True)                 # [B, F, T]
            Xri = torch.view_as_real(Xc)                                                 # [B, F, T, 2]
            Yri = model(Xri)                                                             # [B, F, T, 2]
            Yc  = torch.complex(Yri[...,0], Yri[...,1])                                  # [B, F, T]
            enh = torch.istft(Yc, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                              window=hann, center=True, length=orig_len)                  # [B, T]

            # --- Loss ---
            if use_teacher:
                loss_core = teacher_loss_fn(enh, clean, detector=detector, spk_emb=speaker_emb)
            else:
                loss_core = si_snr_loss(enh, clean)
            loss_aux  = 0.1 * mrstft(enh, clean)  # small MR-STFT regularizer
            loss = loss_core + loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation (SI-SNR only to keep it simple / fast) ----
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(rank, non_blocking=True)
                clean = clean.to(rank, non_blocking=True)
                orig_len = noisy.shape[-1]
                Xc = torch.stft(noisy, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                                window=hann, return_complex=True, center=True)
                Xri = torch.view_as_real(Xc)
                Yri = model(Xri)
                Yc  = torch.complex(Yri[...,0], Yri[...,1])
                enh = torch.istft(Yc, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                                  window=hann, center=True, length=orig_len)
                vloss = si_snr_loss(enh, clean)
                val_total += float(vloss.item())
        avg_val_loss = val_total / len(val_loader)
        score = avg_val_loss  # lower is better

        # ---- Checkpointing ----
        if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
            model_to_save = model.module if hasattr(model, "module") else model
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "last_train_loss": avg_train_loss,
                "last_val_loss": avg_val_loss,
                "stft": {"n_fft": N_FFT, "hop": HOP, "win": WINLEN}
            }
            torch.save(checkpoint, last_path)
            if score < best_score:
                best_score = score
                torch.save(checkpoint, best_path)
                print(f"[Epoch {epoch+1}] New best val_loss: {score:.4f} — saved to '{best_path}'")

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

    cleanup()

def run_train(train, world_size, OUTPUT_DIR, TARGET_SR=16000):
    mp.spawn(train, args=(world_size, OUTPUT_DIR, TARGET_SR), nprocs=world_size, join=True)

if __name__ == "__main__":
    clean_dir = "/home/local1/workspace/danukam/AMD/Data/train-clean-360/LibriSpeech/train-clean-360"
    bg_speaker_dir = "/home/local1/workspace/danukam/AMD/Data/train-clean-360/LibriSpeech/bg-speakers"
    noise_dir = "/home/local1/workspace/danukam/AMD/TSOS/DCCRN_test/data/DEMAND"

    CLEAN_DIR = clean_dir
    NOISE_DIR = noise_dir
    SPEAKER_NOISE_DIR = bg_speaker_dir
    # OUTPUT_DIR = "/home/local1/workspace/danukam/AMD/traindccrn/Training_data"
    OUTPUT_DIR = "/home/local1/workspace/ilias/deepvqe/training_data_with_silence_2"
    CSV_PATH = "metadata_360_silence_2.csv"
    TARGET_SR = 16000
    SNR_RANGE = [5, 10, 15, 20]
    SEED = 42
    SAMPLE_DURATION = 10  # seconds

    generate_dataset = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED)

    if generate_dataset:
        # generate_noisy_dataset()
        generate_noisy_dataset_with_silence()

    world_size = 2
    print("RUN TRAINING ...")
    run_train(train, world_size, OUTPUT_DIR, TARGET_SR=TARGET_SR)
