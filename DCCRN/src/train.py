# Full DCCRN Training Pipeline with Noise Augmentation and CSV Logging

import os
import random
import argparse
import torchaudio
import torch
import torch.optim as optim
import pandas as pd
from glob import glob
from torchaudio.transforms import Resample
from dccrn import DCCRN
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === UTILITY FUNCTIONS ===
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

# === GENERATE NOISY DATASET ===
def generate_noisy_dataset(CLEAN_dir, NOISE_dir, SPEAKER_NOISE_dir):
    clean_files = list_flac_files(CLEAN_dir)
    print(len(clean_files))
    noise_files = list_wav_files(NOISE_dir) + list_flac_files(SPEAKER_NOISE_dir)
    metadata = []

    for i, clean_fp in enumerate(tqdm(clean_files)):
        clean_wav = load_audio(clean_fp, TARGET_SR)
        duration_samples = SAMPLE_DURATION * TARGET_SR
        if clean_wav.shape[1] < duration_samples:
            continue

        clean_wav = clean_wav[:, :duration_samples]
        noise_fp = random.choice(noise_files)
        noise_wav = load_audio(noise_fp, TARGET_SR)
        start = random.randint(0, max(0, noise_wav.shape[1] - duration_samples))
        noise_crop = noise_wav[:, start:start + duration_samples]

        snr = random.choice(SNR_RANGE)
        noisy_wav = mix_audio(clean_wav, noise_crop, snr)

        out_name = f"noisy_{i:05d}.pt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        torch.save({"noisy": noisy_wav, "clean": clean_wav}, out_path)
        torchaudio.save(f"{OUTPUT_DIR}/noisy_{i:05d}.flac", noisy_wav, TARGET_SR)

        metadata.append({
            "noisy_path": out_path,
            "noisy_flac_path": f"{OUTPUT_DIR}/noisy_{i:05d}.flac",
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

# === DATASET CLASS ===
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["noisy_path"])
        return data["noisy"].squeeze(0), data["clean"].squeeze(0)
    
class AudioChunkDataset(Dataset):
    def __init__(self, file_list, chunk_size=16000*4, hop_size=None, sample_rate=16000):
        self.file_list = file_list  # List of (noisy_path, clean_path)
        self.chunk_size = chunk_size
        self.hop_size = hop_size or chunk_size
        self.sample_rate = sample_rate

        self.chunk_metadata = []  # List of (noisy_path, clean_path, start_sample)

        # Precompute all possible chunks
        for noisy_path, clean_path in file_list:
            length = min(self._get_length(noisy_path), self._get_length(clean_path))
            for start in range(0, length - chunk_size + 1, self.hop_size):
                self.chunk_metadata.append((noisy_path, clean_path, start))

    def _get_length(self, path):
        info = torchaudio.info(path)
        return info.num_frames

    def __len__(self):
        return len(self.chunk_metadata)

    def __getitem__(self, idx):
        noisy_path, clean_path, start = self.chunk_metadata[idx]

        noisy, _ = torchaudio.load(noisy_path, frame_offset=start, num_frames=self.chunk_size)
        clean, _ = torchaudio.load(clean_path, frame_offset=start, num_frames=self.chunk_size)

        return noisy.squeeze(0), clean.squeeze(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# === MAIN TRAINING ===
def train(rank, world_size, CSV_PATH, TARGET_SR=16000):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    file_list = read_file_list_from_csv(CSV_PATH)
    dataset = AudioChunkDataset(file_list, chunk_size=TARGET_SR * 4, hop_size=TARGET_SR // 2)  # 1s chunks with 50% overlap
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=40, sampler=sampler, num_workers=10, pin_memory=True)

    model = DCCRN(win_len=320, win_inc=160, fft_len=512, rnn_units=256, masking_mode='E', use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256]).to(rank)

    def loss_function(predictions, targets):
        return model.loss(predictions, targets, loss_mode='SI-SNR')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(rank), clean.to(rank)
            optimizer.zero_grad()
            _, enhanced = model(noisy)
            loss = loss_function(enhanced, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"/home/local1/workspace/danukam/AMD/TSOS/DCCRN_test/src/data/retrain/dccrn_model_3_pll_epoch_{epoch}.pth")
        print("Model saved as 'dccrn_model_3_pll.pth'")
    cleanup()

def run_train(train, world_size, CSV_PATH, TARGET_SR=16000):
    mp.spawn(train,
             args=(world_size, CSV_PATH, TARGET_SR),
             nprocs=world_size,
             join=True)
    
def main():
    parser = argparse.ArgumentParser(description="DCCRN Training with Noisy Dataset")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory of clean audio files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory of noise audio files")
    parser.add_argument("--speaker_noise_dir", type=str, required=True, help="Directory of speaker noise audio files")
    parser.add_argument("--output_dir", type=str, default="data/noisy_dataset", help="Output directory for noisy dataset")
    parser.add_argument("--csv_path", type=str, default="metadata_360_reatrain.csv", help="Path to save metadata CSV")
    args = parser.parse_args()

    # === CONFIGURATION ===
    CLEAN_DIR = args.clean_dir # Root folder containing subfolders of clean audio
    NOISE_DIR = args.noise_dir # Background noise
    SPEAKER_NOISE_DIR = args.bg_speaker_dir  # Speaker noise
    OUTPUT_DIR = args.output_dir
    CSV_PATH = args.csv_path
    TARGET_SR = 16000
    SEED = 42
    n_gpus = 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate the noisy dataset
    if not os.path.exists(CSV_PATH):
        print("Generating noisy dataset...")
        generate_noisy_dataset(CLEAN_DIR, NOISE_DIR, SPEAKER_NOISE_DIR)
    else:
        print(f"Using existing metadata from {CSV_PATH}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs. Running demo on {n_gpus} GPUs.")
        world_size = min(n_gpus, 2) 
    else:
        print("No GPU found. Running DDP on CPU is not the primary use case and may be slow.")
        world_size = 2 # Or 1

    run_train(train, world_size, CSV_PATH, TARGET_SR=TARGET_SR)

if __name__ == "__main__":
    main()
