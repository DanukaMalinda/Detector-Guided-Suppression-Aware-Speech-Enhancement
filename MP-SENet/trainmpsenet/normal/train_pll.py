# Full DCCRN Training Pipeline with Noise Augmentation and CSV Logging

import os
import random
import torchaudio
import torch
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
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dataset import Dataset, mag_pha_stft, mag_pha_istft, get_dataset_filelist
from models.model import MPNet, pesq_score, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils import scan_checkpoint, load_checkpoint, save_checkpoint
import time
import argparse

torch.backends.cudnn.benchmark = True

# from teacher_loss import loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEAKER_EMB_PATH = "train_speaker_embeddings.pt"

# === UTILITY FUNCTIONS ===
def load_embeds(path: str, nested_key: str | None = None) -> dict:
    obj = torch.load(path, map_location="cpu")
    if nested_key:
        if not isinstance(obj, dict) or nested_key not in obj or not isinstance(obj[nested_key], dict):
            raise TypeError(f"File does not contain a dict at key '{nested_key}'.")
        return obj[nested_key]
    # Auto-detect a common nested key
    if isinstance(obj, dict) and "speaker_embeds" in obj and isinstance(obj["speaker_embeds"], dict):
        return obj["speaker_embeds"]
    if not isinstance(obj, dict):
        raise TypeError("Loaded object is not a dict of embeddings.")
    return obj

def get_speaker_key(example):
    for k in ("speaker_id"):
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
    """
    file_list item formats supported:
      - Sequence (list/tuple): [x, y, speaker_id]  -> speaker at `speaker_index` (default 2)
      - Dict: must contain one of: speaker_id / spaker_id / spk_id / spk

    Args:
        val_ratio (float): Fraction of speakers to allocate to validation (by speaker).
        seed (int): RNG seed for reproducible shuffling/subsampling.
        speaker_index (int): Index of speaker id in sequence items.
        max_files_per_speaker (int | None): If set, cap the number of files contributed
            by each speaker *within its split* (train or val). Selection is random
            but reproducible via `seed`. Default None = no cap.
    """
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

    # Collect unique speakers
    speakers = sorted({get_spk(ex) for ex in file_list})
    if len(speakers) < 2:
        raise ValueError("Need at least 2 unique speakers to create a train/val split.")

    # RNG for reproducibility
    rng = np.random.default_rng(seed)

    # Shuffle speakers reproducibly
    speakers = speakers.copy()
    rng.shuffle(speakers)

    # Choose validation speakers (ensure non-empty train)
    n_val_spk = max(1, int(round(len(speakers) * val_ratio)))
    n_val_spk = min(n_val_spk, len(speakers) - 1)  # don't empty train
    val_speakers = set(speakers[:n_val_spk])
    train_speakers = set(speakers[n_val_spk:])

    # Partition items by split
    train_files = [ex for ex in file_list if get_spk(ex) in train_speakers]
    val_files   = [ex for ex in file_list if get_spk(ex) in val_speakers]

    # Optionally cap files per speaker within each split
    def cap_per_speaker(files):
        if max_files_per_speaker is None:
            return files
        groups = defaultdict(list)
        for ex in files:
            groups[get_spk(ex)].append(ex)

        capped = []
        for spk, items in groups.items():
            if len(items) > max_files_per_speaker:
                # Random, reproducible subset; keep stable order of chosen indices
                idx = rng.choice(len(items), size=max_files_per_speaker, replace=False)
                selected = [items[i] for i in sorted(idx)]
                capped.extend(selected)
            else:
                capped.extend(items)
        return capped

    train_files = cap_per_speaker(train_files)
    val_files   = cap_per_speaker(val_files)

    # Safety checks
    assert train_speakers.isdisjoint(val_speakers), "Speaker leakage into both splits."
    if not train_files or not val_files:
        raise RuntimeError("Empty split after capping; adjust val_ratio or cap.")

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
            # if len(data_list) == 50:
            #     break

    return data_list

# === GENERATE NOISY DATASET ===
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
        torch.save({"noisy_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac", "clean_path": clean_fp, "speaker_id": speaker_id_str}, out_path)
        torchaudio.save(f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac", noisy_wav, TARGET_SR)

        metadata.append({
            "noisy_path": out_path,
            "noisy_flac_path": f"{OUTPUT_DIR}/{name_without_ext}_noisy.flac",  # Optional: if you save it
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
    def __init__(self, file_list, chunk_size=16000*4, hop_size=None, sample_rate=16000, teaching = False):
        self.file_list = file_list  # List of (noisy_path, clean_path)
        self.chunk_size = chunk_size
        self.hop_size = hop_size or chunk_size  # Use full chunks if hop_size not given
        self.sample_rate = sample_rate
        self.teaching = teaching

        self.chunk_metadata = []  # List of (noisy_path, clean_path, start_sample)

        # Precompute all possible chunks
        for file in file_list:
            noisy_path = file['noisy_path']
            clean_path = file['clean_path']
            speaker_id = file['speaker_id']
            length = min(self._get_length(noisy_path), self._get_length(clean_path))
            for start in range(0, length - chunk_size + 1, self.hop_size):
                self.chunk_metadata.append((noisy_path, clean_path, start, speaker_id))

    def _get_length(self, path):
        info = torchaudio.info(path)
        return info.num_frames

    def __len__(self):
        return len(self.chunk_metadata)

    def __getitem__(self, idx):
        noisy_path, clean_path, start, speaker_id = self.chunk_metadata[idx]

        # Load only the required chunk
        noisy, _ = torchaudio.load(noisy_path, frame_offset=start, num_frames=self.chunk_size)
        clean, _ = torchaudio.load(clean_path, frame_offset=start, num_frames=self.chunk_size)

        speaker_embeds = load_embeds(SPEAKER_EMB_PATH)

        # If your speaker_embeds is a tensor, convert to int
        if isinstance(speaker_embeds, dict):
            speaker_id = speaker_id
        else:
            speaker_id = int(speaker_id)

        if has_id(speaker_embeds, speaker_id):
            embedding = has_id(speaker_embeds, speaker_id)
        else:
            print(f"Warning no speaker embedding found for speaker {speaker_id}")
            exit()
        
        if self.teaching:
            return noisy.squeeze(0), clean.squeeze(0), embedding 

        return noisy.squeeze(0), clean.squeeze(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def has_id(embeds: dict, spk_id: str) -> bool:
    # Fast exact match first
    if spk_id in embeds:
        return True

    # Build normalized key sets once
    key_raw = set(embeds.keys())
    key_str = {str(k) for k in key_raw}

    # Try string-normalized match
    if spk_id in key_str:
        return True

    # If spk_id looks like an int, try int match (many speakers are numeric)
    try:
        spk_int = int(spk_id)
    except ValueError:
        spk_int = None
    if spk_int is not None and spk_int in key_raw:
        return True

    return False
# === MAIN TRAINING ===
def train(rank, a, h, dataset_dir, world_size, TARGET_SR=16000):
    if world_size > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * world_size, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        print(generator)
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print('Total Parameters: {:.3f}M'.format(num_params/1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    file_list = load_pt_files(dataset_dir)

    train_files, val_files = split_by_speaker(file_list, val_ratio=0.1, seed=42, max_files_per_speaker=10)

    chunk_size = TARGET_SR * 4
    hop_size   = TARGET_SR // 2  # change if you don't want heavy overlap

    train_ds = AudioChunkDataset(train_files, chunk_size=chunk_size, hop_size=hop_size, teaching=False)
    val_ds   = AudioChunkDataset(val_files,   chunk_size=chunk_size, hop_size=hop_size, teaching=False)

    # DDP-aware loaders (falls back to single GPU/CPU)
    is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = torch.utils.data.DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        train_loader  = torch.utils.data.DataLoader(train_ds, batch_size=40, sampler=train_sampler,
                                num_workers=10, pin_memory=True)
        validation_loader    = torch.utils.data.DataLoader(val_ds,   batch_size=40, sampler=val_sampler,
                                num_workers=10, pin_memory=True)
    else:
        train_loader  = torch.utils.data.DataLoader(train_ds, batch_size=40, shuffle=True,
                                num_workers=10, pin_memory=True, drop_last=True)
        validation_loader    = torch.utils.data.DataLoader(val_ds,   batch_size=40, shuffle=False,
                                num_workers=10, pin_memory=True)
        
    if rank == 0:
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        

    generator.train()
    discriminator.train()

    best_pesq = 0

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            noisy_audio, clean_audio  = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            # Discriminator
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                print('pesq is None!')
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # L2 Consistency Loss
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = loss_mag * 0.9 + loss_pha * 0.3  + loss_com * 0.1 + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g)
                        pha_error = (ip_error + gd_error + iaf_error).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric loss: {:4.3f}, Magnitude Loss : {:4.3f}, Phase Loss : {:4.3f}, Complex Loss : {:4.3f}, Time Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}'.
                           format(steps, loss_gen_all, loss_disc_all, metric_error, mag_error, pha_error, com_error, time_error, stft_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    t = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, noisy_audio = batch
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))

                            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()
                            val_stft_err_tot += F.mse_loss(com_g, com_g_hat).item()

                            t = j

                        val_mag_err = val_mag_err_tot / (t+1)
                        val_pha_err = val_pha_err_tot / (t+1)
                        val_com_err = val_com_err_tot / (t+1)
                        val_stft_err = val_stft_err_tot / (t+1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)
                    
                    if epoch >= a.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(best_checkpoint_path,
                                        {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))



def run_train(train, a, h, dataset_dir, world_size):
    mp.spawn(train,
             args=(a, h, dataset_dir, world_size),
             nprocs=world_size,
             join=True)
    



if __name__ == "__main__":
    clean_dir = "/home/local1/workspace/danukam/AMD/Data/train-clean-360/LibriSpeech/train-clean-360"
    bg_speaker_dir = "/home/local1/workspace/danukam/AMD/Data/train-clean-360/LibriSpeech/bg-speakers"
    noise_dir = "/home/local1/workspace/danukam/AMD/TSOS/DCCRN_test/data/DEMAND"
    

    # === CONFIGURATION ===
    CLEAN_DIR = clean_dir # Root folder containing subfolders of clean audio
    NOISE_DIR = noise_dir # Background noise
    SPEAKER_NOISE_DIR = bg_speaker_dir  # Speaker noise
    OUTPUT_DIR = "/home/local1/workspace/danukam/AMD/traindccrn/Training_data"
    CSV_PATH = "metadata_360.csv"
    TARGET_SR = 16000
    SNR_RANGE = [5, 10, 15, 20]
    SEED = 42
    SAMPLE_DURATION = 4  # in seconds
    
    generate_dataset = False
    print("Generate dataset: ", generate_dataset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate the noisy dataset
    if generate_dataset:
        generate_noisy_dataset()

    n_gpus = 1
    print('number of GPU is using: ', n_gpus)
    
    # You can check for available GPUs like this:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs. Running demo on {n_gpus} GPUs.")
        # Ensure we don't try to use more GPUs than are available
        world_size = min(n_gpus, 2) 
    else:
        # Fallback to CPU if no GPUs are found
        print("No GPU found. Running DDP on CPU is not the primary use case and may be slow.")
        world_size = 1 # Or 1, for a simple check
    
    print('Training started...')
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='VoiceBank+DEMAND/wavs_clean/clean_trainset_wav')
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/wavs_noisy/noisy_trainset_wav')
    parser.add_argument('--input_training_file', default='VoiceBank+DEMAND/training.txt')
    parser.add_argument('--input_validation_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--checkpoint_path', default='cp_model')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    run_train(train, a, h, OUTPUT_DIR, world_size)