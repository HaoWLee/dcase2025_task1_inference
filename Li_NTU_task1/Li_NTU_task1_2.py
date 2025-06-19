import torch
import torchaudio
from torch import Tensor
from typing import List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import os
import numpy as np
import librosa
from Li_NTU_task1.models.cp_mobile_clean import get_model
from Li_NTU_task1.models.mel import AugmentMelSTFT
from Li_NTU_task1 import ckpts
import importlib.resources as pkg_resources

class Config:
    sample_rate = 32000
    window_size = 3072
    hop_size = 500
    n_fft = 4096
    n_mels = 256
    f_min = 0
    f_max = None
    freqm = 48
    timem = 0
    fmin_aug_range = 1
    fmax_aug_range = 1000
    n_classes = 10
    in_channels = 1
    base_channels = 32
    channels_multiplier = 1.8
    expansion_rate = 2.1
 # Device IDs
    device_ids = ['a', 'b', 'c', 's1', 's2', 's3']

class StudentInference(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
            sr=config.sample_rate,
            win_length=config.window_size,
            hopsize=config.hop_size,
            n_fft=config.n_fft,
            freqm=config.freqm,
            timem=config.timem,
            fmin=config.f_min,
            fmax=config.f_max,
            fmin_aug_range = config.fmin_aug_range,
            fmax_aug_range = config.fmax_aug_range
        )

        self.model = get_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate
        )
        self.model.eval()
        self.model.half()
        self.class_order = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]

    def preprocess(self, waveform: Tensor) -> Tensor:
        old_shape = waveform.size()  # [B, 1, T]
        waveform = waveform.reshape(-1, old_shape[2])  # [B, T]
        mel = self.mel(waveform)  # [B, n_mels, T']
        mel = mel.reshape(old_shape[0], old_shape[1], mel.shape[1], mel.shape[2])
        return mel.half()

    def forward(self, waveform: Tensor) -> Tensor:
        with torch.no_grad():
            mel = self.preprocess(waveform)
            logits = self.model(mel)
        return logits


def load_model(model_file_path: Optional[str] = None) -> StudentInference:
    config = Config()
    model = StudentInference(config)
    if model_file_path is None:
        with pkg_resources.path(ckpts, "general_Li_NTU_sub2.ckpt") as ckpt_path:
            model_file_path = str(ckpt_path)
    ckpt = torch.load(model_file_path, map_location="cpu")

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    clean_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    model.model.load_state_dict(clean_dict, strict=False)
    model.eval()
    model.model.half()  
    return model

def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: StudentInference,
    num_workers: int = 16,
    batch_size: int = 256
) -> List[Tensor]:
    """
    Load and preprocess audio files in parallel with batch-wise resampling and STFT.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for STFT/mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    assert len(file_paths) == len(device_ids)

    device = next(model.parameters()).device
    target_sr = model.config.sample_rate

    # def _load(indexed_path):
    #     path, idx = indexed_path
    #     # sr = 32000
    #     waveform, sr = torchaudio.load(path)
    #     # sig, _ = librosa.load((path), sr=sr, mono=True)
    #     # waveform = torch.from_numpy(sig[np.newaxis])
    #     # [channels, samples]
    #     waveform = waveform.mean(dim=0)                   # mono: [samples]
    #
    #     return idx, waveform, sr
    def _load(indexed_path):
        path, idx = indexed_path
        sig, sr = librosa.load(path, sr=target_sr, mono=True)  # 保留原采样率
        waveform = torch.from_numpy(sig)  # [samples]
        return idx, waveform, sr

    # Step 1: Load & mono-mix in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        loaded = list(tqdm(
            exe.map(_load, zip(file_paths, range(len(file_paths)))),
            total=len(file_paths),
            desc="Loading files"
        ))
    # loaded = list of (original_idx, waveform, sample_rate)

    # Step 2: Group by original sample rate to batch resampling
    sr_groups = defaultdict(list)
    for idx, waveform, sr in loaded:
        sr_groups[sr].append((idx, waveform))

    inputs: List[Tensor] = [None] * len(file_paths)  # final output buffer

    # Step 3: Batch resample and preprocess each group
    print("Batched Resampling ...")
    for sr, items in sr_groups.items():
        print(f"Processing SR={sr} with {len(items)} files")
        for i in tqdm(range(0, len(items), batch_size), desc=f"SR={sr}", leave=False):
            chunk = items[i:i + batch_size]
            indices, waveforms = zip(*chunk)

            # Pad waveforms to the same length → [max_len, B]
            padded = pad_sequence(waveforms, batch_first=False)

            # Reshape → [B, 1, samples]
            batch_wave = padded.transpose(0, 1).unsqueeze(1).to(device)

            # Resample if needed
            # if sr != target_sr:
            #     batch_wave = torchaudio.functional.resample(
            #         batch_wave, orig_freq=sr, new_freq=target_sr
            #     )

            # STFT + mel computation for batch
            with torch.no_grad():
                mel_batch = model.preprocess(batch_wave)  # [B, 1, n_mels, T]

            # Store each preprocessed mel in original order
            for mel, idx in zip(mel_batch, indices):
                inputs[idx] = mel.unsqueeze(0).cpu()  # [1, 1, n_mels, T]

    return inputs
def get_model_for_device(model: StudentInference, device_id: str) -> torch.nn.Module:
    """
    For single-model submission: always return the same model regardless of device ID.
    """
    return model.model  # 所有设备共用一个模型实例
def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using device-specific models.

    Files are grouped by device ID and processed in batches.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.ckpt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

Returns:
    A tuple (logits, class_order)
    - logits: List of tensors, one per file, each of shape [n_classes]
    - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path).to(device)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

    # Step 2: Group by device ID, squeeze each mel to [n_mels, T]
    groups = defaultdict(list)
    for idx, (mel, dev) in enumerate(zip(inputs, device_ids)):
        mel_squeezed = mel.squeeze(0).squeeze(0)  # → [n_mels, T]
        groups[dev].append((mel_squeezed, idx))

    outputs = [None] * len(inputs)  # Placeholder for final predictions

    # Step 3: For each device, batch and infer
    for dev, items in tqdm(groups.items(), desc="Batched inference"):
        submodel = get_model_for_device(model, dev)

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            mels, indices = zip(*chunk)  # List of [n_mels, T_i]

            # Pad to same length → [max_T, batch, n_mels]
            # This is to potentially also support files of varying length. However,
            # all files in the TAU dataset are exactly one second in length.
            padded = pad_sequence([m.T for m in mels], batch_first=False)

            # Reshape → [batch, 1, n_mels, max_T]
            batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)

            with torch.no_grad():
                logits = submodel(batch).cpu()  # [B, n_classes]

            # Scatter outputs back in original file order
            for logit, idx in zip(logits, indices):
                outputs[idx] = logit

    return outputs, model.class_order
