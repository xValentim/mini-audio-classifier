import torch
import numpy as np 
import pandas as pd

def pad_or_truncate_waveform(waveform, max_length=16000):
    num_samples = waveform.shape[-1]
    if num_samples < max_length:
        # Padding
        pad_amount = max_length - num_samples
        # Padding 
        waveform_padded = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        # Trunc
        waveform_padded = waveform[..., :max_length]
    return waveform_padded

def compute_stft_real(waveform, n_fft=400, hop_length=160, win_length=400):
    waveform = pad_or_truncate_waveform(waveform)
    window = torch.hann_window(win_length)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window,
                      return_complex=True)  
    real_part = stft.real
    return real_part

def prepare_features(dataset, max_frames):
    features = []
    labels = []
    for waveform, sample_rate, label in dataset:
        real_part = compute_stft_real(waveform)
        # print(real_part.shape)
        # real_part shape: (n_freq, n_frames)
        n_freq, n_frames = real_part.shape[-2], real_part.shape[-1]
        # Padding ou truncamento
        if n_frames < max_frames:
            # Pad 
            pad_amount = max_frames - n_frames
            real_part_padded = torch.nn.functional.pad(real_part, (0, pad_amount))
        else:
            # Trunc
            real_part_padded = real_part[..., :max_frames]
        feature_vector = real_part_padded 
        features.append(feature_vector.numpy())
        labels.append(label)
    return np.array(features), np.array(labels)