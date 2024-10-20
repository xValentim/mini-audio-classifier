import os
import torch
import torchaudio
from torch.utils.data import Dataset

class MiniSpeechCommandsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([x for x in os.listdir(root_dir) if '.md' not in x])
        self.audio_files = []
        for label in self.classes:
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.wav')]
                self.audio_files.extend([(file, label) for file in files])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path, label = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        label_idx = self.classes.index(label)
        return waveform, sample_rate, label_idx