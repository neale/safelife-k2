import numpy as np
import torch
import torch.utils.data.Dataset as Dataset

class SafeLifeDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.istensor(idx):
            idx = idx.tolist()




