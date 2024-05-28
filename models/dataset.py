import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        point_cloud = data['point_cloud']
        label = data['label']
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
