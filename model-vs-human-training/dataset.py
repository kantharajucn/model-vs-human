import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ModelVSHumanDataset(Dataset):
    """Model vs Human dataset"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images.iloc[idx, 0]
        img = Image.open(os.path.join(img_id)).convert("RGB")
        target = torch.tensor(self.images.iloc[idx, 1], dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return img, target





