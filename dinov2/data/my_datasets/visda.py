import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

# import rasterio
# from rasterio import logging

# log = logging.getLogger()
# log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None


class Visda2017(Dataset):
    def __init__(self, file_path, path_prefix="", transform=None, target_transform=None, split="train"):
        df = pd.read_csv(file_path, delimiter=" ")
        self.image_paths = [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "train" in path]
        if split == "train+val":
            print("Pre-training on train+val")
            self.image_paths += [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "validation" in path]
        elif split == "val":
            print("Validation dataset")
            self.image_paths = [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "validation" in path]
        self.labels = df.iloc[:, 1]
        self.transform = transform
        self.n_classes = 12

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
