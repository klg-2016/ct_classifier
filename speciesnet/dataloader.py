from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re


class SpeciesImageDataset(Dataset):
    def __init__(self, df, image_dir, classifier):
        self.df = df
        self.image_dir = image_dir
        self.classifier = classifier

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        img = Image.open(img_path).convert("RGB")
        preprocessed = self.classifier.preprocess(img)
        img_tensor = torch.tensor(preprocessed.arr).permute(2, 0, 1).float()
        label = int(row["ground_truth_index"])
        return img_tensor, label



