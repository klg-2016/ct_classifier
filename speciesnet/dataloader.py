import os
import io
import time
import socket
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

class GoogleDriveAPIDataset(Dataset):
    def __init__(self, csv_path=None, df=None, credentials_path=None, transform=None, temp_dir_name="temp_images_api"):
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path, low_memory=False)
        else:
            raise ValueError("You must provide either a csv_path or a DataFrame (df).")

        self.transform = transform
        self.temp_dir = os.path.join(os.getcwd(), temp_dir_name)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Setup Drive API client
        scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
        self.service = build('drive', 'v3', credentials=self.creds)

        self.MAX_RETRIES = 3
        self.TIMEOUT_SECONDS = 60

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = self._extract_drive_id(row['filepath'])
        filename = row['filename']

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Missing or invalid filename at index {idx}")

        save_path = os.path.join(self.temp_dir, filename)

        if file_id and not os.path.exists(save_path):
            success = False
            for attempt in range(self.MAX_RETRIES):
                try:
                    print(f"📥 Downloading: {filename} (attempt {attempt + 1})")
                    request = self.service.files().get_media(fileId=file_id)
                    fh = io.FileIO(save_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request)

                    done = False
                    start_time = time.time()
                    while not done:
                        if time.time() - start_time > self.TIMEOUT_SECONDS:
                            raise TimeoutError(f"⏰ Download timed out: {filename}")
                        _, done = downloader.next_chunk()

                    success = True
                    break

                except (socket.timeout, TimeoutError) as e:
                    print(f"⏳ Timeout for {filename}: {e}")
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    time.sleep(2)

                except Exception as e:
                    print(f"⚠️ Failed to download {filename} ({file_id}): {e}")
                    break

            if not success:
                raise RuntimeError(f"❌ Failed to download {filename} after {self.MAX_RETRIES} attempts.")

        # Load image
        try:
            with Image.open(save_path) as img:
                image = img.convert("RGB")
        except UnidentifiedImageError:
            print(f"❌ Invalid image file: {save_path}")
            raise

        print(f"📸 Returning image {idx}: {filename}")

        if self.transform:
            image = self.transform(image)

        return image

    def _extract_drive_id(self, url):
        try:
            return url.split("/d/")[1].split("/")[0]
        except IndexError:
            return None

    def cleanup(self):
        print(f"🧹 Cleaning up: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        
def stratified_site_split(df, stratify_col="species", group_col="site", n_splits=2, random_state=42):
    df = df.copy()
    df["species_cleaned"] = df[stratify_col].str.strip().str.lower()
    df = df.dropna(subset=["species_cleaned", group_col])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in sgkf.split(df.index, df["species_cleaned"], df[group_col]):
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)