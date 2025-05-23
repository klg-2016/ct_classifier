import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.detector import SpeciesNetDetector
from dataloader import GoogleDriveAPIDataset, stratified_site_split
from model import AugmentedSpeciesNet
import wandb

# === Device Selection ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# === Config ===
# Root directory (automatically resolves to the correct home/Desktop path)
base_dir = os.path.expanduser("~/Desktop/Kaitlyn_Catalyst/ct_classifier")

# Construct all paths relative to the base directory
csv_path = os.path.join(base_dir, "notebooks", "full_df_filtered.csv")
label_mapping_path = os.path.join(base_dir, "species_with_label_matching.csv")
target_species_txt = os.path.join(base_dir, "target_species.txt")
credentials_path = os.path.expanduser("~/Desktop/Kaitlyn_Catalyst/credentials.json")

# Model cache path stays the same (already uses home directory notation)
classifier_model_name = os.path.expanduser("~/.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.1a/1")

num_epochs = 5

# === Initialize Weights & Biases ===
wandb.init(
    project="Species-Classification",
    name="speciesnet-v1",
    config={
        "epochs": num_epochs,
        "lr": 1e-4,
        "model": "AugmentedSpeciesNet + SpeciesNetClassifier"
    }
)

print(f"Using '{device}' device", flush=True)

# === Load and Prepare Data ===
full_df = pd.read_csv(csv_path)
label_map_df = pd.read_csv(label_mapping_path)

full_df["species_cleaned"] = full_df["species"].str.strip().str.lower()
label_map_df["CommName"] = label_map_df["CommName"].str.strip().str.lower()
species_to_uuid = dict(zip(label_map_df["CommName"], label_map_df["Matching_JSON_Entries"]))
full_df["ground_truth_uuid"] = full_df["species_cleaned"].map(species_to_uuid)

# === Load Base Classifier ===
classifier = SpeciesNetClassifier(model_name=classifier_model_name, target_species_txt=target_species_txt)
original_outputs = len(classifier.labels)

# === Add extra class index for Mongoose ===
mongoose_index = original_outputs
classifier.model = AugmentedSpeciesNet(classifier.model, original_outputs=original_outputs, extra_outputs=1)
classifier.model = classifier.model.to(device)
classifier.model.train()

# === Enable training on all layers ===
for param in classifier.model.parameters():
    param.requires_grad = True

# === Map UUIDs to Indices ===
uuid_to_idx = {uuid: idx for idx, uuid in enumerate(classifier.target_labels)}
uuid_to_idx["mongoose"] = mongoose_index  # manually add mongoose index
full_df["ground_truth_index"] = full_df["ground_truth_uuid"].map(uuid_to_idx)
full_df = full_df.dropna(subset=["ground_truth_index"])
full_df["ground_truth_index"] = full_df["ground_truth_index"].astype(int)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Split ===
train_df, val_df = stratified_site_split(full_df)
train_dataset = GoogleDriveAPIDataset(df=train_df, credentials_path=credentials_path, transform=None)
val_dataset = GoogleDriveAPIDataset(df=val_df, credentials_path=credentials_path, transform=None)

# === Load Detector ===
detector = SpeciesNetDetector(model_name=classifier_model_name)

# === Loss + Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.model.parameters(), lr=1e-4)

# === Training & Validation ===
for epoch in range(num_epochs):
    print(f"\n🌀 Epoch {epoch + 1}/{num_epochs}")
    classifier.model.train()
    train_loss = 0.0

    for i in range(len(train_dataset)):
        try:
            row = train_dataset.df.iloc[i]
            save_path = os.path.join(train_dataset.temp_dir, row['filename'])

            _ = train_dataset[i]  # download image if needed

            raw_img = Image.open(save_path).convert("RGB")
            preprocessed_image = detector.preprocess(raw_img)
            detections_result = detector.predict(filepath=save_path, img=preprocessed_image)
            detections = detections_result.get("detections", [])
            if not detections:
                print(f"🗑️ No detection — skipping {row['filename']}")
                continue

            img_tensor = transform(raw_img)
            x = img_tensor.unsqueeze(0).to(device)
            label = int(row['ground_truth_index'])
            label_tensor = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            outputs = classifier.model(x)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            wandb.log({"train/loss": loss.item()}, step=i + epoch * len(train_dataset))

        except Exception as e:
            print(f"⚠️ Training error at {row.get('filename', 'unknown')}: {e}")

    avg_train_loss = train_loss / len(train_dataset)
    print(f"✅ Avg Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    classifier.model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for j in range(len(val_dataset)):
            try:
                row = val_dataset.df.iloc[j]
                save_path = os.path.join(val_dataset.temp_dir, row['filename'])

                _ = val_dataset[j]  # download image if needed

                raw_img = Image.open(save_path).convert("RGB")
                preprocessed_image = detector.preprocess(raw_img)
                detections_result = detector.predict(filepath=save_path, img=preprocessed_image)
                detections = detections_result.get("detections", [])
                if not detections:
                    print(f"🗑️ No detection — skipping {row['filename']}")
                    continue

                img_tensor = transform(raw_img)
                x = img_tensor.unsqueeze(0).to(device)
                label = int(row['ground_truth_index'])
                label_tensor = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)

                outputs = classifier.model(x)
                loss = criterion(outputs, label_tensor)
                val_loss += loss.item()

            except Exception as e:
                print(f"⚠️ Validation error at {row.get('filename', 'unknown')}: {e}")

    avg_val_loss = val_loss / len(val_dataset)
    print(f"🔍 Avg Val Loss: {avg_val_loss:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "val/avg_loss": avg_val_loss
    })

# === Cleanup ===
train_dataset.cleanup()
val_dataset.cleanup()

# === Finish W&B ===
wandb.finish()