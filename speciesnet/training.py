import os
from tqdm import tqdm
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.detector import SpeciesNetDetector
from splitting import extract_species_and_site_from_filename, stratified_site_split_from_folder, filter_df_with_detections
from dataloader import SpeciesImageDataset
from model import AugmentedSpeciesNet
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import wandb

# === Device Selection ===
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# === Config ===
base_dir = os.path.expanduser("~/Desktop/Kaitlyn_Catalyst/ct_classifier")
csv_path = os.path.join(base_dir, "notebooks", "full_df_filtered.csv")
target_species_txt = os.path.join(base_dir, "target_species.txt")
image_dir = os.path.join(base_dir, "datasets", "all_species_images")
classifier_model_name = os.path.expanduser("~/.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.1a/1")
num_epochs = 5
batch_size = 16

# === Initialize Weights & Biases ===
wandb.init(
    project="Species-Classification",
    name="speciesnet-v1",
    config={"epochs": num_epochs, "lr": 1e-4, "batch_size": batch_size, "model": "AugmentedSpeciesNet"}
)
wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")

print(f"Using '{device}' device", flush=True)

# === Load Data ===
full_df = pd.read_csv(csv_path)
train_df, val_df = stratified_site_split_from_folder(image_dir, test_size=0.3)

# === Detector ===
detector = SpeciesNetDetector(model_name=classifier_model_name)

# === Filter by detector and cache ===
train_filtered_path = os.path.join(base_dir, "speciesnet/train_filtered.csv")
val_filtered_path = os.path.join(base_dir, "speciesnet/val_filtered.csv")

if os.path.exists(train_filtered_path) and os.path.exists(val_filtered_path):
    print("Using cached filtered CSVs.")
    train_filtered_df = pd.read_csv(train_filtered_path)
    val_filtered_df = pd.read_csv(val_filtered_path)
else:
    print("Running detector to filter train/val...")
    train_filtered_df = filter_df_with_detections(train_df, image_dir, detector)
    val_filtered_df = filter_df_with_detections(val_df, image_dir, detector)
    train_filtered_df.to_csv(train_filtered_path, index=False)
    val_filtered_df.to_csv(val_filtered_path, index=False)
    print("Filtered train/val saved.")

# === Load classifier and wrap model ===
classifier = SpeciesNetClassifier(model_name=classifier_model_name, target_species_txt=target_species_txt)
original_outputs = len(classifier.labels)
target_labels = len(classifier.target_labels)
classifier.model = AugmentedSpeciesNet(classifier.model, original_outputs, target_labels)
classifier.model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(classifier.model.parameters(), lr=1e-4)

# === DataLoaders ===
train_dataset = SpeciesImageDataset(train_filtered_df, image_dir, classifier)
val_dataset = SpeciesImageDataset(val_filtered_df, image_dir, classifier)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# === Training Loop ===
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    classifier.model.train()
    train_loss = 0.0
    train_preds, train_trues = [], []

    for x_batch, y_batch in tqdm(train_loader, desc="Training"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = classifier.model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # === Calculate per-batch accuracy ===
        preds = torch.argmax(outputs, dim=1)
        batch_acc = torch.mean((preds == y_batch).float()).item()

        # === Print batch loss and accuracy ===
        print(f"Batch Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}")

        train_preds.extend(preds.cpu().numpy())
        train_trues.extend(y_batch.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(train_trues, train_preds)
    print(f"✅ Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # === Validation ===
    classifier.model.eval()
    val_loss = 0.0
    val_preds, val_trues = [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc="Validating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = classifier.model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            # === Calculate per-batch accuracy ===
            preds = torch.argmax(outputs, dim=1)
            batch_acc = torch.mean((preds == y_batch).float()).item()

            # === Print batch loss and accuracy ===
            print(f"Batch Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}")
            
            val_preds.extend(preds.cpu().numpy())
            val_trues.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_trues, val_preds)
    print(f"✅ Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
     
    # Load the mapping JSON file
    with open("id_to_species_full.json", "r") as f:
        id_to_species = json.load(f)

    # Build reverse map from target_label to short_name
    targetlabel_to_shortname = {
        info["target_label"]: info["short_name"].replace(" ", "_")
        for info in id_to_species.values()
    }
    
    shortnames = [targetlabel_to_shortname[label] for label in classifier.target_labels]

    # === Logging with wandb ===
    train_report = classification_report(train_trues, train_preds, target_names=shortnames, output_dict=True)
    val_report = classification_report(val_trues, val_preds, target_names=shortnames, output_dict=True)

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "train/accuracy": train_acc,
        "val/avg_loss": avg_val_loss,
        "val/accuracy": val_acc,
    })
    
    # ✅ Loop through class shortnames
    for short_name in shortnames:
        if short_name in train_report:
            wandb.log({
                f"{short_name}/train_precision": train_report[short_name]["precision"],
                f"{short_name}/train_recall": train_report[short_name]["recall"],
                f"{short_name}/train_f1": train_report[short_name]["f1-score"],
            }, step=epoch + 1)

        if short_name in val_report:
            wandb.log({
                f"{short_name}/val_precision": val_report[short_name]["precision"],
                f"{short_name}/val_recall": val_report[short_name]["recall"],
                f"{short_name}/val_f1": val_report[short_name]["f1-score"],
            }, step=epoch + 1)

wandb.finish()