"""Train a ResNet-18 steering model on DonkeyCar-style driving logs.

This script expects a zip archive containing:
  - catalog.csv (with cam/image_array, user/angle, user/throttle)
  - images/ (the image files referenced in the catalog)

It performs basic augmentation, trains a 2-output regression model
(steering + throttle), and saves a .pth checkpoint.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import os
import zipfile
from PIL import Image
import timm
from tqdm import tqdm
import shutil
import random

# --- CONFIGURATION ---
# Input dataset zip and extract location.
ZIP_PATH = './merged_dataset_v1.zip'  
EXTRACT_DIR = './dataset'             
OUTPUT_DIR = './output_models'        

# Training hyperparameters.
MODELS_TO_TRAIN = ['resnet18']
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 1e-4

# 1. Setup Data
# Clean and unzip the dataset to a local working directory.
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)

print("Unzipping data...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-detect dataset root (handles zips that contain a single top-level folder).
DATA_ROOT = EXTRACT_DIR
if 'merged_dataset_v1' in os.listdir(EXTRACT_DIR):
    DATA_ROOT = os.path.join(EXTRACT_DIR, 'merged_dataset_v1')

print(f"Dataset Root: {DATA_ROOT}")

# --- CUSTOM TRANSFORMS ---
# Simple salt-and-pepper noise to make the model robust to sensor artifacts.
class AddSaltPepperNoise(object):
    def __init__(self, amount=0.004):
        self.amount = amount

    def __call__(self, tensor):
        if random.random() < 0.5:
            return tensor

        # Salt
        num_salt = int(self.amount * tensor.numel() * 0.5)
        coords = [torch.randint(0, i, (num_salt,)) for i in tensor.shape]
        tensor[coords[0], coords[1], coords[2]] = 1.0

        # Pepper
        num_pepper = int(self.amount * tensor.numel() * 0.5)
        coords = [torch.randint(0, i, (num_pepper,)) for i in tensor.shape]
        tensor[coords[0], coords[1], coords[2]] = 0.0
        return tensor

# 2. Dataset Class
# Reads the catalog and loads images on-demand to keep memory usage low.
class AutonomousDataset(Dataset):
    def __init__(self, catalog_path, img_dir, transform=None):
        self.data = pd.read_csv(catalog_path)
        self.img_dir = img_dir
        self.transform = transform
        print(f"Loaded {len(self.data)} rows from CSV.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['cam/image_array']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('L')
        except:
            # Fallback if image is corrupt or unreadable.
            return self.__getitem__((idx + 1) % len(self))

        angle = float(row['user/angle'])
        throttle = float(row['user/throttle'])

        # Data augmentation: horizontal flip and steering sign inversion.
        # DonkeyCar steering is signed: left is negative, right is positive.
        # When we mirror the image, we must negate the steering angle.
        if random.random() > 0.5:
            image = TF.hflip(image)
            angle = -angle

        # The model predicts [steering, throttle] as a regression target.
        labels = torch.tensor([angle, throttle], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# 3. Preprocessing Pipeline
# Keep the model's input consistent with inference preprocessing.
train_transform = transforms.Compose([
    transforms.CenterCrop(480),
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
    ], p=0.2),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    AddSaltPepperNoise(amount=0.02),
    # Normalize grayscale to [-1, 1] so it matches inference preprocessing.
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 4. Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

csv_file_path = os.path.join(DATA_ROOT, 'catalog.csv')
img_root_path = os.path.join(DATA_ROOT, 'images')

dataset = AutonomousDataset(
    csv_file_path,
    img_root_path,
    train_transform
)

# 90/10 train/val split.
train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# 5. Training Loop
for MODEL_NAME in MODELS_TO_TRAIN:
    print(f"\n" + "="*30)
    print(f"STARTING TRAINING: {MODEL_NAME}")
    print("="*30)

    # ResNet-18 with a 1-channel input and 2 outputs (angle, throttle).
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2, in_chans=1).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")

    # 6. Save
    local_filename = f"{MODEL_NAME}_merged_csv_v1.pth"
    save_path = os.path.join(OUTPUT_DIR, local_filename)
    torch.save(model.state_dict(), save_path)

    print(f"Saved to: {save_path}")

print("DONE.")
