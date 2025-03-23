import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # Get list of image file paths
        self.image_files = [
            os.path.join(root, filename)
            for root, _, files in os.walk(folder_path)  # Recursively walk through the folder
            for filename in files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png'))  # Filter by image extensions
        ]
        
        print(f"Number of images found: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        img = Image.open(img_path)
        
        if img.mode in ['P', 'LA', 'RGBA']:
            img = img.convert("RGBA")  # Convert to RGBA to preserve transparency
            img = img.convert("RGB")  # Then convert it to RGB (removes alpha channel)
        else:
            img = img.convert("RGB")  # If not in a transparency mode, just convert to RGB

        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)

        return img
