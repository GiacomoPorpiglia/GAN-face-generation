import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from zipfile import ZipFile
from io import BytesIO



import mxnet as mx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RecordDataset(Dataset):
    def __init__(self, rec_file, transform=None):
        self.dataset = mx.image.ImageRecordDataset(rec_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert MXNet NDArray to NumPy
        img = img.asnumpy()

        # Convert to PIL Image for torchvision transforms
        img = transforms.functional.to_pil_image(img)

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        return img, label









class ZipImageDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.image_files = []
        with ZipFile(zip_path, 'r') as zip_file:
            self.image_files = zip_file.namelist()

    def __getitem__(self, idx):
        with ZipFile(self.zip_path, 'r') as zip_file:
            img_file = zip_file.open(self.image_files[idx])
            img = Image.open(img_file).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.image_files)



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
