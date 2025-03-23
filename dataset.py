import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO


class ZipImageDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        
        # Get list of image file names
        self.image_files = [f for f in self.zip_file.namelist() if f.lower().startswith("humans/") and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(len(self.image_files))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Read image from ZIP into memory
        with self.zip_file.open(img_name) as img_file:
            img = Image.open(BytesIO(img_file.read())).convert("RGB")  # Convert to RGB

        # Apply transforms if any
        if self.transform:
            img = self.transform(img)

        return img

