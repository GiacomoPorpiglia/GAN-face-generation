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
    def __init__(self, rec_file, idx_file, transform=None):
        self.recordio_reader = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
        self.transform = transform
        self.indices = []
        with open(idx_file, "r") as f:
            for line in f:
                index = int(line.strip().split('\t')[0])  # Extract index from the index file
                self.indices.append(index)
        self.num_samples = len(self.indices)              # Store the number of images

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        record = self.recordio_reader.read_idx(self.indices[idx])
        
        if record is None:
            return None  # Skip invalid records
    
        try:
            header, img = mx.recordio.unpack(record)
    
            if img is None or len(img) == 0:
                return None  # Skip corrupt images
    
            img = mx.image.imdecode(img).asnumpy()  # Convert MXNet image to NumPy
            img = Image.fromarray(img)              # Convert to PIL image
    
            if self.transform:
                img = self.transform(img)
                
            return img
    
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None

