import torch

batch_size = 32
noise_size = 8
num_channels = 3
image_size = 128
lr = 2e-4

size = 'big'


device = "cuda" if torch.cuda.is_available() else "cpu"