import torch

batch_size = 64
noise_size = 128
num_channels = 3
image_size = 128
lr = 2e-4

lambda_gp = 10  # Weight for gradient penalty

size = 'big'

device = "cuda" if torch.cuda.is_available() else "cpu"