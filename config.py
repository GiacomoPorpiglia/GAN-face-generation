import torch

batch_size = 32
noise_size = 8
num_channels = 3
output_size = 64
input_size = 64
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"