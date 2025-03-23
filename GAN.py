import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, size='small'):
        super(Generator, self).__init__()

        if size=='small':
            self.expansion = 128
            self.reduction1 = 64
            self.reduction2 = 32
        else:
            self.expansion = 192
            self.reduction1 = 96
            self.reduction2 = 48

        self.generate = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3*8*8, self.expansion * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (self.expansion, 8, 8)), # [B, expansion, 8, 8]

            nn.Upsample(scale_factor=2), # [B, expansion, 16, 16]
            nn.Conv2d(self.expansion, self.expansion, kernel_size=3, stride=1, padding=1), #
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2), # [B, expansion, 32, 32]
            nn.Conv2d(self.expansion, self.reduction1, kernel_size=3, stride=1, padding=1), #
            nn.BatchNorm2d(self.reduction1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2), # [B, expansion, 64, 64]
            nn.Conv2d(self.reduction1, self.reduction2, kernel_size=3, stride=1, padding=1), #
            nn.BatchNorm2d(self.reduction2),
            nn.ReLU(),

            nn.Conv2d(self.reduction2, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, z):
        # z: [B, 3, 8, 8]
        x = self.generate(z)
        return x



class Discriminator(nn.Module):
    def __init__(self, size='small'):
        super(Discriminator, self).__init__()

        if size == 'small':
            self.expansion1 = 48
            self.expansion2 = 32
            self.expansion3 = 16
        else:
            self.expansion1 = 128
            self.expansion2 = 64
            self.expansion3 = 32

        self.discriminate = nn.Sequential(
            nn.Conv2d(3, self.expansion1, kernel_size=3, stride=1, padding=1), # [B, expansion, 64, 64]
            nn.MaxPool2d(kernel_size=2),  # [B, 64, 32, 32]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.expansion1, self.expansion2, kernel_size=3, stride=1, padding=1),  # [B, 128, 32, 32]
            nn.MaxPool2d(kernel_size=2),  # [B, 128, 16, 16]
            nn.LeakyReLU(inplace=True),
            nn.Flatten(start_dim=1), # [B, 128*16*16]
            nn.Linear(self.expansion2*16*16, self.expansion3*16*16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.expansion3*16*16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.discriminate(x)
        return x # [B, 1]