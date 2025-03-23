import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, size='small'):
        super(Generator, self).__init__()

        if size=='small':
            self.expansion1 = 16
            self.expansion2 = 32
            self.expansion3 = 64
        else:
            self.expansion1 = 32
            self.expansion2 = 48
            self.expansion3 = 64

        self.generate = nn.Sequential(
            nn.ConvTranspose2d(3, self.expansion1, kernel_size=4, stride=2, padding=1), # [B, expansion1, 16, 16]
            nn.BatchNorm2d(self.expansion1),
            nn.ReLU(),

            nn.ConvTranspose2d(self.expansion1, self.expansion2, kernel_size=4, stride=2, padding=1),  # [B, self.expansion2, 32, 32]
            nn.BatchNorm2d(self.expansion2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(self.expansion2, self.expansion3, kernel_size=4, stride=2, padding=1),  # [B, self.expansion3, 32, 32]
            nn.BatchNorm2d(self.expansion3),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(self.expansion3, 3, kernel_size=3, stride=1, padding=1),  # [B, 3, 64, 64]
            nn.Tanh()  # Output in range [-1, 1]
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
            nn.ReLU(inplace=True),
            nn.Conv2d(self.expansion1, self.expansion2, kernel_size=3, stride=1, padding=1),  # [B, 128, 32, 32]
            nn.MaxPool2d(kernel_size=2),  # [B, 128, 16, 16]
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1), # [B, 128*16*16]
            nn.Linear(self.expansion2*16*16, self.expansion3*16*16),
            nn.ReLU(inplace=True),
            nn.Linear(self.expansion3*16*16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.discriminate(x)
        return x # [B, 1]