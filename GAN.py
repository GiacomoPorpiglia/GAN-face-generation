import torch
import torch.nn as nn
import torch.nn.functional as F
import config



class Generator(nn.Module):
    def __init__(self, size='small', image_size = 128):
        super(Generator, self).__init__()
        if size == 'small':
            self.features = 24
        elif size == 'medium':
            self.features = 32
        elif size == 'big':
            self.features = 48

        self.generate = nn.Sequential(
            self._block(config.noise_size, self.features*16, 4, 1, 0), # [B, features*16, 4, 4]
            self._block(self.features*16, self.features*8, 4, 2, 1), # [B, features*8, 8, 8]
            self._block(self.features*8, self.features*4, 4, 2, 1), # [B, features*4, 16, 16]
            self._block(self.features*4, self.features*2, 4, 2, 1), # [B, features*2, 32, 32]
            nn.ConvTranspose2d(self.features*2, 3, kernel_size=4, stride=2, padding=1), # [B, 3, 64, 64]
            nn.Tanh()
        )

    def _block(self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )



    def forward(self, z):
        # z: [B, 3, 8, 8]
        return self.generate(z)


class Discriminator(nn.Module):
    def __init__(self, size='small', image_size=128):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        if size == 'small':
            self.features = 24
        elif size == 'medium':
            self.features = 32
        elif size == 'big':
            self.features = 48

        self.discriminate = nn.Sequential(

            nn.Conv2d(3, self.features, kernel_size=4, stride=2, padding=1), # [B, features, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),

            self._block(self.features, self.features*2, 4, 2, 1),   # [B, features*2, 32, 32]
            self._block(self.features*2, self.features*4, 4, 2, 1), # [B, features*4, 16, 16]
            self._block(self.features*4, self.features*8, 4, 2, 1), # [B, features*8, 8, 8]
            self._block(self.features*8, self.features*16, 4, 2, 1), # [B, features*16, 4, 4]
            nn.Conv2d(self.features*8, 3, kernel_size=4, stride=2, padding=0), # [B, features*16, 1, 1]
            nn.Sigmoid()

        )


    def _block (self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )


    def forward(self, x):
        x = self.discriminate(x)
        return x # [B, 1]