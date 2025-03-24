import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, size='small', image_size = 64):
        super(Generator, self).__init__()
        self.initial_size = image_size // 16
        if size == 'small':
            self.expansion = 128
            self.reduction1 = 64
            self.reduction2 = 32
            self.reduction3 = 16
        elif size == 'medium':
            self.expansion = 192
            self.reduction1 = 96
            self.reduction2 = 48
            self.reduction3 = 24
        elif size == 'big':
            self.expansion1 = 256
            self.reduction1 = 128
            self.reduction2 = 64
            self.reduction3 = 32

        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3*8*8, self.expansion * self.initial_size * self.initial_size),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (self.expansion, self.initial_size, self.initial_size)), # [B, expansion, initial_size, initial_size]

            nn.ConvTranspose2d(self.expansion, self.expansion, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.expansion),
            nn.ReLU(inplace=True),

            # nn.Upsample(scale_factor=2), # [B, reduction1, 4*initial_size, 4*initial_size]
            # nn.Conv2d(self.expansion, self.reduction1, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.expansion, self.reduction1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.reduction1),
            nn.ReLU(inplace=True),

            # nn.Upsample(scale_factor=2), # [B, reduction2, image_size, image_size]
            # nn.Conv2d(self.reduction1, self.reduction2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.reduction1, self.reduction2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.reduction2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.reduction2, self.reduction3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.reduction3),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.reduction3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )


    def forward(self, z):
        # z: [B, 3, 8, 8]

        for l in self.layers:
            z = l(z)
            print(z.shape)

        # x = self.layers(z)
        return z



class Discriminator(nn.Module):
    def __init__(self, size='small', image_size=64):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        if size == 'small':
            self.expansion1 = 48
            self.expansion2 = 32
            self.expansion3 = 16
            self.expansion4 = 16
        elif size == 'medium':
            self.expansion1 = 128
            self.expansion2 = 64
            self.expansion3 = 32
            self.expansion4 = 16
        elif size == 'big':
            self.expansion1 = 256
            self.expansion2 = 128
            self.expansion3 = 64
            self.expansion4 = 32


        self.discriminate = nn.Sequential(
            nn.Conv2d(3, self.expansion1, kernel_size=3, stride=1, padding=1), # [B, expansion1, image_size, image_size]
            nn.MaxPool2d(kernel_size=2),  # [B, expansion1, image_size/2, image_size/2]
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.expansion1, self.expansion2, kernel_size=3, stride=1, padding=1),  # [B, expansion2, image_size/2, image_size/2]
            nn.MaxPool2d(kernel_size=2),  # [B, expansion2, image_size/4, image_size/4]
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.expansion2, self.expansion3, kernel_size=3, stride=1, padding=1),  # [B, expansion3, image_size/4, image_size/4]
            nn.MaxPool2d(kernel_size=2),  # [B, expansion3, image_size/8, image_size/8]
            nn.LeakyReLU(inplace=True),

            nn.Flatten(start_dim=1), # [B, expansion3 * image_size * image_size / 64]
            nn.Linear(self.expansion3 * self.image_size * self.image_size // 64, self.expansion4 * self.image_size * self.image_size // 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.expansion4 * self.image_size * self.image_size // 64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.discriminate(x)
        return x # [B, 1]