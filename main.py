import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import ImageFolderDataset
import config
from GAN import Generator, Discriminator
from tqdm import tqdm
import torchvision.utils as vutils

checkpoint_path_gen = "./checkpoint_gen.pth.tar"
checkpoint_path_disc = "./checkpoint_disc.pth.tar"

print(config.device)

def save_checkpoint(checkpoint, filename):
    print("=> saving checkpoint...")
    try:
        torch.save(checkpoint, filename)
        print("\t=> checkpoint saved!")
    except:
        print("\tX => Something went wrong in saving the network")



def load_checkpoint(model, optimizer, checkpoint):
    print("=> loading checkpoint...")
    try:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\t=> checkpoint loaded!")
    except:
        print("\tX => Something went wrong in loading the checkpoint")


### ---------------------------------------------------------------


transform = transforms.Compose([
    transforms.Resize((config.input_size, config.input_size)),  # Resize images to 64x64
    transforms.ToTensor(),        # Convert to tensor
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

folder_path = "faces/Humans/"
imgDataset = ImageFolderDataset(folder_path, transform=transform)
dataloader = DataLoader(imgDataset, batch_size=config.batch_size, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)

generator = Generator('small').to(config.device)
discriminator = Discriminator('small').to(config.device)

opt_generator = optim.Adam(generator.parameters(), lr=config.lr)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=config.lr)

criterion = nn.BCELoss()



def train(dataloader, generator, discriminator, test_noise, start_epoch = 0, num_epochs=20):

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        last_loss_gen = 0
        last_loss_disc = 0
        batch_cnt = 0
        for batch_idx, real in enumerate(tqdm(dataloader)):

            real = real.to(config.device, non_blocking=True)

            input_noise = torch.randn(config.batch_size, config.num_channels, config.noise_size, config.noise_size, device=config.device)
            fake = generator(input_noise)

            ### discriminator loss

            disc_real = discriminator(real)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward(retain_graph = True) ### retain_graph=True is to keep "fake" in the memory, because
                                                    ### we'll use it later for the generator
            opt_discriminator.step()

            ### generator loss
            output = discriminator(fake)
            loss_gen = criterion(output, torch.ones_like(output))

            generator.zero_grad()
            loss_gen.backward()
            opt_generator.step()



            last_loss_gen += loss_gen.item()
            last_loss_disc += loss_disc.item()
            batch_cnt += 1


        
        save_generated_image(generator, epoch+1, test_noise)


        print("Loss gen: ", last_loss_gen/(batch_cnt*config.batch_size))
        print("Loss disc: ", last_loss_disc/(batch_cnt*config.batch_size))

        checkpoint_gen = {
            'state_dict': generator.state_dict(),
            'optimizer': opt_generator.state_dict()
        }
        checkpoint_disc = {
            'state_dict': discriminator.state_dict(),
            'optimizer': opt_discriminator.state_dict()
        }
        save_checkpoint(checkpoint_gen, checkpoint_path_gen)    
        save_checkpoint(checkpoint_disc, checkpoint_path_disc)


def save_generated_image(generator, epoch, fixed_noise, save_dir="generated_images"):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        fake_image = generator(fixed_noise).cpu()  # Generate image from noise

    # Convert [-1, 1] range back to [0, 1] for saving
    fake_image = (fake_image + 1) / 2  # Normalize to [0,1]

    # Save image grid
    filename = f"{save_dir}/epoch_{epoch}.png"
    vutils.save_image(fake_image, filename, normalize=True, nrow=4)
    print(f"Saved image: {filename}")


if __name__ == "__main__":
  
    if os.path.exists(checkpoint_path_gen):
        load_checkpoint(generator, opt_generator, torch.load(checkpoint_path_gen))
        
    if os.path.exists(checkpoint_path_disc):
        load_checkpoint(discriminator, opt_discriminator, torch.load(checkpoint_path_disc))
    test_noise = torch.randn(16, config.num_channels, config.noise_size, config.noise_size, device=config.device)
    train(dataloader, generator, discriminator, test_noise, start_epoch=0, num_epochs=150)
    