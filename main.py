import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import ImageFolderDataset, RecordDataset, ZipImageDataset
import config
from GAN import Generator, Discriminator
from tqdm import tqdm
import torchvision.utils as vutils
from generation import *

args = config.get_args()


checkpoint_path_gen = args.gen_path
checkpoint_path_disc = args.disc_path

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


def init_weights(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


### ---------------------------------------------------------------


transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),  # Resize images
    transforms.ToTensor(),        # Convert to tensor
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])


rec_file = "casia-webface/train.rec"
idx_file = "casia-webface/train.idx"
imgDataset = RecordDataset(rec_file, idx_file, transform=transform)
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove None values
    return torch.utils.data.default_collate(batch) if batch else None

dataloader = DataLoader(imgDataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn, num_workers=8, persistent_workers=True, pin_memory=True)

generator = Generator(args.model_size).to(config.device)
generator.apply(init_weights)

discriminator = Discriminator(args.model_size).to(config.device)
discriminator.apply(init_weights)

opt_generator = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()





def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)  # Random weight for interpolation
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),  # Create same shape tensor for gradients
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.shape[0], -1)  # Flatten gradients
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # Enforce ||grad|| = 1

    return gradient_penalty






def train(dataloader, generator, discriminator, test_noise, start_epoch = 0, num_epochs=20):
    total_batches = 0
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        last_loss_gen = 0
        last_loss_disc = 0
        batch_cnt = 0
        for batch_idx, real in enumerate(tqdm(dataloader)):

            real = real.to(config.device, non_blocking=True)

            input_noise = torch.randn(real.shape[0], config.noise_size, 1, 1, device=config.device)
            fake = generator(input_noise)

            ### discriminator loss (WGAN-GP)

            disc_real = discriminator(real)
            disc_fake = discriminator(fake)
            wasserstein_loss = torch.mean(disc_fake) - torch.mean(disc_real)  # WGAN-GP loss

            gradient_penalty = compute_gradient_penalty(discriminator, real, fake, config.device)
            # Add gradient penalty
            loss_disc = wasserstein_loss + config.lambda_gp * gradient_penalty

            discriminator.zero_grad()
            loss_disc.backward(retain_graph = True) ### retain_graph=True is to keep "fake" in the memory, because
                                                    ### we'll use it later for the generator
            opt_discriminator.step()

            ### generator loss
            output = discriminator(fake)
            loss_gen = -torch.mean(output)

            generator.zero_grad()
            loss_gen.backward()
            opt_generator.step()



            last_loss_gen += loss_gen.item()
            last_loss_disc += loss_disc.item()
            batch_cnt += 1
            total_batches+=1


            if(total_batches%1000==0):
                save_training_generation_image(generator, total_batches//1000, test_noise)


                print("Loss gen: ", last_loss_gen/(batch_cnt*args.batch_size))
                print("Loss disc: ", last_loss_disc/(batch_cnt*args.batch_size))
        
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


def save_training_generation_image(generator, epoch, fixed_noise, save_dir="training_gen_results"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        fake_image = generator(fixed_noise).cpu()  # Generate image from noise

    # Convert [-1, 1] range back to [0, 1] for saving
    fake_image = (fake_image + 1) / 2  # Normalize to [0,1]

    # Save image grid
    filename = f"{save_dir}/step_{epoch}.png"
    vutils.save_image(fake_image, filename, normalize=True, nrow=4)
    print(f"Saved image: {filename}")




if __name__ == "__main__":
    
    if os.path.exists(checkpoint_path_gen):
        load_checkpoint(generator, opt_generator, torch.load(checkpoint_path_gen))
        
    if os.path.exists(checkpoint_path_disc):
        load_checkpoint(discriminator, opt_discriminator, torch.load(checkpoint_path_disc))

    if args.mode == 'train':
        test_noise = torch.randn(16, config.noise_size, 1, 1, device=config.device)
        train(dataloader, generator, discriminator, test_noise, start_epoch=0, num_epochs=args.num_epochs)

    elif args.mode == 'generate':
        generate(generator, save_dir="generated_images", num_images=args.num_images)

    
    
    