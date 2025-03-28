import torch
import config
import torchvision.utils as vutils
import os

def generate(generator, save_dir="generated_images", num_images=10):
    generator.eval()

    with torch.no_grad():

        for i in range(num_images):
            
            noise = torch.randn(1, config.noise_size, 1, 1, device=config.device)
            image = generator(noise).cpu()
            image = (image + 1) / 2  # Normalize to [0,1]
            

            index = 1
            while True:
                filename = f"image_{index:04d}.png"  # Format as image_001.png, image_002.png, etc.
                filepath = os.path.join(save_dir, filename)
                
                if not os.path.exists(filepath):
                    break  # Stop when an unused filename is found
                index += 1
                
            filename = f"{save_dir}/image_{index:04d}.png"

            vutils.save_image(image, filename, normalize=True)
            print(f"Saved image: {filename}")


