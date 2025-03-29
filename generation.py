import torch
import config
import torchvision.utils as vutils
import os
import imageio.v2 as imageio
import moviepy.video.io.ImageSequenceClip

def generate(generator, save_dir="generated_images", num_images=10):
    generator.eval()

    with torch.no_grad():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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





def interpolate(generator, noise1, noise2, save_dir="interpolated_images", num_images=500):

    fps = 60

    generator.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(num_images):

            alpha = i / num_images
            noise = alpha * noise1 + (1-alpha) * noise2

            image = generator(noise).cpu()
            image = (image + 1) / 2  # Normalize to [0,1]
            
            filename = f"frame_{i:04d}.png"  # Format as image_001.png, image_002.png, etc.
        
            file_path = f"{save_dir}/{filename}"

            vutils.save_image(image, file_path, normalize=True)
            print(f"Saved image: {filename}")

    ### generate the video

    image_files = [os.path.join(save_dir,img)
                for img in os.listdir(save_dir)
                if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('interpolation_result.mp4')