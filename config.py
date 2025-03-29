import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for training and evaluation modes")
    parser.add_argument("-mode", type=str, choices=["train", "generate", "interpolate"], required=True, help="Mode of operation: 'train', 'generate' or 'interpolate'")
    
    # Additional arguments for training mode
    parser.add_argument("-num_epochs", type=int, default=10, help="Number of epochs for training (only applicable in train mode)")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size for training (only applicable in train mode)")
    parser.add_argument("-learning_rate", type=float, default=2e-4, help="Learning rate for training (only applicable in train mode)")

    parser.add_argument("-gen_path", type=str, default="./generator.pth.tar", help="Path to load the generator model. Must point to a .pth.par file (default: ./generator.pth.tar)")

    parser.add_argument("-disc_path", type=str, default="./discriminator.pth.tar", help="Path to load the discriminator model. Must point to a .pth.par file (default: ./discriminator.pth.tar)")

    parser.add_argument("-model_size", type=str, default="big", required=True, help="Size of the model: 'small', 'medium' or 'big'")

    parser.add_argument("-num_images", type=int, default=10, help="Number of images to generate. (only applicable in generate mode)")

    args = parser.parse_args()
    
    if args.mode == "train":
        print("****************")
        print(f"\tTraining mode selected with {args.num_epochs} epochs and {args.learning_rate} learning rate.")
        print(f"\tPath of the generator model: {args.gen_path}")
        print(f"\tPath of the discriminator model: {args.disc_path}")
        print("****************")
    else:
        print("****************")
        print("\tEvaluation mode selected.")
        print(f"\tPath of the generator model to load: {args.gen_path}")
        print("****************")
    
    return args


noise_size = 128
num_channels = 3
image_size = 128
lambda_gp = 10  # Weight for gradient penalty

device = "cuda" if torch.cuda.is_available() else "cpu"