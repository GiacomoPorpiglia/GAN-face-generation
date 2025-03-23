import os
import glob
import time
import matplotlib.pyplot as plt
import cv2

def display_image_sequence(folder_path, delay=1):
    """
    Displays a sequence of images from a specified folder.
    Assumes filenames follow the pattern 'epoch_{num}.jpg'.
    
    Parameters:
    folder_path (str): Path to the folder containing images.
    delay (int): Time (in seconds) between displaying images.
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, "epoch_*.png")), 
                         key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    if not image_files:
        print("No images found in the specified folder.")
        return
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    for img_file in image_files:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format for correct display
        
        ax.clear()
        ax.imshow(img)
        ax.set_title(os.path.basename(img_file))
        plt.pause(delay)
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Example usage
if __name__ == "__main__":
    folder_path = "generated_images"  # Change this to your folder path
    display_image_sequence(folder_path, delay=0.05)
