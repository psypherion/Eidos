import os
import numpy as np
from PIL import Image
import pandas as pd
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
images_dir = 'kaggle/input/flickr30k/Images/flickr30k_images'                # Directory containing your images (e.g., '1000092795.jpg')
annotations_file = 'kaggle/input/flickr30k/captions.csv'     # Your image-caption CSV file
fourier_out_dir = 'fourier_transforms/'  # Output directory for .npy transforms

os.makedirs(fourier_out_dir, exist_ok=True)

# --- LOAD CSV ---
data = pd.read_csv(annotations_file)
# Get unique image filenames
images = sorted(set(data['image']))

# --- FOURIER TRANSFORM FUNCTION ---
def image_to_fourier(path):
    img = Image.open(path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    fft2 = np.fft.fft2(img_array)
    fft2_shift = np.fft.fftshift(fft2)
    return fft2_shift

# --- PROCESS IMAGES ---
for img_file in tqdm(images):
    img_path = os.path.join(images_dir, img_file)
    try:
        fft = image_to_fourier(img_path)
        out_path = os.path.join(fourier_out_dir, img_file.replace('.jpg', '_fft.npy'))
        np.save(out_path, fft)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("Done. All images processed and Fourier transformed.")
