import pandas as pd
import numpy as np
import os

annotations_file = 'kaggle/input/flickr30k/captions.csv'
fourier_dir = 'fourier_transforms/'
tfidf_vectors_file = 'caption_tfidf_vectors.npy'
output_dir = 'paired_samples/'

os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(annotations_file)
data = data.dropna(subset=['caption'])
tfidf_vectors = np.load(tfidf_vectors_file)

count = 0
for idx, row in data.iterrows():
    img_name = row['image']
    caption_vec = tfidf_vectors[idx]
    fft_path = os.path.join(fourier_dir, img_name.replace('.jpg', '_fft.npy'))
    if os.path.exists(fft_path):
        fft_array = np.load(fft_path)
        np.savez_compressed(os.path.join(output_dir, f'sample_{count}.npz'),
                            caption=caption_vec, fft=fft_array)
        count += 1
    else:
        print(f'Missing FFT for {img_name}; skipping.')

print(f'Saved {count} paired samples in {output_dir}.')
