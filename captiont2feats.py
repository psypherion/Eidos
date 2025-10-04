import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- CONFIGURATION ---
annotations_file = 'kaggle/input/flickr30k/captions.csv'  # CSV as before

# --- LOAD CSV ---
data = pd.read_csv(annotations_file)
data = data.dropna(subset=['caption'])  # Drop rows with NaN captions
captions = data['caption'].tolist()  # All captions


# --- TF-IDF VECTORIZATION ---
vectorizer = TfidfVectorizer(max_features=256)  # Change 256 if you want larger/smaller vectors
tfidf_matrix = vectorizer.fit_transform(captions)  # shape [num_captions, features]
tfidf_array = np.array(tfidf_matrix.todense())

# --- SAVE VECTOR EMBEDDINGS ---
np.save('caption_tfidf_vectors.npy', tfidf_array)
print("TF-IDF vectors saved as 'caption_tfidf_vectors.npy'")
