from kagglefetcher import fetch_dataset
from dotenv import load_dotenv
import os

load_dotenv()

kaggle_src = os.getenv("kaggle_src")
if not kaggle_src:
	raise ValueError("Environment variable 'kaggle_src' is not set or is empty.")
dataset = fetch_dataset(kaggle_src)

print(f"Dataset downloaded to: {dataset}")