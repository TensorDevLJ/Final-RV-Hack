# download_model.py
from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer("all-mpnet-base-v2")
model.save("./models/all-mpnet-base-v2")
print("Model downloaded and saved to ./models/all-mpnet-base-v2")
