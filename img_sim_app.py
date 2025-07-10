import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
import matplotlib.pyplot as plt

# --- Model setup ---
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

preprocess = weights.transforms()

def extract_features_from_image(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().numpy()

# --- Load reference images from local folder ---
reference_folder = input("Enter path to folder of reference images: ").strip()
if not os.path.exists(reference_folder):
    print(f"Error: Folder not found at '{reference_folder}'")
    exit()

features_list = []
image_paths = []

print("Loading reference images and extracting features...")
for filename in os.listdir(reference_folder):
    filepath = os.path.join(reference_folder, filename)
    try:
        image = Image.open(filepath).convert("RGB")
        features = extract_features_from_image(image)
        features_list.append(features)
        image_paths.append(filepath)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

if len(image_paths) == 0:
    print("Error: No valid reference images found in the folder.")
    exit()

print(f"Loaded {len(image_paths)} reference images.")

features_matrix = np.array(features_list).astype('float32')
index = faiss.IndexFlatL2(features_matrix.shape[1])
index.add(features_matrix)

# --- Run similarity search ---
query_path = input("Enter path to your query image: ").strip()
if not os.path.exists(query_path):
    print(f"Error: File not found at '{query_path}'")
    exit()

query_img = Image.open(query_path).convert("RGB")
query_features = extract_features_from_image(query_img).astype('float32')
D, I = index.search(np.array([query_features]), k=5)

print("\nTop 5 similar images:")
for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
    print(f"{rank}. {image_paths[idx]}\n   Distance: {dist:.2f}\n")

# --- Display the results side-by-side ---
fig, axes = plt.subplots(1, 6, figsize=(18, 4))
axes[0].imshow(query_img)
axes[0].set_title("Query Image")
axes[0].axis("off")

for i, idx in enumerate(I[0]):
    sim_img = Image.open(image_paths[idx]).convert("RGB")
    axes[i + 1].imshow(sim_img)
    axes[i + 1].set_title(f"Match {i + 1}\nDist: {D[0][i]:.2f}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()
