import os
pip3 install torchvision 
import torch
import torchvision
pip3 install faiss-cpu
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
import streamlit as st


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

st.title("🔍 Image Similarity Finder")

# Upload reference images
ref_images = st.file_uploader("Upload reference images", type=["jpg", "png"], accept_multiple_files=True)

# Upload query image
query_img_file = st.file_uploader("Upload a query image", type=["jpg", "png"])

if ref_images and query_img_file:
    features_list = []
    filenames = []

    st.write("Extracting features from reference images...")
    for uploaded_file in ref_images:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            features = extract_features_from_image(img)
            features_list.append(features)
            filenames.append((uploaded_file.name, img))
        except Exception as e:
            st.warning(f"Skipping {uploaded_file.name}: {e}")

    if len(features_list) == 0:
        st.error("No valid reference images found.")
    else:
        query_img = Image.open(query_img_file).convert("RGB")
        query_features = extract_features_from_image(query_img).astype('float32')
        features_matrix = np.array(features_list).astype('float32')

        index = faiss.IndexFlatL2(features_matrix.shape[1])
        index.add(features_matrix)
        D, I = index.search(np.array([query_features]), k=min(5, len(filenames)))

        st.image(query_img, caption="Query Image", width=200)
        st.write("### Top Similar Images:")
        for idx, dist in zip(I[0], D[0]):
            name, matched_img = filenames[idx]
            st.image(matched_img, caption=f"{name} — Distance: {dist:.2f}", width=200)
