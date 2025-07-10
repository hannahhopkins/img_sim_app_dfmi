from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
import uuid

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model setup
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Clear uploads folder
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

        # Save reference images
        ref_imgs = request.files.getlist('ref_images')
        ref_paths = []
        features_list = []

        for img in ref_imgs:
            filename = f"{uuid.uuid4().hex}_{img.filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(path)
            try:
                image = Image.open(path).convert("RGB")
                features = extract_features_from_image(image)
                features_list.append(features)
                ref_paths.append(path)
            except Exception:
                continue

        # Save query image
        query = request.files['query_image']
        query_filename = f"query_{uuid.uuid4().hex}_{query.filename}"
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_filename)
        query.save(query_path)
        query_img = Image.open(query_path).convert("RGB")
        query_features = extract_features_from_image(query_img).astype('float32')

        # FAISS index
        if len(features_list) == 0:
            return render_template('index.html', error="No valid reference images.")
        features_matrix = np.array(features_list).astype('float32')
        index = faiss.IndexFlatL2(features_matrix.shape[1])
        index.add(features_matrix)
        D, I = index.search(np.array([query_features]), k=min(5, len(ref_paths)))

        # Prepare results
        matches = [(ref_paths[idx], f"{D[0][i]:.2f}") for i, idx in enumerate(I[0])]
        return render_template('index.html', query_img=query_path, matches=matches)

    return render_template('index.html')
