import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import json
import io
import base64
import numpy as np

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model/class_names.json") as f:
    CLASS_NAMES = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 38)
)
model_path = "model/mobilenetv2_plant.pth"
if os.path.exists(model_path) and os.path.getsize(model_path) < 1000:
    import urllib.request
    print("Detected Git LFS pointer. Downloading real model weights from Hugging Face...")
    url = "https://huggingface.co/Daksh159/plant-disease-mobilenetv2/resolve/main/mobilenetv2_plant.pth"
    urllib.request.urlretrieve(url, model_path)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()

# Image preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 5)

    top_predictions = []

    for prob, idx in zip(top_probs, top_indices):
        top_predictions.append({
            "disease": CLASS_NAMES[idx.item()],
            "confidence_percent": round(float(prob) * 100, 2)
        })

    return (
        CLASS_NAMES[top_indices[0].item()],
        float(top_probs[0]),
        top_indices[0].item(),
        top_predictions
    )

def generate_heatmap(image: Image.Image, predicted_class_idx: int):
    # Disable GradCAM heatmap generation to prevent 502 Bad Gateway (OOM) 
    # on Render's 512MB RAM free tier. The frontend does not currently display it.
    return None