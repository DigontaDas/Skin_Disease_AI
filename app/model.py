import torch
import torch.nn as nn
from torchvision import models
import json
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "efficientnet_skin.pth")
LABELS_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model


def load_model():
    # Load class labels
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Loading on: {device}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = checkpoint.get("num_classes", len(class_names))

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"[Model] Loaded. Classes: {num_classes} | Best Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    return model, class_names, device


def predict(image_tensor, model, class_names, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        top5    = torch.topk(probs, k=min(5, len(class_names)))

    # Best prediction
    best_idx    = top5.indices[0].item()
    best_conf   = top5.values[0].item()
    disease     = class_names[best_idx]

    # Top 5 list
    top5_list = [
        {
            "disease"   : class_names[top5.indices[i].item()],
            "confidence": round(top5.values[i].item(), 4)
        }
        for i in range(top5.indices.shape[0])
    ]

    return disease, best_conf, top5_list