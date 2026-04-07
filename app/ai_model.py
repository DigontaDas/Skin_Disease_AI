import torch
import torch.nn as nn
from torchvision import models
import json, os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "resnet50_skin.pth")
LABELS_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")

def build_model(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(m.fc.in_features, 512),
        nn.ReLU(),        nn.Dropout(0.3), nn.Linear(512, num_classes))
    return m

def load_model():
    with open(LABELS_PATH) as f: class_names = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    m    = build_model(ckpt.get("num_classes", len(class_names)))
    m.load_state_dict(ckpt["model_state"])
    m.to(device).eval()
    print(f"[Model] ResNet50 loaded  val_acc={ckpt.get('val_acc',0):.4f}")
    return m, class_names, device

def predict(tensor, model, class_names, device):
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1)[0]
        top5  = torch.topk(probs, k=min(5, len(class_names)))
    return (class_names[top5.indices[0].item()],
            top5.values[0].item(),
            [{"disease": class_names[top5.indices[i].item()],
              "confidence": round(top5.values[i].item(), 4)} for i in range(top5.indices.shape[0])])
