import torch
from torchvision import transforms
from PIL import Image
import io

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def preprocess_image(image_bytes: bytes):
    return transform(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)
