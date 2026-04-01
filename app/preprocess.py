import torch
from torchvision import transforms
from PIL import Image
import io

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes → normalized tensor ready for model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)   # Add batch dimension
    return tensor