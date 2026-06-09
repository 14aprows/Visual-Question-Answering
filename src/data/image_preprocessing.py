from pathlib import Path
from PIL import Image
from torchvision import transforms

def resolve_image_path(image_dir, image_name):
    image_dir = Path(image_dir)
    image_name = str(image_name)
    image_path = image_dir / image_name
    if image_path.exists():
        return image_path

    if not Path(image_name).suffix:
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir / f"{image_name}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Image {image_name} not found in {image_dir}")

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def get_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
