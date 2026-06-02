"""
Shared loader for deep learning descriptors.
Loads fine-tuned MetricModel if weights exist in models/, else falls back to pretrained timm backbone.
"""
import os
import numpy as np
import torch
import timm
from PIL import Image
import cv2

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

_models: dict = {}

CONFIGS = {
    "mobilenetv2": {"timm_name": "mobilenetv2_100.ra_in1k",              "img_size": 224},
    "resnet50":    {"timm_name": "resnet50.a1_in1k",                      "img_size": 224},
    "vit_base":    {"timm_name": "vit_base_patch16_224.augreg_in1k",      "img_size": 224},
    "dinov2":      {"timm_name": "vit_small_patch14_dinov2.lvd142m",      "img_size": 518},
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _load_finetuned(name: str):
    """Load MetricModel from models/<name>_best.pth if available."""
    weights_path = os.path.join(MODELS_DIR, f"{name}_best.pth")
    if not os.path.exists(weights_path):
        return None
    try:
        from app.training.models import load_for_inference
        model = load_for_inference(name, weights_path)
        print(f"[deep_model] Loaded fine-tuned weights for {name}")
        return model
    except Exception as e:
        print(f"[deep_model] Warning: could not load fine-tuned {name}: {e}")
        return None


def _load_pretrained(name: str):
    cfg = CONFIGS[name]
    model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=0)
    model.eval().float()
    return model


def _get_model(name: str):
    if name not in _models:
        ft = _load_finetuned(name)
        _models[name] = ft if ft is not None else _load_pretrained(name)
    return _models[name]


def _preprocess(image: np.ndarray, img_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)


def extract(image: np.ndarray, model_name: str) -> np.ndarray:
    model = _get_model(model_name)
    # Fine-tuned MetricModel exposes .img_size from the checkpoint; pretrained timm models don't.
    img_size = getattr(model, "img_size", None) or CONFIGS[model_name]["img_size"]
    tensor = _preprocess(image, img_size)
    with torch.no_grad():
        feat = model(tensor).squeeze().float().numpy()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat
