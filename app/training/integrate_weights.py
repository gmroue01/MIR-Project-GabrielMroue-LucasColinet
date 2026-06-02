"""
After training on Colab, download the .pth files to models/ and run this script.
It re-indexes all fine-tuned descriptors and updates indexes/.

Usage:
    py -3 -m app.training.integrate_weights --models mobilenetv2 resnet50 vit_base dinov2
    py -3 -m app.training.integrate_weights --models mobilenetv2   # single model
"""
import os
import sys
import time
import argparse
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.training.models import MetricModel, load_for_inference, MODEL_CONFIGS
from app.indexer import get_class_from_filename

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
INDEX_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "indexes")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def reindex_model(model_name: str):
    weights_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(weights_path):
        print(f"[{model_name}] Weights not found at {weights_path}. Skipping.")
        return

    print(f"\n[{model_name}] Loading fine-tuned weights from {weights_path}")
    model = load_for_inference(model_name, weights_path)
    img_size = model.img_size  # read from checkpoint, not from MODEL_CONFIGS default
    print(f"  img_size = {img_size}")

    import torchvision.transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg")))
    features = []
    extract_times = []

    for path in tqdm(image_paths, desc=f"Indexing {model_name}", ncols=80):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            features.append(np.zeros(512, dtype=np.float32))
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model(tensor).squeeze().numpy()
        extract_times.append(time.perf_counter() - t0)
        features.append(emb.astype(np.float32))

    matrix = np.array(features, dtype=np.float32)
    out_path = os.path.join(INDEX_DIR, f"{model_name}.npz")
    np.savez_compressed(out_path, features=matrix)

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    avg_time = float(np.mean(extract_times))
    print(f"  → Saved {matrix.shape} | {size_mb:.2f} MB | {avg_time*1000:.2f} ms/img")

    # Update metrics.json
    import json
    metrics_path = os.path.join(INDEX_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    metrics[model_name]["descriptor_dim"] = int(matrix.shape[1])
    metrics[model_name]["descriptor_size_mb"] = round(size_mb, 4)
    metrics[model_name]["avg_search_time_s"] = round(avg_time, 6)
    metrics[model_name]["fine_tuned"] = True

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  → metrics.json updated (fine_tuned=True)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["mobilenetv2", "resnet50", "vit_base", "dinov2"],
        default=["mobilenetv2", "resnet50", "vit_base", "dinov2"],
    )
    args = parser.parse_args()

    os.makedirs(INDEX_DIR, exist_ok=True)
    for name in args.models:
        reindex_model(name)

    print("\nDone. Restart the FastAPI server to use the new indexes.")


if __name__ == "__main__":
    main()
