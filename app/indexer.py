"""
Pre-computes all descriptors for every image in the dataset and stores them
in indexes/<descriptor_name>.npz alongside indexing metrics in indexes/metrics.json.

Run once:  python -m app.indexer
"""
import os
import time
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm

from app.descriptors import (
    color_histogram,
    sift,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

# Seuls les descripteurs classiques sont calculés ici.
# Les descripteurs deep learning (dinov2_supcon, mobilenet_arcface, etc.)
# sont des embeddings pré-calculés convertis depuis modelsV2/ via
# scripts/generate_sift_index.py et le script de conversion ad hoc.
DESCRIPTORS = {
    "color_histogram": color_histogram.extract,
    "sift":            sift.extract,
}


def get_class_from_filename(fname: str) -> str:
    """Extract class label (brandId_modelId_Brand_Model) from filename."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # Format: brandId_modelId_Brand_Model_index
    # Class = first 4 parts joined
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return base


def build_index(descriptor_name: str, extract_fn, image_paths: list) -> dict:
    features = []
    failed = 0
    search_times = []

    for path in tqdm(image_paths, desc=descriptor_name, ncols=80):
        img = cv2.imread(path)
        if img is None:
            features.append(np.zeros(1, dtype=np.float32))
            failed += 1
            continue
        t0 = time.perf_counter()
        feat = extract_fn(img)
        search_times.append(time.perf_counter() - t0)
        features.append(feat.astype(np.float32))

    return {
        "features": features,
        "avg_search_time": float(np.mean(search_times)) if search_times else 0.0,
        "failed": failed,
    }


def run():
    os.makedirs(INDEX_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images.")

    filenames = [os.path.basename(p) for p in image_paths]
    classes = [get_class_from_filename(p) for p in image_paths]

    # Save image list and classes once
    np.save(os.path.join(INDEX_DIR, "filenames.npy"), np.array(filenames))
    np.save(os.path.join(INDEX_DIR, "classes.npy"), np.array(classes))

    metrics = {}

    for name, fn in DESCRIPTORS.items():
        out_path = os.path.join(INDEX_DIR, f"{name}.npz")
        print(f"\n[{name}] Indexing...")

        t_start = time.perf_counter()
        result = build_index(name, fn, image_paths)
        indexing_time = time.perf_counter() - t_start

        features = result["features"]

        # Pad to same shape if needed (should not happen with our extractors)
        dim = max(f.shape[0] for f in features)
        matrix = np.zeros((len(features), dim), dtype=np.float32)
        for i, f in enumerate(features):
            matrix[i, : f.shape[0]] = f

        np.savez_compressed(out_path, features=matrix)

        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        metrics[name] = {
            "indexing_time_s": round(indexing_time, 3),
            "descriptor_size_mb": round(size_mb, 4),
            "avg_search_time_s": round(result["avg_search_time"], 6),
            "num_images": len(image_paths),
            "descriptor_dim": int(dim),
            "failed": result["failed"],
        }
        print(
            f"  Done. Time: {indexing_time:.1f}s | Size: {size_mb:.2f} MB | "
            f"Avg/img: {result['avg_search_time']*1000:.2f} ms"
        )

    with open(os.path.join(INDEX_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nIndexing complete. Metrics saved to indexes/metrics.json")


if __name__ == "__main__":
    run()
