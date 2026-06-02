"""
PCA + whitening + L2-norm transform for DL descriptors.

PCA models are pre-fitted by scripts/apply_pca.py and stored as
indexes/pca_{name}.npz (mean, components, whitening_scale).

Use transform_query() to project a raw query embedding into the same
reduced space as the stored index before computing similarities.
"""
import os
import numpy as np

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

# Recommended reduced dimensions (from pca_recommendations_report.md)
REDUCED_DIMS = {
    "dinov2_supcon":     13,
    "dinov2_zeroshot":   13,
    "mobilenet_arcface": 49,
    "vit_b16_zeroshot":  230,
    "mobilenet_zeroshot":257,
    "resnet50_zeroshot": 256,
}

EPS = 1e-3

_model_cache: dict = {}


def _pca_path(name: str) -> str:
    return os.path.join(INDEX_DIR, f"pca_{name}.npz")


def _load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    path = _pca_path(name)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    model = (data["mean"], data["components"], data["whitening_scale"])
    _model_cache[name] = model
    return model


def has_pca(name: str) -> bool:
    return os.path.exists(_pca_path(name))


def transform_query(name: str, vec: np.ndarray) -> np.ndarray:
    """
    Project a raw query vector into the PCA-whitened space.
    Returns L2-normalised reduced vector, or the original vec if no PCA model exists.
    """
    model = _load_model(name)
    if model is None:
        return vec
    mean, components, whitening_scale = model
    v      = (vec.astype(np.float32) - mean) @ components.T
    v      = v * whitening_scale
    norm   = np.linalg.norm(v)
    return v / norm if norm > 0 else v
