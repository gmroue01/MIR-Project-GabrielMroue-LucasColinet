"""
Core search engine. Loads pre-computed indexes and performs nearest-neighbour
search with concatenated descriptors and a chosen similarity measure.
"""
import math
import os
import json
import numpy as np
from typing import List, Tuple

from app.similarity.measures import MEASURES, BATCH_MEASURES

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

_cache: dict = {}
_prepared_cache: dict = {}  # descriptor_name -> L2-normalised matrix


def _load(name: str) -> np.ndarray:
    if name not in _cache:
        path = os.path.join(INDEX_DIR, f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found: {path}. Run indexer first.")
        _cache[name] = np.load(path)["features"]
    return _cache[name]


def _load_meta() -> Tuple[np.ndarray, np.ndarray]:
    if "_meta" not in _cache:
        f_path = os.path.join(INDEX_DIR, "filenames.npy")
        c_path = os.path.join(INDEX_DIR, "classes.npy")
        if not os.path.exists(f_path) or not os.path.exists(c_path):
            return np.array([]), np.array([])
        _cache["_meta"] = (np.load(f_path), np.load(c_path))
    return _cache["_meta"]


def _normalize_l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _prepare_descriptor(name: str, measure: str = "euclidean") -> np.ndarray:
    """Load and L2-normalise the descriptor matrix. Indexes are already PCA-reduced
    offline by scripts/apply_pca.py — no runtime reduction needed."""
    if name in _prepared_cache:
        return _prepared_cache[name]

    mat = _load(name).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    _prepared_cache[name] = mat
    return mat


def build_combined_features(descriptor_names: List[str], measure: str = "euclidean") -> np.ndarray:
    """Concatenate L2-normalised feature matrices."""
    parts = [_prepare_descriptor(name) for name in descriptor_names]
    return np.concatenate(parts, axis=1)


def get_query_vector(query_idx: int, descriptor_names: List[str], measure: str = "euclidean") -> np.ndarray:
    """Return the combined feature vector for a query image from the cached matrix."""
    parts = [_prepare_descriptor(name)[query_idx] for name in descriptor_names]
    return np.concatenate(parts)


def search(
    query_idx: int,
    descriptor_names: List[str],
    measure: str,
    top_k: int = 50,
) -> List[dict]:
    """
    Returns top_k results (excluding the query itself) as a list of dicts:
    {filename, class, rank, distance}
    """
    filenames, classes = _load_meta()
    batch_fn = BATCH_MEASURES[measure]

    db_matrix = build_combined_features(descriptor_names, measure)
    query_vec = get_query_vector(query_idx, descriptor_names, measure)

    distances = batch_fn(query_vec, db_matrix).astype(np.float32)
    distances[query_idx] = np.inf
    ranked = np.argsort(distances)

    results = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        results.append({
            "filename": str(filenames[idx]),
            "class": str(classes[idx]),
            "rank": rank,
            "distance": float(distances[idx]),
            "index": int(idx),
        })
    return results


def search_with_reranking(
    query_idx: int,
    descriptor_names: List[str],
    measure: str,
    top_k: int = 50,
    pool_percent: int = 25,
) -> List[dict]:
    """
    Two-stage retrieval:
      1. Fast embedding search on pool_size = top_k + ceil(top_k * pool_percent / 100)
      2. SIFT-RANSAC reranking — returns top_k best results.
    """
    from app.descriptors.sift_ransac import rerank

    pool_size      = top_k + math.ceil(top_k * pool_percent / 100)
    pool_results   = search(query_idx, descriptor_names, measure, top_k=pool_size)
    filenames, _   = _load_meta()
    query_filename = str(filenames[query_idx])
    reranked       = rerank(query_filename, pool_results)
    return reranked[:top_k]


def get_indexing_metrics() -> dict:
    path = os.path.join(INDEX_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_all_images() -> List[dict]:
    filenames, classes = _load_meta()
    return [
        {"filename": str(filenames[i]), "class": str(classes[i]), "index": i}
        for i in range(len(filenames))
    ]
