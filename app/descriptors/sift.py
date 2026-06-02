import cv2
import numpy as np

_sift = cv2.SIFT_create(nfeatures=500)


def extract(image: np.ndarray) -> np.ndarray:
    """SIFT: mean-pool keypoint descriptors → 128-dim L2-normalized vector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = _sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128, dtype=np.float32)
    feat = descriptors.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat
