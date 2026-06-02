import cv2
import numpy as np


def extract(image: np.ndarray) -> np.ndarray:
    """RGB color histogram, 8 bins per channel, L1-normalized."""
    bins = 8
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([hist_r, hist_g, hist_b])
    s = feat.sum()
    return feat / s if s > 0 else feat
