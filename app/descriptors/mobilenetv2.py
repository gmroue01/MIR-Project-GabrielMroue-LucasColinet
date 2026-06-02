import numpy as np
from .deep_model import extract as _extract


def extract(image: np.ndarray) -> np.ndarray:
    return _extract(image, "mobilenetv2")
