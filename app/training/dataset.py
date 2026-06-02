"""
Car dataset for metric learning.
Returns (image_tensor, class_index) pairs. Online triplet mining is done in the loss.
"""
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T


def get_class_label(filename: str) -> str:
    """Extract class from filename: brandId_modelId_Brand_Model_index.jpg → brandId_modelId_Brand_Model"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return "_".join(parts[:4]) if len(parts) >= 4 else base


def build_splits(dataset_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """Stratified 80/20 split. Returns (train_paths, val_paths)."""
    all_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
    class_to_paths: dict = {}
    for p in all_paths:
        cls = get_class_label(p)
        class_to_paths.setdefault(cls, []).append(p)

    rng = np.random.default_rng(seed)
    train_paths, val_paths = [], []
    for cls, paths in class_to_paths.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    return train_paths, val_paths


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def make_transforms(img_size: int, augment: bool = True) -> T.Compose:
    """Build train (augment=True) or val transforms for any square resolution."""
    if augment:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


# Backward-compat aliases
TRAIN_TRANSFORMS     = make_transforms(224, augment=True)
VAL_TRANSFORMS       = make_transforms(224, augment=False)
TRAIN_TRANSFORMS_518 = make_transforms(518, augment=True)
VAL_TRANSFORMS_518   = make_transforms(518, augment=False)


class CarDataset(Dataset):
    def __init__(self, paths: list, transform=None, class_to_idx: dict | None = None):
        self.paths = paths
        self.transform = transform

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted(set(get_class_label(p) for p in paths))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.labels = [self.class_to_idx[get_class_label(p)] for p in paths]
        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class ClassBalancedSampler(Sampler):
    """
    Each batch contains exactly n_classes classes × n_samples images per class.
    Ensures every batch has enough diversity for hard mining and proxy-anchor loss.
    """
    def __init__(self, dataset: CarDataset, n_classes: int, n_samples: int):
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples

        labels = np.array(dataset.labels)
        self.class_indices = {
            c: np.where(labels == c)[0]
            for c in range(dataset.num_classes)
        }
        # Drop classes with only 1 image (can't form a valid pair)
        self.valid_classes = [c for c, idxs in self.class_indices.items() if len(idxs) >= 2]

        if len(self.valid_classes) < n_classes:
            raise ValueError(
                f"ClassBalancedSampler needs at least {n_classes} valid classes, "
                f"found {len(self.valid_classes)}"
            )

        self._n_batches = len(dataset) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self._n_batches):
            batch = []
            chosen = rng.choice(self.valid_classes, size=self.n_classes, replace=False)
            for c in chosen:
                pool = self.class_indices[c]
                replace = len(pool) < self.n_samples
                batch.extend(rng.choice(pool, size=self.n_samples, replace=replace).tolist())
            yield batch

    def __len__(self):
        return self._n_batches
