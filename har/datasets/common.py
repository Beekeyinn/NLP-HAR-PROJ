"""Dataset-agnostic windowing, subject-independent splitting and torch Dataset."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def make_windows(values: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    """Slice a (n_samples, n_channels) stream into fixed-length overlapping windows."""
    n = len(values)
    return [
        values[start : start + window_size]
        for start in range(0, n - window_size + 1, stride)
    ]


def subject_independent_split(
    subjects: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_seed: int = 42,
):
    """Split window indices into train/val/test by subject (no subject appears in two splits)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        "fractions must sum to 1"
    )

    idx = np.arange(len(subjects))

    splitter1 = GroupShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=random_seed
    )
    train_idx, rest_idx = next(splitter1.split(idx, groups=subjects))

    rest_test_frac = test_frac / (val_frac + test_frac)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=rest_test_frac, random_state=random_seed
    )
    val_idx_rel, test_idx_rel = next(
        splitter2.split(rest_idx, groups=subjects[rest_idx])
    )

    val_idx = rest_idx[val_idx_rel]
    test_idx = rest_idx[test_idx_rel]
    return train_idx, val_idx, test_idx


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a per-channel StandardScaler using training windows only."""
    n_channels = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_channels))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_channels = X.shape[-1]
    return (
        scaler.transform(X.reshape(-1, n_channels)).reshape(X.shape).astype(np.float32)
    )


class HARWindowDataset(Dataset):
    """Windowed sensor dataset; applies an optional augmentation pipeline on `__getitem__`."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.y[idx])
