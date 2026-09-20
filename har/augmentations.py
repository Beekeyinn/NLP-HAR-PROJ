"""Training-time data augmentations for windowed HAR sensor signals.

Each augmentation operates on a numpy array shaped (window_size, n_channels)
and is applied only to the training split.
"""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """Add zero-mean Gaussian noise eps ~ N(0, std^2) to every channel."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return x
        noise = np.random.normal(loc=0.0, scale=self.std, size=x.shape).astype(x.dtype)
        return x + noise


class TimeMasking:
    """SpecAugment-style masking: zero out a random contiguous time span."""

    def __init__(self, max_mask_fraction: float = 0.10, p: float = 0.5):
        self.max_mask_fraction = max_mask_fraction
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return x
        window_size = x.shape[0]
        max_span = max(1, int(self.max_mask_fraction * window_size))
        span = np.random.randint(1, max_span + 1)
        start = np.random.randint(0, window_size - span + 1)
        x = x.copy()
        x[start : start + span, :] = 0.0
        return x


class Compose:
    """Chain multiple augmentations together, applied in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x


def build_train_augmentations(aug_config) -> Compose | None:
    """Build the augmentation pipeline from an AugmentationConfig."""
    transforms = []
    if aug_config.use_time_masking:
        transforms.append(
            TimeMasking(aug_config.time_mask_fraction, aug_config.time_mask_p)
        )
    if aug_config.use_gaussian_noise:
        transforms.append(
            GaussianNoise(aug_config.gaussian_noise_std, aug_config.gaussian_noise_p)
        )
    return Compose(transforms) if transforms else None
