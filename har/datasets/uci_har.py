"""UCI HAR dataset loader.

Expects the standard "UCI HAR Dataset" layout (as distributed by the UCI ML
repository), already pre-segmented into 128-sample / 50Hz / 50%-overlap windows:

    <data_dir>/UCI HAR Dataset/{train,test}/Inertial Signals/*.txt
    <data_dir>/UCI HAR Dataset/{train,test}/y_{train,test}.txt
    <data_dir>/UCI HAR Dataset/{train,test}/subject_{train,test}.txt

Only the raw accelerometer + gyroscope channels (6 total) are used, matching the
WISDM loader's channel order: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _load_split(root: Path, split: str):
    signals_dir = root / split / "Inertial Signals"
    channels = []
    for name in SIGNAL_FILES:
        path = signals_dir / f"{name}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the UCI HAR Dataset and extract it under `data_dir`."
            )
        channels.append(np.loadtxt(path))  # (n_windows, 128)

    X = np.stack(channels, axis=-1).astype(np.float32)  # (n_windows, 128, 6)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int)
    subjects = np.loadtxt(root / split / f"subject_{split}.txt").astype(int)
    return X, y, subjects


def load_uci_har_windows(data_dir: Path):
    """Load and concatenate the train + test splits (re-split subject-independently downstream).

    Returns:
        X: (n_windows, 128, 6) float32 array
        y: (n_windows,) integer activity labels (1-6, see ACTIVITY_LABELS)
        subjects: (n_windows,) subject ids
    """
    root = Path(data_dir) / "UCI HAR Dataset"
    X_train, y_train, subj_train = _load_split(root, "train")
    X_test, y_test, subj_test = _load_split(root, "test")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subjects = np.concatenate([subj_train, subj_test], axis=0)
    return X, y, subjects
