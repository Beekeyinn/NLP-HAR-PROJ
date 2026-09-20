"""WISDM raw dataset loader: merges phone accelerometer + gyroscope streams,
resamples to a common sampling rate, and produces fixed-length sliding windows
with subject ids and activity labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from har.datasets.common import make_windows

COLUMNS = ["id", "activity", "timestamp", "x", "y", "z"]

ACTIVITY_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}
ACTIVITIES = sorted(ACTIVITY_MAP.keys())


def _load_raw_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMNS,
        converters={"z": lambda z: str(z).rstrip(";")},
    )
    df["z"] = df["z"].astype(float)
    return df


def _resample_stream(df: pd.DataFrame, target_hz: int) -> np.ndarray:
    """Linearly resample an (x, y, z) stream from its native rate to `target_hz`."""
    t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    duration_s = (t1 - t0) / 1e9
    n_target = max(2, int(duration_s * target_hz))
    src_t = df["timestamp"].to_numpy(dtype=np.float64)
    tgt_t = np.linspace(t0, t1, n_target)
    out = np.stack(
        [
            np.interp(tgt_t, src_t, df[axis].to_numpy(dtype=np.float64))
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    return out


def load_wisdm_windows(
    data_dir: Path,
    window_size: int,
    stride: int,
    sampling_rate_hz: int,
    device: str = "phone",
):
    """Merge accel + gyro per (subject, activity), resample, and window into (window, 6) arrays.

    Returns:
        X: (n_windows, window_size, 6) float32 array, channel order [ax, ay, az, gx, gy, gz]
        y: (n_windows,) activity letter labels
        subjects: (n_windows,) subject ids
    """
    raw_dir = Path(data_dir) / "wisdm-dataset" / "raw"
    accel_dir = raw_dir / device / "accel"
    gyro_dir = raw_dir / device / "gyro"

    X_list, y_list, subject_list = [], [], []
    for accel_path in sorted(accel_dir.glob("data_*.txt")):
        gyro_path = gyro_dir / accel_path.name.replace("accel", "gyro")
        if not gyro_path.exists():
            continue

        accel_df = _load_raw_file(accel_path)
        gyro_df = _load_raw_file(gyro_path)
        subject_id = int(accel_df["id"].iloc[0])

        for activity in ACTIVITIES:
            a_seg = accel_df[accel_df["activity"] == activity]
            g_seg = gyro_df[gyro_df["activity"] == activity]
            if len(a_seg) < 2 or len(g_seg) < 2:
                continue

            a_resampled = _resample_stream(a_seg, sampling_rate_hz)
            g_resampled = _resample_stream(g_seg, sampling_rate_hz)
            n = min(len(a_resampled), len(g_resampled))
            if n < window_size:
                continue

            combined = np.concatenate(
                [a_resampled[:n], g_resampled[:n]], axis=1
            )  # (n, 6)
            for window in make_windows(combined, window_size, stride):
                X_list.append(window)
                y_list.append(activity)
                subject_list.append(subject_id)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    return X, y, subjects
