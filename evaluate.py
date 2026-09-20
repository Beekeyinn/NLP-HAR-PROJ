"""Standalone evaluation script: loads a trained checkpoint and reports full metrics.

Usage:
    uv run python evaluate.py --dataset wisdm --checkpoint models/cnn_bilstm_attention_har.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from har.config import Config, ModelConfig
from har.datasets.common import (
    HARWindowDataset,
    apply_scaler,
    subject_independent_split,
)
from har.datasets.uci_har import ACTIVITY_LABELS, load_uci_har_windows
from har.datasets.wisdm import ACTIVITY_MAP, load_wisdm_windows
from har.metrics import compute_metrics, dynamic_vs_static_f1, get_confusion_matrix
from har.models.cnn_bilstm_attention import CNNBiLSTMAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN-BiLSTM-Attention checkpoint"
    )
    parser.add_argument("--dataset", choices=["wisdm", "uci_har"], default="wisdm")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default="models/cnn_bilstm_attention_har.pt"
    )
    parser.add_argument("--plot-confusion-matrix", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.dataset = args.dataset
    if args.data_dir:
        cfg.data.data_dir = Path(args.data_dir)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint["label_classes"])
    model_config = ModelConfig(**checkpoint["model_config"])

    if cfg.data.dataset == "wisdm":
        X, y_letters, subjects = load_wisdm_windows(
            cfg.data.data_dir,
            cfg.data.window_size,
            cfg.data.stride,
            cfg.data.sampling_rate_hz,
        )
        y_raw = np.array([ACTIVITY_MAP[letter] for letter in y_letters])
    else:
        X, y_ids, subjects = load_uci_har_windows(cfg.data.data_dir)
        y_raw = np.array([ACTIVITY_LABELS[i] for i in y_ids])

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    y = np.asarray(label_encoder.transform(y_raw))

    _, _, test_idx = subject_independent_split(
        subjects,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.data.random_seed,
    )

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]

    X_test = apply_scaler(X[test_idx], scaler)
    y_test = y[test_idx]
    test_loader = DataLoader(
        HARWindowDataset(X_test, y_test), batch_size=64, shuffle=False
    )

    model = CNNBiLSTMAttention(
        n_channels=cfg.data.n_channels,
        n_classes=len(class_names),
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_proba = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
    proba = np.concatenate(all_proba)
    preds = proba.argmax(axis=1)

    metrics = compute_metrics(y_test, preds, proba)
    print("Overall metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    breakdown = dynamic_vs_static_f1(y_test, preds, class_names)
    print(
        f"\nDynamic (locomotion) activities mean F1: {breakdown['dynamic_f1_mean']:.4f}"
    )
    print(
        f"Static (sedentary) activities mean F1:    {breakdown['static_f1_mean']:.4f}"
    )
    print("\nPer-class F1:")
    for name, score in breakdown["per_class_f1"].items():
        print(f"  {name}: {score:.4f}")

    cm = get_confusion_matrix(y_test, preds, len(class_names))
    if args.plot_confusion_matrix:
        _fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("\nSaved confusion matrix plot to confusion_matrix.png")


if __name__ == "__main__":
    main()
