"""Training script for the CNN-BiLSTM-Attention HAR model.

Usage:
    uv run python train.py --dataset wisdm
    uv run python train.py --dataset uci_har --data-dir data/uci_har --epochs 100
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from har.augmentations import build_train_augmentations
from har.config import Config
from har.datasets.common import (
    HARWindowDataset,
    apply_scaler,
    fit_scaler,
    subject_independent_split,
)
from har.datasets.uci_har import ACTIVITY_LABELS, load_uci_har_windows
from har.datasets.wisdm import ACTIVITY_MAP, load_wisdm_windows
from har.metrics import compute_metrics
from har.models.cnn_bilstm_attention import CNNBiLSTMAttention, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM-Attention for HAR")
    parser.add_argument("--dataset", choices=["wisdm", "uci_har"], default="wisdm")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(cfg: Config):
    if cfg.data.dataset == "wisdm":
        X, y_letters, subjects = load_wisdm_windows(
            data_dir=cfg.data.data_dir,
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            sampling_rate_hz=cfg.data.sampling_rate_hz,
        )
        y_raw = np.array(
            [ACTIVITY_MAP[letter] for letter in y_letters]
        )  # letter -> full name
        class_names = sorted(ACTIVITY_MAP.values())
    elif cfg.data.dataset == "uci_har":
        X, y_ids, subjects = load_uci_har_windows(cfg.data.data_dir)
        y_raw = np.array([ACTIVITY_LABELS[i] for i in y_ids])
        class_names = sorted(ACTIVITY_LABELS.values())
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    y = label_encoder.transform(y_raw)
    return X, y, subjects, label_encoder


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = logits.argmax(axis=1)
    proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    avg_loss = total_loss / len(labels)
    metrics = compute_metrics(labels, preds, proba)
    metrics["loss"] = avg_loss
    return metrics


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.dataset = args.dataset
    if args.data_dir:
        cfg.data.data_dir = Path(args.data_dir)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.patience:
        cfg.train.patience = args.patience

    torch.manual_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading dataset: {cfg.data.dataset}")
    X, y, subjects, label_encoder = load_dataset(cfg)
    print(f"Windows: {X.shape}, classes: {len(label_encoder.classes_)}")

    train_idx, val_idx, test_idx = subject_independent_split(
        subjects,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.data.random_seed,
    )
    print(f"Train/Val/Test windows: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    scaler = fit_scaler(X[train_idx])
    X_train = apply_scaler(X[train_idx], scaler)
    X_val = apply_scaler(X[val_idx], scaler)
    X_test = apply_scaler(X[test_idx], scaler)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_transform = build_train_augmentations(cfg.augmentation)
    train_ds = HARWindowDataset(X_train, y_train, transform=train_transform)
    val_ds = HARWindowDataset(X_val, y_val, transform=None)
    test_ds = HARWindowDataset(X_test, y_test, transform=None)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = CNNBiLSTMAttention(
        n_channels=cfg.data.n_channels,
        n_classes=len(label_encoder.classes_),
        config=cfg.model,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.train.label_smoothing
    )  # categorical cross-entropy over softmax(logits)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch:03d}/{cfg.train.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.patience:
                print(
                    f"Early stopping at epoch {epoch} (best val_f1_macro={best_val_f1:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore_best_weights=True

    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    cfg.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.train.checkpoint_dir / cfg.train.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(cfg.model),
            "data_config": asdict(cfg.data),
            "label_classes": label_encoder.classes_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "history": history,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"Saved best model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
