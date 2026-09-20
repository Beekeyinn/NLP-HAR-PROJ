"""Evaluation metrics for HAR classification: overall scores, confusion matrix,
and per-class F1 grouped by dynamic (locomotion) vs static (sedentary) activities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Activities involving continuous body movement vs mostly-still/sedentary activities.
# Extend/adjust these sets to match the label vocabulary of the dataset in use.
DYNAMIC_ACTIVITIES = {
    "walking",
    "jogging",
    "stairs",
    "kicking",
    "dribbling",
    "clapping",
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
}
STATIC_ACTIVITIES = {
    "sitting",
    "standing",
    "typing",
    "teeth",
    "soup",
    "chips",
    "pasta",
    "drinking",
    "sandwich",
    "catch",
    "writing",
    "folding",
    "SITTING",
    "STANDING",
    "LAYING",
}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict:
    """Compute accuracy, macro precision/recall/F1, and (optionally) macro ROC-AUC."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_macro"] = float("nan")  # e.g. a class missing from y_true
    return metrics


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    scores = f1_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    return dict(zip(class_names, scores))


def dynamic_vs_static_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """Average per-class F1 separately for dynamic/locomotion vs static/sedentary activities."""
    f1s = per_class_f1(y_true, y_pred, class_names)
    dynamic_scores = [s for name, s in f1s.items() if name in DYNAMIC_ACTIVITIES]
    static_scores = [s for name, s in f1s.items() if name in STATIC_ACTIVITIES]
    return {
        "dynamic_f1_mean": float(np.mean(dynamic_scores))
        if dynamic_scores
        else float("nan"),
        "static_f1_mean": float(np.mean(static_scores))
        if static_scores
        else float("nan"),
        "per_class_f1": f1s,
    }


def get_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=range(n_classes))
