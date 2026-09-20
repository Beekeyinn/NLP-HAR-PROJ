# NLP-HAR-PROJ

Human Activity Recognition (HAR) from wearable/smartphone accelerometer + gyroscope
signals, using a lightweight hybrid **CNN-BiLSTM-Attention** model in PyTorch.

## Project Structure

```
.
├── dataset.ipynb          # exploratory data analysis + a self-contained CNN-BiLSTM notebook model
├── train.py                # training script (data loading, augmentation, training loop, checkpointing)
├── evaluate.py              # loads a checkpoint and reports evaluation metrics
├── har/                     # reusable package
│   ├── config.py            # dataclass configs (data, augmentation, model, train)
│   ├── augmentations.py      # GaussianNoise, TimeMasking training-time augmentations
│   ├── metrics.py            # accuracy/precision/recall/F1/ROC-AUC, confusion matrix
│   ├── datasets/
│   │   ├── common.py         # windowing, subject-independent split, torch Dataset
│   │   ├── wisdm.py          # WISDM raw accel+gyro loader (resamples, merges, windows)
│   │   └── uci_har.py        # UCI HAR pre-windowed dataset loader
│   └── models/
│       └── cnn_bilstm_attention.py  # CNNBiLSTMAttention model definition
├── data/wisdm-dataset/       # raw WISDM dataset (phone/watch accel/gyro)
└── models/                   # saved checkpoints (created after training)
```

## Model

`CNNBiLSTMAttention`:
1. **1D-CNN** (dilated conv in the 2nd block) extracts local spatial/temporal features from
   6-channel windows (accel x/y/z + gyro x/y/z).
2. **BiLSTM** captures long-range temporal dependencies over the CNN feature sequence.
3. **Temporal attention pooling** computes a weighted context vector across time steps.
4. **Linear classifier** outputs class logits (softmax applied via `CrossEntropyLoss`
   during training, or explicitly via `model.predict_proba()` at inference).

~0.77M trainable parameters.

## Setup

```bash
uv sync
```

## Training

```bash
uv run python train.py --dataset wisdm
uv run python train.py --dataset uci_har --data-dir data/uci_har --epochs 100
```

- Windows: 2.56s @ 50Hz (128 samples), 50% overlap.
- Per-channel z-score normalization fit on the training split only.
- Subject-independent 70/15/15 train/val/test split.
- Adam optimizer (lr=1e-3, weight_decay=1e-4) + cosine annealing LR schedule.
- Early stopping on validation macro-F1 (patience configurable via `--patience`),
  restoring the best weights before saving the checkpoint.
- Checkpoint saved to `models/cnn_bilstm_attention_har.pt`.

## Evaluation

```bash
uv run python evaluate.py --dataset wisdm --checkpoint models/cnn_bilstm_attention_har.pt --plot-confusion-matrix
```

Reports accuracy, macro precision/recall/F1, macro ROC-AUC, per-class F1, a
dynamic (locomotion) vs. static (sedentary) F1 breakdown, and optionally saves a
confusion matrix plot to `confusion_matrix.png`.

## Datasets

- **WISDM**: raw, variable-rate accelerometer/gyroscope streams under `data/wisdm-dataset/raw/`.
  The loader merges accel+gyro per subject/activity and resamples to a common rate.
- **UCI HAR**: pre-segmented 128-sample/50Hz/50%-overlap windows. Download from the
  [UCI ML repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  and extract as `<data_dir>/UCI HAR Dataset/`.
