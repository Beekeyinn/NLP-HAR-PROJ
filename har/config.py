"""Central configuration for the CNN-BiLSTM-Attention HAR pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    dataset: str = "wisdm"  # "wisdm" or "uci_har"
    data_dir: Path = Path("data")
    sampling_rate_hz: int = 50
    window_seconds: float = 2.56
    window_size: int = 128  # window_seconds * sampling_rate_hz
    overlap: float = 0.5
    stride: int = 64  # window_size * (1 - overlap)
    n_channels: int = 6  # accel x/y/z + gyro x/y/z
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_seed: int = 42


@dataclass
class AugmentationConfig:
    use_time_masking: bool = True
    time_mask_fraction: float = 0.15  # max fraction of the window masked
    time_mask_p: float = 0.7  # probability of applying the augmentation

    use_gaussian_noise: bool = True
    gaussian_noise_std: float = 0.02
    gaussian_noise_p: float = 0.7


@dataclass
class ModelConfig:
    cnn_channels: tuple = (
        32,
        64,
    )  # (conv1_out, conv2_out) - smaller to reduce overfitting
    conv_kernel_size: int = 5
    conv2_dilation: int = 2
    pool_kernel_size: int = 2
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.0  # only applies between stacked LSTM layers
    dropout: float = 0.5


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    patience: int = 10  # early stopping patience, monitored on val macro-F1
    grad_clip_norm: float = 5.0
    checkpoint_dir: Path = Path("models")
    checkpoint_name: str = "cnn_bilstm_attention_har.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
