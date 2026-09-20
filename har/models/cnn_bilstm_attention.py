"""Lightweight CNN-BiLSTM model with temporal attention pooling for HAR."""

from __future__ import annotations

import torch
import torch.nn as nn

from har.config import ModelConfig


class TemporalAttentionPooling(nn.Module):
    """Computes attention weights over time steps and returns a weighted context vector.

    Given encoder outputs h_t of shape (batch, seq_len, hidden), this layer learns a
    score e_t = w^T tanh(W h_t) for each time step, normalizes scores with softmax to
    get attention weights a_t, and returns context = sum_t a_t * h_t.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (batch, seq_len, hidden)
        energy = torch.tanh(self.attn_proj(h))  # (batch, seq_len, hidden)
        scores = self.attn_score(energy).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (batch, hidden)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    """1D-CNN feature extractor -> BiLSTM -> temporal attention pooling -> classifier.

    Input:  (batch, window_size, n_channels)
    Output: class logits (batch, n_classes). Apply softmax (see `predict_proba`) to
    obtain a categorical probability distribution; training uses `nn.CrossEntropyLoss`
    which combines log_softmax + NLL loss for numerical stability.
    """

    def __init__(
        self, n_channels: int, n_classes: int, config: ModelConfig | None = None
    ):
        super().__init__()
        cfg = config or ModelConfig()
        c1, c2 = cfg.cnn_channels
        k = cfg.conv_kernel_size

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, c1, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(cfg.pool_kernel_size),
            # Dilated convolution enlarges the temporal receptive field without extra pooling
            nn.Conv1d(
                c1,
                c2,
                kernel_size=k,
                dilation=cfg.conv2_dilation,
                padding=(k // 2) * cfg.conv2_dilation,
            ),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(cfg.pool_kernel_size),
            nn.Dropout(cfg.dropout),
        )

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = cfg.lstm_hidden * 2
        self.attention = TemporalAttentionPooling(lstm_out_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # (batch, window, channels) -> (batch, channels, window) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, c2) for the LSTM

        h, _ = self.lstm(x)  # (batch, seq_len, 2*hidden)
        context, attn_weights = self.attention(h)
        logits = self.classifier(self.dropout(context))

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CNNBiLSTMAttention(n_channels=6, n_classes=18)
    n_params = count_parameters(m)
    print(f"Total trainable parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")
