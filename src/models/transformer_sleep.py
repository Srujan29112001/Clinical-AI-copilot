"""
Transformer Model for Sleep Stage Classification
Multi-stage sleep classification using attention mechanisms
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class SleepStageConfig:
    """Configuration for sleep stage classification"""
    channels: int = 16
    sampling_rate: int = 256
    epoch_length: int = 7680  # 30 seconds at 256 Hz
    num_classes: int = 5  # Wake, N1, N2, N3, REM

    # Transformer config
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Embedding config
    patch_size: int = 128
    max_seq_len: int = 128


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TransformerSleepStage(nn.Module):
    """Transformer model for sleep stage classification"""

    def __init__(self, config: Optional[SleepStageConfig] = None):
        super().__init__()
        self.config = config or SleepStageConfig()

        # Patch embedding
        self.patch_embed = nn.Conv1d(
            self.config.channels,
            self.config.d_model,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.d_model, self.config.max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_encoder_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, samples)
        x = self.patch_embed(x)  # (batch, d_model, num_patches)
        x = x.permute(0, 2, 1)  # (batch, num_patches, d_model)

        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        return self.classifier(x)
