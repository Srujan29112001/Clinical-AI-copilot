"""
Multimodal Fusion Architecture
Combines EEG signals with clinical text for comprehensive diagnosis
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion"""
    eeg_feature_dim: int = 512
    text_feature_dim: int = 4096  # Llama embedding size
    hidden_dim: int = 256
    num_cross_attention_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.1

    # Mamba2 config for long context
    mamba_d_model: int = 256
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # Output dimensions
    num_diagnoses: int = 500  # ICD-10 codes
    num_severity_levels: int = 5
    num_urgency_levels: int = 3


class CrossModalAttention(nn.Module):
    """Cross-attention between EEG and text modalities"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key_value: Key-value tensor (batch, seq_len, embed_dim)

        Returns:
            Attended output
        """
        attn_out, _ = self.attention(query, key_value, key_value)
        output = self.norm(query + self.dropout(attn_out))
        return output


class SimplifiedMamba2(nn.Module):
    """
    Simplified Mamba2 (State Space Model) for long-context processing
    Full implementation would require the mamba-ssm package
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Simplified SSM using LSTM as placeholder
        # In production, use actual Mamba2 implementation
        self.ssm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Processed tensor
        """
        out, _ = self.ssm(x)
        return self.norm(out + x)


class MultimodalClinicalFusion(nn.Module):
    """
    Multimodal fusion model combining EEG and clinical text

    Features:
    - EEG encoder from CNN-LSTM features
    - Clinical text encoder from Llama embeddings
    - Cross-modal attention for fusion
    - Mamba2 for long medical history processing
    - Multiple prediction heads (diagnosis, severity, urgency)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()

        # EEG feature encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.config.eeg_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.hidden_dim)
        )

        # Text feature encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(self.config.text_feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.hidden_dim)
        )

        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(
                embed_dim=self.config.hidden_dim,
                num_heads=self.config.attention_heads,
                dropout=self.config.dropout
            ) for _ in range(self.config.num_cross_attention_layers)
        ])

        # Mamba2 for long-context processing
        self.mamba = SimplifiedMamba2(
            d_model=self.config.hidden_dim,
            d_state=self.config.mamba_d_state
        )

        # Prediction heads
        self.diagnosis_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.num_diagnoses)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.num_severity_levels)
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_urgency_levels)
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        eeg_features: torch.Tensor,
        text_features: torch.Tensor,
        medical_history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal fusion

        Args:
            eeg_features: EEG features from CNN-LSTM (batch, eeg_feature_dim)
            text_features: Text features from Llama (batch, text_feature_dim)
            medical_history: Optional medical history embeddings (batch, seq_len, hidden_dim)

        Returns:
            Dictionary with diagnosis, severity, urgency, and confidence predictions
        """
        # Encode modalities
        eeg_encoded = self.eeg_encoder(eeg_features)  # (batch, hidden_dim)
        text_encoded = self.text_encoder(text_features)  # (batch, hidden_dim)

        # Add sequence dimension for attention
        eeg_encoded = eeg_encoded.unsqueeze(1)  # (batch, 1, hidden_dim)
        text_encoded = text_encoded.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Cross-modal fusion with attention
        fused = eeg_encoded
        for attention_layer in self.cross_attention_layers:
            # Attend from EEG to text
            fused = attention_layer(fused, text_encoded)
            # Residual connection with original EEG features
            fused = fused + eeg_encoded

        # Process with medical history if available
        if medical_history is not None:
            # Concatenate current features with history
            full_context = torch.cat([fused, medical_history], dim=1)
            # Process with Mamba2 for long-range dependencies
            fused = self.mamba(full_context)
            # Take mean over sequence
            fused = fused.mean(dim=1)
        else:
            fused = fused.squeeze(1)

        # Generate predictions
        diagnosis = self.diagnosis_head(fused)
        severity = self.severity_head(fused)
        urgency = self.urgency_head(fused)
        confidence = self.confidence_head(fused)

        return {
            'diagnosis': diagnosis,
            'severity': severity,
            'urgency': urgency,
            'confidence': confidence,
            'features': fused
        }

    def predict_with_uncertainty(
        self,
        eeg_features: torch.Tensor,
        text_features: torch.Tensor,
        medical_history: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout

        Args:
            eeg_features: EEG features
            text_features: Text features
            medical_history: Optional medical history
            num_samples: Number of MC samples

        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(eeg_features, text_features, medical_history)
                predictions.append(pred)

        self.eval()

        # Calculate mean and std
        diagnosis_preds = torch.stack([p['diagnosis'] for p in predictions])
        severity_preds = torch.stack([p['severity'] for p in predictions])
        urgency_preds = torch.stack([p['urgency'] for p in predictions])

        return {
            'diagnosis_mean': diagnosis_preds.mean(dim=0),
            'diagnosis_std': diagnosis_preds.std(dim=0),
            'severity_mean': severity_preds.mean(dim=0),
            'severity_std': severity_preds.std(dim=0),
            'urgency_mean': urgency_preds.mean(dim=0),
            'urgency_std': urgency_preds.std(dim=0),
        }


def create_multimodal_model(
    config: Optional[FusionConfig] = None,
    checkpoint_path: Optional[str] = None
) -> MultimodalClinicalFusion:
    """
    Create multimodal fusion model

    Args:
        config: Model configuration
        checkpoint_path: Path to checkpoint file

    Returns:
        Initialized model
    """
    model = MultimodalClinicalFusion(config)

    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    eeg_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 4096)
    medical_history = torch.randn(batch_size, 10, 256)

    model = create_multimodal_model()

    output = model(eeg_features, text_features, medical_history)

    print("Output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")

    # Test uncertainty estimation
    uncertain_output = model.predict_with_uncertainty(
        eeg_features, text_features, medical_history, num_samples=5
    )

    print("\nUncertainty estimation shapes:")
    for key, value in uncertain_output.items():
        print(f"  {key}: {value.shape}")
