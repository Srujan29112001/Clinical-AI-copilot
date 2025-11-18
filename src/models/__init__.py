"""
Deep Learning Models for EEG Analysis
Includes CNN-LSTM, Transformer, SNN, and Multimodal Fusion architectures
"""

from .cnn_lstm import HybridCNNLSTM, SeizureDetectionConfig
from .transformer_sleep import TransformerSleepStage, SleepStageConfig
from .snn import SpikingNeuralNetwork, SNNConfig
from .multimodal_fusion import MultimodalClinicalFusion, FusionConfig

__all__ = [
    'HybridCNNLSTM',
    'SeizureDetectionConfig',
    'TransformerSleepStage',
    'SleepStageConfig',
    'SpikingNeuralNetwork',
    'SNNConfig',
    'MultimodalClinicalFusion',
    'FusionConfig'
]

__version__ = '1.0.0'
