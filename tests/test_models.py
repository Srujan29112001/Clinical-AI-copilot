"""
Tests for Deep Learning Models
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_lstm import HybridCNNLSTM, SeizureDetectionConfig
from src.models.transformer_sleep import TransformerSleepStage, SleepStageConfig
from src.models.snn import SpikingNeuralNetwork, SNNConfig
from src.models.multimodal_fusion import MultimodalClinicalFusion, FusionConfig


class TestSeizureDetectionModel:
    """Test CNN-LSTM seizure detection model"""

    @pytest.fixture
    def model(self):
        """Create model"""
        config = SeizureDetectionConfig(
            channels=16,
            num_windows=10
        )
        return HybridCNNLSTM(config)

    @pytest.fixture
    def input_data(self):
        """Create sample input"""
        return torch.randn(2, 10, 16, 256)  # (batch, windows, channels, samples)

    def test_forward_pass(self, model, input_data):
        """Test forward pass"""
        output, attention = model(input_data, return_attention=False)

        assert output.shape == (2, 3)  # (batch, num_classes)
        assert attention is None

    def test_forward_with_attention(self, model, input_data):
        """Test forward pass with attention"""
        output, attention = model(input_data, return_attention=True)

        assert output.shape == (2, 3)
        assert attention is not None

    def test_predict(self, model, input_data):
        """Test prediction"""
        predictions = model.predict(input_data)

        assert predictions.shape == (2,)
        assert predictions.dtype == torch.long

    def test_feature_extraction(self, model, input_data):
        """Test feature extraction"""
        features = model.extract_features(input_data)

        assert features.shape[0] == 2  # batch size


class TestSleepClassificationModel:
    """Test Transformer sleep classification model"""

    @pytest.fixture
    def model(self):
        """Create model"""
        config = SleepStageConfig(
            channels=16,
            epoch_length=7680
        )
        return TransformerSleepStage(config)

    @pytest.fixture
    def input_data(self):
        """Create sample input"""
        return torch.randn(2, 16, 7680)  # (batch, channels, samples)

    def test_forward_pass(self, model, input_data):
        """Test forward pass"""
        output = model(input_data)

        assert output.shape == (2, 5)  # (batch, 5 sleep stages)


class TestSpikingNeuralNetwork:
    """Test SNN"""

    @pytest.fixture
    def model(self):
        """Create SNN"""
        config = SNNConfig(input_size=256, output_size=16)
        return SpikingNeuralNetwork(config)

    @pytest.fixture
    def input_data(self):
        """Create sample input"""
        return torch.randn(4, 256)  # (batch, features)

    def test_forward_pass(self, model, input_data):
        """Test forward pass"""
        output = model(input_data)

        assert output.shape == (4, 16)  # (batch, output_size)


class TestMultimodalFusion:
    """Test multimodal fusion model"""

    @pytest.fixture
    def model(self):
        """Create multimodal model"""
        config = FusionConfig()
        return MultimodalClinicalFusion(config)

    def test_forward_pass(self, model):
        """Test forward pass"""
        eeg_features = torch.randn(2, 512)
        text_features = torch.randn(2, 4096)

        output = model(eeg_features, text_features)

        assert 'diagnosis' in output
        assert 'severity' in output
        assert 'urgency' in output
        assert 'confidence' in output
        assert output['diagnosis'].shape == (2, 500)

    def test_with_medical_history(self, model):
        """Test with medical history"""
        eeg_features = torch.randn(2, 512)
        text_features = torch.randn(2, 4096)
        medical_history = torch.randn(2, 10, 256)

        output = model(eeg_features, text_features, medical_history)

        assert output['diagnosis'].shape == (2, 500)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
