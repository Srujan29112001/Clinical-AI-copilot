"""
Tests for EEG Processing Pipeline
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signal_processing.eeg_processor import EEGProcessingPipeline
from src.signal_processing.filters import SignalFilters
from src.signal_processing.feature_extraction import (
    PowerSpectralDensity,
    ConnectivityAnalyzer,
    EntropyCalculator,
    HjorthParameters
)


class TestEEGProcessor:
    """Test EEG processing pipeline"""

    @pytest.fixture
    def processor(self):
        """Create EEG processor"""
        return EEGProcessingPipeline(sampling_rate=256, channels=16)

    @pytest.fixture
    def sample_eeg(self):
        """Generate sample EEG data"""
        np.random.seed(42)
        return np.random.randn(16, 256).astype(np.float32)

    def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.sampling_rate == 256
        assert processor.channels == 16
        assert processor.ica is not None

    def test_process_chunk(self, processor, sample_eeg):
        """Test chunk processing"""
        features, coeffs = processor.process_chunk(sample_eeg, calibrate=True)

        assert features is not None
        assert 'psd' in features
        assert 'entropy' in features
        assert 'hjorth' in features
        assert coeffs is not None

    def test_anomaly_detection(self, processor, sample_eeg):
        """Test anomaly detection"""
        features, _ = processor.process_chunk(sample_eeg)
        anomalies = processor.detect_anomalies(features)

        assert isinstance(anomalies, dict)
        assert 'high_delta' in anomalies
        assert all(isinstance(v, (bool, np.bool_)) for v in anomalies.values())


class TestSignalFilters:
    """Test signal filtering"""

    @pytest.fixture
    def filters(self):
        """Create signal filters"""
        return SignalFilters(sampling_rate=256)

    @pytest.fixture
    def signal(self):
        """Generate test signal"""
        t = np.linspace(0, 1, 256)
        # 10 Hz sine wave
        return np.sin(2 * np.pi * 10 * t).astype(np.float32)

    def test_bandpass_filter(self, filters, signal):
        """Test bandpass filtering"""
        filtered = filters.bandpass_filter(signal)
        assert filtered.shape == signal.shape
        assert not np.array_equal(filtered, signal)

    def test_notch_filter(self, filters, signal):
        """Test notch filtering"""
        filtered = filters.notch_filter(signal)
        assert filtered.shape == signal.shape


class TestFeatureExtraction:
    """Test feature extraction"""

    @pytest.fixture
    def eeg_data(self):
        """Generate EEG data"""
        np.random.seed(42)
        return np.random.randn(16, 256).astype(np.float32)

    def test_psd_extraction(self, eeg_data):
        """Test PSD feature extraction"""
        psd = PowerSpectralDensity(sampling_rate=256)
        features = psd.extract(eeg_data)

        assert 'delta' in features
        assert 'theta' in features
        assert 'alpha' in features
        assert 'beta' in features
        assert 'gamma' in features
        assert all(len(v) == 16 for v in features.values() if isinstance(v, np.ndarray))

    def test_connectivity_analysis(self, eeg_data):
        """Test connectivity analysis"""
        conn = ConnectivityAnalyzer(sampling_rate=256)
        features = conn.extract(eeg_data)

        assert 'coherence' in features or 'plv' in features
        for v in features.values():
            assert v.shape == (16, 16)  # Connectivity matrix

    def test_entropy_calculation(self, eeg_data):
        """Test entropy calculation"""
        entropy = EntropyCalculator()
        features = entropy.extract(eeg_data)

        assert 'shannon' in features
        assert len(features['shannon']) == 16

    def test_hjorth_parameters(self, eeg_data):
        """Test Hjorth parameters"""
        hjorth = HjorthParameters()
        features = hjorth.extract(eeg_data)

        assert 'activity' in features
        assert 'mobility' in features
        assert 'complexity' in features
        assert all(len(v) == 16 for v in features.values())


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
