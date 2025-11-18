"""
EEG Processing Pipeline
Real-time EEG signal processing with ICA, filtering, and feature extraction
"""

import numpy as np
import pywt
from scipy import signal
from sklearn.decomposition import FastICA
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .feature_extraction import (
    PowerSpectralDensity,
    ConnectivityAnalyzer,
    EntropyCalculator,
    HjorthParameters
)
from .filters import SignalFilters

logger = logging.getLogger(__name__)


@dataclass
class EEGConfig:
    """Configuration for EEG processing"""
    sampling_rate: int = 256  # Hz
    channels: int = 16
    window_size: int = 256  # 1 second at 256 Hz
    overlap: float = 0.5

    # Filter parameters
    bandpass_low: float = 0.5
    bandpass_high: float = 50.0
    notch_freq: float = 50.0  # or 60.0 for US
    notch_quality: float = 30.0

    # Wavelet parameters
    wavelet: str = 'db4'
    decomposition_level: int = 5

    # ICA parameters
    ica_n_components: Optional[int] = None  # None means use all channels
    ica_max_iter: int = 200


class EEGProcessingPipeline:
    """
    Comprehensive EEG processing pipeline for real-time analysis

    Features:
    - ICA artifact removal (eye blinks, muscle artifacts)
    - Bandpass filtering (0.5-50 Hz)
    - Notch filtering (50/60 Hz power line noise)
    - Wavelet decomposition for time-frequency analysis
    - Multi-domain feature extraction

    Example:
        >>> processor = EEGProcessingPipeline(sampling_rate=256, channels=16)
        >>> features, coefficients = processor.process_chunk(eeg_data)
        >>> print(features['psd']['alpha_power'])
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        channels: int = 16,
        config: Optional[EEGConfig] = None
    ):
        """
        Initialize EEG processing pipeline

        Args:
            sampling_rate: Sampling rate in Hz
            channels: Number of EEG channels
            config: Optional EEGConfig object for custom configuration
        """
        self.config = config or EEGConfig(
            sampling_rate=sampling_rate,
            channels=channels
        )

        self.sampling_rate = self.config.sampling_rate
        self.channels = self.config.channels

        # Initialize signal filters
        self.filters = SignalFilters(
            sampling_rate=self.sampling_rate,
            bandpass_low=self.config.bandpass_low,
            bandpass_high=self.config.bandpass_high,
            notch_freq=self.config.notch_freq,
            notch_quality=self.config.notch_quality
        )

        # Initialize ICA for artifact removal
        ica_components = self.config.ica_n_components or self.channels
        self.ica = FastICA(
            n_components=ica_components,
            algorithm='parallel',
            whiten=True,
            max_iter=self.config.ica_max_iter,
            random_state=42
        )

        # Wavelet configuration
        self.wavelet = pywt.Wavelet(self.config.wavelet)
        self.decomposition_level = self.config.decomposition_level

        # Initialize feature extractors
        self.feature_extractors = {
            'psd': PowerSpectralDensity(
                sampling_rate=self.sampling_rate,
                freq_bands={
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 50)
                }
            ),
            'connectivity': ConnectivityAnalyzer(
                sampling_rate=self.sampling_rate,
                methods=['coherence', 'phase_locking_value', 'mutual_information']
            ),
            'entropy': EntropyCalculator(
                types=['shannon', 'sample', 'permutation']
            ),
            'hjorth': HjorthParameters()
        }

        # Buffer for continuous processing
        self.buffer = np.zeros((self.channels, self.sampling_rate * 2))
        self.buffer_index = 0

        # ICA fitted flag
        self.ica_fitted = False
        self.calibration_data = []
        self.calibration_samples_needed = self.sampling_rate * 30  # 30 seconds

        logger.info(f"EEG Processing Pipeline initialized: {self.channels} channels @ {self.sampling_rate} Hz")

    def calibrate_ica(self, eeg_data: np.ndarray) -> None:
        """
        Calibrate ICA using initial clean EEG data

        Args:
            eeg_data: Clean EEG data for ICA calibration, shape (channels, samples)
        """
        if eeg_data.shape[0] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {eeg_data.shape[0]}")

        self.calibration_data.append(eeg_data)
        total_samples = sum(d.shape[1] for d in self.calibration_data)

        if total_samples >= self.calibration_samples_needed and not self.ica_fitted:
            # Concatenate calibration data
            cal_data = np.concatenate(self.calibration_data, axis=1)

            # Fit ICA
            logger.info("Fitting ICA with calibration data...")
            self.ica.fit(cal_data.T)
            self.ica_fitted = True
            logger.info("ICA calibration complete")

            # Clear calibration data to free memory
            self.calibration_data = []

    def remove_artifacts(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Remove artifacts using ICA

        Args:
            eeg_data: Raw EEG data, shape (channels, samples)

        Returns:
            Cleaned EEG data
        """
        if not self.ica_fitted:
            logger.warning("ICA not fitted yet, skipping artifact removal")
            return eeg_data

        # Transform to ICA space
        ica_sources = self.ica.transform(eeg_data.T)

        # Identify artifact components (simplified - in production use proper detection)
        # Here we remove components with very high variance (likely artifacts)
        variances = np.var(ica_sources, axis=0)
        threshold = np.mean(variances) + 3 * np.std(variances)
        artifact_components = np.where(variances > threshold)[0]

        # Zero out artifact components
        ica_sources[:, artifact_components] = 0

        # Reconstruct cleaned signal
        cleaned = self.ica.inverse_transform(ica_sources).T

        logger.debug(f"Removed {len(artifact_components)} artifact components")
        return cleaned

    def apply_filters(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters

        Args:
            eeg_data: EEG data, shape (channels, samples)

        Returns:
            Filtered EEG data
        """
        # Apply bandpass filter
        filtered = self.filters.bandpass_filter(eeg_data)

        # Apply notch filter for power line noise
        filtered = self.filters.notch_filter(filtered)

        return filtered

    def wavelet_decomposition(self, eeg_data: np.ndarray) -> List[List[np.ndarray]]:
        """
        Perform wavelet decomposition for time-frequency analysis

        Args:
            eeg_data: Filtered EEG data, shape (channels, samples)

        Returns:
            List of wavelet coefficients for each channel
        """
        coefficients = []

        for channel_idx in range(eeg_data.shape[0]):
            channel_data = eeg_data[channel_idx, :]

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(
                channel_data,
                self.wavelet,
                level=self.decomposition_level
            )

            coefficients.append(coeffs)

        return coefficients

    def extract_features(self, filtered_data: np.ndarray) -> Dict:
        """
        Extract multiple features from filtered EEG data

        Args:
            filtered_data: Filtered EEG data, shape (channels, samples)

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Extract features using all feature extractors
        for name, extractor in self.feature_extractors.items():
            try:
                features[name] = extractor.extract(filtered_data)
            except Exception as e:
                logger.error(f"Error extracting {name} features: {e}")
                features[name] = None

        # Add basic statistics
        features['statistics'] = {
            'mean': np.mean(filtered_data, axis=1),
            'std': np.std(filtered_data, axis=1),
            'max': np.max(filtered_data, axis=1),
            'min': np.min(filtered_data, axis=1),
            'range': np.ptp(filtered_data, axis=1)
        }

        return features

    def process_chunk(
        self,
        eeg_chunk: np.ndarray,
        calibrate: bool = False
    ) -> Tuple[Dict, List[List[np.ndarray]]]:
        """
        Process a single chunk of EEG data (typically 1 second)

        Args:
            eeg_chunk: Raw EEG data chunk, shape (channels, samples)
            calibrate: If True, use this chunk for ICA calibration

        Returns:
            Tuple of (features_dict, wavelet_coefficients)
        """
        # Validate input
        expected_shape = (self.channels, self.config.window_size)
        if eeg_chunk.shape != expected_shape:
            raise ValueError(
                f"Expected chunk shape {expected_shape}, got {eeg_chunk.shape}"
            )

        # Calibration mode
        if calibrate:
            self.calibrate_ica(eeg_chunk)

        # 1. Artifact removal with ICA
        if self.ica_fitted:
            cleaned = self.remove_artifacts(eeg_chunk)
        else:
            cleaned = eeg_chunk
            logger.debug("Processing without ICA (not calibrated yet)")

        # 2. Apply filters
        filtered = self.apply_filters(cleaned)

        # 3. Wavelet decomposition
        wavelet_coeffs = self.wavelet_decomposition(filtered)

        # 4. Feature extraction
        features = self.extract_features(filtered)

        # Add metadata
        features['metadata'] = {
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'window_size': eeg_chunk.shape[1],
            'ica_applied': self.ica_fitted,
            'timestamp': np.datetime64('now')
        }

        return features, wavelet_coeffs

    def process_continuous(
        self,
        eeg_stream: np.ndarray,
        overlap: float = 0.5
    ) -> List[Tuple[Dict, List[List[np.ndarray]]]]:
        """
        Process continuous EEG stream with overlapping windows

        Args:
            eeg_stream: Continuous EEG data, shape (channels, samples)
            overlap: Overlap fraction between windows (0-1)

        Returns:
            List of (features, wavelet_coefficients) tuples for each window
        """
        results = []

        window_size = self.config.window_size
        step_size = int(window_size * (1 - overlap))

        num_windows = (eeg_stream.shape[1] - window_size) // step_size + 1

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size

            if end > eeg_stream.shape[1]:
                break

            chunk = eeg_stream[:, start:end]
            features, coeffs = self.process_chunk(chunk)
            results.append((features, coeffs))

        logger.info(f"Processed {len(results)} windows from continuous stream")
        return results

    def detect_anomalies(self, features: Dict) -> Dict[str, bool]:
        """
        Detect potential anomalies in EEG signals

        Args:
            features: Extracted features dictionary

        Returns:
            Dictionary of anomaly flags
        """
        anomalies = {}

        # Check for excessive power in specific bands (potential seizure)
        if features['psd']:
            psd = features['psd']
            # Seizure often shows increased delta/theta power
            avg_delta = np.mean(psd['delta'])
            avg_theta = np.mean(psd['theta'])
            avg_alpha = np.mean(psd['alpha'])

            # Simple threshold-based detection (improve with ML model)
            anomalies['high_delta'] = avg_delta > 50  # uV^2/Hz
            anomalies['high_theta'] = avg_theta > 30
            anomalies['alpha_suppression'] = avg_alpha < 5

        # Check entropy (lower entropy might indicate seizure)
        if features['entropy']:
            entropy = features['entropy']
            avg_entropy = np.mean(entropy.get('shannon', []))
            anomalies['low_entropy'] = avg_entropy < 2.0

        # Check Hjorth parameters
        if features['hjorth']:
            hjorth = features['hjorth']
            # High mobility might indicate artifacts
            avg_mobility = np.mean(hjorth['mobility'])
            anomalies['high_mobility'] = avg_mobility > 2.0

        return anomalies

    def reset(self):
        """Reset the processor state"""
        self.buffer = np.zeros((self.channels, self.sampling_rate * 2))
        self.buffer_index = 0
        self.ica_fitted = False
        self.calibration_data = []
        logger.info("EEG processor reset")


def main():
    """Example usage of EEG Processing Pipeline"""
    # Simulate EEG data (16 channels, 1 second at 256 Hz)
    np.random.seed(42)
    eeg_data = np.random.randn(16, 256) * 10  # Simulate in microvolts

    # Add some realistic EEG patterns
    t = np.linspace(0, 1, 256)
    for ch in range(16):
        # Add alpha rhythm (8-13 Hz)
        eeg_data[ch, :] += 20 * np.sin(2 * np.pi * 10 * t)
        # Add some slow wave
        eeg_data[ch, :] += 30 * np.sin(2 * np.pi * 2 * t)

    # Initialize processor
    processor = EEGProcessingPipeline(sampling_rate=256, channels=16)

    # Calibrate ICA with clean data
    print("Calibrating ICA...")
    for _ in range(30):  # 30 seconds of calibration data
        calibration_chunk = np.random.randn(16, 256) * 10
        processor.calibrate_ica(calibration_chunk)

    # Process a chunk
    print("\nProcessing EEG chunk...")
    features, coefficients = processor.process_chunk(eeg_data)

    # Display results
    print("\nExtracted Features:")
    print(f"PSD - Alpha Power: {np.mean(features['psd']['alpha']):.2f} uV^2/Hz")
    print(f"PSD - Beta Power: {np.mean(features['psd']['beta']):.2f} uV^2/Hz")
    print(f"Entropy - Shannon: {np.mean(features['entropy']['shannon']):.2f}")
    print(f"Hjorth - Mobility: {np.mean(features['hjorth']['mobility']):.2f}")

    # Detect anomalies
    anomalies = processor.detect_anomalies(features)
    print(f"\nAnomalies Detected: {sum(anomalies.values())} / {len(anomalies)}")
    for anomaly, detected in anomalies.items():
        if detected:
            print(f"  - {anomaly}: WARNING")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
