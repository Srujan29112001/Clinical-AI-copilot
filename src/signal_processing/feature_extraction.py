"""
Feature Extraction Module
Extract multiple features from EEG signals for analysis and classification
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Tuple, Optional
import logging

try:
    import antropy as ant
except ImportError:
    ant = None
    logging.warning("antropy not installed, some entropy features will be unavailable")

logger = logging.getLogger(__name__)


class PowerSpectralDensity:
    """
    Extract Power Spectral Density (PSD) features from EEG signals
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'welch',
        nperseg: int = 256
    ):
        """
        Initialize PSD feature extractor

        Args:
            sampling_rate: Sampling rate in Hz
            freq_bands: Dictionary of frequency bands (name: (low, high))
            method: 'welch' or 'periodogram'
            nperseg: Length of each segment for Welch method
        """
        self.sampling_rate = sampling_rate
        self.method = method
        self.nperseg = nperseg

        self.freq_bands = freq_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract PSD features

        Args:
            data: EEG data, shape (channels, samples)

        Returns:
            Dictionary of band powers for each channel
        """
        n_channels = data.shape[0]
        band_powers = {band: np.zeros(n_channels) for band in self.freq_bands}

        for ch in range(n_channels):
            # Compute PSD
            if self.method == 'welch':
                freqs, psd = signal.welch(
                    data[ch, :],
                    fs=self.sampling_rate,
                    nperseg=self.nperseg
                )
            else:
                freqs, psd = signal.periodogram(
                    data[ch, :],
                    fs=self.sampling_rate
                )

            # Extract band powers
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                # Find frequency indices
                idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                # Calculate band power (integrate PSD)
                band_powers[band_name][ch] = np.trapz(psd[idx], freqs[idx])

        # Calculate additional metrics
        total_power = sum(band_powers.values())
        relative_powers = {
            f'{band}_relative': powers / (total_power + 1e-10)
            for band, powers in band_powers.items()
        }

        # Combine absolute and relative powers
        features = {**band_powers, **relative_powers}

        # Add ratios (commonly used in clinical analysis)
        features['theta_beta_ratio'] = band_powers['theta'] / (band_powers['beta'] + 1e-10)
        features['alpha_theta_ratio'] = band_powers['alpha'] / (band_powers['theta'] + 1e-10)
        features['delta_alpha_ratio'] = band_powers['delta'] / (band_powers['alpha'] + 1e-10)

        return features


class ConnectivityAnalyzer:
    """
    Analyze functional connectivity between EEG channels
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        methods: List[str] = None
    ):
        """
        Initialize connectivity analyzer

        Args:
            sampling_rate: Sampling rate in Hz
            methods: List of methods to compute ['coherence', 'phase_locking_value', 'mutual_information']
        """
        self.sampling_rate = sampling_rate
        self.methods = methods or ['coherence', 'phase_locking_value']

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract connectivity features

        Args:
            data: EEG data, shape (channels, samples)

        Returns:
            Dictionary of connectivity matrices
        """
        n_channels = data.shape[0]
        features = {}

        if 'coherence' in self.methods:
            features['coherence'] = self._compute_coherence(data, n_channels)

        if 'phase_locking_value' in self.methods:
            features['plv'] = self._compute_plv(data, n_channels)

        if 'mutual_information' in self.methods:
            features['mutual_info'] = self._compute_mutual_information(data, n_channels)

        if 'cross_correlation' in self.methods:
            features['cross_correlation'] = self._compute_cross_correlation(data, n_channels)

        return features

    def _compute_coherence(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Compute coherence matrix"""
        coherence_matrix = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                freqs, coh = signal.coherence(
                    data[i, :],
                    data[j, :],
                    fs=self.sampling_rate,
                    nperseg=min(256, data.shape[1] // 4)
                )
                # Average coherence across frequencies
                coherence_matrix[i, j] = np.mean(coh)
                coherence_matrix[j, i] = coherence_matrix[i, j]

        return coherence_matrix

    def _compute_plv(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Compute Phase Locking Value matrix"""
        plv_matrix = np.zeros((n_channels, n_channels))

        # Hilbert transform to get instantaneous phase
        analytic_signals = signal.hilbert(data, axis=1)
        phases = np.angle(analytic_signals)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Phase difference
                phase_diff = phases[i, :] - phases[j, :]
                # PLV
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv

        return plv_matrix

    def _compute_mutual_information(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Compute Mutual Information matrix"""
        mi_matrix = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Discretize signals
                bins = 20
                hist_2d, _, _ = np.histogram2d(
                    data[i, :],
                    data[j, :],
                    bins=bins
                )

                # Normalize to get joint probability
                pxy = hist_2d / np.sum(hist_2d)

                # Marginal probabilities
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)

                # Compute mutual information
                mi = 0
                for ii in range(bins):
                    for jj in range(bins):
                        if pxy[ii, jj] > 0:
                            mi += pxy[ii, jj] * np.log(
                                pxy[ii, jj] / (px[ii] * py[jj] + 1e-10)
                            )

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _compute_cross_correlation(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Compute cross-correlation matrix"""
        corr_matrix = np.corrcoef(data)
        return corr_matrix


class EntropyCalculator:
    """
    Calculate various entropy measures from EEG signals
    """

    def __init__(self, types: List[str] = None):
        """
        Initialize entropy calculator

        Args:
            types: List of entropy types ['shannon', 'sample', 'permutation', 'spectral', 'svd']
        """
        self.types = types or ['shannon', 'sample', 'permutation']

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract entropy features

        Args:
            data: EEG data, shape (channels, samples)

        Returns:
            Dictionary of entropy measures
        """
        n_channels = data.shape[0]
        features = {}

        if 'shannon' in self.types:
            features['shannon'] = self._shannon_entropy(data, n_channels)

        if 'sample' in self.types and ant:
            features['sample'] = self._sample_entropy(data, n_channels)

        if 'permutation' in self.types and ant:
            features['permutation'] = self._permutation_entropy(data, n_channels)

        if 'spectral' in self.types and ant:
            features['spectral'] = self._spectral_entropy(data, n_channels)

        if 'svd' in self.types and ant:
            features['svd'] = self._svd_entropy(data, n_channels)

        if 'approximate' in self.types and ant:
            features['approximate'] = self._approximate_entropy(data, n_channels)

        return features

    def _shannon_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate Shannon entropy"""
        entropies = np.zeros(n_channels)

        for ch in range(n_channels):
            # Discretize signal
            hist, _ = np.histogram(data[ch, :], bins=50, density=True)
            # Remove zero bins
            hist = hist[hist > 0]
            # Calculate entropy
            entropies[ch] = -np.sum(hist * np.log2(hist + 1e-10))

        return entropies

    def _sample_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate sample entropy"""
        entropies = np.zeros(n_channels)

        if ant is None:
            logger.warning("antropy not available, skipping sample entropy")
            return entropies

        for ch in range(n_channels):
            try:
                entropies[ch] = ant.sample_entropy(data[ch, :])
            except Exception as e:
                logger.debug(f"Error computing sample entropy for channel {ch}: {e}")
                entropies[ch] = 0

        return entropies

    def _permutation_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate permutation entropy"""
        entropies = np.zeros(n_channels)

        if ant is None:
            logger.warning("antropy not available, skipping permutation entropy")
            return entropies

        for ch in range(n_channels):
            try:
                entropies[ch] = ant.perm_entropy(data[ch, :], normalize=True)
            except Exception as e:
                logger.debug(f"Error computing permutation entropy for channel {ch}: {e}")
                entropies[ch] = 0

        return entropies

    def _spectral_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate spectral entropy"""
        entropies = np.zeros(n_channels)

        if ant is None:
            logger.warning("antropy not available, skipping spectral entropy")
            return entropies

        for ch in range(n_channels):
            try:
                entropies[ch] = ant.spectral_entropy(data[ch, :], sf=256, method='welch')
            except Exception as e:
                logger.debug(f"Error computing spectral entropy for channel {ch}: {e}")
                entropies[ch] = 0

        return entropies

    def _svd_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate SVD entropy"""
        entropies = np.zeros(n_channels)

        if ant is None:
            logger.warning("antropy not available, skipping SVD entropy")
            return entropies

        for ch in range(n_channels):
            try:
                entropies[ch] = ant.svd_entropy(data[ch, :], normalize=True)
            except Exception as e:
                logger.debug(f"Error computing SVD entropy for channel {ch}: {e}")
                entropies[ch] = 0

        return entropies

    def _approximate_entropy(self, data: np.ndarray, n_channels: int) -> np.ndarray:
        """Calculate approximate entropy"""
        entropies = np.zeros(n_channels)

        if ant is None:
            logger.warning("antropy not available, skipping approximate entropy")
            return entropies

        for ch in range(n_channels):
            try:
                entropies[ch] = ant.app_entropy(data[ch, :])
            except Exception as e:
                logger.debug(f"Error computing approximate entropy for channel {ch}: {e}")
                entropies[ch] = 0

        return entropies


class HjorthParameters:
    """
    Calculate Hjorth parameters: Activity, Mobility, and Complexity
    """

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract Hjorth parameters

        Args:
            data: EEG data, shape (channels, samples)

        Returns:
            Dictionary with activity, mobility, and complexity arrays
        """
        n_channels = data.shape[0]

        activity = np.zeros(n_channels)
        mobility = np.zeros(n_channels)
        complexity = np.zeros(n_channels)

        for ch in range(n_channels):
            signal_data = data[ch, :]

            # Activity: variance of the signal
            activity[ch] = np.var(signal_data)

            # First derivative
            d1 = np.diff(signal_data)
            # Second derivative
            d2 = np.diff(d1)

            # Mobility: square root of variance of first derivative / variance of signal
            var_d1 = np.var(d1)
            mobility[ch] = np.sqrt(var_d1 / (activity[ch] + 1e-10))

            # Complexity: mobility of first derivative / mobility of signal
            var_d2 = np.var(d2)
            mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-10))
            complexity[ch] = mobility_d1 / (mobility[ch] + 1e-10)

        return {
            'activity': activity,
            'mobility': mobility,
            'complexity': complexity
        }


class StatisticalFeatures:
    """
    Extract statistical features from EEG signals
    """

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract statistical features

        Args:
            data: EEG data, shape (channels, samples)

        Returns:
            Dictionary of statistical features
        """
        features = {
            'mean': np.mean(data, axis=1),
            'std': np.std(data, axis=1),
            'var': np.var(data, axis=1),
            'max': np.max(data, axis=1),
            'min': np.min(data, axis=1),
            'range': np.ptp(data, axis=1),
            'median': np.median(data, axis=1),
            'mad': np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1),
            'kurtosis': self._kurtosis(data),
            'skewness': self._skewness(data),
            'rms': np.sqrt(np.mean(data**2, axis=1)),
            'zero_crossings': self._zero_crossings(data),
            'peak_to_peak': np.ptp(data, axis=1)
        }

        return features

    @staticmethod
    def _kurtosis(data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-10))**4, axis=1) - 3

    @staticmethod
    def _skewness(data: np.ndarray) -> np.ndarray:
        """Calculate skewness"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-10))**3, axis=1)

    @staticmethod
    def _zero_crossings(data: np.ndarray) -> np.ndarray:
        """Count zero crossings"""
        n_channels = data.shape[0]
        zc = np.zeros(n_channels)

        for ch in range(n_channels):
            zc[ch] = np.sum(np.diff(np.sign(data[ch, :])) != 0)

        return zc


class FractalDimension:
    """
    Calculate fractal dimension of EEG signals
    """

    def extract(self, data: np.ndarray, method: str = 'higuchi') -> Dict[str, np.ndarray]:
        """
        Extract fractal dimension

        Args:
            data: EEG data, shape (channels, samples)
            method: 'higuchi' or 'katz'

        Returns:
            Dictionary with fractal dimension
        """
        n_channels = data.shape[0]
        fd = np.zeros(n_channels)

        for ch in range(n_channels):
            if method == 'higuchi':
                fd[ch] = self._higuchi_fd(data[ch, :])
            elif method == 'katz':
                fd[ch] = self._katz_fd(data[ch, :])

        return {'fractal_dimension': fd}

    @staticmethod
    def _higuchi_fd(signal_data: np.ndarray, kmax: int = 10) -> float:
        """Calculate Higuchi fractal dimension"""
        N = len(signal_data)
        L = np.zeros(kmax)

        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lm = 0
                for i in range(1, int((N - m) / k)):
                    Lm += np.abs(signal_data[m + i * k] - signal_data[m + (i - 1) * k])
                Lm = Lm * (N - 1) / (((N - m) / k) * k)
                Lk.append(Lm)
            L[k - 1] = np.mean(Lk)

        # Fit line to log-log plot
        x = np.log(np.arange(1, kmax + 1))
        y = np.log(L)
        coeffs = np.polyfit(x, y, 1)

        return -coeffs[0]

    @staticmethod
    def _katz_fd(signal_data: np.ndarray) -> float:
        """Calculate Katz fractal dimension"""
        N = len(signal_data)
        L = np.sum(np.abs(np.diff(signal_data)))
        d = np.max(np.abs(signal_data - signal_data[0]))

        if d == 0 or L == 0:
            return 0

        return np.log10(N) / (np.log10(d / L) + np.log10(N))


def extract_all_features(
    data: np.ndarray,
    sampling_rate: int = 256
) -> Dict:
    """
    Extract all available features from EEG data

    Args:
        data: EEG data, shape (channels, samples)
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary containing all extracted features
    """
    all_features = {}

    # PSD features
    psd_extractor = PowerSpectralDensity(sampling_rate=sampling_rate)
    all_features['psd'] = psd_extractor.extract(data)

    # Connectivity features
    conn_extractor = ConnectivityAnalyzer(sampling_rate=sampling_rate)
    all_features['connectivity'] = conn_extractor.extract(data)

    # Entropy features
    entropy_extractor = EntropyCalculator()
    all_features['entropy'] = entropy_extractor.extract(data)

    # Hjorth parameters
    hjorth_extractor = HjorthParameters()
    all_features['hjorth'] = hjorth_extractor.extract(data)

    # Statistical features
    stats_extractor = StatisticalFeatures()
    all_features['statistics'] = stats_extractor.extract(data)

    # Fractal dimension
    fractal_extractor = FractalDimension()
    all_features['fractal'] = fractal_extractor.extract(data)

    return all_features
