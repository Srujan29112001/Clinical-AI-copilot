"""
Signal Filtering Module
Implements various filters for EEG signal preprocessing
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SignalFilters:
    """
    Collection of signal filters for EEG preprocessing

    Includes:
    - Bandpass filter (0.5-50 Hz)
    - Notch filter (50/60 Hz power line noise)
    - High-pass filter
    - Low-pass filter
    - Savitzky-Golay filter for smoothing
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        bandpass_low: float = 0.5,
        bandpass_high: float = 50.0,
        notch_freq: float = 50.0,
        notch_quality: float = 30.0,
        filter_order: int = 4
    ):
        """
        Initialize signal filters

        Args:
            sampling_rate: Sampling rate in Hz
            bandpass_low: Low cutoff frequency for bandpass filter
            bandpass_high: High cutoff frequency for bandpass filter
            notch_freq: Frequency to notch out (50 or 60 Hz)
            notch_quality: Quality factor for notch filter
            filter_order: Order of Butterworth filters
        """
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order

        # Design bandpass filter
        nyquist = sampling_rate / 2
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist

        self.bandpass_b, self.bandpass_a = signal.butter(
            filter_order,
            [low, high],
            btype='band'
        )

        # Design notch filter
        notch_norm = notch_freq / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(
            notch_norm,
            notch_quality
        )

        # Design high-pass filter (for DC removal)
        high_pass_cutoff = 0.5 / nyquist
        self.highpass_b, self.highpass_a = signal.butter(
            filter_order,
            high_pass_cutoff,
            btype='high'
        )

        # Design low-pass filter (for anti-aliasing)
        low_pass_cutoff = 50.0 / nyquist
        self.lowpass_b, self.lowpass_a = signal.butter(
            filter_order,
            low_pass_cutoff,
            btype='low'
        )

        logger.info(f"Filters initialized: BP [{bandpass_low}-{bandpass_high}] Hz, Notch {notch_freq} Hz")

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to data

        Args:
            data: Input data, shape (channels, samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            return signal.filtfilt(self.bandpass_b, self.bandpass_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = signal.filtfilt(
                    self.bandpass_b,
                    self.bandpass_a,
                    data[ch, :]
                )
            return filtered

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise

        Args:
            data: Input data, shape (channels, samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            return signal.filtfilt(self.notch_b, self.notch_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = signal.filtfilt(
                    self.notch_b,
                    self.notch_a,
                    data[ch, :]
                )
            return filtered

    def highpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter for DC removal

        Args:
            data: Input data, shape (channels, samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            return signal.filtfilt(self.highpass_b, self.highpass_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = signal.filtfilt(
                    self.highpass_b,
                    self.highpass_a,
                    data[ch, :]
                )
            return filtered

    def lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter for anti-aliasing

        Args:
            data: Input data, shape (channels, samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            return signal.filtfilt(self.lowpass_b, self.lowpass_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = signal.filtfilt(
                    self.lowpass_b,
                    self.lowpass_a,
                    data[ch, :]
                )
            return filtered

    def savgol_filter(
        self,
        data: np.ndarray,
        window_length: int = 51,
        polyorder: int = 3
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing

        Args:
            data: Input data, shape (channels, samples) or (samples,)
            window_length: Length of filter window (must be odd)
            polyorder: Order of polynomial fit

        Returns:
            Smoothed data with same shape
        """
        if window_length % 2 == 0:
            window_length += 1  # Must be odd

        if data.ndim == 1:
            return signal.savgol_filter(data, window_length, polyorder)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = signal.savgol_filter(
                    data[ch, :],
                    window_length,
                    polyorder
                )
            return filtered

    def adaptive_filter(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        mu: float = 0.01,
        order: int = 32
    ) -> np.ndarray:
        """
        Apply adaptive LMS filter for artifact removal

        Args:
            data: Primary input (contaminated signal)
            reference: Reference input (artifact template)
            mu: Step size parameter
            order: Filter order

        Returns:
            Filtered data
        """
        if reference is None:
            # Use delayed version of signal as reference
            reference = np.roll(data, order)

        if data.ndim == 1:
            return self._lms_filter(data, reference, mu, order)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = self._lms_filter(
                    data[ch, :],
                    reference[ch, :] if reference.ndim > 1 else reference,
                    mu,
                    order
                )
            return filtered

    @staticmethod
    def _lms_filter(
        primary: np.ndarray,
        reference: np.ndarray,
        mu: float,
        order: int
    ) -> np.ndarray:
        """
        Least Mean Squares adaptive filter implementation

        Args:
            primary: Primary input signal
            reference: Reference signal
            mu: Step size
            order: Filter order

        Returns:
            Filtered signal
        """
        n_samples = len(primary)
        weights = np.zeros(order)
        output = np.zeros(n_samples)

        for i in range(order, n_samples):
            # Get reference window
            ref_window = reference[i-order:i][::-1]

            # Filter output
            y = np.dot(weights, ref_window)

            # Error signal
            error = primary[i] - y

            # Update weights
            weights += 2 * mu * error * ref_window

            # Store output
            output[i] = error

        return output

    def detrend(self, data: np.ndarray, type: str = 'linear') -> np.ndarray:
        """
        Remove trend from data

        Args:
            data: Input data, shape (channels, samples) or (samples,)
            type: 'linear' or 'constant'

        Returns:
            Detrended data
        """
        if data.ndim == 1:
            return signal.detrend(data, type=type)
        else:
            detrended = np.zeros_like(data)
            for ch in range(data.shape[0]):
                detrended[ch, :] = signal.detrend(data[ch, :], type=type)
            return detrended

    def resample(self, data: np.ndarray, target_rate: int) -> np.ndarray:
        """
        Resample data to target sampling rate

        Args:
            data: Input data, shape (channels, samples) or (samples,)
            target_rate: Target sampling rate

        Returns:
            Resampled data
        """
        n_samples = int(data.shape[-1] * target_rate / self.sampling_rate)

        if data.ndim == 1:
            return signal.resample(data, n_samples)
        else:
            resampled = np.zeros((data.shape[0], n_samples))
            for ch in range(data.shape[0]):
                resampled[ch, :] = signal.resample(data[ch, :], n_samples)
            return resampled


class AdaptiveThresholdFilter:
    """
    Adaptive threshold filter for spike detection and artifact removal
    """

    def __init__(self, window_size: int = 100, threshold_multiplier: float = 3.0):
        """
        Initialize adaptive threshold filter

        Args:
            window_size: Size of moving window for threshold calculation
            threshold_multiplier: Multiplier for standard deviation threshold
        """
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier

    def filter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive threshold filtering

        Args:
            data: Input data, shape (channels, samples) or (samples,)

        Returns:
            Tuple of (filtered_data, artifact_mask)
        """
        if data.ndim == 1:
            return self._filter_1d(data)
        else:
            filtered = np.zeros_like(data)
            mask = np.zeros_like(data, dtype=bool)

            for ch in range(data.shape[0]):
                filtered[ch, :], mask[ch, :] = self._filter_1d(data[ch, :])

            return filtered, mask

    def _filter_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive threshold to 1D data

        Args:
            data: 1D input data

        Returns:
            Tuple of (filtered_data, artifact_mask)
        """
        n_samples = len(data)
        filtered = data.copy()
        mask = np.zeros(n_samples, dtype=bool)

        for i in range(self.window_size, n_samples - self.window_size):
            # Calculate local statistics
            window = data[i - self.window_size:i + self.window_size]
            mean = np.mean(window)
            std = np.std(window)

            # Adaptive threshold
            threshold = self.threshold_multiplier * std

            # Mark artifacts
            if np.abs(data[i] - mean) > threshold:
                mask[i] = True
                # Replace with interpolated value
                filtered[i] = mean

        return filtered, mask


def apply_common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to EEG data

    Args:
        data: EEG data, shape (channels, samples)

    Returns:
        CAR-referenced data
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D (channels, samples)")

    # Calculate average across channels
    avg = np.mean(data, axis=0, keepdims=True)

    # Subtract average from each channel
    car_data = data - avg

    return car_data


def apply_laplacian_reference(
    data: np.ndarray,
    channel_neighbors: dict
) -> np.ndarray:
    """
    Apply Laplacian (surface) reference to EEG data

    Args:
        data: EEG data, shape (channels, samples)
        channel_neighbors: Dictionary mapping channel index to neighbor indices

    Returns:
        Laplacian-referenced data
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D (channels, samples)")

    laplacian_data = np.zeros_like(data)

    for ch, neighbors in channel_neighbors.items():
        if len(neighbors) > 0:
            # Average of neighbors
            neighbor_avg = np.mean(data[neighbors, :], axis=0)
            # Laplacian: channel - average of neighbors
            laplacian_data[ch, :] = data[ch, :] - neighbor_avg
        else:
            # No neighbors, keep original
            laplacian_data[ch, :] = data[ch, :]

    return laplacian_data
