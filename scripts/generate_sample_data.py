"""
Generate Sample EEG Data for Testing
Creates synthetic EEG data with realistic characteristics
"""

import numpy as np
import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_realistic_eeg(
    duration_seconds: int = 60,
    sampling_rate: int = 256,
    channels: int = 16,
    add_seizure: bool = False,
    seizure_start: float = 30.0,
    seizure_duration: float = 10.0
) -> np.ndarray:
    """
    Generate realistic EEG data

    Args:
        duration_seconds: Duration in seconds
        sampling_rate: Sampling rate in Hz
        channels: Number of channels
        add_seizure: Whether to add seizure activity
        seizure_start: Seizure start time (seconds)
        seizure_duration: Seizure duration (seconds)

    Returns:
        EEG data array, shape (channels, samples)
    """
    num_samples = duration_seconds * sampling_rate
    t = np.linspace(0, duration_seconds, num_samples)

    eeg_data = np.zeros((channels, num_samples))

    for ch in range(channels):
        # Background activity
        # Alpha rhythm (8-13 Hz) - dominant in posterior regions
        alpha_freq = 10 + np.random.randn() * 0.5
        alpha_amp = 20 + np.random.randn() * 5
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

        # Beta rhythm (13-30 Hz) - active thinking
        beta_freq = 20 + np.random.randn() * 2
        beta_amp = 10 + np.random.randn() * 2
        beta = beta_amp * np.sin(2 * np.pi * beta_freq * t)

        # Theta rhythm (4-8 Hz) - drowsiness
        theta_freq = 6 + np.random.randn() * 0.5
        theta_amp = 15 + np.random.randn() * 3
        theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)

        # Delta rhythm (0.5-4 Hz) - deep sleep
        delta_freq = 2 + np.random.randn() * 0.3
        delta_amp = 25 + np.random.randn() * 5
        delta = delta_amp * np.sin(2 * np.pi * delta_freq * t)

        # Combine rhythms
        eeg_data[ch, :] = alpha + 0.5 * beta + 0.3 * theta + 0.2 * delta

        # Add pink noise (1/f noise common in EEG)
        noise = np.random.randn(num_samples)
        from scipy import signal
        noise_filtered = signal.lfilter([1], [1, -0.95], noise)
        eeg_data[ch, :] += 5 * noise_filtered

    # Add seizure activity if requested
    if add_seizure:
        seizure_start_sample = int(seizure_start * sampling_rate)
        seizure_end_sample = int((seizure_start + seizure_duration) * sampling_rate)

        for ch in range(channels):
            # High-frequency oscillations during seizure
            seizure_freq = 15 + np.random.randn() * 3
            seizure_amp = 100 + np.random.randn() * 20

            t_seizure = t[seizure_start_sample:seizure_end_sample]
            seizure_signal = seizure_amp * np.sin(2 * np.pi * seizure_freq * t_seizure)

            # Add spike-wave pattern
            spike_freq = 3  # 3 Hz spike-wave
            spikes = 50 * signal.square(2 * np.pi * spike_freq * t_seizure)

            eeg_data[ch, seizure_start_sample:seizure_end_sample] += seizure_signal + spikes

        logger.info(f"Added seizure from {seizure_start}s to {seizure_start + seizure_duration}s")

    return eeg_data.astype(np.float32)


def main():
    """Generate and save sample EEG data"""
    parser = argparse.ArgumentParser(description='Generate Sample EEG Data')
    parser.add_argument('--output', type=str, default='./data/sample/eeg_sample.raw')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--channels', type=int, default=16, help='Number of channels')
    parser.add_argument('--sampling-rate', type=int, default=256, help='Sampling rate (Hz)')
    parser.add_argument('--with-seizure', action='store_true', help='Add seizure activity')
    parser.add_argument('--num-files', type=int, default=1, help='Number of files to generate')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_files):
        # Generate EEG data
        eeg_data = generate_realistic_eeg(
            duration_seconds=args.duration,
            sampling_rate=args.sampling_rate,
            channels=args.channels,
            add_seizure=args.with_seizure
        )

        # Save to file
        if args.num_files > 1:
            filename = output_path.parent / f"{output_path.stem}_{i+1}{output_path.suffix}"
        else:
            filename = output_path

        eeg_data.tofile(filename)

        logger.info(
            f"Generated EEG data: {filename}\n"
            f"  Shape: {eeg_data.shape}\n"
            f"  Duration: {args.duration}s\n"
            f"  Channels: {args.channels}\n"
            f"  Sampling Rate: {args.sampling_rate} Hz\n"
            f"  Seizure: {'Yes' if args.with_seizure else 'No'}"
        )

    logger.info(f"Generated {args.num_files} EEG file(s) successfully!")


if __name__ == "__main__":
    main()
