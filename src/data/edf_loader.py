"""
EDF (European Data Format) Loader for EEG Data

Supports:
- EDF and EDF+ formats
- Multiple channel configurations
- Annotation extraction
- Resampling and channel selection
"""

import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EDFDataLoader:
    """
    Load and preprocess EEG data from EDF files

    Features:
    - Automatic channel detection
    - Resampling to target frequency
    - Annotation extraction (seizure markers, artifacts, etc.)
    - Standard montage mapping
    """

    def __init__(
        self,
        target_sfreq: int = 256,
        channels: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize EDF loader

        Args:
            target_sfreq: Target sampling frequency
            channels: List of channel names to extract (None = all)
            verbose: Whether to print loading information
        """
        self.target_sfreq = target_sfreq
        self.channels = channels
        self.verbose = verbose

        # Standard EEG channel names (10-20 system)
        self.standard_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz'
        ]

    def load_edf(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load EDF file

        Args:
            file_path: Path to EDF file

        Returns:
            Tuple of (data, metadata)
            - data: numpy array (channels, samples)
            - metadata: dict with sampling rate, channel names, annotations
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")

        logger.info(f"Loading EDF file: {file_path}")

        try:
            # Load raw EDF
            raw = mne.io.read_raw_edf(
                file_path,
                preload=True,
                verbose=self.verbose
            )

            # Get channel information
            available_channels = raw.ch_names
            logger.info(f"Available channels: {len(available_channels)}")

            # Select channels
            if self.channels:
                # Try to find matching channels
                selected_channels = [
                    ch for ch in self.channels if ch in available_channels
                ]
                if not selected_channels:
                    logger.warning(
                        f"None of the requested channels found. "
                        f"Using all available channels."
                    )
                    selected_channels = available_channels
            else:
                # Try to use standard EEG channels
                selected_channels = [
                    ch for ch in self.standard_channels if ch in available_channels
                ]
                if not selected_channels:
                    # Use all channels
                    selected_channels = available_channels

            # Pick channels
            raw.pick_channels(selected_channels)

            # Resample if needed
            current_sfreq = raw.info['sfreq']
            if current_sfreq != self.target_sfreq:
                logger.info(f"Resampling from {current_sfreq} Hz to {self.target_sfreq} Hz")
                raw.resample(self.target_sfreq)

            # Get data
            data = raw.get_data()  # (channels, samples)

            # Extract annotations (seizures, artifacts, etc.)
            annotations = self._extract_annotations(raw)

            # Metadata
            metadata = {
                'sfreq': self.target_sfreq,
                'channels': selected_channels,
                'n_channels': len(selected_channels),
                'duration': raw.times[-1],
                'annotations': annotations,
                'file_path': str(file_path),
            }

            logger.info(
                f"Loaded EDF: {metadata['n_channels']} channels, "
                f"{metadata['duration']:.1f} seconds"
            )

            return data, metadata

        except Exception as e:
            logger.error(f"Error loading EDF file: {e}")
            raise

    def _extract_annotations(self, raw: mne.io.Raw) -> List[Dict]:
        """Extract annotations from EDF file"""

        annotations = []

        if raw.annotations:
            for ann in raw.annotations:
                annotations.append({
                    'onset': ann['onset'],
                    'duration': ann['duration'],
                    'description': ann['description'],
                })

        return annotations

    def load_tuh_eeg(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load Temple University Hospital (TUH) EEG dataset format

        The TUH EEG corpus is a major public EEG dataset
        """
        # TUH EEG uses EDF+ format with specific naming conventions
        data, metadata = self.load_edf(file_path)

        # Extract seizure labels if present
        seizure_intervals = []
        for ann in metadata['annotations']:
            desc = ann['description'].lower()
            if 'seiz' in desc or 'bckg' in desc:
                seizure_intervals.append({
                    'start': ann['onset'],
                    'end': ann['onset'] + ann['duration'],
                    'type': 'seizure' if 'seiz' in desc else 'background',
                })

        metadata['seizure_intervals'] = seizure_intervals

        return data, metadata

    def load_chb_mit(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load CHB-MIT Scalp EEG Database

        Pediatric seizure database from Boston Children's Hospital
        """
        data, metadata = self.load_edf(file_path)

        # CHB-MIT uses specific channel naming
        # Map to standard 10-20 if needed
        channel_mapping = self._get_chb_mit_channel_mapping()

        # Extract seizure times from annotations or summary file
        summary_file = Path(file_path).parent / f"{Path(file_path).stem}-summary.txt"
        if summary_file.exists():
            seizure_times = self._parse_chb_mit_summary(summary_file)
            metadata['seizure_times'] = seizure_times

        return data, metadata

    def _get_chb_mit_channel_mapping(self) -> Dict[str, str]:
        """Map CHB-MIT channel names to standard 10-20"""
        return {
            'FP1-F7': 'Fp1',
            'F7-T7': 'F7',
            'T7-P7': 'T3',
            'P7-O1': 'O1',
            'FP1-F3': 'F3',
            'F3-C3': 'C3',
            'C3-P3': 'P3',
            'P3-O1': 'P3',
            'FP2-F4': 'Fp2',
            'F4-C4': 'F4',
            'C4-P4': 'C4',
            'P4-O2': 'P4',
            'FP2-F8': 'F8',
            'F8-T8': 'T4',
            'T8-P8': 'T6',
            'P8-O2': 'O2',
        }

    def _parse_chb_mit_summary(self, summary_file: Path) -> List[Dict]:
        """Parse CHB-MIT summary file for seizure times"""
        seizures = []

        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                if 'Seizure' in lines[i]:
                    # Extract start and end times
                    start = None
                    end = None

                    if 'Start Time' in lines[i + 1]:
                        start = int(lines[i + 1].split(':')[1].strip().split()[0])

                    if 'End Time' in lines[i + 2]:
                        end = int(lines[i + 2].split(':')[1].strip().split()[0])

                    if start is not None and end is not None:
                        seizures.append({
                            'start': start,
                            'end': end,
                            'duration': end - start,
                        })

                i += 1

        except Exception as e:
            logger.warning(f"Could not parse summary file: {e}")

        return seizures

    def segment_by_annotations(
        self,
        data: np.ndarray,
        metadata: Dict,
        window_size: int = 256,
        label_type: str = 'seizure'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment EEG data into windows with labels

        Args:
            data: EEG data (channels, samples)
            metadata: Metadata with annotations
            window_size: Window size in samples
            label_type: Type of label to extract

        Returns:
            Tuple of (segments, labels)
            - segments: (n_windows, channels, window_size)
            - labels: (n_windows,) binary labels
        """
        sfreq = metadata['sfreq']
        total_samples = data.shape[1]

        segments = []
        labels = []

        # Create windows
        for start in range(0, total_samples - window_size, window_size // 2):  # 50% overlap
            end = start + window_size

            if end > total_samples:
                break

            segment = data[:, start:end]

            # Determine label
            time_start = start / sfreq
            time_end = end / sfreq

            label = self._get_label_for_window(
                time_start, time_end, metadata, label_type
            )

            segments.append(segment)
            labels.append(label)

        segments = np.array(segments)
        labels = np.array(labels)

        return segments, labels

    def _get_label_for_window(
        self,
        time_start: float,
        time_end: float,
        metadata: Dict,
        label_type: str
    ) -> int:
        """Determine label for a time window"""

        # Check seizure intervals
        if 'seizure_intervals' in metadata:
            for interval in metadata['seizure_intervals']:
                if interval['type'] == label_type:
                    # Check overlap
                    if (time_start < interval['end'] and time_end > interval['start']):
                        return 1  # Seizure

        # Check annotations
        for ann in metadata['annotations']:
            desc = ann['description'].lower()
            ann_start = ann['onset']
            ann_end = ann['onset'] + ann['duration']

            if label_type.lower() in desc:
                if (time_start < ann_end and time_end > ann_start):
                    return 1

        return 0  # Normal/background


def load_edf_dataset(
    data_dir: str,
    target_sfreq: int = 256,
    channels: Optional[List[str]] = None,
    window_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load multiple EDF files from a directory

    Args:
        data_dir: Directory containing EDF files
        target_sfreq: Target sampling frequency
        channels: Channel names to extract
        window_size: Window size for segmentation

    Returns:
        Tuple of (all_segments, all_labels, metadata_list)
    """
    loader = EDFDataLoader(target_sfreq=target_sfreq, channels=channels)

    data_dir = Path(data_dir)
    edf_files = list(data_dir.glob('**/*.edf'))

    logger.info(f"Found {len(edf_files)} EDF files in {data_dir}")

    all_segments = []
    all_labels = []
    metadata_list = []

    for edf_file in edf_files:
        try:
            data, metadata = loader.load_edf(str(edf_file))
            segments, labels = loader.segment_by_annotations(
                data, metadata, window_size=window_size
            )

            all_segments.append(segments)
            all_labels.append(labels)
            metadata_list.append(metadata)

        except Exception as e:
            logger.error(f"Error loading {edf_file}: {e}")
            continue

    # Concatenate all data
    if all_segments:
        all_segments = np.concatenate(all_segments, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    return all_segments, all_labels, metadata_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    loader = EDFDataLoader(target_sfreq=256)

    # Load a single EDF file
    # data, metadata = loader.load_edf('path/to/file.edf')
    # print(f"Loaded data shape: {data.shape}")
    # print(f"Metadata: {metadata}")

    print("EDF loader ready")
