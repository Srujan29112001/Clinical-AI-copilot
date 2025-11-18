"""
HDF5 Data Loader for Preprocessed EEG Data

HDF5 is efficient for large-scale medical datasets
Supports hierarchical organization of patient data
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class HDF5DataLoader:
    """
    Load preprocessed EEG data from HDF5 files

    HDF5 structure:
    /
    ├── patient_001/
    │   ├── eeg_data (channels, samples)
    │   ├── labels (samples,)
    │   ├── metadata (attributes)
    │   └── features/ (optional)
    │       ├── psd
    │       ├── connectivity
    │       └── entropy
    ├── patient_002/
    │   └── ...
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize HDF5 loader

        Args:
            chunk_size: Number of samples to load at a time
        """
        self.chunk_size = chunk_size

    def load_patient_data(
        self,
        file_path: str,
        patient_id: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Load data for a specific patient

        Args:
            file_path: Path to HDF5 file
            patient_id: Patient identifier

        Returns:
            Tuple of (eeg_data, labels, metadata)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            if patient_id not in f:
                raise ValueError(f"Patient {patient_id} not found in {file_path}")

            patient_group = f[patient_id]

            # Load EEG data
            eeg_data = patient_group['eeg_data'][:]  # (channels, samples)

            # Load labels if available
            labels = None
            if 'labels' in patient_group:
                labels = patient_group['labels'][:]

            # Load metadata
            metadata = dict(patient_group.attrs)
            metadata['patient_id'] = patient_id

            # Load features if available
            if 'features' in patient_group:
                features = {}
                for feature_name in patient_group['features'].keys():
                    features[feature_name] = patient_group['features'][feature_name][:]
                metadata['features'] = features

        logger.info(
            f"Loaded patient {patient_id}: "
            f"{eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples"
        )

        return eeg_data, labels, metadata

    def load_all_patients(
        self,
        file_path: str
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray], Dict]]:
        """
        Load data for all patients in HDF5 file

        Args:
            file_path: Path to HDF5 file

        Returns:
            List of (eeg_data, labels, metadata) tuples
        """
        file_path = Path(file_path)

        all_data = []

        with h5py.File(file_path, 'r') as f:
            patient_ids = list(f.keys())
            logger.info(f"Found {len(patient_ids)} patients in {file_path}")

            for patient_id in patient_ids:
                try:
                    data = self.load_patient_data(str(file_path), patient_id)
                    all_data.append(data)
                except Exception as e:
                    logger.error(f"Error loading patient {patient_id}: {e}")
                    continue

        return all_data

    def create_hdf5_from_arrays(
        self,
        output_path: str,
        patient_data: Dict[str, Dict],
        compression: str = 'gzip',
        compression_opts: int = 9
    ):
        """
        Create HDF5 file from numpy arrays

        Args:
            output_path: Output HDF5 file path
            patient_data: Dictionary of patient data
                {
                    'patient_001': {
                        'eeg_data': array,
                        'labels': array,
                        'metadata': dict,
                        'features': dict (optional)
                    },
                    ...
                }
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            for patient_id, data in patient_data.items():
                # Create patient group
                patient_group = f.create_group(patient_id)

                # Store EEG data
                patient_group.create_dataset(
                    'eeg_data',
                    data=data['eeg_data'],
                    compression=compression,
                    compression_opts=compression_opts
                )

                # Store labels if available
                if 'labels' in data and data['labels'] is not None:
                    patient_group.create_dataset(
                        'labels',
                        data=data['labels'],
                        compression=compression
                    )

                # Store metadata as attributes
                if 'metadata' in data:
                    for key, value in data['metadata'].items():
                        if isinstance(value, (str, int, float, bool)):
                            patient_group.attrs[key] = value

                # Store features
                if 'features' in data and data['features']:
                    features_group = patient_group.create_group('features')
                    for feature_name, feature_data in data['features'].items():
                        features_group.create_dataset(
                            feature_name,
                            data=feature_data,
                            compression=compression
                        )

        logger.info(f"Created HDF5 file: {output_path}")

    def load_batch_iterator(
        self,
        file_path: str,
        patient_id: str,
        batch_size: int = 256
    ):
        """
        Create an iterator for loading data in batches

        Useful for large datasets that don't fit in memory

        Args:
            file_path: Path to HDF5 file
            patient_id: Patient identifier
            batch_size: Number of samples per batch

        Yields:
            Batches of (eeg_data, labels) or (eeg_data, None)
        """
        with h5py.File(file_path, 'r') as f:
            patient_group = f[patient_id]
            eeg_data = patient_group['eeg_data']
            labels = patient_group.get('labels', None)

            n_samples = eeg_data.shape[1]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                batch_eeg = eeg_data[:, start:end]
                batch_labels = labels[start:end] if labels is not None else None

                yield batch_eeg, batch_labels

    def append_patient_data(
        self,
        file_path: str,
        patient_id: str,
        eeg_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        features: Optional[Dict] = None
    ):
        """
        Append new patient data to existing HDF5 file

        Args:
            file_path: Path to HDF5 file
            patient_id: Patient identifier
            eeg_data: EEG data array
            labels: Optional labels
            metadata: Optional metadata dictionary
            features: Optional features dictionary
        """
        with h5py.File(file_path, 'a') as f:
            if patient_id in f:
                logger.warning(f"Patient {patient_id} already exists, skipping")
                return

            patient_group = f.create_group(patient_id)

            patient_group.create_dataset(
                'eeg_data',
                data=eeg_data,
                compression='gzip'
            )

            if labels is not None:
                patient_group.create_dataset(
                    'labels',
                    data=labels,
                    compression='gzip'
                )

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        patient_group.attrs[key] = value

            if features:
                features_group = patient_group.create_group('features')
                for feature_name, feature_data in features.items():
                    features_group.create_dataset(
                        feature_name,
                        data=feature_data,
                        compression='gzip'
                    )

        logger.info(f"Appended patient {patient_id} to {file_path}")

    def get_dataset_statistics(self, file_path: str) -> Dict:
        """
        Get statistics about the HDF5 dataset

        Args:
            file_path: Path to HDF5 file

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_patients': 0,
            'total_samples': 0,
            'total_size_gb': 0,
            'channels': [],
            'sampling_rates': [],
            'durations': [],
        }

        with h5py.File(file_path, 'r') as f:
            stats['n_patients'] = len(f.keys())

            for patient_id in f.keys():
                patient_group = f[patient_id]
                eeg_data = patient_group['eeg_data']

                stats['total_samples'] += eeg_data.shape[1]
                stats['channels'].append(eeg_data.shape[0])

                if 'sfreq' in patient_group.attrs:
                    stats['sampling_rates'].append(patient_group.attrs['sfreq'])

                if 'duration' in patient_group.attrs:
                    stats['durations'].append(patient_group.attrs['duration'])

            # Calculate file size
            file_size = Path(file_path).stat().st_size
            stats['total_size_gb'] = file_size / (1024**3)

        return stats


def convert_edf_to_hdf5(
    edf_dir: str,
    output_hdf5: str,
    edf_loader,
    window_size: int = 256
):
    """
    Convert EDF files to HDF5 format

    Args:
        edf_dir: Directory containing EDF files
        output_hdf5: Output HDF5 file path
        edf_loader: EDFDataLoader instance
        window_size: Window size for segmentation
    """
    edf_dir = Path(edf_dir)
    edf_files = list(edf_dir.glob('**/*.edf'))

    logger.info(f"Converting {len(edf_files)} EDF files to HDF5")

    hdf5_loader = HDF5DataLoader()
    patient_data = {}

    for idx, edf_file in enumerate(edf_files):
        try:
            data, metadata = edf_loader.load_edf(str(edf_file))
            segments, labels = edf_loader.segment_by_annotations(
                data, metadata, window_size=window_size
            )

            patient_id = f"patient_{idx:04d}"

            patient_data[patient_id] = {
                'eeg_data': data,
                'labels': labels,
                'metadata': {
                    'sfreq': metadata['sfreq'],
                    'n_channels': metadata['n_channels'],
                    'duration': metadata['duration'],
                    'source_file': str(edf_file),
                }
            }

        except Exception as e:
            logger.error(f"Error converting {edf_file}: {e}")
            continue

    # Create HDF5 file
    hdf5_loader.create_hdf5_from_arrays(output_hdf5, patient_data)

    logger.info(f"HDF5 file created: {output_hdf5}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    loader = HDF5DataLoader()

    # Get dataset statistics
    # stats = loader.get_dataset_statistics('path/to/data.h5')
    # print(f"Dataset statistics: {stats}")

    print("HDF5 loader ready")
