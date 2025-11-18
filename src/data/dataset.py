"""
PyTorch Dataset for Clinical EEG Data

Integrates EDF, HDF5, and FHIR loaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class ClinicalEEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data with clinical context

    Supports:
    - EEG signals (from EDF or HDF5)
    - Clinical labels (seizure, sleep stage, etc.)
    - Patient metadata (from FHIR)
    - Data augmentation
    """

    def __init__(
        self,
        eeg_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        clinical_data: Optional[List[Dict]] = None,
        transform: Optional[callable] = None,
        augment: bool = False,
    ):
        """
        Initialize dataset

        Args:
            eeg_data: EEG data (n_samples, channels, time_samples)
            labels: Labels (n_samples,)
            clinical_data: List of clinical metadata dicts
            transform: Optional transform function
            augment: Whether to apply data augmentation
        """
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.clinical_data = clinical_data
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.eeg_data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""

        eeg = self.eeg_data[idx]

        # Apply augmentation
        if self.augment and self.training:
            eeg = self._augment_eeg(eeg)

        # Apply transform
        if self.transform:
            eeg = self.transform(eeg)

        sample = {'eeg': eeg}

        # Add label
        if self.labels is not None:
            sample['label'] = self.labels[idx]

        # Add clinical data
        if self.clinical_data:
            sample['clinical'] = self.clinical_data[idx]

        return sample

    def _augment_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to EEG signal

        Augmentations:
        - Time shifting
        - Amplitude scaling
        - Gaussian noise
        - Channel dropout
        """
        # Time shifting (random offset)
        if torch.rand(1) < 0.5:
            shift = torch.randint(-20, 20, (1,)).item()
            eeg = torch.roll(eeg, shift, dims=-1)

        # Amplitude scaling
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(0.9, 1.1)
            eeg = eeg * scale

        # Gaussian noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(eeg) * 0.01
            eeg = eeg + noise

        # Channel dropout
        if torch.rand(1) < 0.2:
            n_channels = eeg.shape[0]
            dropout_mask = torch.rand(n_channels) > 0.1
            eeg = eeg * dropout_mask.unsqueeze(1)

        return eeg


class MultimodalEEGDataset(Dataset):
    """
    Dataset with both EEG and text (clinical notes)
    For training multimodal fusion model
    """

    def __init__(
        self,
        eeg_data: np.ndarray,
        text_data: List[str],
        labels: Optional[np.ndarray] = None,
        text_encoder: Optional[callable] = None,
    ):
        """
        Initialize multimodal dataset

        Args:
            eeg_data: EEG data (n_samples, channels, time_samples)
            text_data: List of clinical text strings
            labels: Labels (n_samples,)
            text_encoder: Function to encode text (e.g., BERT, Llama)
        """
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.text_data = text_data
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.text_encoder = text_encoder

    def __len__(self) -> int:
        return len(self.eeg_data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""

        eeg = self.eeg_data[idx]
        text = self.text_data[idx]

        # Encode text if encoder provided
        if self.text_encoder:
            text_encoded = self.text_encoder(text)
        else:
            # Return raw text
            text_encoded = text

        sample = {
            'eeg': eeg,
            'text': text_encoded,
        }

        if self.labels is not None:
            sample['label'] = self.labels[idx]

        return sample


class StreamingEEGDataset(Dataset):
    """
    Dataset for streaming/online learning
    Loads data on-the-fly from HDF5
    """

    def __init__(
        self,
        hdf5_path: str,
        patient_ids: List[str],
        hdf5_loader,
        window_size: int = 256,
    ):
        """
        Initialize streaming dataset

        Args:
            hdf5_path: Path to HDF5 file
            patient_ids: List of patient IDs to include
            hdf5_loader: HDF5DataLoader instance
            window_size: Window size in samples
        """
        self.hdf5_path = hdf5_path
        self.patient_ids = patient_ids
        self.hdf5_loader = hdf5_loader
        self.window_size = window_size

        # Build index of (patient_id, window_idx) pairs
        self._build_index()

    def _build_index(self):
        """Build index of all available windows"""
        self.index = []

        for patient_id in self.patient_ids:
            # Get patient data dimensions without loading full data
            import h5py
            with h5py.File(self.hdf5_path, 'r') as f:
                if patient_id not in f:
                    continue

                eeg_shape = f[patient_id]['eeg_data'].shape
                n_samples = eeg_shape[1]

                # Calculate number of windows
                n_windows = (n_samples - self.window_size) // (self.window_size // 2)

                for win_idx in range(n_windows):
                    self.index.append((patient_id, win_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """Load a single window on-the-fly"""

        patient_id, window_idx = self.index[idx]

        # Load patient data
        eeg_data, labels, metadata = self.hdf5_loader.load_patient_data(
            self.hdf5_path,
            patient_id
        )

        # Extract window
        start = window_idx * (self.window_size // 2)
        end = start + self.window_size

        eeg_window = eeg_data[:, start:end]

        sample = {
            'eeg': torch.FloatTensor(eeg_window),
            'patient_id': patient_id,
        }

        if labels is not None:
            # Get label for this window
            window_labels = labels[start:end]
            # Majority vote
            label = int(window_labels.mean() > 0.5)
            sample['label'] = torch.LongTensor([label])

        return sample


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create PyTorch DataLoader with optimal settings

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test

    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    from torch.utils.data import random_split

    generator = torch.Generator().manual_seed(random_seed)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test dataset
    n_samples = 100
    n_channels = 16
    time_samples = 256

    eeg_data = np.random.randn(n_samples, n_channels, time_samples)
    labels = np.random.randint(0, 2, n_samples)

    dataset = ClinicalEEGDataset(eeg_data, labels, augment=True)

    dataloader = create_dataloader(dataset, batch_size=8)

    # Test batch
    for batch in dataloader:
        print(f"Batch EEG shape: {batch['eeg'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        break

    print("Dataset ready")
