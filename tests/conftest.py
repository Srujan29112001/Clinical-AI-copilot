"""
Pytest Configuration and Fixtures
Shared fixtures for all tests
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get available device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data"""
    np.random.seed(42)
    return np.random.randn(16, 256).astype(np.float32)


@pytest.fixture
def sample_eeg_windows():
    """Generate sample EEG windows for seizure detection"""
    np.random.seed(42)
    return np.random.randn(10, 16, 256).astype(np.float32)


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        'patient_id': 'TEST_PAT_001',
        'age': 45,
        'sex': 'M',
        'symptoms': 'recurrent seizures',
        'history': 'epilepsy diagnosis 5 years ago'
    }


@pytest.fixture(scope='session')
def temp_model_path(tmp_path_factory):
    """Temporary path for saving models during tests"""
    return tmp_path_factory.mktemp('models')


@pytest.fixture(scope='session')
def temp_data_path(tmp_path_factory):
    """Temporary path for test data"""
    return tmp_path_factory.mktemp('data')


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
