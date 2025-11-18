"""
EEG Signal Processing Module
Handles real-time EEG data acquisition, preprocessing, and feature extraction
"""

from .eeg_processor import EEGProcessingPipeline
from .feature_extraction import (
    PowerSpectralDensity,
    ConnectivityAnalyzer,
    EntropyCalculator,
    HjorthParameters
)
from .filters import SignalFilters

__all__ = [
    'EEGProcessingPipeline',
    'PowerSpectralDensity',
    'ConnectivityAnalyzer',
    'EntropyCalculator',
    'HjorthParameters',
    'SignalFilters'
]

__version__ = '1.0.0'
