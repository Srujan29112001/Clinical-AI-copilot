"""
Kafka Streaming Module
Real-time EEG data streaming and processing
"""

from .producer import EEGStreamProducer
from .consumer import EEGStreamConsumer
from .processor import StreamProcessor

__all__ = [
    'EEGStreamProducer',
    'EEGStreamConsumer',
    'StreamProcessor'
]

__version__ = '1.0.0'
