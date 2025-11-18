"""
Kafka Producer for EEG Streaming
Publishes real-time EEG data to Kafka topics
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

try:
    from aiokafka import AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("aiokafka not available, using mock producer")

logger = logging.getLogger(__name__)


class EEGStreamProducer:
    """
    Asynchronous Kafka producer for real-time EEG streaming

    Features:
    - Async/await for non-blocking operations
    - Automatic reconnection
    - Batching for efficiency
    - Error handling and retries
    """

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'eeg-stream',
        batch_size: int = 16384,
        compression_type: str = 'gzip'
    ):
        """
        Initialize EEG stream producer

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic name
            batch_size: Batch size for sending
            compression_type: Compression algorithm
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        self.compression_type = compression_type

        self.producer: Optional[AIOKafkaProducer] = None
        self.is_running = False

    async def start(self):
        """Start the producer"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, running in mock mode")
            self.is_running = True
            return

        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type=self.compression_type,
                max_batch_size=self.batch_size,
                linger_ms=10  # Wait 10ms for batching
            )

            await self.producer.start()
            self.is_running = True
            logger.info(f"EEG producer started, topic: {self.topic}")

        except Exception as e:
            logger.error(f"Failed to start producer: {e}")
            raise

    async def stop(self):
        """Stop the producer"""
        if self.producer:
            await self.producer.stop()
        self.is_running = False
        logger.info("EEG producer stopped")

    async def send_eeg_chunk(
        self,
        patient_id: str,
        eeg_data: np.ndarray,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send EEG data chunk to Kafka

        Args:
            patient_id: Patient identifier
            eeg_data: EEG data array (channels, samples)
            timestamp: Data timestamp
            metadata: Additional metadata
        """
        if not self.is_running:
            raise RuntimeError("Producer not started")

        # Prepare message
        message = {
            'patient_id': patient_id,
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'eeg_data': eeg_data.tolist(),  # Convert to list for JSON
            'shape': eeg_data.shape,
            'metadata': metadata or {}
        }

        if KAFKA_AVAILABLE and self.producer:
            try:
                # Send to Kafka
                await self.producer.send_and_wait(self.topic, message)
                logger.debug(f"Sent EEG chunk for patient {patient_id}")

            except Exception as e:
                logger.error(f"Failed to send EEG chunk: {e}")
                raise
        else:
            # Mock mode
            logger.debug(f"[MOCK] Sent EEG chunk for patient {patient_id}")

    async def stream_eeg_file(
        self,
        patient_id: str,
        eeg_file_path: str,
        chunk_size: int = 256,
        sampling_rate: int = 256,
        channels: int = 16,
        real_time: bool = True
    ) -> None:
        """
        Stream EEG data from file in chunks

        Args:
            patient_id: Patient identifier
            eeg_file_path: Path to EEG file
            chunk_size: Samples per chunk
            sampling_rate: Sampling rate in Hz
            channels: Number of channels
            real_time: If True, simulate real-time streaming with delays
        """
        # Load EEG file
        try:
            eeg_data = np.fromfile(eeg_file_path, dtype=np.float32)
            eeg_data = eeg_data.reshape(-1, channels).T  # (channels, samples)
        except Exception as e:
            logger.error(f"Failed to load EEG file: {e}")
            raise

        # Calculate delay for real-time simulation
        delay_seconds = chunk_size / sampling_rate if real_time else 0

        # Stream in chunks
        num_samples = eeg_data.shape[1]
        for i in range(0, num_samples, chunk_size):
            chunk = eeg_data[:, i:i+chunk_size]

            # Pad last chunk if needed
            if chunk.shape[1] < chunk_size:
                padding = np.zeros((channels, chunk_size - chunk.shape[1]))
                chunk = np.hstack([chunk, padding])

            # Send chunk
            await self.send_eeg_chunk(
                patient_id=patient_id,
                eeg_data=chunk,
                metadata={
                    'chunk_index': i // chunk_size,
                    'sampling_rate': sampling_rate,
                    'source': eeg_file_path
                }
            )

            # Delay for real-time simulation
            if real_time:
                await asyncio.sleep(delay_seconds)

        logger.info(f"Finished streaming EEG file for patient {patient_id}")

    async def send_alert(
        self,
        patient_id: str,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Send alert to alerts topic

        Args:
            patient_id: Patient identifier
            alert_type: Type of alert (SEIZURE, ANOMALY, etc.)
            severity: Alert severity (LOW, MEDIUM, HIGH, CRITICAL)
            message: Alert message
            data: Additional data
        """
        alert_topic = 'clinical-alerts'

        alert_message = {
            'patient_id': patient_id,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }

        if KAFKA_AVAILABLE and self.producer:
            try:
                await self.producer.send_and_wait(alert_topic, alert_message)
                logger.info(f"Sent {severity} alert for patient {patient_id}: {alert_type}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            logger.info(f"[MOCK] Alert: {alert_type} - {severity} - {message}")


async def main():
    """Example usage"""
    producer = EEGStreamProducer()

    try:
        await producer.start()

        # Simulate EEG data
        patient_id = "PAT12345"
        eeg_chunk = np.random.randn(16, 256).astype(np.float32)

        # Send chunk
        await producer.send_eeg_chunk(patient_id, eeg_chunk)

        # Send alert
        await producer.send_alert(
            patient_id=patient_id,
            alert_type='SEIZURE_DETECTED',
            severity='HIGH',
            message='Seizure activity detected with 94% confidence',
            data={'confidence': 0.94, 'duration': 30}
        )

        print("Sent EEG chunk and alert successfully")

    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
