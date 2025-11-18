"""
Stream Processor for Real-Time EEG Analysis
Integrates Kafka streaming with EEG processing and model inference
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
from datetime import datetime

from .consumer import EEGStreamConsumer
from .producer import EEGStreamProducer
from src.signal_processing.eeg_processor import EEGProcessingPipeline
from src.models.cnn_lstm import HybridCNNLSTM
from src.utils.audit_log import HIPAALogger

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Real-time EEG stream processor

    Integrates:
    - Kafka consumer (receives EEG data)
    - EEG processing pipeline (feature extraction)
    - Deep learning models (seizure detection)
    - Kafka producer (sends alerts)
    - Audit logging (HIPAA compliance)
    """

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        eeg_topic: str = 'eeg-stream',
        alert_topic: str = 'clinical-alerts',
        consumer_group: str = 'eeg-processor-group',
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize stream processor

        Args:
            bootstrap_servers: Kafka servers
            eeg_topic: Input topic for EEG data
            alert_topic: Output topic for alerts
            consumer_group: Kafka consumer group
            model_path: Path to trained model
            device: Device for inference
        """
        # Kafka components
        self.consumer = EEGStreamConsumer(
            bootstrap_servers=bootstrap_servers,
            topic=eeg_topic,
            group_id=consumer_group
        )

        self.producer = EEGStreamProducer(
            bootstrap_servers=bootstrap_servers,
            topic=alert_topic
        )

        # Processing components
        self.eeg_processor = EEGProcessingPipeline(
            sampling_rate=256,
            channels=16
        )

        # Model
        self.device = device
        self.model: Optional[HybridCNNLSTM] = None
        self.model_path = model_path

        # Audit logging
        self.audit_logger = HIPAALogger()

        # Buffers for windowing
        self.patient_buffers: Dict[str, list] = {}
        self.window_size = 10  # 10 chunks = 10 seconds

        # Statistics
        self.processed_count = 0
        self.alert_count = 0

    async def start(self):
        """Start the stream processor"""
        logger.info("Starting stream processor...")

        # Start Kafka components
        await self.consumer.start()
        await self.producer.start()

        # Load model if path provided
        if self.model_path:
            self._load_model()

        # Set message callback
        self.consumer.set_callback(self._process_eeg_message)

        logger.info("Stream processor started successfully")

    async def stop(self):
        """Stop the stream processor"""
        logger.info("Stopping stream processor...")
        await self.consumer.stop()
        await self.producer.stop()
        logger.info(f"Processed {self.processed_count} chunks, sent {self.alert_count} alerts")

    def _load_model(self):
        """Load trained seizure detection model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            from src.models.cnn_lstm import SeizureDetectionConfig
            config = SeizureDetectionConfig()
            self.model = HybridCNNLSTM(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    async def _process_eeg_message(self, message: Dict[str, Any]):
        """
        Process incoming EEG message

        Args:
            message: EEG message from Kafka
        """
        try:
            patient_id = message['patient_id']
            eeg_data = np.array(message['eeg_data'], dtype=np.float32)
            timestamp = message['timestamp']
            metadata = message.get('metadata', {})

            logger.debug(f"Processing EEG chunk for patient {patient_id}")

            # 1. Process chunk with signal processor
            features, wavelets = self.eeg_processor.process_chunk(eeg_data)

            # 2. Detect anomalies
            anomalies = self.eeg_processor.detect_anomalies(features)

            # 3. Add to buffer for windowing
            if patient_id not in self.patient_buffers:
                self.patient_buffers[patient_id] = []

            self.patient_buffers[patient_id].append(eeg_data)

            # Keep only last N chunks
            if len(self.patient_buffers[patient_id]) > self.window_size:
                self.patient_buffers[patient_id].pop(0)

            # 4. Run model inference if we have enough data
            seizure_probability = 0.0
            if len(self.patient_buffers[patient_id]) == self.window_size and self.model:
                seizure_probability = await self._run_seizure_detection(patient_id)

            # 5. Send alerts if needed
            await self._check_and_send_alerts(
                patient_id=patient_id,
                seizure_probability=seizure_probability,
                anomalies=anomalies,
                features=features
            )

            # 6. Audit log
            self.audit_logger.log_access(
                user_id="SYSTEM_STREAM_PROCESSOR",
                patient_id=patient_id,
                action="EEG_STREAM_PROCESSING",
                ip_address="127.0.0.1",
                resource="EEG_CHUNK",
                additional_data={
                    'timestamp': timestamp,
                    'seizure_probability': float(seizure_probability),
                    'anomalies_detected': sum(anomalies.values())
                }
            )

            self.processed_count += 1

        except Exception as e:
            logger.error(f"Error processing EEG message: {e}", exc_info=True)

    async def _run_seizure_detection(self, patient_id: str) -> float:
        """
        Run seizure detection model on buffered data

        Args:
            patient_id: Patient identifier

        Returns:
            Seizure probability (0-1)
        """
        try:
            # Get buffered data
            buffer = self.patient_buffers[patient_id]

            # Stack into tensor (batch=1, time_windows=10, channels=16, samples=256)
            data_tensor = torch.tensor(
                np.array(buffer),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output, _ = self.model(data_tensor)
                probs = torch.softmax(output, dim=1)
                seizure_prob = float(probs[0][2])  # Ictal class

            return seizure_prob

        except Exception as e:
            logger.error(f"Error in seizure detection: {e}")
            return 0.0

    async def _check_and_send_alerts(
        self,
        patient_id: str,
        seizure_probability: float,
        anomalies: Dict[str, bool],
        features: Dict
    ):
        """
        Check conditions and send alerts

        Args:
            patient_id: Patient identifier
            seizure_probability: Seizure probability from model
            anomalies: Detected anomalies
            features: Extracted features
        """
        # High seizure probability
        if seizure_probability > 0.8:
            await self.producer.send_alert(
                patient_id=patient_id,
                alert_type='SEIZURE_DETECTED',
                severity='CRITICAL',
                message=f'Seizure detected with {seizure_probability*100:.1f}% confidence',
                data={
                    'confidence': seizure_probability,
                    'timestamp': datetime.now().isoformat()
                }
            )
            self.alert_count += 1
            logger.warning(f"CRITICAL: Seizure detected for patient {patient_id}")

        # Medium seizure probability
        elif seizure_probability > 0.5:
            await self.producer.send_alert(
                patient_id=patient_id,
                alert_type='SEIZURE_WARNING',
                severity='HIGH',
                message=f'Possible seizure activity ({seizure_probability*100:.1f}% confidence)',
                data={'confidence': seizure_probability}
            )
            self.alert_count += 1
            logger.warning(f"HIGH: Seizure warning for patient {patient_id}")

        # Anomaly alerts
        anomaly_count = sum(anomalies.values())
        if anomaly_count >= 3:
            await self.producer.send_alert(
                patient_id=patient_id,
                alert_type='EEG_ANOMALY',
                severity='MEDIUM',
                message=f'Multiple EEG anomalies detected ({anomaly_count})',
                data={
                    'anomalies': {k: v for k, v in anomalies.items() if v},
                    'count': anomaly_count
                }
            )
            self.alert_count += 1

    async def run(self):
        """Run the stream processor continuously"""
        await self.start()

        try:
            logger.info("Stream processor running... (Ctrl+C to stop)")
            await self.consumer.consume()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Stream processor error: {e}", exc_info=True)
        finally:
            await self.stop()

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'processed_chunks': self.processed_count,
            'alerts_sent': self.alert_count,
            'patients_tracked': len(self.patient_buffers),
            'model_loaded': self.model is not None
        }


async def main():
    """Example usage"""
    processor = StreamProcessor(
        bootstrap_servers='localhost:9092',
        eeg_topic='eeg-stream',
        alert_topic='clinical-alerts'
    )

    try:
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
