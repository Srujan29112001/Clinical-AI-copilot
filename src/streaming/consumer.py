"""
Kafka Consumer for EEG Streaming
Consumes and processes real-time EEG data from Kafka topics
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Dict, Any
import numpy as np

try:
    from aiokafka import AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("aiokafka not available, using mock consumer")

logger = logging.getLogger(__name__)


class EEGStreamConsumer:
    """
    Asynchronous Kafka consumer for real-time EEG processing

    Features:
    - Async/await for non-blocking operations
    - Automatic offset management
    - Error handling and retries
    - Callback-based processing
    """

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'eeg-stream',
        group_id: str = 'eeg-processor-group',
        auto_offset_reset: str = 'latest'
    ):
        """
        Initialize EEG stream consumer

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic name
            group_id: Consumer group ID
            auto_offset_reset: Offset reset strategy
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_running = False
        self.message_callback: Optional[Callable] = None

    async def start(self):
        """Start the consumer"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, running in mock mode")
            self.is_running = True
            return

        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                enable_auto_commit=True,
                max_poll_records=100
            )

            await self.consumer.start()
            self.is_running = True
            logger.info(f"EEG consumer started, topic: {self.topic}, group: {self.group_id}")

        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise

    async def stop(self):
        """Stop the consumer"""
        self.is_running = False
        if self.consumer:
            await self.consumer.stop()
        logger.info("EEG consumer stopped")

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for message processing

        Args:
            callback: Async function to process messages
        """
        self.message_callback = callback

    async def consume(self):
        """
        Consume messages from Kafka and process with callback

        This method runs continuously until stopped
        """
        if not self.is_running:
            raise RuntimeError("Consumer not started")

        if not self.message_callback:
            raise RuntimeError("No callback set for message processing")

        if not KAFKA_AVAILABLE or not self.consumer:
            logger.info("Running in mock mode, no actual consumption")
            return

        try:
            async for message in self.consumer:
                try:
                    # Process message with callback
                    await self.message_callback(message.value)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages

        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            raise

    async def consume_with_timeout(self, timeout_seconds: float = 60.0):
        """
        Consume messages with timeout

        Args:
            timeout_seconds: Maximum time to consume
        """
        try:
            await asyncio.wait_for(self.consume(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.info(f"Consumer timeout after {timeout_seconds} seconds")


class AlertConsumer:
    """Consumer for clinical alerts"""

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'clinical-alerts',
        group_id: str = 'alert-processor-group'
    ):
        self.consumer = EEGStreamConsumer(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            group_id=group_id
        )

    async def start(self):
        """Start alert consumer"""
        await self.consumer.start()

    async def stop(self):
        """Stop alert consumer"""
        await self.consumer.stop()

    async def process_alerts(self, alert_handler: Callable):
        """
        Process alerts with custom handler

        Args:
            alert_handler: Async function to handle alerts
        """
        self.consumer.set_callback(alert_handler)
        await self.consumer.consume()


async def example_eeg_processor(message: Dict[str, Any]):
    """
    Example EEG message processor

    Args:
        message: EEG message from Kafka
    """
    patient_id = message.get('patient_id')
    eeg_data = np.array(message.get('eeg_data'))
    timestamp = message.get('timestamp')

    logger.info(f"Processing EEG chunk for patient {patient_id}")
    logger.info(f"  Shape: {eeg_data.shape}")
    logger.info(f"  Timestamp: {timestamp}")

    # Here you would call your EEG processing pipeline
    # For example:
    # from src.signal_processing import EEGProcessingPipeline
    # processor = EEGProcessingPipeline()
    # features, coeffs = processor.process_chunk(eeg_data)

    # Simulate processing
    await asyncio.sleep(0.1)

    logger.info(f"Finished processing chunk for patient {patient_id}")


async def example_alert_handler(alert: Dict[str, Any]):
    """
    Example alert handler

    Args:
        alert: Alert message from Kafka
    """
    patient_id = alert.get('patient_id')
    alert_type = alert.get('alert_type')
    severity = alert.get('severity')
    message = alert.get('message')

    logger.warning(f"ALERT [{severity}] for patient {patient_id}: {alert_type}")
    logger.warning(f"  Message: {message}")

    # Here you would send notifications, update dashboards, etc.
    # For example:
    # - Send email to physician
    # - Push notification to mobile app
    # - Update patient dashboard
    # - Log to database


async def main():
    """Example usage"""
    # EEG stream consumer
    eeg_consumer = EEGStreamConsumer()
    eeg_consumer.set_callback(example_eeg_processor)

    # Alert consumer
    alert_consumer = AlertConsumer()

    try:
        # Start consumers
        await eeg_consumer.start()
        await alert_consumer.start()

        # Start processing
        consume_task = asyncio.create_task(eeg_consumer.consume())
        alert_task = asyncio.create_task(
            alert_consumer.process_alerts(example_alert_handler)
        )

        # Run for 60 seconds
        await asyncio.sleep(60)

        # Cancel tasks
        consume_task.cancel()
        alert_task.cancel()

    finally:
        await eeg_consumer.stop()
        await alert_consumer.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
