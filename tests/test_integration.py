"""
Integration Tests for Clinical AI Copilot
Tests the complete workflow from EEG upload to diagnosis
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.signal_processing.eeg_processor import EEGProcessingPipeline
from src.models.cnn_lstm import HybridCNNLSTM, SeizureDetectionConfig
from src.data.edf_loader import EDFDataLoader
from src.clinical.drug_interactions import check_drug_interactions


client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Clinical AI Copilot" in response.json()["message"]

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_eeg_analysis_unauthorized(self):
        """Test EEG analysis without authentication"""
        # Create fake EEG data
        eeg_data = np.random.randn(16, 2560).astype(np.float32)

        response = client.post(
            "/api/v1/analyze_eeg",
            files={
                "eeg_file": ("test.eeg", eeg_data.tobytes(), "application/octet-stream"),
                "patient_context": ('{"patient_id": "TEST001", "symptoms": "seizures"}')
            }
        )

        assert response.status_code == 401  # Unauthorized

    def test_eeg_analysis_with_auth(self):
        """Test EEG analysis with authentication"""
        import json

        # Create fake EEG data
        eeg_data = np.random.randn(16, 2560).astype(np.float32)

        patient_context = {
            "patient_id": "TEST001",
            "symptoms": "seizures",
            "history": "No prior conditions",
            "age": 35,
            "sex": "F"
        }

        response = client.post(
            "/api/v1/analyze_eeg",
            headers={"Authorization": "Bearer demo_testuser"},
            files={
                "eeg_file": ("test.eeg", eeg_data.tobytes(), "application/octet-stream"),
                "patient_context": json.dumps(patient_context)
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert "primary_diagnosis" in data
        assert "eeg_findings" in data

    def test_patient_history(self):
        """Test patient history endpoint"""
        response = client.get(
            "/api/v1/patient/TEST001/history",
            headers={"Authorization": "Bearer demo_testuser"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "patient_id_hash" in data


class TestEEGProcessing:
    """Test EEG signal processing"""

    def test_eeg_processor_initialization(self):
        """Test EEG processor initialization"""
        processor = EEGProcessingPipeline(sampling_rate=256, channels=16)
        assert processor.sampling_rate == 256
        assert processor.channels == 16

    def test_process_chunk(self):
        """Test processing a single EEG chunk"""
        processor = EEGProcessingPipeline(sampling_rate=256, channels=16)

        # Create calibration data
        calibration_data = np.random.randn(16, 7680)  # 30 seconds
        processor.calibrate_ica(calibration_data)

        # Process a chunk
        chunk = np.random.randn(16, 256)
        features, coeffs = processor.process_chunk(chunk)

        assert 'psd' in features
        assert 'entropy' in features
        assert 'hjorth' in features
        assert len(coeffs) == 16  # One per channel

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        processor = EEGProcessingPipeline(sampling_rate=256, channels=16)

        # Create normal features
        chunk = np.random.randn(16, 256) * 10  # Normal amplitude
        features, _ = processor.process_chunk(chunk)

        anomalies = processor.detect_anomalies(features)

        assert isinstance(anomalies, dict)
        assert 'high_amplitude' in anomalies
        assert 'spike_detected' in anomalies


class TestModels:
    """Test deep learning models"""

    def test_cnn_lstm_forward(self):
        """Test CNN-LSTM forward pass"""
        config = SeizureDetectionConfig()
        model = HybridCNNLSTM(config)
        model.eval()

        # Create input: (batch, time_windows, channels, samples)
        x = torch.randn(2, 10, 16, 256)

        with torch.no_grad():
            output, attention = model(x, return_attention=True)

        assert output.shape == (2, 3)  # 3 classes
        assert attention is not None

    def test_cnn_lstm_uncertainty(self):
        """Test uncertainty estimation"""
        config = SeizureDetectionConfig()
        model = HybridCNNLSTM(config)

        x = torch.randn(1, 10, 16, 256)

        output_mean, output_std = model.predict_with_uncertainty(x, num_samples=5)

        assert output_mean.shape == (1, 3)
        assert output_std.shape == (1, 3)
        assert torch.all(output_std >= 0)  # Std should be non-negative


class TestDataLoaders:
    """Test data loading"""

    def test_edf_loader_initialization(self):
        """Test EDF loader initialization"""
        loader = EDFDataLoader(target_sfreq=256)
        assert loader.target_sfreq == 256

    def test_clinical_dataset(self):
        """Test PyTorch dataset"""
        from src.data.dataset import ClinicalEEGDataset

        n_samples = 10
        eeg_data = np.random.randn(n_samples, 16, 256)
        labels = np.random.randint(0, 2, n_samples)

        dataset = ClinicalEEGDataset(eeg_data, labels)

        assert len(dataset) == n_samples

        sample = dataset[0]
        assert 'eeg' in sample
        assert 'label' in sample
        assert sample['eeg'].shape == (16, 256)


class TestDrugInteractions:
    """Test drug interaction checking"""

    def test_no_interactions(self):
        """Test medications with no interactions"""
        medications = ["Levetiracetam"]
        interactions, summary = check_drug_interactions(medications)

        assert len(interactions) == 0
        assert summary["total_interactions"] == 0

    def test_moderate_interaction(self):
        """Test moderate interaction"""
        medications = ["Levetiracetam", "Valproic Acid"]
        interactions, summary = check_drug_interactions(medications)

        assert len(interactions) == 1
        assert summary["moderate"] == 1

    def test_major_interaction(self):
        """Test major interaction"""
        medications = ["Valproic Acid", "Lamotrigine"]
        interactions, summary = check_drug_interactions(medications)

        assert len(interactions) == 1
        assert summary["major"] == 1
        assert summary["requires_action"] == True

    def test_contraindicated(self):
        """Test contraindicated combination"""
        medications = ["Diazepam", "Alcohol"]
        interactions, summary = check_drug_interactions(medications)

        assert len(interactions) == 1
        assert summary["contraindicated"] == 1
        assert summary["highest_severity"] == "CONTRAINDICATED"


class TestAuthentication:
    """Test authentication and authorization"""

    def test_token_verification(self):
        """Test token verification"""
        from src.api.auth import verify_token, TokenData

        # Create a test token (simplified)
        from src.api.auth import create_access_token

        token = create_access_token(
            data={"sub": "test_user", "email": "test@example.com", "roles": ["physician"]}
        )

        token_data = verify_token(token)

        assert isinstance(token_data, TokenData)
        assert token_data.user_id == "test_user"
        assert "physician" in token_data.roles

    def test_invalid_token(self):
        """Test invalid token"""
        from src.api.auth import verify_token
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            verify_token("invalid_token")


class TestEncryption:
    """Test encryption utilities"""

    def test_field_encryption(self):
        """Test field encryption and decryption"""
        from src.utils.encryption import EncryptionManager

        manager = EncryptionManager()
        original = "Sensitive patient data"

        encrypted = manager.encrypt_field(original)
        decrypted = manager.decrypt_field(encrypted)

        assert encrypted != original
        assert decrypted == original

    def test_hash_identifier(self):
        """Test identifier hashing"""
        from src.utils.encryption import EncryptionManager

        manager = EncryptionManager()
        patient_id = "PATIENT-12345"

        hash1 = manager.hash_identifier(patient_id)
        hash2 = manager.hash_identifier(patient_id)

        # Same input should produce same hash
        assert hash1 == hash2

        # Different input should produce different hash
        hash3 = manager.hash_identifier("PATIENT-54321")
        assert hash1 != hash3


class TestRateLimiting:
    """Test rate limiting"""

    def test_rate_limiter(self):
        """Test rate limiter"""
        from src.api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=5)

        # First 5 requests should be allowed
        for i in range(5):
            allowed, retry_after = limiter.is_allowed("test_user")
            assert allowed == True
            assert retry_after is None

        # 6th request should be blocked
        allowed, retry_after = limiter.is_allowed("test_user")
        assert allowed == False
        assert retry_after is not None and retry_after > 0


@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection"""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        with client.websocket_connect("/ws/eeg/stream/TEST001") as websocket:
            # Send authentication
            websocket.send_json({"token": "demo_testuser"})

            # Receive ready message
            data = websocket.receive_json()
            assert data["status"] == "connected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
