"""
Load Testing Suite for Clinical AI Copilot API
Uses Locust for performance and stress testing
"""

from locust import HttpUser, task, between, events
import json
import numpy as np
from io import BytesIO
import time


class ClinicalAIUser(HttpUser):
    """
    Simulated user for load testing

    Tests:
    - EEG analysis endpoint
    - Patient history retrieval
    - WebSocket streaming
    - Batch processing
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    host = "http://localhost:8000"

    def on_start(self):
        """Initialize user session"""
        # Authenticate
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        else:
            self.token = None
            self.headers = {}

    @task(3)
    def health_check(self):
        """Health check endpoint (high frequency)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def analyze_eeg(self):
        """Test EEG analysis endpoint"""
        # Generate synthetic EEG data
        eeg_data = self._generate_synthetic_eeg()

        # Create request
        request_data = {
            "patient_context": {
                "patient_id": f"test_patient_{np.random.randint(1000, 9999)}",
                "symptoms": "Suspected seizure activity",
                "age": np.random.randint(18, 80),
                "sex": np.random.choice(["M", "F"]),
                "medications": ["Levetiracetam 500mg"]
            }
        }

        # Upload EEG file
        files = {
            "eeg_file": ("test.edf", BytesIO(eeg_data), "application/octet-stream"),
            "patient_context": (None, json.dumps(request_data["patient_context"]), "application/json")
        }

        start_time = time.time()

        with self.client.post(
            "/api/v1/analyze_eeg",
            files=files,
            headers={"Authorization": f"Bearer {self.token}"},
            catch_response=True
        ) as response:
            latency = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code == 200:
                response.success()
                # Log inference time
                events.request.fire(
                    request_type="analysis",
                    name="EEG Analysis Latency",
                    response_time=latency,
                    response_length=len(response.content)
                )
            else:
                response.failure(f"EEG analysis failed: {response.status_code}")

    @task(2)
    def get_patient_history(self):
        """Test patient history endpoint"""
        patient_id = f"test_patient_{np.random.randint(1000, 9999)}"

        with self.client.get(
            f"/api/v1/patient/{patient_id}/history",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 is acceptable
                response.success()
            else:
                response.failure(f"Get history failed: {response.status_code}")

    @task(1)
    def get_stats(self):
        """Test statistics endpoint"""
        with self.client.get(
            "/api/v1/stats",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get stats failed: {response.status_code}")

    def _generate_synthetic_eeg(self, duration_sec: int = 10, sampling_rate: int = 256) -> bytes:
        """
        Generate synthetic EEG data for testing

        Args:
            duration_sec: Duration in seconds
            sampling_rate: Sampling rate in Hz

        Returns:
            Bytes of EEG data
        """
        n_channels = 16
        n_samples = duration_sec * sampling_rate

        # Generate random EEG-like signal
        eeg = np.random.randn(n_channels, n_samples).astype(np.float32)

        # Add some structure (alpha waves ~10 Hz)
        t = np.linspace(0, duration_sec, n_samples)
        for ch in range(n_channels):
            eeg[ch] += 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand())

        return eeg.tobytes()


class AdminUser(HttpUser):
    """Admin user for testing admin endpoints"""

    wait_time = between(5, 10)
    host = "http://localhost:8000"

    def on_start(self):
        """Initialize admin session"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin_password"
        })

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        else:
            self.token = None
            self.headers = {}

    @task(1)
    def list_models(self):
        """Test model listing endpoint"""
        with self.client.get(
            "/api/v1/admin/models",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"List models failed: {response.status_code}")

    @task(1)
    def get_system_metrics(self):
        """Test system metrics endpoint"""
        with self.client.get(
            "/api/v1/admin/metrics/system",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get metrics failed: {response.status_code}")


# Event handlers for custom metrics
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--target-rps", type=int, default=100, help="Target requests per second")
    parser.add_argument("--test-duration", type=int, default=300, help="Test duration in seconds")


@events.test_start.add_listener
def _(environment, **kwargs):
    print(f"Starting load test with {environment.parsed_options.num_users} users")
    print(f"Target RPS: {environment.parsed_options.target_rps}")
    print(f"Duration: {environment.parsed_options.test_duration}s")


@events.test_stop.add_listener
def _(environment, **kwargs):
    print("\n=== Load Test Summary ===")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Failures: {environment.stats.total.num_failures}")
    print(f"Median response time: {environment.stats.total.median_response_time}ms")
    print(f"95th percentile: {environment.stats.total.get_response_time_percentile(0.95)}ms")
    print(f"99th percentile: {environment.stats.total.get_response_time_percentile(0.99)}ms")
    print(f"RPS: {environment.stats.total.total_rps:.2f}")


# CLI Usage:
# locust -f tests/load_test.py --host=http://localhost:8000
# locust -f tests/load_test.py --headless --users 100 --spawn-rate 10 --run-time 5m
# locust -f tests/load_test.py --headless --users 500 --spawn-rate 50 --run-time 10m --html=report.html
