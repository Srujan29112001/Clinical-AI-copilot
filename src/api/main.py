"""
HIPAA-Compliant FastAPI Main Server
Clinical AI Copilot API with security, encryption, and audit logging
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from datetime import datetime
import logging
import json as json_lib
import asyncio

from src.signal_processing.eeg_processor import EEGProcessingPipeline
from src.models.cnn_lstm import HybridCNNLSTM, SeizureDetectionConfig
from src.models.multimodal_fusion import MultimodalClinicalFusion
from src.rag.graph_rag import ClinicalGraphRAG
from src.rag.vector_store import ClinicalVectorStore
from src.utils.encryption import EncryptionManager
from src.utils.audit_log import HIPAALogger, AuditContext
from src.utils.gpu_manager import GPUMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical AI Copilot",
    description="HIPAA-compliant EEG analysis and clinical decision support",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
encryption_manager = EncryptionManager()
audit_logger = HIPAALogger(log_dir="./logs/audit")
gpu_manager = GPUMemoryManager()

# Initialize models (lazy loading)
eeg_processor = None
seizure_model = None
multimodal_model = None
graph_rag = None
vector_store = None


# Pydantic models
class PatientContext(BaseModel):
    """Patient context for EEG analysis"""
    patient_id: str = Field(..., description="Patient identifier")
    symptoms: str = Field(..., description="Current symptoms")
    history: Optional[str] = Field(None, description="Medical history")
    medications: Optional[List[str]] = Field(None, description="Current medications")
    age: Optional[int] = Field(None, description="Patient age")
    sex: Optional[str] = Field(None, description="Patient sex")


class EEGAnalysisRequest(BaseModel):
    """Request for EEG analysis"""
    patient_context: PatientContext
    eeg_data_url: Optional[str] = Field(None, description="URL to EEG data file")
    real_time: bool = Field(False, description="Real-time streaming mode")


class DiagnosisResponse(BaseModel):
    """Response with diagnosis and recommendations"""
    status: str
    report_id: str
    patient_id_hash: str
    timestamp: str

    # Diagnosis
    primary_diagnosis: Dict[str, Any]
    differential_diagnoses: List[Dict[str, Any]]
    severity: str
    urgency: str
    confidence: float

    # EEG Analysis
    eeg_findings: Dict[str, Any]
    seizure_probability: float
    anomaly_flags: Dict[str, bool]

    # Recommendations
    recommended_treatments: List[Dict[str, Any]]
    follow_up: str

    # Supporting evidence
    supporting_evidence: List[Dict[str, Any]]
    clinical_guidelines: List[str]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    gpu_available: bool
    models_loaded: List[str]


# Dependency: Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify JWT token (simplified for demo)
    In production, implement proper JWT verification with user roles
    """
    token = credentials.credentials

    # Simplified token verification (implement proper JWT in production)
    if token.startswith("demo_"):
        user_id = token.replace("demo_", "")
        return user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )


# Dependency: Initialize models
def get_models():
    """Initialize and return models (lazy loading)"""
    global eeg_processor, seizure_model, multimodal_model, graph_rag, vector_store

    if eeg_processor is None:
        logger.info("Initializing EEG processor...")
        eeg_processor = EEGProcessingPipeline(sampling_rate=256, channels=16)

    if seizure_model is None and torch.cuda.is_available():
        logger.info("Loading seizure detection model...")
        config = SeizureDetectionConfig()
        seizure_model = HybridCNNLSTM(config)
        seizure_model = gpu_manager.load_model(
            seizure_model,
            gpu_manager.ModelConfig(name='seizure_model', size_gb=1.5, precision='fp16')
        )
        seizure_model.eval()

    if multimodal_model is None and torch.cuda.is_available():
        logger.info("Loading multimodal fusion model...")
        multimodal_model = MultimodalClinicalFusion()
        multimodal_model = gpu_manager.load_model(
            multimodal_model,
            gpu_manager.ModelConfig(name='multimodal_model', size_gb=2.0, precision='fp16')
        )
        multimodal_model.eval()

    if vector_store is None:
        logger.info("Initializing vector store...")
        vector_store = ClinicalVectorStore(use_memory=True)

    if graph_rag is None:
        logger.info("Initializing Graph RAG...")
        try:
            graph_rag = ClinicalGraphRAG()
        except Exception as e:
            logger.warning(f"Graph RAG initialization failed: {e}")
            graph_rag = None

    return {
        'eeg_processor': eeg_processor,
        'seizure_model': seizure_model,
        'multimodal_model': multimodal_model,
        'graph_rag': graph_rag,
        'vector_store': vector_store
    }


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Clinical AI Copilot API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    models = get_models()

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        gpu_available=torch.cuda.is_available(),
        models_loaded=[k for k, v in models.items() if v is not None]
    )


@app.post("/api/v1/analyze_eeg", response_model=DiagnosisResponse)
async def analyze_eeg(
    request: Request,
    eeg_file: UploadFile = File(...),
    patient_context: str = File(...),  # JSON string
    user_id: str = Depends(verify_token)
):
    """
    Analyze EEG data and provide diagnosis with recommendations

    This endpoint:
    1. Processes uploaded EEG file
    2. Extracts features
    3. Detects seizures and anomalies
    4. Retrieves relevant clinical knowledge
    5. Generates comprehensive diagnosis with supporting evidence
    """
    # Parse patient context
    import json
    patient_data = PatientContext(**json.loads(patient_context))

    # Audit log
    with AuditContext(
        audit_logger,
        user_id=user_id,
        patient_id=patient_data.patient_id,
        action="EEG_ANALYSIS",
        ip_address=request.client.host,
        resource="EEG_FILE"
    ):
        try:
            # Initialize models
            models = get_models()

            # Read EEG file
            eeg_data = await eeg_file.read()
            eeg_array = np.frombuffer(eeg_data, dtype=np.float32)

            # Reshape to (channels, samples)
            # Assuming 16 channels, reshape accordingly
            expected_samples = len(eeg_array) // 16
            eeg_array = eeg_array.reshape(16, expected_samples)

            # Take first 10 seconds for analysis (2560 samples at 256 Hz)
            if expected_samples > 2560:
                eeg_array = eeg_array[:, :2560]

            # Process EEG with signal processor
            logger.info("Processing EEG signal...")
            chunks = []
            for i in range(0, eeg_array.shape[1], 256):
                if i + 256 <= eeg_array.shape[1]:
                    chunk = eeg_array[:, i:i+256]
                    features, coeffs = models['eeg_processor'].process_chunk(chunk)
                    chunks.append(features)

            # Extract overall features
            eeg_features = chunks[-1] if chunks else {}

            # Detect anomalies
            anomalies = models['eeg_processor'].detect_anomalies(eeg_features)

            # Seizure detection (if model is available)
            seizure_probability = 0.0
            if models['seizure_model'] is not None:
                # Prepare input for seizure model: (batch, time_windows, channels, samples)
                time_windows = min(10, eeg_array.shape[1] // 256)
                seizure_input = []
                for i in range(time_windows):
                    window = eeg_array[:, i*256:(i+1)*256]
                    seizure_input.append(window)

                if len(seizure_input) == time_windows:
                    seizure_tensor = torch.tensor(seizure_input, dtype=torch.float32)
                    seizure_tensor = seizure_tensor.unsqueeze(0)  # Add batch dimension

                    with torch.no_grad():
                        seizure_output, _ = models['seizure_model'](seizure_tensor)
                        seizure_probs = torch.softmax(seizure_output, dim=1)
                        seizure_probability = float(seizure_probs[0][2])  # Ictal class

            # Retrieve clinical context
            clinical_context = models['vector_store'].search(
                query=patient_data.symptoms,
                top_k=5
            )

            # Get graph-based recommendations
            graph_results = []
            if models['graph_rag'] is not None:
                graph_results = models['graph_rag'].retrieve_context(
                    query=patient_data.symptoms,
                    eeg_features=eeg_features
                )

            # Generate diagnosis
            # Simplified logic - in production use multimodal model
            primary_diagnosis = {
                'condition': 'Epilepsy' if seizure_probability > 0.5 else 'Normal',
                'icd10': 'G40.9' if seizure_probability > 0.5 else 'Z00.00',
                'confidence': float(seizure_probability if seizure_probability > 0.5 else 1 - seizure_probability)
            }

            # Determine severity and urgency
            severity = "High" if seizure_probability > 0.8 else "Medium" if seizure_probability > 0.5 else "Low"
            urgency = "Immediate" if seizure_probability > 0.8 else "Routine"

            # Recommendations
            treatments = []
            if seizure_probability > 0.5:
                treatments = [
                    {'name': 'Levetiracetam', 'type': 'Antiepileptic', 'dosage': '500mg BID'},
                    {'name': 'Valproic Acid', 'type': 'Antiepileptic', 'dosage': '250mg TID'}
                ]

            # Create response
            report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            patient_id_hash = encryption_manager.hash_identifier(patient_data.patient_id)

            response = DiagnosisResponse(
                status="success",
                report_id=report_id,
                patient_id_hash=patient_id_hash,
                timestamp=datetime.now().isoformat(),
                primary_diagnosis=primary_diagnosis,
                differential_diagnoses=[],
                severity=severity,
                urgency=urgency,
                confidence=primary_diagnosis['confidence'],
                eeg_findings={
                    'alpha_power': float(np.mean(eeg_features.get('psd', {}).get('alpha', [0]))),
                    'beta_power': float(np.mean(eeg_features.get('psd', {}).get('beta', [0]))),
                    'theta_beta_ratio': float(np.mean(eeg_features.get('psd', {}).get('theta_beta_ratio', [0])))
                },
                seizure_probability=seizure_probability,
                anomaly_flags=anomalies,
                recommended_treatments=treatments,
                follow_up="Schedule follow-up EEG in 3 months" if seizure_probability > 0.5 else "Routine monitoring",
                supporting_evidence=[
                    {'source': doc.metadata.get('title', 'Clinical Study'), 'score': score}
                    for doc, score in clinical_context[:3]
                ],
                clinical_guidelines=["AAN Practice Guidelines for Epilepsy"]
            )

            return response

        except Exception as e:
            logger.error(f"Error analyzing EEG: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error analyzing EEG: {str(e)}"
            )


@app.get("/api/v1/patient/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    request: Request,
    user_id: str = Depends(verify_token)
):
    """Get patient analysis history"""
    # Audit log
    audit_logger.log_access(
        user_id=user_id,
        patient_id=patient_id,
        action="READ_HISTORY",
        ip_address=request.client.host,
        resource="PATIENT_HISTORY"
    )

    # In production, query database
    return {
        "patient_id_hash": encryption_manager.hash_identifier(patient_id),
        "analyses": [],
        "message": "History retrieval not yet implemented"
    }


@app.get("/api/v1/stats")
async def get_stats(user_id: str = Depends(verify_token)):
    """Get system statistics"""
    gpu_stats = gpu_manager.get_memory_stats()

    return {
        "gpu_memory": gpu_stats,
        "models_loaded": list(gpu_manager.loaded_models.keys()),
        "timestamp": datetime.now().isoformat()
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# WebSocket Connection Manager
class ConnectionManager:
    """Manage WebSocket connections for real-time EEG streaming"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.patient_buffers: Dict[str, List[np.ndarray]] = {}

    async def connect(self, websocket: WebSocket, patient_id: str):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections[patient_id] = websocket
        self.patient_buffers[patient_id] = []
        logger.info(f"WebSocket connected for patient {patient_id}")

    def disconnect(self, patient_id: str):
        """Remove WebSocket connection"""
        if patient_id in self.active_connections:
            del self.active_connections[patient_id]
        if patient_id in self.patient_buffers:
            del self.patient_buffers[patient_id]
        logger.info(f"WebSocket disconnected for patient {patient_id}")

    async def send_message(self, patient_id: str, message: Dict):
        """Send message to specific patient connection"""
        if patient_id in self.active_connections:
            await self.active_connections[patient_id].send_json(message)

    def add_to_buffer(self, patient_id: str, data: np.ndarray):
        """Add EEG data to patient buffer"""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = []
        self.patient_buffers[patient_id].append(data)

    def get_buffer(self, patient_id: str, clear: bool = True) -> Optional[np.ndarray]:
        """Get buffered data for patient"""
        if patient_id in self.patient_buffers and self.patient_buffers[patient_id]:
            buffer = np.concatenate(self.patient_buffers[patient_id], axis=-1)
            if clear:
                self.patient_buffers[patient_id] = []
            return buffer
        return None


manager = ConnectionManager()


@app.websocket("/ws/eeg/stream/{patient_id}")
async def websocket_eeg_stream(websocket: WebSocket, patient_id: str):
    """
    WebSocket endpoint for real-time EEG streaming and analysis

    Protocol:
    1. Client sends authentication token
    2. Client streams EEG chunks: {"channels": 16, "samples": 256, "data": [...]}
    3. Server processes in real-time and sends back:
       - Features: {"psd": {...}, "entropy": {...}}
       - Alerts: {"type": "seizure_detected", "probability": 0.95}
       - Status: {"processed_chunks": 10, "total_time": 2.5}
    """
    await manager.connect(websocket, patient_id)

    try:
        # Initialize models
        models = get_models()

        # First message should be authentication
        auth_message = await websocket.receive_json()
        token = auth_message.get("token")

        if not token or not token.startswith("demo_"):
            await websocket.send_json({
                "error": "Authentication failed",
                "code": 401
            })
            await websocket.close()
            return

        user_id = token.replace("demo_", "")

        # Audit log connection
        audit_logger.log_access(
            user_id=user_id,
            patient_id=patient_id,
            action="WEBSOCKET_CONNECT",
            ip_address=websocket.client.host if hasattr(websocket, 'client') else "unknown",
            resource="EEG_STREAM"
        )

        # Send ready message
        await websocket.send_json({
            "status": "connected",
            "patient_id": patient_id,
            "ready": True,
            "timestamp": datetime.now().isoformat()
        })

        processed_chunks = 0
        alert_threshold = 0.8

        while True:
            # Receive EEG chunk
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "status": "timeout",
                    "message": "No data received in 30 seconds"
                })
                continue

            # Parse EEG data
            channels = data.get("channels", 16)
            samples = data.get("samples", 256)
            eeg_data = np.array(data.get("data"), dtype=np.float32)

            # Reshape to (channels, samples)
            try:
                eeg_chunk = eeg_data.reshape(channels, samples)
            except ValueError:
                await websocket.send_json({
                    "error": "Invalid data shape",
                    "expected": f"({channels}, {samples})",
                    "received": eeg_data.shape
                })
                continue

            # Process chunk
            try:
                features, coeffs = models['eeg_processor'].process_chunk(eeg_chunk)

                # Detect anomalies
                anomalies = models['eeg_processor'].detect_anomalies(features)

                # Add to buffer for seizure detection
                manager.add_to_buffer(patient_id, eeg_chunk)

                # Run seizure detection every 10 chunks (10 seconds)
                seizure_probability = 0.0
                if processed_chunks > 0 and processed_chunks % 10 == 0:
                    buffer = manager.get_buffer(patient_id, clear=False)
                    if buffer is not None and buffer.shape[1] >= 2560:  # 10 seconds
                        # Take last 10 seconds
                        buffer_10s = buffer[:, -2560:]

                        # Create windows for seizure model
                        time_windows = 10
                        seizure_input = []
                        for i in range(time_windows):
                            window = buffer_10s[:, i*256:(i+1)*256]
                            seizure_input.append(window)

                        if len(seizure_input) == time_windows and models['seizure_model'] is not None:
                            seizure_tensor = torch.tensor(seizure_input, dtype=torch.float32)
                            seizure_tensor = seizure_tensor.unsqueeze(0)

                            with torch.no_grad():
                                seizure_output, _ = models['seizure_model'](seizure_tensor)
                                seizure_probs = torch.softmax(seizure_output, dim=1)
                                seizure_probability = float(seizure_probs[0][2])

                # Send analysis results
                response = {
                    "status": "processed",
                    "chunk_id": processed_chunks,
                    "timestamp": datetime.now().isoformat(),
                    "features": {
                        "psd": {
                            band: float(np.mean(features['psd'].get(band, [0])))
                            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
                        },
                        "entropy": {
                            "shannon": float(np.mean(features['entropy'].get('shannon', [0]))),
                            "sample": float(np.mean(features['entropy'].get('sample', [0])))
                        },
                        "hjorth": {
                            "activity": float(np.mean(features['hjorth'].get('activity', [0]))),
                            "mobility": float(np.mean(features['hjorth'].get('mobility', [0]))),
                            "complexity": float(np.mean(features['hjorth'].get('complexity', [0])))
                        }
                    },
                    "anomalies": anomalies,
                    "seizure_probability": seizure_probability if processed_chunks % 10 == 0 else None
                }

                await websocket.send_json(response)

                # Send alert if seizure detected
                if seizure_probability > alert_threshold:
                    alert = {
                        "type": "ALERT",
                        "severity": "CRITICAL",
                        "message": "Seizure activity detected",
                        "probability": seizure_probability,
                        "timestamp": datetime.now().isoformat(),
                        "action_required": "Immediate medical attention"
                    }
                    await websocket.send_json(alert)

                    # Log critical event
                    audit_logger.log_access(
                        user_id=user_id,
                        patient_id=patient_id,
                        action="SEIZURE_ALERT",
                        ip_address=websocket.client.host if hasattr(websocket, 'client') else "unknown",
                        resource="EEG_STREAM",
                        additional_data={"probability": seizure_probability}
                    )

                processed_chunks += 1

            except Exception as e:
                logger.error(f"Error processing chunk: {e}", exc_info=True)
                await websocket.send_json({
                    "error": "Processing error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for patient {patient_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        manager.disconnect(patient_id)

        # Audit log disconnection
        try:
            audit_logger.log_access(
                user_id=user_id if 'user_id' in locals() else "unknown",
                patient_id=patient_id,
                action="WEBSOCKET_DISCONNECT",
                ip_address=websocket.client.host if hasattr(websocket, 'client') else "unknown",
                resource="EEG_STREAM"
            )
        except:
            pass


@app.get("/ws/eeg/status/{patient_id}")
async def websocket_status(patient_id: str, user_id: str = Depends(verify_token)):
    """Get WebSocket connection status for a patient"""
    is_connected = patient_id in manager.active_connections
    buffer_size = len(manager.patient_buffers.get(patient_id, []))

    return {
        "patient_id": patient_id,
        "connected": is_connected,
        "buffer_chunks": buffer_size,
        "timestamp": datetime.now().isoformat()
    }


# Include admin endpoints
try:
    from src.api.admin_endpoints import router as admin_router
    app.include_router(admin_router)
    logger.info("Admin endpoints loaded successfully")
except Exception as e:
    logger.error(f"Failed to load admin endpoints: {str(e)}")


def main():
    """Main entry point"""
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting Clinical AI Copilot API on {host}:{port}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,  # Single worker for GPU management
        log_level="info"
    )


if __name__ == "__main__":
    main()
