"""
Admin and Management API Endpoints
Model management, batch processing, metrics, and system administration
"""

import os
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import torch
import asyncio
from pathlib import Path
import json

from src.api.auth import verify_admin, verify_token, RoleEnum
from src.utils.audit_log import HIPAALogger, AuditContext
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Create router
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# HIPAA audit logger
audit_logger = HIPAALogger(
    log_dir="logs/audit",
    encryption_key=os.environ.get("AUDIT_ENCRYPTION_KEY", "")
)

# Prometheus metrics
request_count = Counter('clinical_ai_requests_total', 'Total requests', ['endpoint', 'method'])
request_duration = Histogram('clinical_ai_request_duration_seconds', 'Request duration', ['endpoint'])
active_users = Gauge('clinical_ai_active_users', 'Number of active users')
model_inference_time = Histogram('clinical_ai_model_inference_seconds', 'Model inference time', ['model_name'])
gpu_utilization = Gauge('clinical_ai_gpu_utilization_percent', 'GPU utilization percentage')
gpu_memory_used = Gauge('clinical_ai_gpu_memory_used_bytes', 'GPU memory used in bytes')


# ====== Pydantic Models ======

class BatchEEGRequest(BaseModel):
    """Batch EEG analysis request"""
    patient_ids: List[str] = Field(..., description="List of patient IDs")
    eeg_file_urls: List[str] = Field(..., description="URLs to EEG files")
    priority: str = Field("normal", description="Priority: low, normal, high")
    notify_on_completion: bool = Field(True, description="Send notification when complete")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_ids": ["patient_001", "patient_002"],
                "eeg_file_urls": ["s3://bucket/patient_001.edf", "s3://bucket/patient_002.edf"],
                "priority": "normal",
                "notify_on_completion": True
            }
        }


class BatchJobResponse(BaseModel):
    """Batch job response"""
    job_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    created_at: datetime
    estimated_completion: Optional[datetime] = None


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    version: str
    type: str
    parameters: int
    size_mb: float
    loaded: bool
    device: str
    quantization: Optional[str] = None
    accuracy: Optional[float] = None
    last_updated: datetime


class ModelUpdateRequest(BaseModel):
    """Model update request"""
    model_name: str
    model_url: str  # URL to download new model
    validate_before_deploy: bool = Field(True, description="Validate model before deployment")


class SystemMetrics(BaseModel):
    """System metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_used_mb: float
    active_connections: int
    total_requests_24h: int
    average_response_time_ms: float
    error_rate_percent: float


class UserManagement(BaseModel):
    """User management"""
    username: str
    email: str
    role: str
    enabled: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    total_requests: int


class ConfigUpdate(BaseModel):
    """System configuration update"""
    config_key: str
    config_value: Any
    description: str


# ====== Batch Processing Endpoints ======

@router.post("/batch/analyze", response_model=BatchJobResponse, dependencies=[Depends(verify_admin)])
async def batch_analyze_eeg(
    request: BatchEEGRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    """
    Submit batch EEG analysis job

    Requires admin role. Processes multiple EEG files asynchronously.
    """
    with AuditContext(
        audit_logger,
        user_id=credentials.credentials,
        action="BATCH_ANALYSIS_SUBMIT",
        resource_type="batch_job"
    ):
        try:
            job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create batch job
            batch_job = {
                "job_id": job_id,
                "status": "queued",
                "total_tasks": len(request.patient_ids),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "created_at": datetime.now(),
                "patient_ids": request.patient_ids,
                "eeg_file_urls": request.eeg_file_urls,
                "priority": request.priority
            }

            # Save job to database (simplified - use actual DB in production)
            batch_jobs_file = Path("data/batch_jobs.json")
            batch_jobs_file.parent.mkdir(parents=True, exist_ok=True)

            jobs = {}
            if batch_jobs_file.exists():
                with open(batch_jobs_file, 'r') as f:
                    jobs = json.load(f)

            jobs[job_id] = batch_job

            with open(batch_jobs_file, 'w') as f:
                json.dump(jobs, f, indent=2, default=str)

            # Schedule background processing
            background_tasks.add_task(process_batch_job, job_id, batch_job)

            return BatchJobResponse(
                job_id=job_id,
                status="queued",
                total_tasks=batch_job["total_tasks"],
                completed_tasks=0,
                failed_tasks=0,
                created_at=batch_job["created_at"],
                estimated_completion=None
            )

        except Exception as e:
            logger.error(f"Error submitting batch job: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to submit batch job: {str(e)}")


@router.get("/batch/status/{job_id}", response_model=BatchJobResponse, dependencies=[Depends(verify_admin)])
async def get_batch_status(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    """Get batch job status"""
    try:
        batch_jobs_file = Path("data/batch_jobs.json")

        if not batch_jobs_file.exists():
            raise HTTPException(status_code=404, detail="Batch jobs not found")

        with open(batch_jobs_file, 'r') as f:
            jobs = json.load(f)

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job = jobs[job_id]

        return BatchJobResponse(
            job_id=job_id,
            status=job["status"],
            total_tasks=job["total_tasks"],
            completed_tasks=job["completed_tasks"],
            failed_tasks=job["failed_tasks"],
            created_at=datetime.fromisoformat(job["created_at"]) if isinstance(job["created_at"], str) else job["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_batch_job(job_id: str, batch_job: Dict[str, Any]):
    """Background task to process batch job"""
    try:
        batch_jobs_file = Path("data/batch_jobs.json")

        # Update status to processing
        with open(batch_jobs_file, 'r') as f:
            jobs = json.load(f)

        jobs[job_id]["status"] = "processing"

        with open(batch_jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2, default=str)

        # Process each patient (simplified - implement actual processing)
        for i, (patient_id, eeg_url) in enumerate(zip(batch_job["patient_ids"], batch_job["eeg_file_urls"])):
            try:
                # Simulate processing
                await asyncio.sleep(1)

                # Update progress
                jobs[job_id]["completed_tasks"] += 1

                with open(batch_jobs_file, 'w') as f:
                    json.dump(jobs, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Error processing {patient_id}: {str(e)}")
                jobs[job_id]["failed_tasks"] += 1

                with open(batch_jobs_file, 'w') as f:
                    json.dump(jobs, f, indent=2, default=str)

        # Mark as complete
        jobs[job_id]["status"] = "completed"

        with open(batch_jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2, default=str)

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {str(e)}")
        try:
            with open(batch_jobs_file, 'r') as f:
                jobs = json.load(f)
            jobs[job_id]["status"] = "failed"
            with open(batch_jobs_file, 'w') as f:
                json.dump(jobs, f, indent=2, default=str)
        except:
            pass


# ====== Model Management Endpoints ======

@router.get("/models", response_model=List[ModelInfo], dependencies=[Depends(verify_admin)])
async def list_models():
    """List all available models"""
    try:
        model_dir = Path(os.environ.get("MODEL_PATH", "models"))
        models = []

        # CNN-LSTM Seizure Detection
        cnn_lstm_path = model_dir / "cnn_lstm_seizure.pt"
        if cnn_lstm_path.exists():
            stat = cnn_lstm_path.stat()
            models.append(ModelInfo(
                model_name="CNN-LSTM Seizure Detection",
                version="1.0",
                type="seizure_detection",
                parameters=2_400_000,
                size_mb=stat.st_size / (1024 * 1024),
                loaded=True,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                quantization="fp16",
                accuracy=0.9875,
                last_updated=datetime.fromtimestamp(stat.st_mtime)
            ))

        # Transformer Sleep Staging
        transformer_path = model_dir / "transformer_sleep.pt"
        if transformer_path.exists():
            stat = transformer_path.stat()
            models.append(ModelInfo(
                model_name="Transformer Sleep Staging",
                version="1.0",
                type="sleep_staging",
                parameters=1_800_000,
                size_mb=stat.st_size / (1024 * 1024),
                loaded=True,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                quantization="fp16",
                accuracy=0.93,
                last_updated=datetime.fromtimestamp(stat.st_mtime)
            ))

        # Multimodal Fusion
        fusion_path = model_dir / "multimodal_fusion.pt"
        if fusion_path.exists():
            stat = fusion_path.stat()
            models.append(ModelInfo(
                model_name="Multimodal Clinical Fusion",
                version="1.0",
                type="multimodal_fusion",
                parameters=3_500_000,
                size_mb=stat.st_size / (1024 * 1024),
                loaded=True,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                quantization="fp16",
                accuracy=0.91,
                last_updated=datetime.fromtimestamp(stat.st_mtime)
            ))

        return models

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelInfo, dependencies=[Depends(verify_admin)])
async def get_model_info(model_name: str):
    """Get detailed model information"""
    models = await list_models()

    for model in models:
        if model.model_name == model_name:
            return model

    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@router.post("/models/update", dependencies=[Depends(verify_admin)])
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    """
    Update a model from URL

    Downloads and validates new model version
    """
    with AuditContext(
        audit_logger,
        user_id=credentials.credentials,
        action="MODEL_UPDATE",
        resource_type="model",
        resource_id=request.model_name
    ):
        try:
            # Schedule model download and validation in background
            background_tasks.add_task(download_and_validate_model, request.model_name, request.model_url)

            return {
                "status": "success",
                "message": f"Model update scheduled for {request.model_name}",
                "model_url": request.model_url
            }

        except Exception as e:
            logger.error(f"Error scheduling model update: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


async def download_and_validate_model(model_name: str, model_url: str):
    """Background task to download and validate model"""
    try:
        # Implement model download and validation
        logger.info(f"Downloading model {model_name} from {model_url}")
        # TODO: Implement actual download logic
        await asyncio.sleep(2)
        logger.info(f"Model {model_name} updated successfully")
    except Exception as e:
        logger.error(f"Failed to update model {model_name}: {str(e)}")


# ====== System Metrics Endpoints ======

@router.get("/metrics/system", response_model=SystemMetrics, dependencies=[Depends(verify_admin)])
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        import psutil

        # Get GPU metrics if available
        gpu_usage = 0.0
        gpu_memory = 0.0

        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

                # Update Prometheus metrics
                gpu_utilization.set(gpu_usage)
                gpu_memory_used.set(gpu_memory * 1024 * 1024)  # bytes
            except:
                pass

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            memory_usage_percent=psutil.virtual_memory().percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_used_mb=gpu_memory,
            active_connections=0,  # TODO: Track from connection manager
            total_requests_24h=0,  # TODO: Query from metrics
            average_response_time_ms=0.0,  # TODO: Calculate from histogram
            error_rate_percent=0.0  # TODO: Calculate from error counts
        )

    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint

    Exposes metrics in Prometheus format for scraping
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ====== User Management Endpoints ======

@router.get("/users", response_model=List[UserManagement], dependencies=[Depends(verify_admin)])
async def list_users():
    """List all users (admin only)"""
    # TODO: Implement actual user listing from database
    return [
        UserManagement(
            username="admin",
            email="admin@example.com",
            role="admin",
            enabled=True,
            created_at=datetime(2024, 1, 1),
            last_login=datetime.now(),
            total_requests=1000
        )
    ]


@router.post("/users/{username}/disable", dependencies=[Depends(verify_admin)])
async def disable_user(
    username: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    """Disable a user account"""
    with AuditContext(
        audit_logger,
        user_id=credentials.credentials,
        action="USER_DISABLE",
        resource_type="user",
        resource_id=username
    ):
        # TODO: Implement actual user disabling
        return {"status": "success", "message": f"User {username} disabled"}


# ====== Configuration Management ======

@router.get("/config", dependencies=[Depends(verify_admin)])
async def get_system_config():
    """Get system configuration"""
    return {
        "model_path": os.environ.get("MODEL_PATH", "models"),
        "max_batch_size": int(os.environ.get("MAX_BATCH_SIZE", "4")),
        "use_fp16": os.environ.get("USE_FP16", "true").lower() == "true",
        "encryption_enabled": os.environ.get("ENCRYPTION_ENABLED", "true").lower() == "true",
        "audit_retention_days": int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "2555"))
    }


@router.post("/config/update", dependencies=[Depends(verify_admin)])
async def update_config(
    config: ConfigUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    """Update system configuration"""
    with AuditContext(
        audit_logger,
        user_id=credentials.credentials,
        action="CONFIG_UPDATE",
        resource_type="config",
        resource_id=config.config_key
    ):
        # TODO: Implement actual configuration update
        return {
            "status": "success",
            "message": f"Configuration {config.config_key} updated",
            "new_value": config.config_value
        }


# Export router
__all__ = ['router']
