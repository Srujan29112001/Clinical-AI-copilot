"""
PostgreSQL Database Models using SQLAlchemy ORM
HIPAA-compliant patient data storage with encryption
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class Patient(Base):
    """Patient demographics and information"""
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hash

    # Encrypted demographics
    encrypted_data = Column(Text, nullable=False)  # Stores encrypted JSON

    # Non-PHI metadata
    age_group = Column(String(20))  # e.g., "30-40", not exact age
    gender = Column(String(10))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    eeg_analyses = relationship("EEGAnalysis", back_populates="patient")
    medical_history = relationship("MedicalHistory", back_populates="patient")
    alerts = relationship("ClinicalAlert", back_populates="patient")

    def __repr__(self):
        return f"<Patient(id={self.id}, age_group={self.age_group})>"


class EEGAnalysis(Base):
    """EEG analysis results and reports"""
    __tablename__ = "eeg_analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False, index=True)
    report_id = Column(String(50), unique=True, nullable=False, index=True)

    # Analysis metadata
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    duration_seconds = Column(Float)
    channels_count = Column(Integer)
    sampling_rate = Column(Integer)

    # Results (encrypted)
    encrypted_results = Column(Text, nullable=False)

    # Non-sensitive summary
    primary_diagnosis_code = Column(String(20))  # ICD-10
    severity = Column(String(20))
    urgency = Column(String(20))
    confidence = Column(Float)
    seizure_probability = Column(Float)

    # Audit fields
    analyzed_by_user_id = Column(String(36))
    model_version = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="eeg_analyses")
    features = relationship("EEGFeatures", back_populates="analysis", uselist=False)

    def __repr__(self):
        return f"<EEGAnalysis(report_id={self.report_id}, diagnosis={self.primary_diagnosis_code})>"


class EEGFeatures(Base):
    """Extracted EEG features"""
    __tablename__ = "eeg_features"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String(36), ForeignKey("eeg_analyses.id"), nullable=False, unique=True)

    # Power Spectral Density
    delta_power = Column(Float)
    theta_power = Column(Float)
    alpha_power = Column(Float)
    beta_power = Column(Float)
    gamma_power = Column(Float)

    # Band ratios
    theta_beta_ratio = Column(Float)
    alpha_theta_ratio = Column(Float)

    # Entropy measures
    shannon_entropy = Column(Float)
    sample_entropy = Column(Float)
    permutation_entropy = Column(Float)

    # Hjorth parameters
    hjorth_activity = Column(Float)
    hjorth_mobility = Column(Float)
    hjorth_complexity = Column(Float)

    # Connectivity
    coherence_mean = Column(Float)
    phase_locking_value = Column(Float)

    # Additional features as JSON
    additional_features = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    analysis = relationship("EEGAnalysis", back_populates="features")

    def __repr__(self):
        return f"<EEGFeatures(analysis_id={self.analysis_id})>"


class MedicalHistory(Base):
    """Patient medical history"""
    __tablename__ = "medical_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False, index=True)

    # Encrypted medical data
    encrypted_conditions = Column(Text)
    encrypted_medications = Column(Text)
    encrypted_procedures = Column(Text)

    # Non-sensitive metadata
    entry_date = Column(DateTime(timezone=True))
    entry_type = Column(String(50))  # condition, medication, procedure, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="medical_history")

    def __repr__(self):
        return f"<MedicalHistory(patient_id={self.patient_id}, type={self.entry_type})>"


class ClinicalAlert(Base):
    """Clinical alerts and notifications"""
    __tablename__ = "clinical_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False, index=True)

    alert_type = Column(String(50), nullable=False)  # SEIZURE, ANOMALY, etc.
    severity = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    message = Column(Text, nullable=False)
    probability = Column(Float)

    # Status tracking
    status = Column(String(20), default="NEW")  # NEW, ACKNOWLEDGED, RESOLVED
    acknowledged_by = Column(String(36))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metadata = Column(JSON)

    # Relationships
    patient = relationship("Patient", back_populates="alerts")

    def __repr__(self):
        return f"<ClinicalAlert(type={self.alert_type}, severity={self.severity}, status={self.status})>"


class AuditLog(Base):
    """Audit log for HIPAA compliance"""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_id = Column(String(36), nullable=False, index=True)
    patient_id_hash = Column(String(64), index=True)

    action = Column(String(50), nullable=False)  # READ, WRITE, DELETE, etc.
    resource = Column(String(100), nullable=False)
    ip_address = Column(String(45))  # IPv6 max length
    user_agent = Column(String(255))

    result = Column(String(20), nullable=False)  # SUCCESS, FAILURE
    error_message = Column(Text)

    # Encrypted additional data
    encrypted_data = Column(Text)

    # Integrity check
    checksum = Column(String(64), nullable=False)  # SHA-256

    def __repr__(self):
        return f"<AuditLog(user={self.user_id}, action={self.action}, resource={self.resource})>"


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # GPU metrics
    gpu_utilization = Column(Float)
    gpu_memory_used_gb = Column(Float)
    gpu_memory_total_gb = Column(Float)
    gpu_temperature = Column(Float)

    # API metrics
    api_requests_total = Column(Integer)
    api_requests_failed = Column(Integer)
    avg_response_time_ms = Column(Float)

    # Model metrics
    model_inference_count = Column(Integer)
    avg_inference_time_ms = Column(Float)

    # Alerts
    alerts_generated = Column(Integer)
    critical_alerts = Column(Integer)

    # Additional metrics as JSON
    additional_metrics = Column(JSON)

    def __repr__(self):
        return f"<SystemMetrics(timestamp={self.timestamp})>"
