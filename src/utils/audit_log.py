"""
HIPAA-Compliant Audit Logging
Tracks all access to PHI (Protected Health Information)
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path
import logging

from .encryption import EncryptionManager

logger = logging.getLogger(__name__)


class HIPAALogger:
    """
    HIPAA-compliant audit logger

    Requirements:
    - Track all PHI access (who, what, when, where, why)
    - Immutable logs
    - Encrypted storage
    - 7-year retention (2555 days)
    - Tamper detection
    """

    def __init__(
        self,
        log_dir: str = "./logs/audit",
        encryption_key: Optional[str] = None,
        retention_days: int = 2555  # 7 years
    ):
        """
        Initialize HIPAA audit logger

        Args:
            log_dir: Directory for audit logs
            encryption_key: Encryption key for log storage
            retention_days: Log retention period (default: 7 years)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days

        # Initialize encryption
        self.encryption = EncryptionManager(encryption_key)

        # Current log file
        self.current_log_file = self._get_log_file()

        logger.info(f"HIPAA audit logger initialized: {self.log_dir}")

    def _get_log_file(self) -> Path:
        """Get current log file path"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.log"

    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA-256 checksum for tamper detection"""
        return hashlib.sha256(data.encode()).hexdigest()

    def log_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        ip_address: str,
        resource: Optional[str] = None,
        reason: Optional[str] = None,
        result: str = "SUCCESS",
        additional_data: Optional[Dict] = None
    ) -> None:
        """
        Log PHI access event

        Args:
            user_id: ID of user accessing PHI
            patient_id: Patient identifier (will be hashed)
            action: Action performed (READ, WRITE, DELETE, etc.)
            ip_address: IP address of request
            resource: Resource accessed (e.g., EEG_DATA, DIAGNOSIS)
            reason: Reason for access (treatment, billing, etc.)
            result: Access result (SUCCESS, DENIED, ERROR)
            additional_data: Additional contextual data
        """
        # Create audit entry
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'patient_id_hash': self.encryption.hash_identifier(patient_id),
            'action': action,
            'resource': resource,
            'reason': reason,
            'result': result,
            'ip_address': ip_address,
            'additional_data': additional_data or {}
        }

        # Add checksum
        entry_json = json.dumps(audit_entry, sort_keys=True)
        audit_entry['checksum'] = self._calculate_checksum(entry_json)

        # Encrypt entry
        encrypted_entry = self.encryption.encrypt(json.dumps(audit_entry))

        # Write to log file
        log_file = self._get_log_file()

        with open(log_file, 'a') as f:
            f.write(encrypted_entry + '\n')

        # Log to standard logger (without PHI)
        logger.info(
            f"Audit: {action} by {user_id} on patient {audit_entry['patient_id_hash'][:8]}... "
            f"from {ip_address} - {result}"
        )

    def log_authentication(
        self,
        user_id: str,
        ip_address: str,
        success: bool,
        method: str = "PASSWORD",
        failure_reason: Optional[str] = None
    ) -> None:
        """Log authentication attempt"""
        self.log_access(
            user_id=user_id,
            patient_id="N/A",
            action="AUTHENTICATION",
            ip_address=ip_address,
            resource=f"AUTH_{method}",
            result="SUCCESS" if success else "FAILURE",
            additional_data={
                'method': method,
                'failure_reason': failure_reason
            }
        )

    def log_data_export(
        self,
        user_id: str,
        patient_id: str,
        ip_address: str,
        export_type: str,
        record_count: int,
        destination: str
    ) -> None:
        """Log data export event"""
        self.log_access(
            user_id=user_id,
            patient_id=patient_id,
            action="DATA_EXPORT",
            ip_address=ip_address,
            resource=export_type,
            additional_data={
                'record_count': record_count,
                'destination': destination
            }
        )

    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        additional_data: Optional[Dict] = None
    ) -> None:
        """Log system-level event"""
        self.log_access(
            user_id="SYSTEM",
            patient_id="N/A",
            action=f"SYSTEM_{event_type}",
            ip_address="127.0.0.1",
            resource="SYSTEM",
            additional_data={
                'description': description,
                'severity': severity,
                **(additional_data or {})
            }
        )

    def query_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        action: Optional[str] = None
    ) -> list:
        """
        Query audit logs (requires authorization)

        Args:
            start_date: Start date for query
            end_date: End date for query
            user_id: Filter by user ID
            patient_id: Filter by patient ID (will be hashed for comparison)
            action: Filter by action type

        Returns:
            List of matching audit entries
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        if end_date is None:
            end_date = datetime.now()

        # Hash patient ID for comparison if provided
        patient_id_hash = None
        if patient_id:
            patient_id_hash = self.encryption.hash_identifier(patient_id)

        matching_entries = []

        # Iterate through log files
        current_date = start_date
        while current_date <= end_date:
            log_file = self.log_dir / f"audit_{current_date.strftime('%Y-%m-%d')}.log"

            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            # Decrypt entry
                            decrypted = self.encryption.decrypt(line.strip())
                            entry = json.loads(decrypted)

                            # Apply filters
                            if user_id and entry.get('user_id') != user_id:
                                continue

                            if patient_id_hash and entry.get('patient_id_hash') != patient_id_hash:
                                continue

                            if action and entry.get('action') != action:
                                continue

                            # Verify checksum
                            checksum = entry.pop('checksum')
                            entry_json = json.dumps(entry, sort_keys=True)
                            if self._calculate_checksum(entry_json) != checksum:
                                logger.warning(f"Checksum mismatch in log entry: {entry}")
                                continue

                            matching_entries.append(entry)

                        except Exception as e:
                            logger.error(f"Error processing log entry: {e}")

            current_date += timedelta(days=1)

        return matching_entries

    def cleanup_old_logs(self) -> None:
        """Remove logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for log_file in self.log_dir.glob("audit_*.log"):
            # Extract date from filename
            try:
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")

            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")


class AuditContext:
    """Context manager for audit logging"""

    def __init__(
        self,
        logger: HIPAALogger,
        user_id: str,
        patient_id: str,
        action: str,
        ip_address: str,
        **kwargs
    ):
        self.logger = logger
        self.user_id = user_id
        self.patient_id = patient_id
        self.action = action
        self.ip_address = ip_address
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = "SUCCESS" if exc_type is None else "ERROR"
        duration = (datetime.now() - self.start_time).total_seconds()

        additional_data = self.kwargs.get('additional_data', {})
        additional_data['duration_seconds'] = duration

        if exc_type:
            additional_data['error'] = str(exc_val)

        self.logger.log_access(
            user_id=self.user_id,
            patient_id=self.patient_id,
            action=self.action,
            ip_address=self.ip_address,
            result=result,
            additional_data=additional_data,
            **{k: v for k, v in self.kwargs.items() if k != 'additional_data'}
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test audit logging
    audit_logger = HIPAALogger(log_dir="./test_logs")

    # Log access
    audit_logger.log_access(
        user_id="doctor123",
        patient_id="patient456",
        action="READ",
        ip_address="192.168.1.100",
        resource="EEG_DATA",
        reason="Treatment",
        result="SUCCESS"
    )

    # Log authentication
    audit_logger.log_authentication(
        user_id="doctor123",
        ip_address="192.168.1.100",
        success=True,
        method="PASSWORD"
    )

    # Query logs
    logs = audit_logger.query_logs(
        user_id="doctor123",
        action="READ"
    )

    print(f"\nFound {len(logs)} matching audit entries")
    for entry in logs:
        print(f"  {entry['timestamp']}: {entry['action']} - {entry['result']}")

    # Test context manager
    with AuditContext(
        audit_logger,
        user_id="doctor123",
        patient_id="patient456",
        action="ANALYZE_EEG",
        ip_address="192.168.1.100",
        resource="EEG_ANALYSIS"
    ):
        print("\nPerforming EEG analysis...")
        # Simulated work
        import time
        time.sleep(0.1)

    print("\nAudit logging test complete")
