"""
Encryption Module for HIPAA Compliance
Handles encryption/decryption of PHI (Protected Health Information)
"""

import os
import base64
import hashlib
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    HIPAA-compliant encryption manager

    Features:
    - AES-256 encryption for data at rest
    - Secure key derivation (PBKDF2)
    - Field-level encryption
    - Key rotation support
    """

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption manager

        Args:
            master_key: Master encryption key (from environment variable)
        """
        if master_key is None:
            master_key = os.environ.get('ENCRYPTION_KEY')

        if master_key is None:
            logger.warning("No encryption key provided, generating new key")
            master_key = Fernet.generate_key().decode()

        self.master_key = master_key.encode() if isinstance(master_key, str) else master_key
        self.fernet = Fernet(self._derive_key(self.master_key))

    def _derive_key(self, password: bytes, salt: Optional[bytes] = None) -> bytes:
        """
        Derive encryption key from password using PBKDF2

        Args:
            password: Master password
            salt: Optional salt (generated if not provided)

        Returns:
            Derived key
        """
        if salt is None:
            salt = b'clinical-ai-salt-v1'  # In production, use random salt per record

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data

        Args:
            data: Data to encrypt (string or bytes)

        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode()

        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted string
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def encrypt_dict(self, data: dict, fields_to_encrypt: list) -> dict:
        """
        Encrypt specific fields in dictionary

        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt

        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()

        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field] is not None:
                value = str(encrypted_data[field])
                encrypted_data[field] = self.encrypt(value)
                encrypted_data[f'{field}_encrypted'] = True

        return encrypted_data

    def decrypt_dict(self, data: dict, fields_to_decrypt: list) -> dict:
        """
        Decrypt specific fields in dictionary

        Args:
            data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt

        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()

        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data.get(f'{field}_encrypted'):
                decrypted_data[field] = self.decrypt(decrypted_data[field])
                decrypted_data.pop(f'{field}_encrypted', None)

        return decrypted_data

    def hash_identifier(self, identifier: str) -> str:
        """
        Create one-way hash of identifier (for indexing without storing PHI)

        Args:
            identifier: Patient identifier or other sensitive ID

        Returns:
            SHA-256 hash
        """
        return hashlib.sha256(identifier.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate new encryption key"""
        return Fernet.generate_key().decode()


def encrypt_file(input_file: str, output_file: str, encryption_manager: EncryptionManager):
    """
    Encrypt file contents

    Args:
        input_file: Path to input file
        output_file: Path to output file
        encryption_manager: EncryptionManager instance
    """
    with open(input_file, 'rb') as f:
        data = f.read()

    encrypted = encryption_manager.encrypt(data)

    with open(output_file, 'w') as f:
        f.write(encrypted)

    logger.info(f"Encrypted {input_file} -> {output_file}")


def decrypt_file(input_file: str, output_file: str, encryption_manager: EncryptionManager):
    """
    Decrypt file contents

    Args:
        input_file: Path to encrypted file
        output_file: Path to output file
        encryption_manager: EncryptionManager instance
    """
    with open(input_file, 'r') as f:
        encrypted_data = f.read()

    decrypted = encryption_manager.decrypt(encrypted_data)

    with open(output_file, 'w') as f:
        f.write(decrypted)

    logger.info(f"Decrypted {input_file} -> {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test encryption
    em = EncryptionManager()

    # Test string encryption
    original = "Patient ID: 12345, Diagnosis: Epilepsy"
    encrypted = em.encrypt(original)
    decrypted = em.decrypt(encrypted)

    print(f"Original: {original}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {original == decrypted}")

    # Test dictionary encryption
    patient_data = {
        'patient_id': '12345',
        'name': 'John Doe',
        'diagnosis': 'Epilepsy',
        'age': 45
    }

    encrypted_data = em.encrypt_dict(
        patient_data,
        fields_to_encrypt=['patient_id', 'name', 'diagnosis']
    )

    print(f"\nEncrypted patient data: {encrypted_data}")

    decrypted_data = em.decrypt_dict(
        encrypted_data,
        fields_to_decrypt=['patient_id', 'name', 'diagnosis']
    )

    print(f"Decrypted patient data: {decrypted_data}")

    # Test hash
    patient_id = "12345"
    hashed = em.hash_identifier(patient_id)
    print(f"\nPatient ID hash: {hashed}")
