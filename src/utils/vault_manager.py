"""
HashiCorp Vault Integration for Secrets Management
HIPAA-compliant secrets storage and retrieval
"""

import os
import logging
from typing import Dict, Any, Optional
import hvac
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VaultSecretsManager:
    """
    HashiCorp Vault client for managing secrets

    Provides secure storage and retrieval of sensitive data:
    - Database credentials
    - API keys
    - Encryption keys
    - JWT secrets
    - SSL/TLS certificates
    """

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        vault_namespace: Optional[str] = None,
        mount_point: str = "secret"
    ):
        """
        Initialize Vault client

        Args:
            vault_addr: Vault server address (default: from VAULT_ADDR env)
            vault_token: Vault token (default: from VAULT_TOKEN env)
            vault_namespace: Vault namespace for enterprise
            mount_point: KV secrets engine mount point
        """
        self.vault_addr = vault_addr or os.environ.get("VAULT_ADDR", "http://localhost:8200")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN")
        self.vault_namespace = vault_namespace or os.environ.get("VAULT_NAMESPACE")
        self.mount_point = mount_point

        # Initialize client
        self.client = hvac.Client(
            url=self.vault_addr,
            token=self.vault_token,
            namespace=self.vault_namespace
        )

        # Verify authentication
        if not self.client.is_authenticated():
            raise Exception("Failed to authenticate with Vault")

        logger.info(f"Vault client initialized: {self.vault_addr}")

    def get_secret(self, path: str, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve secret from Vault

        Args:
            path: Secret path (e.g., "clinical-ai/database")
            version: Specific version to retrieve (None = latest)

        Returns:
            Dictionary containing secret data
        """
        try:
            # KV v2 engine
            if version:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point,
                    version=version
                )
            else:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point
                )

            return response['data']['data']

        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {str(e)}")
            raise

    def set_secret(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store secret in Vault

        Args:
            path: Secret path
            data: Secret data to store

        Returns:
            Response from Vault
        """
        try:
            response = self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.mount_point
            )

            logger.info(f"Secret stored at {path}")
            return response

        except Exception as e:
            logger.error(f"Failed to store secret at {path}: {str(e)}")
            raise

    def delete_secret(self, path: str, versions: Optional[list] = None):
        """
        Delete secret versions

        Args:
            path: Secret path
            versions: Versions to delete (None = all)
        """
        try:
            if versions:
                self.client.secrets.kv.v2.delete_secret_versions(
                    path=path,
                    versions=versions,
                    mount_point=self.mount_point
                )
            else:
                self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path,
                    mount_point=self.mount_point
                )

            logger.info(f"Secret deleted at {path}")

        except Exception as e:
            logger.error(f"Failed to delete secret at {path}: {str(e)}")
            raise

    def get_database_credentials(self) -> Dict[str, str]:
        """Get PostgreSQL database credentials"""
        return self.get_secret("clinical-ai/database/postgres")

    def get_neo4j_credentials(self) -> Dict[str, str]:
        """Get Neo4j credentials"""
        return self.get_secret("clinical-ai/database/neo4j")

    def get_encryption_keys(self) -> Dict[str, str]:
        """Get encryption keys for HIPAA compliance"""
        return self.get_secret("clinical-ai/encryption/keys")

    def get_jwt_secret(self) -> str:
        """Get JWT secret key"""
        secret = self.get_secret("clinical-ai/auth/jwt")
        return secret['secret_key']

    def get_api_keys(self) -> Dict[str, str]:
        """Get external API keys (OpenAI, Hugging Face, etc.)"""
        return self.get_secret("clinical-ai/api-keys")

    def rotate_secret(self, path: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate a secret (creates new version)

        Args:
            path: Secret path
            new_data: New secret data

        Returns:
            Response from Vault
        """
        return self.set_secret(path, new_data)

    def get_dynamic_database_credentials(
        self,
        role: str,
        ttl: timedelta = timedelta(hours=1)
    ) -> Dict[str, str]:
        """
        Get dynamic database credentials (auto-expiring)

        Requires Vault database secrets engine configured

        Args:
            role: Database role name
            ttl: Time-to-live for credentials

        Returns:
            Temporary database credentials
        """
        try:
            response = self.client.secrets.database.generate_credentials(
                name=role,
                mount_point="database"
            )

            return {
                'username': response['data']['username'],
                'password': response['data']['password'],
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration']
            }

        except Exception as e:
            logger.error(f"Failed to generate dynamic credentials for role {role}: {str(e)}")
            raise

    def revoke_lease(self, lease_id: str):
        """
        Revoke a dynamic secret lease

        Args:
            lease_id: Lease ID to revoke
        """
        try:
            self.client.sys.revoke_lease(lease_id=lease_id)
            logger.info(f"Lease {lease_id} revoked")

        except Exception as e:
            logger.error(f"Failed to revoke lease {lease_id}: {str(e)}")
            raise

    def create_pki_certificate(
        self,
        common_name: str,
        role: str = "clinical-ai",
        ttl: str = "8760h"  # 1 year
    ) -> Dict[str, str]:
        """
        Generate SSL/TLS certificate from Vault PKI

        Args:
            common_name: Certificate common name (e.g., "api.clinical-ai.com")
            role: PKI role name
            ttl: Certificate TTL

        Returns:
            Certificate, private key, and CA chain
        """
        try:
            response = self.client.secrets.pki.generate_certificate(
                name=role,
                common_name=common_name,
                ttl=ttl,
                mount_point="pki"
            )

            return {
                'certificate': response['data']['certificate'],
                'private_key': response['data']['private_key'],
                'issuing_ca': response['data']['issuing_ca'],
                'ca_chain': response['data']['ca_chain'],
                'serial_number': response['data']['serial_number']
            }

        except Exception as e:
            logger.error(f"Failed to generate certificate for {common_name}: {str(e)}")
            raise


# Convenience function for initializing Vault
def get_vault_client() -> VaultSecretsManager:
    """
    Get initialized Vault client

    Returns:
        Configured VaultSecretsManager instance
    """
    return VaultSecretsManager()


# Example usage for migrating secrets to Vault
def initialize_vault_secrets():
    """
    Initialize Vault with Clinical AI secrets

    Run this once to populate Vault with initial secrets
    """
    vault = get_vault_client()

    # Database credentials
    vault.set_secret("clinical-ai/database/postgres", {
        "username": os.environ.get("POSTGRES_USER", "clinical_admin"),
        "password": os.environ.get("POSTGRES_PASSWORD", "CHANGE_ME"),
        "host": os.environ.get("POSTGRES_HOST", "postgres-service"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "database": os.environ.get("POSTGRES_DB", "clinical_ai")
    })

    vault.set_secret("clinical-ai/database/neo4j", {
        "username": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "CHANGE_ME"),
        "uri": os.environ.get("NEO4J_URI", "bolt://neo4j-service:7687")
    })

    # Encryption keys
    vault.set_secret("clinical-ai/encryption/keys", {
        "encryption_key": os.environ.get("ENCRYPTION_KEY", ""),
        "audit_encryption_key": os.environ.get("AUDIT_ENCRYPTION_KEY", "")
    })

    # JWT secrets
    vault.set_secret("clinical-ai/auth/jwt", {
        "secret_key": os.environ.get("JWT_SECRET_KEY", ""),
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
    })

    # API keys
    vault.set_secret("clinical-ai/api-keys", {
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "huggingface_token": os.environ.get("HUGGINGFACE_TOKEN", "")
    })

    logger.info("Vault secrets initialized successfully")


if __name__ == "__main__":
    # Initialize secrets
    initialize_vault_secrets()
