"""
Model Checkpoint Management and Versioning System
Handles model saving, loading, versioning, and deployment
"""

import os
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelCheckpointManager:
    """
    Manage model checkpoints with versioning and metadata

    Features:
    - Automatic checkpoint saving
    - Version control
    - Metadata tracking (accuracy, hyperparameters, training date)
    - Model comparison
    - Rollback capability
    - S3/cloud storage integration
    """

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": {}}

    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_name: str = "model",
        version: Optional[str] = None,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint

        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            metrics: Performance metrics (accuracy, loss, etc.)
            hyperparameters: Model hyperparameters
            model_name: Name of model
            version: Version string (auto-generated if None)
            is_best: Whether this is the best model

        Returns:
            Checkpoint version string
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_id = f"{model_name}_v{version}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "model_name": model_name
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Calculate checksum
        checkpoint_bytes = json.dumps(checkpoint, default=str).encode()
        checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint["checksum"] = checksum

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        self.metadata["checkpoints"][checkpoint_id] = {
            "path": str(checkpoint_path),
            "version": version,
            "model_name": model_name,
            "epoch": epoch,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "timestamp": checkpoint["timestamp"],
            "checksum": checksum,
            "is_best": is_best,
            "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024)
        }

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / f"{model_name}_best.pt"
            shutil.copy(checkpoint_path, best_path)
            self.metadata["best_checkpoint"] = checkpoint_id

        self._save_metadata()

        logger.info(f"Checkpoint saved: {checkpoint_id}")
        logger.info(f"Metrics: {metrics}")

        return version

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        version: Optional[str] = None,
        model_name: str = "model",
        load_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            model: PyTorch model to load weights into
            version: Specific version to load (None = latest)
            model_name: Name of model
            load_best: Load best checkpoint instead of latest
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load model onto

        Returns:
            Checkpoint metadata
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Best checkpoint for {model_name} not found")
        elif version:
            checkpoint_id = f"{model_name}_v{version}"
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        else:
            # Load latest
            checkpoints = self.get_checkpoints(model_name=model_name)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found for {model_name}")

            latest = max(checkpoints, key=lambda x: x["timestamp"])
            checkpoint_path = Path(latest["path"])

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path.name}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_checkpoints(
        self,
        model_name: Optional[str] = None,
        sort_by: str = "timestamp",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of checkpoints

        Args:
            model_name: Filter by model name
            sort_by: Sort by field (timestamp, metrics.accuracy, etc.)
            limit: Limit number of results

        Returns:
            List of checkpoint metadata
        """
        checkpoints = list(self.metadata["checkpoints"].values())

        # Filter by model name
        if model_name:
            checkpoints = [c for c in checkpoints if c["model_name"] == model_name]

        # Sort
        if sort_by == "timestamp":
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        elif sort_by.startswith("metrics."):
            metric_name = sort_by.split(".", 1)[1]
            checkpoints.sort(
                key=lambda x: x["metrics"].get(metric_name, 0),
                reverse=True
            )

        # Limit
        if limit:
            checkpoints = checkpoints[:limit]

        return checkpoints

    def delete_checkpoint(self, version: str, model_name: str):
        """Delete a checkpoint"""
        checkpoint_id = f"{model_name}_v{version}"

        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint_path = Path(self.metadata["checkpoints"][checkpoint_id]["path"])

        # Delete file
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Remove from metadata
        del self.metadata["checkpoints"][checkpoint_id]
        self._save_metadata()

        logger.info(f"Checkpoint deleted: {checkpoint_id}")

    def cleanup_old_checkpoints(
        self,
        model_name: str,
        keep_last_n: int = 5,
        keep_best: bool = True
    ):
        """
        Clean up old checkpoints, keeping only recent ones

        Args:
            model_name: Model name to clean up
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Always keep best checkpoint
        """
        checkpoints = self.get_checkpoints(model_name=model_name, sort_by="timestamp")

        # Identify checkpoints to delete
        to_delete = checkpoints[keep_last_n:]

        if keep_best:
            best_checkpoint_id = self.metadata.get("best_checkpoint")
            to_delete = [c for c in to_delete if f"{c['model_name']}_v{c['version']}" != best_checkpoint_id]

        # Delete
        for checkpoint in to_delete:
            try:
                self.delete_checkpoint(checkpoint["version"], model_name)
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")

        logger.info(f"Cleaned up {len(to_delete)} old checkpoints for {model_name}")

    def export_checkpoint(
        self,
        version: str,
        model_name: str,
        export_path: str,
        format: str = "pytorch"
    ):
        """
        Export checkpoint to different format

        Args:
            version: Checkpoint version
            model_name: Model name
            export_path: Export destination
            format: Export format (pytorch, onnx, torchscript)
        """
        checkpoint_id = f"{model_name}_v{version}"
        checkpoint_path = Path(self.metadata["checkpoints"][checkpoint_id]["path"])

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "pytorch":
            shutil.copy(checkpoint_path, export_path)
        elif format == "onnx":
            # TODO: Implement ONNX export
            raise NotImplementedError("ONNX export not implemented")
        elif format == "torchscript":
            # TODO: Implement TorchScript export
            raise NotImplementedError("TorchScript export not implemented")
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Checkpoint exported to {export_path}")

    def compare_checkpoints(
        self,
        checkpoint_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple checkpoints

        Args:
            checkpoint_ids: List of checkpoint IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Comparison results
        """
        comparison = {}

        for checkpoint_id in checkpoint_ids:
            if checkpoint_id not in self.metadata["checkpoints"]:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                continue

            checkpoint_meta = self.metadata["checkpoints"][checkpoint_id]

            if metrics:
                comparison[checkpoint_id] = {
                    metric: checkpoint_meta["metrics"].get(metric)
                    for metric in metrics
                }
            else:
                comparison[checkpoint_id] = checkpoint_meta["metrics"]

        return comparison


# Convenience functions
def save_model(
    model: torch.nn.Module,
    model_name: str,
    metrics: Dict[str, float],
    **kwargs
) -> str:
    """Quick save model checkpoint"""
    manager = ModelCheckpointManager()
    return manager.save_checkpoint(model, model_name=model_name, metrics=metrics, **kwargs)


def load_model(
    model: torch.nn.Module,
    model_name: str,
    version: Optional[str] = None,
    load_best: bool = False
) -> Dict[str, Any]:
    """Quick load model checkpoint"""
    manager = ModelCheckpointManager()
    return manager.load_checkpoint(model, model_name=model_name, version=version, load_best=load_best)
