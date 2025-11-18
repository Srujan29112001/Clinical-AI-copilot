"""
Model Training Script for Clinical AI Copilot
Trains seizure detection, sleep classification, and multimodal models
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from tqdm import tqdm
import wandb
from datetime import datetime

from src.models.cnn_lstm import HybridCNNLSTM, SeizureDetectionConfig, SeizureDetectionLoss
from src.models.transformer_sleep import TransformerSleepStage
from src.models.multimodal_fusion import MultimodalClinicalFusion
from src.utils.gpu_manager import GPUMemoryManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer for all models

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = self._create_criterion()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['use_amp'] else None

        # Metrics
        self.best_metric = 0.0
        self.patience_counter = 0

        # Logging
        if config['experiment'].get('wandb', {}).get('enabled'):
            wandb.init(
                project=config['experiment']['wandb']['project'],
                config=config,
                name=config['experiment']['name']
            )

    def _create_optimizer(self):
        """Create optimizer based on config"""
        opt_config = self.config['training']['optimizer']

        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['type'] == 'adam_8bit':
            try:
                import bitsandbytes as bnb
                return bnb.optim.Adam8bit(
                    self.model.parameters(),
                    lr=opt_config['lr'],
                    weight_decay=opt_config['weight_decay']
                )
            except ImportError:
                logger.warning("bitsandbytes not available, using standard Adam")
                return optim.Adam(self.model.parameters(), lr=opt_config['lr'])
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config['training']['scheduler']

        if sched_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=sched_config['min_lr']
            )
        elif sched_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None

    def _create_criterion(self):
        """Create loss function"""
        loss_config = self.config['loss'].get('seizure_detection', {})

        if loss_config.get('type') == 'focal_loss':
            return SeizureDetectionLoss(
                focal_alpha=loss_config['alpha'],
                focal_gamma=loss_config['gamma'],
                use_focal=True
            )
        else:
            return nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output, _ = self.model(data)
                    loss = self.criterion(output, target)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def save_checkpoint(self, epoch: int, metric: float, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.config
        }

        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint: {filename}")

    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            if epoch % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(
                    epoch, val_acc,
                    f"./models/checkpoint_epoch_{epoch}.pth"
                )

            # Save best model
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.save_checkpoint(epoch, val_acc, "./models/best_model.pth")
                self.patience_counter = 0
                logger.info(f"New best model! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                    logger.info(f"Early stopping after {epoch} epochs")
                    break

        logger.info(f"Training complete! Best Val Acc: {self.best_metric:.2f}%")


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_data_loaders(config: dict):
    """
    Create data loaders

    NOTE: This is a placeholder. In production, implement proper
    EEG dataset loading from files (e.g., EDF, HDF5, etc.)
    """
    from torch.utils.data import TensorDataset

    # Dummy data for demonstration
    logger.warning("Using dummy data - implement proper data loading!")

    # Simulate training data (1000 samples)
    train_data = torch.randn(1000, 10, 16, 256)  # (N, time_windows, channels, samples)
    train_labels = torch.randint(0, 3, (1000,))  # 3 classes

    # Validation data (200 samples)
    val_data = torch.randn(200, 10, 16, 256)
    val_labels = torch.randint(0, 3, (200,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Clinical AI Models')
    parser.add_argument('--model', type=str, default='seizure', choices=['seizure', 'sleep', 'multimodal'])
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml')
    parser.add_argument('--model-config', type=str, default='./configs/model_config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configs
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)

    # Override epochs if specified
    if args.epochs:
        train_config['training']['epochs'] = args.epochs

    # Create model
    if args.model == 'seizure':
        from src.models.cnn_lstm import SeizureDetectionConfig
        config = SeizureDetectionConfig(**model_config['seizure_detection'])
        model = HybridCNNLSTM(config)
        logger.info("Created seizure detection model")
    elif args.model == 'sleep':
        from src.models.transformer_sleep import SleepStageConfig
        config = SleepStageConfig(**model_config['sleep_classification'])
        model = TransformerSleepStage(config)
        logger.info("Created sleep stage classification model")
    else:
        logger.error(f"Model {args.model} training not fully implemented yet")
        return

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device
    )

    # Train
    trainer.train(train_config['training']['epochs'])

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
