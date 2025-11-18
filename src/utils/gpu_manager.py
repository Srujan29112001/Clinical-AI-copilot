"""
GPU Memory Management for RTX 3060 (12GB VRAM)
Optimized model loading and inference for limited GPU memory
"""

import torch
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str
    size_gb: float
    precision: str = 'fp32'  # fp32, fp16, int8, int4
    offload_to_cpu: bool = False
    use_gradient_checkpointing: bool = False


class GPUMemoryManager:
    """
    Unified GPU memory management for RTX 3060

    Features:
    - Dynamic model loading/unloading
    - Automatic precision selection
    - CPU offloading for large models
    - Memory monitoring and optimization
    - Batch size optimization
    """

    def __init__(self, device: str = 'cuda:0', reserve_gb: float = 2.0):
        """
        Initialize GPU memory manager

        Args:
            device: CUDA device
            reserve_gb: GB to reserve for system (default: 2GB)
        """
        self.device = device
        self.reserve_gb = reserve_gb

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            self.total_memory_gb = 0
            self.available_memory_gb = 0
        else:
            # Get GPU properties
            props = torch.cuda.get_device_properties(device)
            self.total_memory_gb = props.total_memory / (1024**3)
            self.available_memory_gb = self.total_memory_gb - reserve_gb

            logger.info(
                f"GPU: {props.name}, "
                f"Total Memory: {self.total_memory_gb:.2f} GB, "
                f"Available: {self.available_memory_gb:.2f} GB"
            )

        # Track loaded models
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_configs: Dict[str, ModelConfig] = {}

        # Enable memory optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if self.device == 'cpu':
            return {'total_gb': 0, 'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}

        stats = {
            'total_gb': self.total_memory_gb,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
        }
        stats['free_gb'] = stats['total_gb'] - stats['reserved_gb']

        return stats

    def optimize_model(
        self,
        model: torch.nn.Module,
        config: ModelConfig
    ) -> torch.nn.Module:
        """
        Optimize model based on configuration

        Args:
            model: PyTorch model
            config: Model configuration

        Returns:
            Optimized model
        """
        # Apply precision optimization
        if config.precision == 'fp16':
            model = model.half()
            logger.info(f"Converted {config.name} to FP16")

        elif config.precision == 'int8':
            try:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info(f"Quantized {config.name} to INT8")
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {e}")

        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info(f"Enabled gradient checkpointing for {config.name}")

        # Move to device (with CPU offloading if needed)
        if config.offload_to_cpu:
            # Keep model on CPU, move to GPU only during inference
            logger.info(f"CPU offloading enabled for {config.name}")
        else:
            model = model.to(self.device)

        return model

    def load_model(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        force: bool = False
    ) -> torch.nn.Module:
        """
        Load model with memory optimization

        Args:
            model: Model to load
            config: Model configuration
            force: Force loading even if memory is insufficient

        Returns:
            Loaded and optimized model
        """
        # Check if already loaded
        if config.name in self.loaded_models:
            logger.info(f"Model {config.name} already loaded")
            return self.loaded_models[config.name]

        # Check available memory
        stats = self.get_memory_stats()
        required_memory = config.size_gb

        # Adjust required memory based on precision
        if config.precision == 'fp16':
            required_memory *= 0.5
        elif config.precision == 'int8':
            required_memory *= 0.25
        elif config.precision == 'int4':
            required_memory *= 0.125

        if stats['free_gb'] < required_memory and not force:
            logger.warning(
                f"Insufficient memory for {config.name}: "
                f"required {required_memory:.2f} GB, available {stats['free_gb']:.2f} GB"
            )

            # Try to free memory
            self.clear_cache()
            stats = self.get_memory_stats()

            if stats['free_gb'] < required_memory:
                # Suggest CPU offloading
                config.offload_to_cpu = True
                logger.info(f"Enabling CPU offloading for {config.name}")

        # Optimize model
        optimized_model = self.optimize_model(model, config)

        # Store model and config
        self.loaded_models[config.name] = optimized_model
        self.model_configs[config.name] = config

        logger.info(
            f"Loaded {config.name}: "
            f"{required_memory:.2f} GB ({config.precision})"
        )

        return optimized_model

    def unload_model(self, model_name: str) -> None:
        """Unload model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.model_configs[model_name]

            self.clear_cache()
            logger.info(f"Unloaded model: {model_name}")

    def clear_cache(self) -> None:
        """Clear GPU cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_optimal_batch_size(
        self,
        model_name: str,
        input_size: tuple,
        max_batch_size: int = 64
    ) -> int:
        """
        Determine optimal batch size for model

        Args:
            model_name: Name of the model
            input_size: Input tensor size (without batch dimension)
            max_batch_size: Maximum batch size to try

        Returns:
            Optimal batch size
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not loaded")
            return 1

        model = self.loaded_models[model_name]
        config = self.model_configs[model_name]

        if config.offload_to_cpu:
            # For CPU-offloaded models, use smaller batches
            return min(4, max_batch_size)

        # Binary search for optimal batch size
        model.eval()
        device = self.device if self.device != 'cpu' else 'cpu'

        optimal_batch = 1
        left, right = 1, max_batch_size

        while left <= right:
            mid = (left + right) // 2

            try:
                # Test with dummy input
                dummy_input = torch.randn(mid, *input_size).to(device)

                with torch.no_grad():
                    _ = model(dummy_input)

                # Success, try larger batch
                optimal_batch = mid
                left = mid + 1

                del dummy_input
                self.clear_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM, try smaller batch
                    right = mid - 1
                    self.clear_cache()
                else:
                    raise e

        logger.info(f"Optimal batch size for {model_name}: {optimal_batch}")
        return optimal_batch

    def profile_model(self, model_name: str, input_example: torch.Tensor) -> Dict:
        """
        Profile model memory and compute usage

        Args:
            model_name: Name of the model
            input_example: Example input tensor

        Returns:
            Profiling statistics
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not loaded")
            return {}

        model = self.loaded_models[model_name]
        model.eval()

        # Memory before
        stats_before = self.get_memory_stats()

        # Run inference
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            _ = model(input_example.to(self.device))
            end_event.record()

        torch.cuda.synchronize()

        # Memory after
        stats_after = self.get_memory_stats()

        # Calculate statistics
        inference_time_ms = start_event.elapsed_time(end_event)
        memory_used_gb = stats_after['allocated_gb'] - stats_before['allocated_gb']

        profile = {
            'model_name': model_name,
            'inference_time_ms': inference_time_ms,
            'memory_used_gb': memory_used_gb,
            'throughput_samples_per_sec': 1000 / inference_time_ms if inference_time_ms > 0 else 0
        }

        logger.info(f"Model profile for {model_name}: {profile}")
        return profile


def get_project_optimization_config(project_type: str) -> Dict[str, Any]:
    """
    Get optimized configuration for specific project

    Args:
        project_type: 'healthcare', 'robotics', or 'space'

    Returns:
        Optimization configuration
    """
    configs = {
        'healthcare': {
            'batch_size': 4,
            'precision': 'mixed',
            'use_amp': True,
            'pin_memory': True,
            'num_workers': 4,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'description': 'Optimized for streaming EEG data processing'
        },
        'robotics': {
            'batch_size': 1,
            'precision': 'fp16',
            'tensorrt': True,
            'cudnn_benchmark': True,
            'gradient_accumulation_steps': 8,
            'cache_vision_features': True,
            'async_execution': True,
            'description': 'Optimized for real-time inference'
        },
        'space': {
            'batch_size': 16,
            'precision': 'fp16',
            'gradient_accumulation_steps': 4,
            'compile_model': True,
            'fused_adam': True,
            'channels_last': True,
            'description': 'Optimized for batch processing throughput'
        }
    }

    return configs.get(project_type, configs['healthcare'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize GPU manager
    gpu_manager = GPUMemoryManager()

    # Check memory stats
    print("\nGPU Memory Stats:")
    stats = gpu_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f} GB")

    # Test model loading
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )

        def forward(self, x):
            return self.layers(x)

    model = DummyModel()

    config = ModelConfig(
        name='test_model',
        size_gb=0.5,
        precision='fp16'
    )

    optimized_model = gpu_manager.load_model(model, config)

    # Test optimal batch size
    batch_size = gpu_manager.get_optimal_batch_size(
        'test_model',
        input_size=(256,),
        max_batch_size=64
    )

    print(f"\nOptimal batch size: {batch_size}")

    # Profile model
    dummy_input = torch.randn(batch_size, 256)
    profile = gpu_manager.profile_model('test_model', dummy_input)

    print("\nModel Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")

    # Get project-specific config
    print("\nHealthcare Project Optimization:")
    config = get_project_optimization_config('healthcare')
    for key, value in config.items():
        print(f"  {key}: {value}")
