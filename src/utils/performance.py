"""
Performance monitoring utilities
"""

import time
import psutil
import tensorflow as tf
from functools import wraps
import logging

class PerformanceMonitor:
    """Monitor system performance during training and inference"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def monitor_training(self, func):
        """Decorator to monitor training performance"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start metrics
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent
            start_cpu = psutil.cpu_percent()

            self.logger.info("Training started - Performance monitoring enabled")
            self.logger.info(f"Initial Memory Usage: {start_memory:.1f}%")
            self.logger.info(f"Initial CPU Usage: {start_cpu:.1f}%")

            # Execute function
            result = func(*args, **kwargs)

            # Record end metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            end_cpu = psutil.cpu_percent()

            # Calculate metrics
            training_time = end_time - start_time
            memory_increase = end_memory - start_memory

            self.logger.info("Training completed - Performance summary:")
            self.logger.info(f"Total Training Time: {training_time:.2f} seconds")
            self.logger.info(f"Memory Usage Change: {memory_increase:+.1f}%")
            self.logger.info(f"Final Memory Usage: {end_memory:.1f}%")
            self.logger.info(f"Final CPU Usage: {end_cpu:.1f}%")

            return result

        return wrapper

    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
            'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'memory_percent': psutil.virtual_memory().percent,
            'tensorflow_version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')]
        }

        self.logger.info("System Information:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")

        return info

    def estimate_training_time(self,
                               model: tf.keras.Model,
                               train_samples: int,
                               batch_size: int,
                               epochs: int) -> Dict:
        """
        Estimate training time based on model complexity

        Args:
            model: Keras model
            train_samples: Number of training samples
            batch_size: Batch size
            epochs: Number of epochs

        Returns:
            Dictionary with time estimates
        """
        # Calculate model complexity
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

        # Estimate batches per epoch
        batches_per_epoch = train_samples // batch_size
        total_batches = batches_per_epoch * epochs

        # Rough estimation based on parameters (very approximate)
        # This is a heuristic and actual time will vary significantly
        seconds_per_batch = (total_params / 1000000) * 0.1  # Very rough estimate
        estimated_total_seconds = total_batches * seconds_per_batch

        estimated_hours = estimated_total_seconds / 3600
        estimated_minutes = (estimated_total_seconds % 3600) / 60

        estimates = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'batches_per_epoch': batches_per_epoch,
            'total_batches': total_batches,
            'estimated_seconds_per_batch': seconds_per_batch,
            'estimated_total_seconds': estimated_total_seconds,
            'estimated_hours': estimated_hours,
            'estimated_minutes': estimated_minutes
        }

        self.logger.info("Training Time Estimation:")
        self.logger.info(f"  Model Parameters: {total_params:,}")
        self.logger.info(f"  Batches per Epoch: {batches_per_epoch}")
        self.logger.info(f"  Total Batches: {total_batches}")
        self.logger.info(f"  Estimated Time: {estimated_hours:.1f}h {estimated_minutes:.0f}m")
        self.logger.warning("Note: This is a rough estimate. Actual time may vary significantly.")

        return estimates