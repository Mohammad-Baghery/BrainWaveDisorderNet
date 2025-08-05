"""
GPU utilities for M1 Mac compatibility
"""

import tensorflow as tf
import logging

def setup_gpu():
    """Setup GPU acceleration for M1 Mac"""
    logger = logging.getLogger("GPU_Setup")

    try:
        # Enable memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU found and configured: {len(gpus)} device(s)")
            except RuntimeError as e:
                logger.warning(f"GPU setup error: {e}")
        else:
            logger.info("No GPU found, using CPU")

        # Set mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision enabled")

    except Exception as e:
        logger.warning(f"GPU setup failed: {e}")
        logger.info("Falling back to CPU")

    # Print device information
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Available devices: {tf.config.list_physical_devices()}")