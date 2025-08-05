"""
BrainWaveDisorderNet - EEG Signal Analysis for Neurological Disorder Detection
Main application file
Author: Mohammad Baghery
Compatible with: macOS M1, TensorFlow Metal
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config.config_manager import ConfigManager
from src.data.data_processor import EEGDataProcessor
from src.models.cnn_model import BrainWaveCNN
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.gpu_utils import setup_gpu

def main():
    """Main application entry point"""

    # Setup logging
    logger = setup_logger("BrainWaveDisorderNet", level=logging.INFO)
    logger.info("Starting BrainWaveDisorderNet application...")

    try:
        # Setup GPU acceleration for M1 Mac
        setup_gpu()

        # Load configuration
        config = ConfigManager()

        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = EEGDataProcessor(config)

        # Check if data file exists
        data_path = config.get('data', 'csv_path')
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at: {data_path}")
            logger.info("Please download the dataset from Kaggle and place it in the data/raw/ directory")
            return

        # Load and preprocess data
        logger.info("Loading and preprocessing EEG data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.load_and_preprocess()

        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Initialize model
        logger.info("Initializing CNN model...")
        input_shape = X_train.shape[1:]
        num_classes = len(data_processor.get_class_names())

        model_builder = BrainWaveCNN(config)
        model = model_builder.build_model(input_shape, num_classes)

        # Initialize trainer
        logger.info("Setting up model trainer...")
        trainer = ModelTrainer(config)

        # Train model
        logger.info("Starting model training...")
        history = trainer.train(
            model,
            X_train, y_train,
            X_val, y_val,
            class_names=data_processor.get_class_names()
        )

        # Evaluate model
        logger.info("Evaluating model performance...")
        evaluator = ModelEvaluator(config)

        # Test set evaluation
        test_results = evaluator.evaluate(
            model,
            X_test, y_test,
            class_names=data_processor.get_class_names(),
            save_plots=True
        )

        logger.info("Training completed successfully!")
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test Loss: {test_results['loss']:.4f}")

        # Save final model
        model_save_path = config.get('model', 'save_path')
        model.save(model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()