"""
Model Trainer for BrainWaveDisorderNet
src/training/trainer.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime

class ModelTrainer:
    """Handles model training and monitoring"""

    def __init__(self, config):
        """
        Initialize model trainer

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = None

    def train(self,
              model: tf.keras.Model,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              class_names: List[str] = None) -> tf.keras.callbacks.History:
        """
        Train the model

        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_names: List of class names

        Returns:
            Training history
        """
        self.logger.info("Starting model training...")

        # Training parameters
        epochs = self.config.get('training', 'epochs')
        batch_size = self.config.get('training', 'batch_size')

        # Create callbacks
        from src.models.cnn_model import BrainWaveCNN
        model_builder = BrainWaveCNN(self.config)
        callbacks = model_builder.get_callbacks(
            save_path=self.config.get('model', 'save_path')
        )

        # Add custom callback for logging
        callbacks.append(TrainingLogger(self.logger))

        # Calculate class weights for imbalanced dataset
        class_weights = self._calculate_class_weights(y_train)

        self.logger.info(f"Training parameters:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Training samples: {len(X_train)}")
        self.logger.info(f"  Validation samples: {len(X_val)}")

        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        self.history = history

        # Save training history
        self._save_training_history(history)

        # Plot training history
        self._plot_training_history(history)

        self.logger.info("Training completed successfully!")

        return history

    def _calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalanced dataset

        Args:
            y_train: Training labels

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )

        class_weight_dict = dict(zip(classes, class_weights))

        self.logger.info("Class weights calculated:")
        for class_idx, weight in class_weight_dict.items():
            self.logger.info(f"  Class {class_idx}: {weight:.3f}")

        return class_weight_dict

    def _save_training_history(self, history: tf.keras.callbacks.History):
        """
        Save training history to file

        Args:
            history: Training history
        """
        import pickle

        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Save history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(results_dir, f"training_history_{timestamp}.pkl")

        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        self.logger.info(f"Training history saved to: {history_path}")

    def _plot_training_history(self, history: tf.keras.callbacks.History):
        """
        Plot and save training history

        Args:
            history: Training history
        """
        # Create plots directory
        plot_dir = self.config.get('evaluation', 'plot_dir')
        os.makedirs(plot_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)

        # Plot training and validation loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot training and validation accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot learning rate if available
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], linewidth=2, color='red')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # Plot loss difference (overfitting indicator)
        loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
        axes[1, 1].plot(loss_diff, linewidth=2, color='orange')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Indicator (Val Loss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plot_dir, f"training_history_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        self.logger.info(f"Training history plot saved to: {plot_path}")

    def resume_training(self,
                       model: tf.keras.Model,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       initial_epoch: int = 0) -> tf.keras.callbacks.History:
        """
        Resume training from a checkpoint

        Args:
            model: Pre-trained Keras model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            initial_epoch: Epoch to start from

        Returns:
            Training history
        """
        self.logger.info(f"Resuming training from epoch {initial_epoch}")

        epochs = self.config.get('training', 'epochs')
        batch_size = self.config.get('training', 'batch_size')

        # Create callbacks
        from src.models.cnn_model import BrainWaveCNN
        model_builder = BrainWaveCNN(self.config)
        callbacks = model_builder.get_callbacks(
            save_path=self.config.get('model', 'save_path')
        )

        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)

        # Resume training
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return history


class TrainingLogger(tf.keras.callbacks.Callback):
    """Custom callback for detailed training logging"""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch"""
        if logs:
            self.logger.info(
                f"Epoch {epoch + 1}: "
                f"loss={logs.get('loss', 0):.4f}, "
                f"accuracy={logs.get('accuracy', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}, "
                f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
            )

    def on_train_begin(self, logs=None):
        """Log training start"""
        self.logger.info("Training started...")

    def on_train_end(self, logs=None):
        """Log training completion"""
        self.logger.info("Training completed!")