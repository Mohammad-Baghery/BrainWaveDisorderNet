"""
CNN Model Architecture for BrainWaveDisorderNet
src/models/cnn_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple, List
import logging


class BrainWaveCNN:
    """1D CNN model for EEG signal classification"""

    def __init__(self, config):
        """
        Initialize CNN model builder

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """
        Build 1D CNN model for EEG classification

        Args:
            input_shape: Shape of input data (time_steps, channels)
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building CNN model with input shape: {input_shape}")
        self.logger.info(f"Number of classes: {num_classes}")

        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=input_shape))

        # Convolutional layers
        conv_layers = self.config.get('model', 'conv_layers')
        l2_reg = self.config.get('model', 'l2_regularization')

        for i, layer_config in enumerate(conv_layers):
            # Convolutional layer
            model.add(layers.Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config['activation'],
                padding='same',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'conv1d_{i + 1}'
            ))

            # Batch normalization
            model.add(layers.BatchNormalization(name=f'batch_norm_{i + 1}'))

            # Max pooling
            model.add(layers.MaxPooling1D(
                pool_size=2,
                padding='same',
                name=f'maxpool_{i + 1}'
            ))

            # Dropout
            dropout_rate = self.config.get('model', 'dropout_rate')
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i + 1}'))

        # Global average pooling instead of flatten to reduce parameters
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))

        # Dense layers
        dense_units = self.config.get('model', 'dense_units')
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i + 1}'
            ))
            model.add(layers.BatchNormalization(name=f'dense_batch_norm_{i + 1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dense_dropout_{i + 1}'))

        # Output layer
        model.add(layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        ))

        # Compile model
        self._compile_model(model)

        # Print model summary
        model.summary()

        return model

    def _compile_model(self, model: tf.keras.Model):
        """
        Compile the model with optimizer, loss, and metrics

        Args:
            model: Keras model to compile
        """
        optimizer_name = self.config.get('training', 'optimizer')
        learning_rate = self.config.get('training', 'learning_rate')
        loss = self.config.get('training', 'loss')
        metrics = self.config.get('training', 'metrics')

        # Create optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}. Using Adam.")
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        self.logger.info(f"Model compiled with {optimizer_name} optimizer")

    def get_callbacks(self, save_path: str = None) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks

        Args:
            save_path: Path to save best model

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Early stopping
        early_stopping_config = self.config.get('training', 'early_stopping')
        if early_stopping_config:
            callbacks.append(EarlyStopping(
                monitor=early_stopping_config['monitor'],
                patience=early_stopping_config['patience'],
                restore_best_weights=early_stopping_config['restore_best_weights'],
                verbose=1
            ))

        # Reduce learning rate on plateau
        reduce_lr_config = self.config.get('training', 'reduce_lr')
        if reduce_lr_config:
            callbacks.append(ReduceLROnPlateau(
                monitor=reduce_lr_config['monitor'],
                factor=reduce_lr_config['factor'],
                patience=reduce_lr_config['patience'],
                min_lr=reduce_lr_config['min_lr'],
                verbose=1
            ))

        # Model checkpoint
        if save_path:
            callbacks.append(ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))

        return callbacks

    def create_advanced_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """
        Create a more advanced CNN model with residual connections

        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes

        Returns:
            Compiled advanced Keras model
        """
        inputs = layers.Input(shape=input_shape)

        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)

        # Residual blocks
        for filters in [64, 128, 256]:
            x = self._residual_block(x, filters)
            x = layers.MaxPooling1D(2)(x)

        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        self._compile_model(model)

        return model

    def _residual_block(self, x, filters: int):
        """
        Create a residual block

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Output tensor
        """
        shortcut = x

        # First conv layer
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second conv layer
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add residual connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        return x