"""
Model utilities
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Any
import pickle
import os

class ModelUtils:
    """Utility functions for model management"""

    @staticmethod
    def save_model_artifacts(model: tf.keras.Model,
                             scaler: Any,
                             label_encoder: Any,
                             config: Dict,
                             save_dir: str):
        """
        Save model and associated artifacts

        Args:
            model: Trained Keras model
            scaler: Fitted data scaler
            label_encoder: Fitted label encoder
            config: Configuration dictionary
            save_dir: Directory to save artifacts
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, 'model.h5')
        model.save(model_path)

        # Save scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save label encoder
        encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)

        # Save config
        config_path = os.path.join(save_dir, 'model_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        print(f"Model artifacts saved to: {save_dir}")

    @staticmethod
    def load_model_artifacts(save_dir: str) -> tuple:
        """
        Load model and associated artifacts

        Args:
            save_dir: Directory containing saved artifacts

        Returns:
            Tuple of (model, scaler, label_encoder, config)
        """
        # Load model
        model_path = os.path.join(save_dir, 'model.h5')
        model = tf.keras.models.load_model(model_path)

        # Load scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Load label encoder
        encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Load config
        config_path = os.path.join(save_dir, 'model_config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        return model, scaler, label_encoder, config

    @staticmethod
    def predict_single_sample(model: tf.keras.Model,
                              sample: np.ndarray,
                              scaler: Any,
                              label_encoder: Any,
                              class_names: list) -> Dict:
        """
        Make prediction on a single EEG sample

        Args:
            model: Trained model
            sample: EEG sample to predict
            scaler: Fitted scaler
            label_encoder: Fitted label encoder
            class_names: List of class names

        Returns:
            Dictionary with prediction results
        """
        # Preprocess sample
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)

        # Scale the sample
        sample_scaled = scaler.transform(sample)

        # Reshape for CNN
        sample_reshaped = sample_scaled.reshape(sample_scaled.shape[0], sample_scaled.shape[1], 1)

        # Make prediction
        prediction_proba = model.predict(sample_reshaped, verbose=0)
        predicted_class = np.argmax(prediction_proba, axis=1)[0]
        confidence = np.max(prediction_proba, axis=1)[0]

        # Get class name
        class_name = class_names[predicted_class] if class_names else f"Class_{predicted_class}"

        return {
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': prediction_proba[0],
            'all_class_probs': {
                class_names[i] if class_names else f"Class_{i}": prob
                for i, prob in enumerate(prediction_proba[0])
            }
        }