"""
EEG Data Processor for BrainWaveDisorderNet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import logging


class EEGDataProcessor:
    """Handles EEG data loading, preprocessing, and preparation"""

    def __init__(self, config):
        """
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = []

    def load_data(self) -> pd.DataFrame:
        """
        Load EEG data from CSV file
        Returns:
            DataFrame containing EEG data
        """
        csv_path = self.config.get('data', 'csv_path')
        self.logger.info(f"Loading data from: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise e

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess EEG data
        Args:
            df: Raw EEG dataframe
        Returns:
            Tuple of preprocessed features and labels
        """
        self.logger.info("Starting data preprocessing...")

        # Assuming the last column is the target variable
        # and the rest are EEG channel data
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values

        # Map class labels to meaningful names
        self.class_names = self._get_class_mapping()

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Normalize features if specified
        if self.config.get('data', 'normalize'):
            self.logger.info("Normalizing features...")
            X = self.scaler.fit_transform(X)

        # Reshape for CNN (add channel dimension)
        sequence_length = self.config.get('data', 'sequence_length')
        if X.shape[1] != sequence_length:
            self.logger.warning(f"Expected {sequence_length} features, got {X.shape[1]}")

        # Reshape to (samples, time_steps, channels)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.logger.info(f"Preprocessing completed. X shape: {X.shape}, y shape: {y_encoded.shape}")
        self.logger.info(f"Number of classes: {len(np.unique(y_encoded))}")

        return X, y_encoded

    def _get_class_mapping(self) -> List[str]:
        """
        Get meaningful class names for epileptic seizure dataset
        Returns:
            List of class names
        """
        # Based on the Kaggle dataset description
        return [
            "Normal_Activity",
            "Tumor_Area",
            "Healthy_Area",
            "Eyes_Closed",
            "Seizure_Activity"
        ]

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets
        Args:
            X: Features
            y: Labels
        Returns:
            Tuple of train, validation, and test sets
        """
        test_size = self.config.get('data', 'test_size')
        val_size = self.config.get('data', 'val_size')
        random_state = self.config.get('data', 'random_state')

        self.logger.info(f"Splitting data - Test: {test_size}, Val: {val_size}")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )

        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Train: {X_train.shape[0]} samples")
        self.logger.info(f"  Val: {X_val.shape[0]} samples")
        self.logger.info(f"  Test: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_and_preprocess(self) -> Tuple[np.ndarray, ...]:
        """
        Complete data loading and preprocessing pipeline
        Returns:
            Tuple of train, validation, and test sets
        """
        # Load data
        df = self.load_data()

        # Preprocess data
        X, y = self.preprocess_data(df)

        # Split data
        return self.split_data(X, y)

    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.class_names

    def get_scaler(self) -> StandardScaler:
        """Get fitted scaler for inference"""
        return self.scaler

    def get_label_encoder(self) -> LabelEncoder:
        """Get fitted label encoder for inference"""
        return self.label_encoder