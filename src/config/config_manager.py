import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            project_root = Path(__file__).parent
            config_path = project_root / "config"/"config.json"

        self.config_path = config_path
        self.config = self._load_default_config()

        if os.path.exists(config_path):
            self._load_config()
        else:
            self._save_config()

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            "data": {
                "csv_path": "data/raw/Epileptic Seizure Recognition.csv",
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": 42,
                "normalize": True,
                "sequence_length": 178,
                "overlap": 0.1
            },
            "model": {
                "architecture": "cnn_1d",
                "conv_layers": [
                    {"filters": 64, "kernel_size": 3, "activation": "relu"},
                    {"filters": 128, "kernel_size": 3, "activation": "relu"},
                    {"filters": 256, "kernel_size": 3, "activation": "relu"}
                ],
                "dropout_rate": 0.5,
                "dense_units": [512, 256],
                "l2_regularization": 0.001,
                "save_path": "models/brainwave_cnn.h5"
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss": "sparse_categorical_crossentropy",
                "metrics": ["accuracy"],
                "early_stopping": {
                    "patience": 15,
                    "restore_best_weights": True,
                    "monitor": "val_loss"
                },
                "reduce_lr": {
                    "factor": 0.5,
                    "patience": 7,
                    "min_lr": 1e-7,
                    "monitor": "val_loss"
                }
            },
            "evaluation": {
                "save_plots": True,
                "plot_dir": "results/plots",
                "report_dir": "results/reports"
            },
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "log_file": "brainwave.log"
            }
        }

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                custom_config = json.load(f)
                self._merge_configs(self.config, custom_config)
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")

    def _merge_configs(self, default: Dict, custom: Dict):
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value

    def _save_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, *keys) -> Any:
        """
        Get configuration value using dot notation
        Args:
            *keys: Configuration keys
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set(self, value: Any, *keys):
        """
        Set configuration value using dot notation
        Args:
            value: Value to set
            *keys: Configuration keys
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config()

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()