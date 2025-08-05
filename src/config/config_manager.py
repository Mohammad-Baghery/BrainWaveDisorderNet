import json
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: Path):
        if config_path is None:
            project_root = Path(__file__).parent
            config_path = project_root / "config"/"config.json"

        self.config_path = config_path
        self.config = self._load_default_config()

