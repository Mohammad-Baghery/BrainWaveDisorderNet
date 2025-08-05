"""
BrainWaveDisorderNet - EEG Signal Analysis for Neurological Disorder Detection
Main application file
Author: AI Assistant
Compatible with: TensorFlow Metal
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))