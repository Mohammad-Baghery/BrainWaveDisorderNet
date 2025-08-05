"""
Data visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
import os


class EEGVisualizer:
    """Utilities for visualizing EEG data and results"""

    def __init__(self):
        # Set default style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    @staticmethod
    def plot_eeg_signals(signals: np.ndarray,
                         sampling_rate: int = 256,
                         duration: float = 1.0,
                         save_path: str = None):
        """
        Plot EEG signals

        Args:
            signals: EEG signals array (channels, time_points)
            sampling_rate: Sampling rate in Hz
            duration: Duration to plot in seconds
            save_path: Path to save plot
        """
        n_channels, n_points = signals.shape
        time_points = int(sampling_rate * duration)
        time_points = min(time_points, n_points)

        time_axis = np.linspace(0, duration, time_points)

        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

        for i in range(n_channels):
            axes[i].plot(time_axis, signals[i, :time_points], linewidth=1)
            axes[i].set_ylabel(f'Channel {i + 1}')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle('EEG Signals')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

    @staticmethod
    def plot_frequency_spectrum(signal: np.ndarray,
                                sampling_rate: int = 256,
                                save_path: str = None):
        """
        Plot frequency spectrum of EEG signal

        Args:
            signal: EEG signal
            sampling_rate: Sampling rate in Hz
            save_path: Path to save plot
        """
        # Calculate FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

        # Take positive frequencies only
        positive_freqs = freqs[:len(freqs) // 2]
        positive_fft = np.abs(fft[:len(fft) // 2])

        plt.figure(figsize=(12, 6))
        plt.plot(positive_freqs, positive_fft)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('EEG Signal Frequency Spectrum')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 50)  # Focus on relevant EEG frequencies

        # Mark important frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for (band_name, (low, high)), color in zip(bands.items(), colors):
            plt.axvspan(low, high, alpha=0.2, color=color, label=band_name)

        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

    @staticmethod
    def plot_class_distribution(labels: np.ndarray,
                                class_names: List[str] = None,
                                save_path: str = None):
        """
        Plot distribution of classes in dataset

        Args:
            labels: Array of labels
            class_names: List of class names
            save_path: Path to save plot
        """
        unique_labels, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique_labels)), counts, alpha=0.8)

        # Color bars differently
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Dataset')

        if class_names:
            plt.xticks(range(len(unique_labels)),
                       [class_names[i] for i in unique_labels],
                       rotation=45)
        else:
            plt.xticks(range(len(unique_labels)), unique_labels)

        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count),
                     ha='center', va='bottom')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()