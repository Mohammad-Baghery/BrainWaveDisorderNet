"""
Model Evaluator for BrainWaveDisorderNet
src/evaluation/evaluator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime
import pandas as pd


class ModelEvaluator:
    """Handles model evaluation and metrics visualization"""

    def __init__(self, config):
        """
        Initialize model evaluator

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self,
                 model: tf.keras.Model,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 class_names: List[str] = None,
                 save_plots: bool = True) -> Dict:
        """
        Comprehensive model evaluation

        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
            save_plots: Whether to save evaluation plots

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting model evaluation...")

        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate basic metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Create results dictionary
        results = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        # Generate detailed classification report
        report = self._generate_classification_report(y_test, y_pred, class_names)
        results['classification_report'] = report

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm

        # Calculate ROC-AUC for multiclass
        if len(np.unique(y_test)) > 2:
            roc_auc = self._calculate_multiclass_roc_auc(y_test, y_pred_proba)
            results['roc_auc'] = roc_auc

        # Save detailed results
        if save_plots:
            self._save_evaluation_plots(y_test, y_pred, y_pred_proba, class_names, results)
            self._save_evaluation_report(results, class_names)

        self.logger.info("Model evaluation completed!")

        return results

    def _generate_classification_report(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        class_names: List[str] = None) -> str:
        """
        Generate detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names

        Returns:
            Classification report string
        """
        target_names = class_names if class_names else None

        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4
        )

        self.logger.info("Classification Report:")
        self.logger.info(f"\n{report}")

        return report

    def _calculate_multiclass_roc_auc(self,
                                      y_true: np.ndarray,
                                      y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate ROC-AUC for multiclass classification

        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary with ROC-AUC scores
        """
        n_classes = y_pred_proba.shape[1]

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Calculate ROC-AUC for each class
        roc_auc = {}
        fpr = {}
        tpr = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate macro and micro average ROC-AUC
        try:
            roc_auc['macro'] = roc_auc_score(y_true_bin, y_pred_proba,
                                             average='macro', multi_class='ovr')
            roc_auc['micro'] = roc_auc_score(y_true_bin, y_pred_proba,
                                             average='micro', multi_class='ovr')
        except ValueError as e:
            self.logger.warning(f"Could not calculate macro/micro ROC-AUC: {e}")

        roc_auc['fpr'] = fpr
        roc_auc['tpr'] = tpr

        return roc_auc

    def _save_evaluation_plots(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: np.ndarray,
                               class_names: List[str],
                               results: Dict):
        """
        Save evaluation plots

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            results: Evaluation results
        """
        plot_dir = self.config.get('evaluation', 'plot_dir')
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot confusion matrix
        self._plot_confusion_matrix(
            results['confusion_matrix'],
            class_names,
            os.path.join(plot_dir, f"confusion_matrix_{timestamp}.png")
        )

        # Plot ROC curves if available
        if 'roc_auc' in results:
            self._plot_roc_curves(
                results['roc_auc'],
                class_names,
                os.path.join(plot_dir, f"roc_curves_{timestamp}.png")
            )

        # Plot prediction distribution
        self._plot_prediction_distribution(
            y_true, y_pred, class_names,
            os.path.join(plot_dir, f"prediction_distribution_{timestamp}.png")
        )

        # Plot confidence distribution
        self._plot_confidence_distribution(
            y_pred_proba,
            os.path.join(plot_dir, f"confidence_distribution_{timestamp}.png")
        )

    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(cm_normalized,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=class_names or range(cm.shape[1]),
                    yticklabels=class_names or range(cm.shape[0]))

        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved to: {save_path}")

    def _plot_roc_curves(self, roc_data: Dict, class_names: List[str], save_path: str):
        """Plot and save ROC curves"""
        plt.figure(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

        for i, color in enumerate(colors):
            if i in roc_data['fpr'] and i in roc_data['tpr']:
                class_name = class_names[i] if class_names else f"Class {i}"
                plt.plot(roc_data['fpr'][i], roc_data['tpr'][i],
                         color=color, linewidth=2,
                         label=f'{class_name} (AUC = {roc_data[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"ROC curves saved to: {save_path}")

    def _plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      class_names: List[str], save_path: str):
        """Plot prediction distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # True label distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[0].bar(range(len(unique)), counts, alpha=0.7, color='skyblue')
        axes[0].set_title('True Label Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        if class_names:
            axes[0].set_xticks(range(len(unique)))
            axes[0].set_xticklabels([class_names[i] for i in unique], rotation=45)

        # Predicted label distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        axes[1].bar(range(len(unique_pred)), counts_pred, alpha=0.7, color='lightcoral')
        axes[1].set_title('Predicted Label Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        if class_names:
            axes[1].set_xticks(range(len(unique_pred)))
            axes[1].set_xticklabels([class_names[i] for i in unique_pred], rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Prediction distribution saved to: {save_path}")

    def _plot_confidence_distribution(self, y_pred_proba: np.ndarray, save_path: str):
        """Plot confidence distribution"""
        plt.figure(figsize=(10, 6))

        # Get maximum probabilities (confidence scores)
        max_probs = np.max(y_pred_proba, axis=1)

        plt.hist(max_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Maximum Probability (Confidence)')
        plt.ylabel('Frequency')
        plt.axvline(x=np.mean(max_probs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(max_probs):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confidence distribution saved to: {save_path}")

    def _save_evaluation_report(self, results: Dict, class_names: List[str]):
        """Save detailed evaluation report"""
        report_dir = self.config.get('evaluation', 'report_dir')
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"evaluation_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("BrainWaveDisorderNet - Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Overall Metrics:\n")
            f.write(f"  Test Loss: {results['loss']:.4f}\n")
            f.write(f"  Test Accuracy: {results['accuracy']:.4f}\n\n")

            if 'roc_auc' in results:
                f.write("ROC-AUC Scores:\n")
                for i in range(len(class_names) if class_names else 5):
                    if i in results['roc_auc']:
                        class_name = class_names[i] if class_names else f"Class {i}"
                        f.write(f"  {class_name}: {results['roc_auc'][i]:.4f}\n")

                if 'macro' in results['roc_auc']:
                    f.write(f"  Macro Average: {results['roc_auc']['macro']:.4f}\n")
                if 'micro' in results['roc_auc']:
                    f.write(f"  Micro Average: {results['roc_auc']['micro']:.4f}\n")
                f.write("\n")

            f.write("Detailed Classification Report:\n")
            f.write(results['classification_report'])

        self.logger.info(f"Evaluation report saved to: {report_path}")

    def compare_models(self, models: List[tf.keras.Model], model_names: List[str],
                       X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            models: List of trained models
            model_names: List of model names
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison results
        """
        comparison_results = []

        for model, name in zip(models, model_names):
            results = self.evaluate(model, X_test, y_test, save_plots=False)
            comparison_results.append({
                'Model': name,
                'Test_Loss': results['loss'],
                'Test_Accuracy': results['accuracy']
            })

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)

        self.logger.info("Model Comparison Results:")
        self.logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df