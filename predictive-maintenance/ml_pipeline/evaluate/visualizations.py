"""
Visualization Tools for Model Evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class EvaluationVisualizer:
    """
    Visualization tools for model evaluation
    """

    def __init__(self):
        logger.info("Evaluation Visualizer initialized")

    def plot_rul_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equipment_ids: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot RUL predictions vs actual

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            equipment_ids: Optional equipment identifiers
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0, 0].plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        axes[0, 0].set_xlabel("True RUL (cycles)", fontsize=12)
        axes[0, 0].set_ylabel("Predicted RUL (cycles)", fontsize=12)
        axes[0, 0].set_title(
            "RUL Predictions vs Actual", fontsize=14, fontweight="bold"
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_true, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("True RUL (cycles)", fontsize=12)
        axes[0, 1].set_ylabel("Residuals (cycles)", fontsize=12)
        axes[0, 1].set_title("Residual Plot", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Prediction Error (cycles)", fontsize=12)
        axes[1, 0].set_ylabel("Frequency", fontsize=12)
        axes[1, 0].set_title("Error Distribution", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # Absolute error by RUL range
        abs_errors = np.abs(residuals)
        rul_ranges = ["0-30", "30-50", "50-100", "100+"]
        range_errors = []

        for low, high in [(0, 30), (30, 50), (50, 100), (100, 1000)]:
            mask = (y_true >= low) & (y_true < high)
            if np.any(mask):
                range_errors.append(np.mean(abs_errors[mask]))
            else:
                range_errors.append(0)

        axes[1, 1].bar(rul_ranges, range_errors, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("RUL Range (cycles)", fontsize=12)
        axes[1, 1].set_ylabel("Mean Absolute Error", fontsize=12)
        axes[1, 1].set_title("MAE by RUL Range", fontsize=14, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"RUL prediction plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(
        self, cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            class_names: Class names
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )

        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

        # Add accuracy on diagonal
        for i in range(len(class_names)):
            total = cm[i, :].sum()
            if total > 0:
                acc = cm[i, i] / total * 100
                plt.text(
                    i + 0.5,
                    i - 0.3,
                    f"{acc:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Plot ROC curves for multi-class classification

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Class names
            save_path: Optional path to save figure
        """
        n_classes = len(class_names)

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)

            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Plot Precision-Recall curves

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Class names
            save_path: Optional path to save figure
        """
        n_classes = len(class_names)

        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)

            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])

            plt.plot(recall, precision, lw=2, label=class_names[i])

        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-Recall curves saved to {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Plot feature importance

        Args:
            feature_names: Feature names
            importances: Importance scores
            top_n: Number of top features to show
            save_path: Optional path to save figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(12, 8))

        plt.barh(range(top_n), top_importances, edgecolor="black", alpha=0.7)
        plt.yticks(range(top_n), top_features)
        plt.xlabel("Importance Score", fontsize=12)
        plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_time_series_predictions(
        self,
        cycles: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equipment_id: str,
        save_path: Optional[str] = None,
    ):
        """
        Plot predictions over time for single equipment

        Args:
            cycles: Cycle numbers
            y_true: True RUL values
            y_pred: Predicted RUL values
            equipment_id: Equipment identifier
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(14, 6))

        plt.plot(cycles, y_true, "b-", label="True RUL", linewidth=2)
        plt.plot(cycles, y_pred, "r--", label="Predicted RUL", linewidth=2, alpha=0.7)

        # Highlight critical zone
        plt.axhline(
            y=30, color="orange", linestyle=":", label="Critical Threshold", linewidth=2
        )
        plt.axhline(
            y=50, color="yellow", linestyle=":", label="Warning Threshold", linewidth=2
        )

        plt.xlabel("Cycle Number", fontsize=12)
        plt.ylabel("RUL (cycles)", fontsize=12)
        plt.title(
            f"RUL Prediction Over Time - {equipment_id}", fontsize=14, fontweight="bold"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Time series plot saved to {save_path}")

        plt.show()

    def plot_comprehensive_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Create comprehensive evaluation dashboard

        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Metrics dictionary
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main scatter plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
        ax1.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax1.set_xlabel("True RUL (cycles)", fontsize=12)
        ax1.set_ylabel("Predicted RUL (cycles)", fontsize=12)
        ax1.set_title("RUL Predictions", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Metrics summary
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        metrics_text = "Performance Metrics\n" + "=" * 30 + "\n\n"
        metrics_text += (
            f"RMSE: {metrics.get('regression_metrics', {}).get('rmse', 0):.2f}\n"
        )
        metrics_text += (
            f"MAE: {metrics.get('regression_metrics', {}).get('mae', 0):.2f}\n"
        )
        metrics_text += (
            f"R²: {metrics.get('regression_metrics', {}).get('r2', 0):.4f}\n"
        )
        metrics_text += (
            f"MAPE: {metrics.get('regression_metrics', {}).get('mape', 0):.2f}%\n\n"
        )
        metrics_text += "Early Warning\n" + "-" * 30 + "\n"
        metrics_text += f"Critical F1: {metrics.get('early_warning_score', {}).get('critical_f1', 0):.4f}\n"
        metrics_text += f"Missed: {metrics.get('early_warning_score', {}).get('missed_critical_warnings', 0)}\n"
        ax2.text(
            0.1,
            0.9,
            metrics_text,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Residual plot
        ax3 = fig.add_subplot(gs[1, 2])
        residuals = y_pred - y_true
        ax3.scatter(y_true, residuals, alpha=0.5, s=20)
        ax3.axhline(y=0, color="r", linestyle="--", lw=2)
        ax3.set_xlabel("True RUL", fontsize=10)
        ax3.set_ylabel("Residuals", fontsize=10)
        ax3.set_title("Residuals", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Error histogram
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
        ax4.axvline(x=0, color="r", linestyle="--", lw=2)
        ax4.set_xlabel("Error (cycles)", fontsize=10)
        ax4.set_ylabel("Frequency", fontsize=10)
        ax4.set_title("Error Distribution", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # MAE by RUL range
        ax5 = fig.add_subplot(gs[2, 1])
        abs_errors = np.abs(residuals)
        rul_ranges = ["0-30", "30-50", "50-100", "100+"]
        range_errors = []
        for low, high in [(0, 30), (30, 50), (50, 100), (100, 1000)]:
            mask = (y_true >= low) & (y_true < high)
            range_errors.append(np.mean(abs_errors[mask]) if np.any(mask) else 0)
        ax5.bar(rul_ranges, range_errors, edgecolor="black", alpha=0.7)
        ax5.set_xlabel("RUL Range", fontsize=10)
        ax5.set_ylabel("MAE", fontsize=10)
        ax5.set_title("MAE by RUL Range", fontsize=12, fontweight="bold")
        ax5.grid(True, alpha=0.3, axis="y")

        # Cost analysis
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis("off")
        cost_text = "Business Impact\n" + "=" * 30 + "\n\n"
        cost_text += f"Prevented Failures: {metrics.get('cost_savings', {}).get('prevented_failures', 0)}\n"
        cost_text += (
            f"False Alarms: {metrics.get('cost_savings', {}).get('false_alarms', 0)}\n"
        )
        cost_text += f"Net Savings: ${metrics.get('cost_savings', {}).get('net_savings', 0):,.0f}\n"
        ax6.text(
            0.1,
            0.9,
            cost_text,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comprehensive evaluation saved to {save_path}")

        plt.show()
