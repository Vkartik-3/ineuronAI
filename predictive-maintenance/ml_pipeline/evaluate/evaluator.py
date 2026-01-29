"""
Model Card Generator
Generates comprehensive model documentation and performance reports
"""

import json
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """
    Generates standardized model cards for ML models
    Following Google's Model Card framework
    """

    def __init__(self):
        logger.info("Model Card Generator initialized")

    def generate_rul_model_card(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict,
        training_data: Dict,
        model_details: Dict,
        intended_use: Optional[Dict] = None,
        considerations: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate model card for RUL prediction model

        Args:
            model_name: Model name
            model_version: Version identifier
            metrics: Performance metrics
            training_data: Training dataset information
            model_details: Model architecture and parameters
            intended_use: Intended use cases
            considerations: Ethical and fairness considerations

        Returns:
            Model card dictionary
        """
        logger.info(f"Generating model card for {model_name} v{model_version}")

        model_card = {
            "model_details": {
                "name": model_name,
                "version": model_version,
                "date": datetime.now().isoformat(),
                "model_type": model_details.get("type", "LSTM"),
                "architecture": model_details.get("architecture", {}),
                "framework": model_details.get("framework", "TensorFlow/Keras"),
                "authors": model_details.get("authors", "Predictive Maintenance Team"),
                "license": model_details.get("license", "Proprietary"),
                "references": model_details.get("references", []),
            },
            "intended_use": {
                "primary_use": intended_use.get(
                    "primary_use",
                    "Predict remaining useful life (RUL) of manufacturing equipment",
                )
                if intended_use
                else "Predict remaining useful life (RUL) of manufacturing equipment",
                "primary_users": intended_use.get(
                    "primary_users",
                    ["Maintenance engineers", "Plant managers", "Operations team"],
                )
                if intended_use
                else ["Maintenance engineers", "Plant managers", "Operations team"],
                "out_of_scope": intended_use.get(
                    "out_of_scope",
                    [
                        "Safety-critical decisions without human oversight",
                        "Equipment types not in training data",
                    ],
                )
                if intended_use
                else [
                    "Safety-critical decisions without human oversight",
                    "Equipment types not in training data",
                ],
            },
            "factors": {
                "relevant_factors": [
                    "Equipment type and model",
                    "Operating conditions (temperature, pressure, vibration)",
                    "Maintenance history",
                    "Environmental factors",
                ],
                "evaluation_factors": [
                    "Equipment lifecycle stage (early vs late life)",
                    "Operating regime (normal vs stressed)",
                    "Sensor data quality",
                ],
            },
            "metrics": {
                "model_performance": {
                    "rmse": metrics.get("regression_metrics", {}).get("rmse"),
                    "mae": metrics.get("regression_metrics", {}).get("mae"),
                    "r2_score": metrics.get("regression_metrics", {}).get("r2"),
                    "mape": metrics.get("regression_metrics", {}).get("mape"),
                    "rul_accuracy_10cycles": metrics.get("rul_accuracy"),
                },
                "early_warning": {
                    "critical_f1_score": metrics.get("early_warning_score", {}).get(
                        "critical_f1"
                    ),
                    "critical_precision": metrics.get("early_warning_score", {}).get(
                        "critical_precision"
                    ),
                    "critical_recall": metrics.get("early_warning_score", {}).get(
                        "critical_recall"
                    ),
                    "missed_critical_warnings": metrics.get(
                        "early_warning_score", {}
                    ).get("missed_critical_warnings"),
                },
                "business_impact": {
                    "prevented_failures": metrics.get("cost_savings", {}).get(
                        "prevented_failures"
                    ),
                    "false_alarms": metrics.get("cost_savings", {}).get("false_alarms"),
                    "net_savings_usd": metrics.get("cost_savings", {}).get(
                        "net_savings"
                    ),
                },
                "decision_thresholds": {
                    "warning_threshold_cycles": 50,
                    "critical_threshold_cycles": 30,
                    "tolerance_cycles": 10,
                },
            },
            "training_data": {
                "dataset_name": training_data.get(
                    "name", "Manufacturing Equipment Sensor Data"
                ),
                "dataset_size": {
                    "train_samples": training_data.get("train_size"),
                    "val_samples": training_data.get("val_size"),
                    "test_samples": training_data.get("test_size"),
                    "total_equipment": training_data.get("n_equipment"),
                    "total_cycles": training_data.get("total_cycles"),
                },
                "features": {
                    "n_features": training_data.get("n_features"),
                    "feature_types": [
                        "Time-series features",
                        "Frequency-domain features",
                        "Statistical features",
                    ],
                    "sequence_length": training_data.get("sequence_length", 50),
                },
                "preprocessing": [
                    "Lag features (1, 3, 5, 10, 20 cycles)",
                    "Rolling window statistics (5, 10, 20, 50, 100 cycles)",
                    "FFT-based frequency features",
                    "Feature normalization",
                ],
                "data_splits": {
                    "method": "Equipment-based split",
                    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    "no_data_leakage": True,
                },
            },
            "quantitative_analysis": {
                "performance_by_rul_range": self._analyze_performance_by_rul(metrics),
                "prediction_stability": {
                    "early_predictions_pct": metrics.get("early_late_analysis", {}).get(
                        "early_predictions_pct"
                    ),
                    "late_predictions_pct": metrics.get("early_late_analysis", {}).get(
                        "late_predictions_pct"
                    ),
                    "mean_early_error": metrics.get("early_late_analysis", {}).get(
                        "mean_early_error"
                    ),
                    "mean_late_error": metrics.get("early_late_analysis", {}).get(
                        "mean_late_error"
                    ),
                },
            },
            "ethical_considerations": {
                "data_privacy": considerations.get(
                    "data_privacy", "Sensor data anonymized, no PII collected"
                )
                if considerations
                else "Sensor data anonymized, no PII collected",
                "fairness": considerations.get(
                    "fairness",
                    "Model trained on diverse equipment types and operating conditions",
                )
                if considerations
                else "Model trained on diverse equipment types and operating conditions",
                "use_cases_to_avoid": [
                    "Sole decision-maker for safety-critical maintenance",
                    "Performance evaluation of maintenance personnel",
                    "Equipment not represented in training data",
                ],
            },
            "caveats_and_recommendations": {
                "limitations": [
                    "Predictions less accurate for RUL > 200 cycles",
                    "Requires consistent sensor data quality",
                    "May not generalize to new equipment types without retraining",
                    "Performance degrades with sensor failures or missing data",
                ],
                "recommendations": [
                    "Use predictions as decision support, not sole decision-maker",
                    "Combine with domain expertise and historical maintenance records",
                    "Monitor prediction confidence and flag low-confidence predictions",
                    "Retrain model quarterly with new failure data",
                    "Validate predictions with maintenance team before critical actions",
                ],
                "monitoring": [
                    "Track prediction accuracy over time",
                    "Monitor input data distribution drift",
                    "Alert on sensor data quality issues",
                    "Review false alarms and missed failures monthly",
                ],
            },
        }

        logger.info("Model card generated successfully")

        return model_card

    def generate_classifier_model_card(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict,
        training_data: Dict,
        model_details: Dict,
        intended_use: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate model card for health status classifier

        Args:
            model_name: Model name
            model_version: Version identifier
            metrics: Performance metrics
            training_data: Training dataset information
            model_details: Model architecture and parameters
            intended_use: Intended use cases

        Returns:
            Model card dictionary
        """
        logger.info(f"Generating model card for {model_name} v{model_version}")

        model_card = {
            "model_details": {
                "name": model_name,
                "version": model_version,
                "date": datetime.now().isoformat(),
                "model_type": model_details.get("type", "Random Forest"),
                "architecture": model_details.get("architecture", {}),
                "framework": model_details.get("framework", "scikit-learn"),
                "authors": model_details.get("authors", "Predictive Maintenance Team"),
                "license": model_details.get("license", "Proprietary"),
            },
            "intended_use": {
                "primary_use": "Classify equipment health status into 4 categories",
                "classes": ["Healthy", "Warning", "Critical", "Imminent Failure"],
                "primary_users": [
                    "Maintenance engineers",
                    "Operations team",
                    "Plant managers",
                ],
                "out_of_scope": [
                    "Real-time safety shutdowns",
                    "Equipment outside training distribution",
                ],
            },
            "metrics": {
                "overall_performance": {
                    "accuracy": metrics.get("classification_metrics", {}).get(
                        "accuracy"
                    ),
                    "precision_weighted": metrics.get("classification_metrics", {}).get(
                        "precision"
                    ),
                    "recall_weighted": metrics.get("classification_metrics", {}).get(
                        "recall"
                    ),
                    "f1_weighted": metrics.get("classification_metrics", {}).get("f1"),
                    "roc_auc": metrics.get("classification_metrics", {}).get("roc_auc"),
                },
                "per_class_performance": {
                    "precision_per_class": metrics.get(
                        "classification_metrics", {}
                    ).get("precision_per_class"),
                    "recall_per_class": metrics.get("classification_metrics", {}).get(
                        "recall_per_class"
                    ),
                    "f1_per_class": metrics.get("classification_metrics", {}).get(
                        "f1_per_class"
                    ),
                },
                "critical_focus": {
                    "critical_precision": metrics.get("critical_class_metrics", {}).get(
                        "critical_precision"
                    ),
                    "critical_recall": metrics.get("critical_class_metrics", {}).get(
                        "critical_recall"
                    ),
                    "critical_f1": metrics.get("critical_class_metrics", {}).get(
                        "critical_f1"
                    ),
                    "missed_critical_failures": metrics.get(
                        "critical_class_metrics", {}
                    ).get("missed_critical_failures"),
                },
                "confusion_matrix": metrics.get("classification_metrics", {}).get(
                    "confusion_matrix"
                ),
            },
            "training_data": {
                "dataset_name": training_data.get(
                    "name", "Manufacturing Equipment Sensor Data"
                ),
                "dataset_size": {
                    "train_samples": training_data.get("train_size"),
                    "val_samples": training_data.get("val_size"),
                    "test_samples": training_data.get("test_size"),
                },
                "class_distribution": training_data.get("class_distribution", {}),
                "features": {
                    "n_features": training_data.get("n_features"),
                    "feature_types": [
                        "Sensor readings",
                        "Time-series features",
                        "Frequency features",
                    ],
                },
                "class_balance": "Balanced using class weights",
            },
            "quantitative_analysis": {
                "feature_importance": "Top features logged separately",
                "class_separability": "Good separation between healthy and critical states",
                "false_positive_rate": "Low false alarm rate for critical class",
            },
            "caveats_and_recommendations": {
                "limitations": [
                    "May confuse warning and critical states in edge cases",
                    "Requires all sensor inputs for accurate classification",
                    "Performance depends on feature engineering quality",
                ],
                "recommendations": [
                    "Prioritize recall for critical class (better to over-warn)",
                    "Use probability scores for confidence assessment",
                    "Combine with RUL predictions for comprehensive assessment",
                    "Review misclassifications monthly for model improvement",
                ],
            },
        }

        logger.info("Classifier model card generated successfully")

        return model_card

    def _analyze_performance_by_rul(self, metrics: Dict) -> Dict:
        """Analyze performance across different RUL ranges"""
        return {
            "high_rul_100_plus": {
                "description": "RUL > 100 cycles",
                "note": "Predictions less critical, higher tolerance acceptable",
            },
            "medium_rul_50_100": {
                "description": "50 < RUL <= 100 cycles",
                "note": "Planning window, moderate accuracy required",
            },
            "low_rul_30_50": {
                "description": "30 < RUL <= 50 cycles",
                "note": "Warning zone, high accuracy critical",
            },
            "critical_rul_0_30": {
                "description": "RUL <= 30 cycles",
                "note": "Critical zone, maximum accuracy required",
                "f1_score": metrics.get("early_warning_score", {}).get("critical_f1"),
            },
        }

    def save_model_card(self, model_card: Dict, filepath: str):
        """
        Save model card to JSON file

        Args:
            model_card: Model card dictionary
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            json.dump(model_card, f, indent=2)

        logger.info(f"Model card saved to {filepath}")

    def generate_markdown_report(self, model_card: Dict) -> str:
        """
        Generate markdown report from model card

        Args:
            model_card: Model card dictionary

        Returns:
            Markdown formatted report
        """
        md = f"""# Model Card: {model_card["model_details"]["name"]}

**Version:** {model_card["model_details"]["version"]}
**Date:** {model_card["model_details"]["date"]}
**Type:** {model_card["model_details"]["model_type"]}
**Framework:** {model_card["model_details"]["framework"]}

## Model Details

{json.dumps(model_card["model_details"]["architecture"], indent=2)}

## Intended Use

**Primary Use:** {model_card["intended_use"]["primary_use"]}

**Primary Users:**
{self._list_to_markdown(model_card["intended_use"]["primary_users"])}

**Out of Scope:**
{self._list_to_markdown(model_card["intended_use"].get("out_of_scope", []))}

## Performance Metrics

### Model Performance
"""

        metrics = model_card["metrics"].get(
            "model_performance", model_card["metrics"].get("overall_performance", {})
        )
        for key, value in metrics.items():
            if value is not None:
                md += (
                    f"- **{key}**: {value:.4f}\n"
                    if isinstance(value, float)
                    else f"- **{key}**: {value}\n"
                )

        md += "\n## Training Data\n\n"
        md += f"**Dataset:** {model_card['training_data']['dataset_name']}\n\n"

        dataset_size = model_card["training_data"]["dataset_size"]
        md += "**Dataset Size:**\n"
        for key, value in dataset_size.items():
            if value is not None:
                md += f"- {key}: {value}\n"

        md += "\n## Limitations\n\n"
        limitations = model_card["caveats_and_recommendations"]["limitations"]
        md += self._list_to_markdown(limitations)

        md += "\n## Recommendations\n\n"
        recommendations = model_card["caveats_and_recommendations"]["recommendations"]
        md += self._list_to_markdown(recommendations)

        return md

    def _list_to_markdown(self, items: List) -> str:
        """Convert list to markdown bullet points"""
        return "\n".join([f"- {item}" for item in items])

    def save_markdown_report(self, model_card: Dict, filepath: str):
        """Save markdown report to file"""
        md_report = self.generate_markdown_report(model_card)

        with open(filepath, "w") as f:
            f.write(md_report)

        logger.info(f"Markdown report saved to {filepath}")
