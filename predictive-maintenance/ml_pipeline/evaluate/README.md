# Model Evaluation

Comprehensive evaluation metrics for predictive maintenance models.

## Overview

This module provides:

- **Regression Metrics**: RMSE, MAE, R², MAPE for RUL prediction
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC for health status
- **Custom PM Metrics**: Early warning detection, cost savings analysis, lead time metrics

## Installation

```bash
pip install scikit-learn numpy matplotlib seaborn
```

## Quick Start

### Evaluate RUL Model

```python
from evaluate.metrics import ModelEvaluator
import numpy as np

# Initialize evaluator
evaluator = ModelEvaluator()

# Predictions
y_true = np.array([100, 80, 60, 40, 20, 10, 5])
y_pred = np.array([95, 75, 58, 42, 22, 12, 8])

# Evaluate
results = evaluator.evaluate_rul_model(
    y_true, y_pred,
    tolerance=10,
    warning_threshold=50,
    critical_threshold=30
)

# Results
print(f"RMSE: {results['regression_metrics']['rmse']:.2f}")
print(f"MAE: {results['regression_metrics']['mae']:.2f}")
print(f"R²: {results['regression_metrics']['r2']:.4f}")
print(f"Accuracy (±10 cycles): {results['rul_accuracy']:.1f}%")
print(f"Critical F1-Score: {results['early_warning_score']['critical_f1']:.4f}")
print(f"Net Savings: ${results['cost_savings']['net_savings']:,.2f}")
```

### Evaluate Classification Model

```python
# Health status predictions
y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 0=healthy, 1=warning, 2=critical, 3=imminent
y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
y_proba = np.random.rand(8, 4)  # Probabilities for 4 classes

# Evaluate
results = evaluator.evaluate_health_classifier(
    y_true, y_pred, y_proba,
    class_names=['healthy', 'warning', 'critical', 'imminent_failure']
)

print(f"Accuracy: {results['classification_metrics']['accuracy']:.4f}")
print(f"F1-Score: {results['classification_metrics']['f1']:.4f}")
print(f"Critical Precision: {results['critical_class_metrics']['critical_precision']:.4f}")
print(f"Critical Recall: {results['critical_class_metrics']['critical_recall']:.4f}")
```

## Metrics Reference

### Regression Metrics (RUL)

| Metric           | Description                     | Good Value  |
| ---------------- | ------------------------------- | ----------- |
| **RMSE**         | Root Mean Squared Error         | < 15 cycles |
| **MAE**          | Mean Absolute Error             | < 10 cycles |
| **R²**           | Coefficient of Determination    | > 0.85      |
| **MAPE**         | Mean Absolute Percentage Error  | < 15%       |
| **RUL Accuracy** | % within tolerance (±10 cycles) | > 85%       |

### Classification Metrics (Health Status)

| Metric        | Description                       | Good Value |
| ------------- | --------------------------------- | ---------- |
| **Accuracy**  | Overall classification accuracy   | > 90%      |
| **Precision** | True positives / All positives    | > 0.88     |
| **Recall**    | True positives / Actual positives | > 0.85     |
| **F1-Score**  | Harmonic mean of precision/recall | > 0.86     |
| **ROC-AUC**   | Area under ROC curve              | > 0.92     |

### Custom PM Metrics

**Early Warning Detection**:

- Critical Precision: % of critical warnings that are correct
- Critical Recall: % of actual critical states detected
- Missed Critical Warnings: Count of undetected critical failures

**Cost Savings**:

- Prevented Failures: Failures avoided through early intervention
- False Alarms: Unnecessary maintenance actions
- Net Savings: Total cost benefit after accounting for false alarms

**Lead Time**:

- Timely Predictions: Predictions with sufficient lead time (≥10 cycles)
- Late Predictions: Warnings too close to failure

## Usage Examples

### Custom Thresholds

```python
from evaluate.metrics import CustomPredictiveMaintenanceMetrics

custom_metrics = CustomPredictiveMaintenanceMetrics()

# Early warning with custom thresholds
warning_metrics = custom_metrics.calculate_early_warning_score(
    y_true, y_pred,
    warning_threshold=60,    # Warn at 60 cycles
    critical_threshold=20    # Critical at 20 cycles
)

print(f"Warning F1: {warning_metrics['warning_f1']:.4f}")
print(f"Critical F1: {warning_metrics['critical_f1']:.4f}")
```

### Cost Analysis

```python
# Custom cost structure
cost_metrics = custom_metrics.calculate_maintenance_cost_savings(
    y_true, y_pred,
    cost_per_failure=15000,      # $15K per unplanned failure
    cost_per_maintenance=2000,   # $2K per planned maintenance
    rul_threshold=25             # Maintain at 25 cycles RUL
)

print(f"Prevented failures: {cost_metrics['prevented_failures']}")
print(f"False alarms: {cost_metrics['false_alarms']}")
print(f"Net savings: ${cost_metrics['net_savings']:,.2f}")
```

### Per-Class Metrics

```python
from evaluate.metrics import ClassificationMetrics

clf_metrics = ClassificationMetrics()

# Full classification report
results = clf_metrics.calculate_all_metrics(
    y_true, y_pred, y_proba,
    class_names=['healthy', 'warning', 'critical', 'imminent_failure']
)

# Per-class precision
for i, name in enumerate(['healthy', 'warning', 'critical', 'imminent']):
    precision = results['precision_per_class'][i]
    recall = results['recall_per_class'][i]
    f1 = results['f1_per_class'][i]
    print(f"{name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
```

## Integration with Training Pipeline

```python
from train.train_pipeline import TrainingPipeline
from evaluate.metrics import ModelEvaluator

# Train model
pipeline = TrainingPipeline('train/config/training_config.yaml')
results = pipeline.run_full_pipeline(
    equipment_ids=['EQ001', 'EQ002', 'EQ003'],
    train_lstm=True
)

# Evaluate
evaluator = ModelEvaluator()
lstm_model = results['lstm']['model']

# Get test predictions
y_pred = lstm_model.predict(X_test)

# Comprehensive evaluation
eval_results = evaluator.evaluate_rul_model(
    y_test, y_pred,
    tolerance=10,
    warning_threshold=50,
    critical_threshold=30
)

# Log to MLflow
from train.tracking.mlflow_tracker import MLflowTracker
mlflow_tracker = MLflowTracker(config)
mlflow_tracker.start_run(run_name="evaluation")
mlflow_tracker.log_metrics(eval_results['regression_metrics'])
mlflow_tracker.log_metrics(eval_results['early_warning_score'])
mlflow_tracker.log_metrics(eval_results['cost_savings'])
mlflow_tracker.end_run()
```

## Visualization

### RUL Predictions Plot

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', label='Perfect Prediction')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title('RUL Prediction Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Confusion Matrix

```python
import seaborn as sns

cm = results['classification_metrics']['confusion_matrix']
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Warning', 'Critical', 'Imminent'],
            yticklabels=['Healthy', 'Warning', 'Critical', 'Imminent'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Health Status Confusion Matrix')
plt.show()
```

### Error Distribution

```python
errors = y_pred - y_true

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, edgecolor='black')
plt.xlabel('Prediction Error (cycles)')
plt.ylabel('Frequency')
plt.title('Error Distribution')

plt.subplot(1, 2, 2)
plt.scatter(y_true, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Prediction Error (cycles)')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()
```

## API Reference

### ModelEvaluator

Main evaluation interface.

**Methods**:

- `evaluate_rul_model(y_true, y_pred, ...)`: Complete RUL model evaluation
- `evaluate_health_classifier(y_true, y_pred, y_proba, ...)`: Complete classifier evaluation

### RegressionMetrics

Regression-specific metrics.

**Methods**:

- `calculate_all_metrics(y_true, y_pred)`: All regression metrics
- `calculate_rul_accuracy(y_true, y_pred, tolerance)`: Accuracy within tolerance
- `calculate_early_late_predictions(y_true, y_pred)`: Early vs late prediction analysis

### ClassificationMetrics

Classification-specific metrics.

**Methods**:

- `calculate_all_metrics(y_true, y_pred, y_proba, class_names)`: All classification metrics
- `calculate_critical_class_metrics(y_true, y_pred)`: Focus on critical classes

### CustomPredictiveMaintenanceMetrics

Domain-specific metrics.

**Methods**:

- `calculate_early_warning_score(y_true, y_pred, ...)`: Early warning detection
- `calculate_maintenance_cost_savings(y_true, y_pred, ...)`: Cost analysis
- `calculate_lead_time_metrics(y_true, y_pred, ...)`: Lead time analysis

## Best Practices

1. **Always evaluate on held-out test set** - Never use training or validation data
2. **Use multiple metrics** - Single metric doesn't tell full story
3. **Focus on critical failures** - Prioritize recall for critical class
4. **Consider business metrics** - Cost savings, lead time matter to stakeholders
5. **Visualize results** - Plots reveal patterns that metrics miss
6. **Compare to baseline** - Show improvement over naive approaches

## Troubleshooting

**High RMSE on long RUL values**:

- Use log-transformed RUL
- Clip predictions to reasonable range
- Focus metrics on critical region (RUL < 100)

**Low recall for critical class**:

- Adjust decision threshold
- Use class weights during training
- Increase critical class samples (SMOTE)

**Many false alarms**:

- Increase maintenance threshold
- Use confidence intervals
- Require consecutive warnings

## Model Cards

### Generate Model Card

```python
from evaluate.evaluator import ModelCardGenerator

# Initialize generator
card_gen = ModelCardGenerator()

# Generate RUL model card
model_card = card_gen.generate_rul_model_card(
    model_name="LSTM RUL Predictor",
    model_version="v1.0.0",
    metrics=eval_results,  # From ModelEvaluator
    training_data={
        "name": "Manufacturing Sensor Data",
        "train_size": 5000,
        "val_size": 1000,
        "test_size": 1000,
        "n_equipment": 20,
        "n_features": 150,
        "sequence_length": 50
    },
    model_details={
        "type": "LSTM",
        "architecture": {
            "lstm_layers": [128, 64],
            "attention": True,
            "dense_layers": [64, 32, 16]
        },
        "framework": "TensorFlow 2.15"
    }
)

# Save as JSON
card_gen.save_model_card(model_card, "models/lstm_rul_model_card.json")

# Save as Markdown
card_gen.save_markdown_report(model_card, "models/lstm_rul_model_card.md")
```

### Model Card Contents

Model cards include:

- **Model Details**: Architecture, version, authors, license
- **Intended Use**: Primary use cases, target users, out-of-scope uses
- **Performance Metrics**: RMSE, MAE, R², business impact
- **Training Data**: Dataset size, features, preprocessing, splits
- **Ethical Considerations**: Privacy, fairness, limitations
- **Recommendations**: Best practices, monitoring, retraining schedule

## Visualizations

### Comprehensive Dashboard

```python
from evaluate.visualizations import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Create comprehensive evaluation dashboard
visualizer.plot_comprehensive_evaluation(
    y_true, y_pred,
    metrics=eval_results,
    save_path='reports/evaluation_dashboard.png'
)
```

### Individual Plots

```python
# RUL predictions
visualizer.plot_rul_predictions(y_true, y_pred, save_path='reports/rul_predictions.png')

# Confusion matrix
visualizer.plot_confusion_matrix(cm, ['Healthy', 'Warning', 'Critical', 'Imminent'],
                                 save_path='reports/confusion_matrix.png')

# ROC curves
visualizer.plot_roc_curves(y_true, y_proba, class_names, save_path='reports/roc_curves.png')

# Feature importance
visualizer.plot_feature_importance(feature_names, importances, top_n=20,
                                   save_path='reports/feature_importance.png')

# Time series predictions
visualizer.plot_time_series_predictions(cycles, y_true, y_pred, 'EQ001',
                                       save_path='reports/time_series_EQ001.png')
```

## Backtesting

### Walk-Forward Validation

```python
from evaluate.backtesting import BacktestFramework

backtest = BacktestFramework()

# Walk-forward validation
wf_results = backtest.walk_forward_validation(
    model=lstm_model,
    X=X,
    y=y,
    equipment_ids=eq_ids,
    cycle_numbers=cycles,
    train_size=200,
    test_size=50,
    step=25
)

print(f"Mean MAE: {wf_results['mean_mae']:.2f} (+/- {wf_results['std_mae']:.2f})")
```

### Expanding Window Validation

```python
# Expanding window (growing training set)
ew_results = backtest.expanding_window_validation(
    model=lstm_model,
    X=X,
    y=y,
    initial_train_size=200,
    test_size=50,
    step=25
)
```

### Leave-One-Equipment-Out

```python
# Test generalization to new equipment
eq_results = backtest.equipment_hold_out_validation(
    model=lstm_model,
    X=X,
    y=y,
    equipment_ids=eq_ids
)

print(f"Equipment generalization MAE: {eq_results['mean_mae']:.2f}")
```

### Temporal Performance Analysis

```python
# Analyze model degradation over time
temporal_results = backtest.temporal_degradation_analysis(
    y_true, y_pred, timestamps
)

if temporal_results['requires_retraining']:
    print(f"⚠️ Model degradation detected: {temporal_results['degradation_pct']:.1f}%")
    print("Recommendation: Retrain model with recent data")
```

### Generate Backtest Report

```python
# Comprehensive backtest report
report = backtest.generate_backtest_report(
    wf_results,
    ew_results,
    eq_results
)

print(report)

# Save report
with open('reports/backtest_report.txt', 'w') as f:
    f.write(report)
```

## Complete Evaluation Workflow

```python
from train.train_pipeline import TrainingPipeline
from evaluate.metrics import ModelEvaluator
from evaluate.evaluator import ModelCardGenerator
from evaluate.visualizations import EvaluationVisualizer
from evaluate.backtesting import BacktestFramework

# 1. Train model
pipeline = TrainingPipeline('train/config/training_config.yaml')
results = pipeline.run_full_pipeline(equipment_ids, train_lstm=True)
lstm_model = results['lstm']['model']

# 2. Evaluate
evaluator = ModelEvaluator()
eval_results = evaluator.evaluate_rul_model(
    y_test, y_pred,
    tolerance=10,
    warning_threshold=50,
    critical_threshold=30
)

# 3. Generate model card
card_gen = ModelCardGenerator()
model_card = card_gen.generate_rul_model_card(
    "LSTM RUL Predictor", "v1.0.0",
    eval_results, training_data, model_details
)
card_gen.save_model_card(model_card, "models/model_card.json")
card_gen.save_markdown_report(model_card, "models/model_card.md")

# 4. Create visualizations
visualizer = EvaluationVisualizer()
visualizer.plot_comprehensive_evaluation(
    y_test, y_pred, eval_results,
    save_path='reports/evaluation_dashboard.png'
)

# 5. Backtest
backtest = BacktestFramework()
wf_results = backtest.walk_forward_validation(lstm_model, X, y, eq_ids, cycles)
report = backtest.generate_backtest_report(wf_results, ew_results, eq_results)

print("✅ Evaluation complete!")
print(f"📊 Model card: models/model_card.md")
print(f"📈 Visualizations: reports/evaluation_dashboard.png")
print(f"📋 Backtest report: reports/backtest_report.txt")
```

## Next Steps

Module 6 (Model Evaluation) complete! ✅

**Phase 3: Inference & Alerting** starts next:

- **Module 7**: Inference API (inference_service/) - REST API for real-time predictions
- **Module 8**: Alerting system for critical warnings
- **Module 9**: Dashboard and visualization

## References

- [Google Model Cards](https://modelcards.withgoogle.com/about)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Prognostics Performance Metrics](https://www.phmsociety.org/)
- [Time Series Backtesting](https://otexts.com/fpp3/accuracy.html)
