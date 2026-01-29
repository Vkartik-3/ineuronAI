# ML Training Pipeline

Train LSTM and Random Forest models for predictive maintenance.

## Overview

This module trains two complementary models:

1. **LSTM**: Sequence-based RUL (Remaining Useful Life) prediction
2. **Random Forest**: Health status classification (healthy/warning/critical/imminent_failure)

## Features

- **LSTM with Attention**: Multi-layer LSTM with attention mechanism for temporal patterns
- **Random Forest**: Ensemble classifier for anomaly detection
- **Hyperparameter Tuning**: Grid search and random search for both models
- **Cross-Validation**: Time-series aware CV to prevent data leakage
- **MLflow Tracking**: Comprehensive experiment tracking, model registry
- **Automated Pipeline**: End-to-end training orchestration

## Installation

```bash
cd ml_pipeline/train
pip install -r requirements.txt
```

## Quick Start

### 1. Train LSTM Model

```python
from train_pipeline import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline('config/training_config.yaml')

# Load data from feature store
equipment_ids = ['EQ001', 'EQ002', 'EQ003', 'EQ004', 'EQ005']
df = pipeline.load_data_from_feature_store(equipment_ids)

# Prepare LSTM sequences
lstm_data = pipeline.prepare_lstm_data(df, sequence_length=50)
X, y = lstm_data['X'], lstm_data['y']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train
results = pipeline.train_lstm(X_train, y_train, X_val, y_val, X_test, y_test)

print(f"Test RMSE: {results['test_metrics']['rmse']:.2f}")
print(f"Test MAE: {results['test_metrics']['mae']:.2f}")
print(f"Test R²: {results['test_metrics']['r2']:.4f}")
```

### 2. Train Random Forest

```python
# Prepare RF data (non-sequential)
X_train, y_train, feature_names = pipeline.prepare_rf_data(train_df, target_col='health_status_code')
X_val, y_val, _ = pipeline.prepare_rf_data(val_df, target_col='health_status_code')
X_test, y_test, _ = pipeline.prepare_rf_data(test_df, target_col='health_status_code')

# Train
rf_results = pipeline.train_random_forest(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names
)

print(f"Test Accuracy: {rf_results['test_metrics']['accuracy']:.4f}")
print(f"Test F1-Score: {rf_results['test_metrics']['f1']:.4f}")
```

### 3. Full Pipeline

```python
# Train both models with single call
results = pipeline.run_full_pipeline(
    equipment_ids=['EQ001', 'EQ002', 'EQ003', 'EQ004', 'EQ005'],
    train_lstm=True,
    train_rf=True,
    tune_hyperparams=False
)

lstm_results = results['lstm']
rf_results = results['random_forest']
```

## Configuration

Edit `config/training_config.yaml`:

### LSTM Architecture

```yaml
lstm:
  architecture:
    lstm_layers:
      - units: 128
        return_sequences: true
        dropout: 0.2
      - units: 64
        return_sequences: false
        dropout: 0.2

    attention:
      enabled: true
      units: 64

    dense_layers:
      - units: 64
        activation: relu
        dropout: 0.3
      - units: 32
        activation: relu

  training:
    learning_rate: 0.001
    epochs: 200
    batch_size: 32

    early_stopping:
      enabled: true
      patience: 20
```

### Random Forest

```yaml
random_forest:
  model_params:
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5
    class_weight: balanced
```

## Hyperparameter Tuning

### LSTM Tuning

```python
from tuning.hyperparameter_tuner import HyperparameterTuner

tuner = HyperparameterTuner(config)

# Random search
tuning_results = tuner.tune_lstm(
    X_train, y_train,
    X_val, y_val,
    n_trials=20
)

best_params = tuning_results['best_params']
print(f"Best LSTM params: {best_params}")
print(f"Best val loss: {tuning_results['best_score']:.4f}")
```

### Random Forest Tuning

```python
# Random search for RF
rf_tuning = tuner.tune_random_forest(
    X_train, y_train,
    method='random_search',
    n_trials=50,
    cv_folds=5
)

print(f"Best RF params: {rf_tuning['best_params']}")
print(f"Best CV score: {rf_tuning['best_score']:.4f}")
```

## Cross-Validation

```python
from validation.cross_validator import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(config)

# Time-series CV (respects temporal order)
cv_results = cv.cross_validate(
    model=lstm_model,
    X=X,
    y=y,
    method='time_series',
    n_splits=5
)

print(f"CV MAE: {cv_results['mean_mae']:.2f} (+/- {cv_results['std_mae']:.2f})")

# Equipment-based CV (no data leakage across equipment)
cv_results_eq = cv.cross_validate(
    model=rf_model,
    X=X,
    y=y,
    method='equipment_based',
    equipment_ids=equipment_ids
)
```

## MLflow Tracking

### View Experiments

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Navigate to http://localhost:5000
```

### Access Runs

```python
import mlflow

# Search runs
experiment = mlflow.get_experiment_by_name("predictive_maintenance_rul")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get best run
best_run = runs.loc[runs['metrics.test_rmse'].idxmin()]
print(f"Best RMSE: {best_run['metrics.test_rmse']:.2f}")

# Load model
model_uri = f"runs:/{best_run.run_id}/model"
loaded_model = mlflow.tensorflow.load_model(model_uri)
```

## Model Outputs

### LSTM Model

- **Input**: (batch_size, sequence_length=50, n_features)
- **Output**: RUL prediction (scalar)
- **Architecture**: 128 → 64 → Attention → Dense(64,32,16) → 1
- **Loss**: MSE
- **Metrics**: MAE, RMSE, R², MAPE

### Random Forest

- **Input**: (n_samples, n_features)
- **Output**: Health status class (0-3)
  - 0: Healthy
  - 1: Warning
  - 2: Critical
  - 3: Imminent Failure
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

## Model Performance

Expected performance on test set:

- **LSTM RUL**: RMSE < 15 cycles, MAE < 10 cycles, R² > 0.85
- **Random Forest**: Accuracy > 90%, F1-Score > 0.88

## Saved Artifacts

```
ml_pipeline/train/
├── models/
│   ├── lstm_rul_20260129_153045.keras
│   └── rf_health_20260129_153210.joblib
├── checkpoints/
│   └── lstm_20260129_153045.keras
├── logs/
│   └── training.log
└── mlruns/
    └── [experiment data]
```

## Advanced Usage

### Custom LSTM Architecture

```python
custom_arch = {
    'lstm_layers': [
        {'units': 256, 'return_sequences': True, 'dropout': 0.3},
        {'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'units': 64, 'return_sequences': False, 'dropout': 0.2}
    ],
    'attention': {'enabled': True, 'units': 128},
    'dense_layers': [
        {'units': 128, 'activation': 'relu', 'dropout': 0.3},
        {'units': 64, 'activation': 'relu', 'dropout': 0.2}
    ],
    'output': {'units': 1, 'activation': 'linear'}
}

lstm_model.build_model(input_shape, custom_arch)
```

### Transfer Learning

```python
# Load pretrained model
lstm_model.load_model('models/lstm_rul_pretrained.keras')

# Fine-tune on new equipment
history = lstm_model.train(
    X_train_new, y_train_new,
    X_val_new, y_val_new,
    checkpoint_path='checkpoints/lstm_finetuned.keras'
)
```

## Troubleshooting

### Memory Issues (LSTM)

- Reduce batch_size: 32 → 16
- Reduce sequence_length: 50 → 30
- Use gradient accumulation

### Overfitting

- Increase dropout: 0.2 → 0.3-0.4
- Add L2 regularization
- Reduce model complexity
- Increase training data

### Class Imbalance (RF)

- Use class_weight='balanced'
- SMOTE oversampling
- Adjust decision thresholds

## Next Steps

1. ✅ Training Pipeline complete
2. ⏭️ **Module 6**: Model Evaluation - Advanced metrics, backtesting
3. ⏭️ **Module 7**: Inference API - Real-time predictions

## References

- [LSTM for Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Attention Mechanisms](https://arxiv.org/abs/1409.0473)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
