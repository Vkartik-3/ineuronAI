# Model Retraining Module

Automated model retraining pipeline with drift detection and intelligent deployment.

## 📋 Overview

This module handles automated model retraining for the predictive maintenance system, triggered by:

- **Data drift detection** (distribution changes in input features)
- **Concept drift detection** (changes in feature-target relationships)
- **Scheduled retraining** (weekly/monthly)
- **Manual triggers** (on-demand retraining)

## 🏗️ Architecture

```
[Scheduled Check / Manual Trigger]
          ↓
[Drift Detector]
├─ Data Drift (KS Test)
└─ Concept Drift (Error Increase)
          ↓
[Retraining Pipeline]
├─ Load training data (90 days)
├─ Train new model
└─ Log to MLflow
          ↓
[Model Comparator]
├─ Evaluate on test set
├─ Compare metrics (MAE, RMSE, R2)
└─ Statistical significance test
          ↓
[Deployment Manager]
├─ Promote to production (if approved)
├─ Archive previous version
└─ Rollback capability
```

## 📦 Components

### 1. Drift Detector (`drift_detector.py`)

Monitors for data and concept drift to trigger retraining.

**Features:**

- **Data Drift**: Kolmogorov-Smirnov test on feature distributions
- **Concept Drift**: Performance degradation on recent data
- Configurable thresholds and window sizes
- MLflow logging of drift metrics

**Usage:**

```python
from ml_pipeline.retrain import DriftDetector

detector = DriftDetector(config_path="config/retrain_config.yaml")

# Compare reference (training) vs current data
drift_report = detector.detect_drift(
    reference_data=historical_df,
    current_data=recent_df,
    target_col='rul'
)

print(f"Drift detected: {drift_report.drift_detected}")
print(f"Drift type: {drift_report.drift_type}")  # 'data', 'concept', 'both', 'none'
print(f"Drift score: {drift_report.drift_score:.3f}")
print(f"Affected features: {drift_report.affected_features}")
print(f"Recommendation: {drift_report.recommendation}")

# Log to MLflow
detector.log_drift_report(drift_report)
```

### 2. Model Comparator (`model_comparator.py`)

Compares candidate model against production model.

**Features:**

- Comprehensive metrics (MAE, RMSE, R2, MAPE)
- Statistical significance testing (paired t-test)
- Configurable improvement thresholds
- Promotion recommendations

**Usage:**

```python
from ml_pipeline.retrain import ModelComparator

comparator = ModelComparator(config_path="config/retrain_config.yaml")

# Compare models on test data
comparison = comparator.compare_models(
    champion_model_uri="models:/predictive_maintenance_model/Production",
    challenger_model_uri="runs:/abc123/model",
    test_data=test_df,
    target_col='rul'
)

print(f"Winner: {comparison.winner}")
print(f"Improvement: {comparison.improvement_pct:.1f}%")
print(f"Should promote: {comparison.should_promote}")
print(f"Recommendation: {comparison.recommendation}")

# Metrics comparison
print("\nChampion metrics:", comparison.metrics_comparison['champion'])
print("Challenger metrics:", comparison.metrics_comparison['challenger'])

# Log to MLflow
comparator.log_comparison_report(comparison)
```

### 3. Deployment Manager (`deployment_manager.py`)

Handles model promotion to production with rollback capabilities.

**Features:**

- Safe deployment with version archival
- Rollback to previous version
- MLflow Model Registry integration
- Deployment logging and tracking

**Usage:**

```python
from ml_pipeline.retrain import DeploymentManager

manager = DeploymentManager(config_path="config/retrain_config.yaml")

# Promote model to production
deployment = manager.promote_to_production(
    model_uri="runs:/abc123/model",
    archive_existing=True
)

print(f"Deployment status: {deployment.deployment_status}")
print(f"New version: {deployment.model_version}")
print(f"Previous version: {deployment.previous_version}")

# Rollback if needed
if issues_detected:
    rollback = manager.rollback_to_previous()
    print(f"Rolled back to version: {rollback.model_version}")

# Check current production model
info = manager.get_production_model_info()
print(f"Production model: {info['model_name']} v{info['version']}")
```

### 4. Retraining Pipeline (`retrain_pipeline.py`)

Orchestrates the complete automated retraining workflow.

**Features:**

- Scheduled drift checks
- Automated retraining triggers
- End-to-end workflow orchestration
- MLflow experiment tracking

**Usage:**

```python
from ml_pipeline.retrain import RetrainingPipeline

pipeline = RetrainingPipeline(config_path="config/retrain_config.yaml")

# Scheduled check (run as cron job)
result = pipeline.run_scheduled_check()

if result['drift_detected']:
    print("Drift detected, retraining triggered")
    print(f"New model URI: {result['retrain_result']['new_model_uri']}")
    print(f"Deployed: {result['retrain_result']['deployed']}")

# Manual retraining trigger
result = pipeline.trigger_retraining(reason="manual")
print(f"Retraining status: {result['status']}")
print(f"Comparison: {result['comparison_report']}")
```

## ⚙️ Configuration

Edit `config/retrain_config.yaml`:

```yaml
pipeline:
  auto_deploy: false # Automatically deploy if model improves
  require_approval: true # Require manual approval
  schedule_cron: "0 2 * * 0" # Weekly on Sunday at 2 AM

drift_detection:
  data_drift_threshold: 0.05 # p-value threshold for KS test
  concept_drift_threshold: 0.15 # Error increase threshold (15%)
  window_size_days: 7 # Recent data window
  min_samples: 1000 # Minimum samples for drift detection

model_comparison:
  primary_metric: "mae" # Primary metric for comparison
  min_improvement_pct: 5.0 # Minimum 5% improvement to promote
  test_size: 0.2 # 20% test set

deployment:
  model_name: "predictive_maintenance_model"
  strategy: "direct" # direct, shadow, canary
  enable_rollback: true # Enable automatic rollback
```

## 🚀 Deployment Strategies

### Direct Deployment (default)

Immediate replacement of production model.

### Shadow Deployment

Run new model in parallel, collect metrics without serving predictions.

### Canary Deployment

Gradual rollout to percentage of traffic (10% → 50% → 100%).

## 📊 Monitoring

### Drift Metrics

- **Data drift score**: Max KS statistic across features
- **Concept drift score**: Relative MAE increase
- **Affected features**: List of drifted features
- **Drift history**: Time series of drift scores

### Model Performance

- **MAE, RMSE, R2, MAPE**: Regression metrics
- **Improvement percentage**: Relative to champion
- **Statistical significance**: p-value from paired t-test

### Deployment Tracking

- **Version history**: All production versions
- **Deployment timestamps**: When models were promoted
- **Rollback events**: When and why rollbacks occurred

## 🔄 Automation with Airflow

Create `dags/retrain_pipeline_dag.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from ml_pipeline.retrain import RetrainingPipeline

def run_drift_check():
    pipeline = RetrainingPipeline()
    result = pipeline.run_scheduled_check()
    return result

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'predictive_maintenance_retrain',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval='0 2 * * 0',  # Weekly on Sunday at 2 AM
    catchup=False,
)

drift_check_task = PythonOperator(
    task_id='drift_check_and_retrain',
    python_callable=run_drift_check,
    dag=dag,
)
```

## 📈 MLflow Experiments

The retraining pipeline creates several MLflow experiments:

1. **drift_monitoring**: Drift detection results
2. **model_comparison**: Model comparison results
3. **model_deployment**: Deployment events
4. **retraining_pipeline**: Complete workflow logs
5. **automated_retraining**: Retraining runs

## 🧪 Testing

```bash
# Test drift detector
cd ml_pipeline/retrain
python drift_detector.py

# Test model comparator
python model_comparator.py

# Test deployment manager
python deployment_manager.py

# Test complete pipeline
python retrain_pipeline.py
```

## 📋 Best Practices

1. **Monitor Drift Continuously**

   - Set up daily/weekly drift checks
   - Alert on high drift scores
   - Review drift reports regularly

2. **Conservative Thresholds**

   - Require significant improvement (5-10%)
   - Use statistical significance tests
   - Manual approval for production deployment

3. **Test Thoroughly**

   - Use holdout test set for comparison
   - Check multiple metrics, not just one
   - Validate on diverse equipment types

4. **Safe Deployment**

   - Always archive previous model
   - Enable rollback capability
   - Monitor post-deployment performance

5. **Version Control**
   - Tag models with metadata
   - Track training data versions
   - Document configuration changes

## 🔍 Troubleshooting

### Drift Not Detected But Model Performance Degrading

- Check concept drift threshold (may be too high)
- Verify production model is being evaluated correctly
- Ensure sufficient recent data samples

### Model Comparison Shows No Improvement

- Review training data quality and size
- Check feature engineering pipeline
- Verify hyperparameter tuning settings
- Ensure training data includes recent patterns

### Deployment Fails

- Check MLflow Model Registry connection
- Verify model artifact exists
- Ensure sufficient permissions
- Review deployment logs in MLflow

## 📚 Related Documentation

- [Training Pipeline](../train/README.md) - Initial model training
- [Model Evaluation](../evaluate/README.md) - Evaluation metrics and validation
- [Inference API](../../inference_service/README.md) - Production inference service
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 🔗 Integration Points

- **Feature Store**: Loads training data
- **TimescaleDB**: Queries sensor data for drift detection
- **MLflow**: Model versioning and tracking
- **Training Pipeline**: Reuses training logic
- **Inference API**: Loads deployed models
- **Alert Engine**: Notifies on drift/deployment events
