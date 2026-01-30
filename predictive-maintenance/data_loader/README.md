# NASA C-MAPSS Data Loader

Integration of NASA C-MAPSS Turbofan Engine Degradation Dataset into the predictive maintenance system.

## 📋 Overview

This module loads the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset and streams it to Kafka, replacing synthetic data with real turbofan engine run-to-failure trajectories.

**Dataset Reference:**
A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, October 2008.

**Dataset Source:**
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## 📦 Dataset Details

### Four Subdatasets (FD001-FD004)

| Dataset   | Train Engines | Test Engines | Operating Conditions | Fault Modes               |
| --------- | ------------- | ------------ | -------------------- | ------------------------- |
| **FD001** | 100           | 100          | 1 (Sea Level)        | 1 (HPC Degradation)       |
| **FD002** | 260           | 259          | 6 (Mixed)            | 1 (HPC Degradation)       |
| **FD003** | 100           | 100          | 1 (Sea Level)        | 2 (HPC + Fan Degradation) |
| **FD004** | 248           | 249          | 6 (Mixed)            | 2 (HPC + Fan Degradation) |

### Data Format

Each row represents one operational cycle with **26 columns**:

```
1. unit_id          - Engine unit number
2. time_cycle       - Operational cycle number
3-5. op_setting_1-3 - Operational settings (altitude, throttle, etc.)
6-26. sensor_1-21   - 21 sensor measurements
```

### Sensor Descriptions

| Sensor    | Description                              | Unit    |
| --------- | ---------------------------------------- | ------- |
| sensor_1  | T2 - Total temperature at fan inlet      | °R      |
| sensor_2  | T24 - Total temperature at LPC outlet    | °R      |
| sensor_3  | T30 - Total temperature at HPC outlet    | °R      |
| sensor_4  | T50 - Total temperature at LPT outlet    | °R      |
| sensor_5  | P2 - Pressure at fan inlet               | psia    |
| sensor_6  | P15 - Total pressure in bypass-duct      | psia    |
| sensor_7  | P30 - Total pressure at HPC outlet       | psia    |
| sensor_8  | Nf - Physical fan speed                  | rpm     |
| sensor_9  | Nc - Physical core speed                 | rpm     |
| sensor_10 | epr - Engine pressure ratio (P50/P2)     | -       |
| sensor_11 | Ps30 - Static pressure at HPC outlet     | psia    |
| sensor_12 | phi - Ratio of fuel flow to Ps30         | pps/psi |
| sensor_13 | NRf - Corrected fan speed                | rpm     |
| sensor_14 | NRc - Corrected core speed               | rpm     |
| sensor_15 | BPR - Bypass Ratio                       | -       |
| sensor_16 | farB - Burner fuel-air ratio             | -       |
| sensor_17 | htBleed - Bleed Enthalpy                 | -       |
| sensor_18 | Nf_dmd - Demanded fan speed              | rpm     |
| sensor_19 | PCNfR_dmd - Demanded corrected fan speed | rpm     |
| sensor_20 | W31 - HPT coolant bleed                  | lbm/s   |
| sensor_21 | W32 - LPT coolant bleed                  | lbm/s   |

## 🏗️ Module Structure

```
data_loader/
├── __init__.py                  # Module initialization
├── cmapss_loader.py             # Dataset loader and preprocessor
├── kafka_streamer.py            # Real-time Kafka streaming
├── cmapss_sensor_mapper.py      # Sensor mapping to standard names
├── config/
│   └── kafka_config.yaml        # Kafka configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Dataset

The dataset is located at `../archive/CMaps/`:

```
archive/CMaps/
├── readme.txt
├── train_FD001.txt              # Training data (run-to-failure)
├── test_FD001.txt               # Test data (stops before failure)
├── RUL_FD001.txt                # True RUL values for test data
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
└── ... (FD003, FD004)
```

### 2. Load Data

```python
from data_loader import CMAPSSLoader

# Initialize loader
loader = CMAPSSLoader(
    dataset_path="../archive/CMaps",
    dataset_id="FD001"
)

# Load training data with RUL labels
train_df = loader.load_train_data()
print(f"Training: {len(train_df)} records, {train_df['unit_id'].nunique()} engines")

# Load test data with RUL labels
test_df = loader.load_test_data()
print(f"Test: {len(test_df)} records, {test_df['unit_id'].nunique()} engines")

# Normalize sensors
train_norm, test_norm, norm_params = loader.normalize_sensors(train_df, test_df)
```

### 3. Stream to Kafka

```python
from data_loader import CMAPSSKafkaStreamer

# Initialize streamer
streamer = CMAPSSKafkaStreamer(
    dataset_path="../archive/CMaps",
    dataset_id="FD001",
    use_train_data=True  # or False for test data
)

# Stream all engines at 1 reading/second per engine
streamer.stream_all_engines(rate_per_engine=1.0)

# Or stream single engine
streamer.stream_single_engine(unit_id=1, rate=1.0)
```

### 4. Command Line Usage

```bash
# Stream FD001 training data to Kafka
cd data_loader
python kafka_streamer.py --dataset FD001 --train --rate 1.0

# Stream single engine
python kafka_streamer.py --dataset FD001 --train --engine 5 --rate 2.0

# Stream in loop (continuous replay)
python kafka_streamer.py --dataset FD001 --train --rate 1.0 --loop
```

## 🔧 Configuration

Edit `config/kafka_config.yaml`:

```yaml
# Kafka broker settings
bootstrap_servers:
  - localhost:9092

# Topic configuration
topic: raw_sensor_data

# Streaming settings
streaming_rate: 1.0 # readings per second per engine
```

## 🗺️ Sensor Mapping

C-MAPSS sensors are mapped to standard feature names:

```python
from data_loader.cmapss_sensor_mapper import CMAPSSSensorMapper

# Map sensors to standard names
mapped_df = CMAPSSSensorMapper.map_cmapss_to_standard(train_df)

# Get available features
features = CMAPSSSensorMapper.get_feature_names()
primary_features = CMAPSSSensorMapper.get_primary_features()
```

**Primary Sensor Mapping:**

- `sensor_2` (T24) → `temperature`
- `sensor_7` (P30) → `pressure`
- `sensor_8` (Nf) → `rpm`
- `sensor_10` (epr) → `engine_pressure_ratio`
- `sensor_12` (phi) → `fuel_flow_ratio`

## 📊 Kafka Message Format

Messages published to `raw_sensor_data` topic:

```json
{
  "equipment_id": "ENGINE_0001",
  "unit_id": 1,
  "equipment_type": "turbofan_engine",
  "timestamp": "2026-01-30T10:00:00",
  "time_cycle": 42,
  "dataset": "FD001",
  "data_source": "cmapss",

  "operating_settings": {
    "op_setting_1": -0.0007,
    "op_setting_2": -0.0004,
    "op_setting_3": 100.0
  },

  "sensors": {
    "sensor_1": 518.67,
    "sensor_2": 643.02,
    "sensor_3": 1589.70,
    ...
    "sensor_21": 23.4190
  },

  "rul": 100  // Optional: only for evaluation
}
```

## 🔄 Integration with Existing Pipeline

The C-MAPSS data integrates seamlessly with the existing pipeline:

### 1. Stream Processor

- Automatically detects `data_source: "cmapss"` in messages
- Preserves `unit_id` in metadata for multi-engine tracking
- Handles C-MAPSS sensor format (21 sensors + 3 op settings)

### 2. Feature Store

- Extracts time-domain and frequency-domain features
- Stores features with unit_id for per-engine analysis

### 3. ML Pipeline (Training)

```python
# Train on C-MAPSS data
from data_loader import CMAPSSLoader

loader = CMAPSSLoader(dataset_path="../archive/CMaps", dataset_id="FD001")
train_df = loader.load_train_data()
test_df = loader.load_test_data()

# Use for model training
# train_df has RUL labels computed from run-to-failure trajectories
```

### 4. ML Pipeline (Evaluation)

```python
# Evaluate using NASA's official scoring function
def nasa_score_function(y_true, y_pred):
    """
    NASA PHM08 Challenge scoring function.
    Penalizes late predictions more than early predictions.
    """
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,  # Early prediction (less penalty)
        np.exp(diff / 10) - 1     # Late prediction (more penalty)
    )
    return np.sum(score)
```

## 📈 Data Preprocessing

### Remove Constant Sensors

Some sensors have near-constant values:

```python
constant_sensors = loader.identify_constant_sensors(train_df, threshold=0.001)
# Typically: sensor_1, sensor_5, sensor_6, sensor_18, sensor_19
train_df = train_df.drop(columns=constant_sensors)
```

### Normalization

```python
# Normalize using training statistics
train_norm, test_norm, norm_params = loader.normalize_sensors(train_df, test_df)

# Save normalization parameters
import json
with open('norm_params.json', 'w') as f:
    json.dump(norm_params, f)
```

### Operating Condition Clustering

For FD002/FD004 (6 operating conditions):

```python
from sklearn.cluster import KMeans

# Cluster operating settings to identify conditions
op_settings = train_df[['op_setting_1', 'op_setting_2', 'op_setting_3']]
kmeans = KMeans(n_clusters=6, random_state=42)
train_df['operating_condition'] = kmeans.fit_predict(op_settings)
```

## 🧪 Testing

```bash
# Test data loader
cd data_loader
python cmapss_loader.py

# Test sensor mapping
python cmapss_sensor_mapper.py

# Test Kafka streaming (requires Kafka running)
python kafka_streamer.py --dataset FD001 --train --engine 1 --rate 10.0
```

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:

- pandas>=2.0.0
- numpy>=1.24.0
- pyyaml>=6.0
- kafka-python>=2.0.2

## 🔗 Integration Checklist

- [x] Data loader for train/test sets
- [x] RUL label computation
- [x] Kafka streaming with rate control
- [x] Sensor mapping to standard names
- [x] Multi-unit concurrent streaming
- [x] Stream processor compatibility
- [ ] ML pipeline training with C-MAPSS data
- [ ] NASA score function for evaluation
- [ ] Dashboard visualization of engine units
- [ ] Alert rules for turbofan-specific thresholds

## 📚 Related Documentation

- [Stream Processor](../stream_processor/README.md) - Consumes C-MAPSS Kafka messages
- [Training Pipeline](../ml_pipeline/train/README.md) - Model training on C-MAPSS data
- [Evaluation](../ml_pipeline/evaluate/README.md) - NASA scoring metrics
- [NASA C-MAPSS Paper](https://ti.arc.nasa.gov/publications/4047/download/) - Original dataset publication

## 🎯 Next Steps

1. **Train models on FD001**: Use `CMAPSSLoader` to load data, train LSTM/RF models
2. **Evaluate with NASA score**: Implement PHM08 challenge scoring function
3. **Stream live data**: Use `CMAPSSKafkaStreamer` to replay engine trajectories
4. **Visualize degradation**: Dashboard showing per-engine health over cycles
5. **Multi-dataset training**: Combine FD001-FD004 for robustness

## ⚠️ Important Notes

- **Training data**: Complete run-to-failure trajectories
- **Test data**: Stops before failure; RUL provided separately
- **Temperature units**: Sensors are in Rankine (°R); converted to Fahrenheit automatically
- **Pressure units**: psia (pounds per square inch absolute)
- **RPM scaling**: Fan/core speeds are in actual rpm (not percentage)
- **RUL labels**: Cycles remaining until failure (0 = failure)

---

**C-MAPSS dataset provides real turbofan engine degradation data for production-grade predictive maintenance models!** 🚀
