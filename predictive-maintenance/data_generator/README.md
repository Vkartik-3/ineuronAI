# Data Generator Module

## Overview

The Data Generator simulates IoT sensor data from manufacturing equipment (turbofan engines, pumps, compressors) with realistic degradation patterns. It publishes sensor readings to Kafka for consumption by downstream processing systems.

## Architecture

```
data_generator/
├── config/                      # Configuration files
│   ├── equipment_config.yaml    # Sensor specifications and baselines
│   ├── degradation_config.yaml  # Failure modes and patterns
│   └── kafka_config.yaml        # Kafka connection settings
├── simulator/                   # Core simulation logic
│   ├── sensor_simulator.py      # Generates sensor readings
│   ├── degradation_engine.py    # Applies failure patterns
│   └── equipment_simulator.py   # Equipment lifecycle management
├── publisher/                   # Kafka integration
│   └── kafka_publisher.py       # Publishes to Kafka topics
├── utils/                       # Utilities
│   └── config_loader.py         # Configuration management
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

## Features

### 1. **Realistic Sensor Simulation**

- **Equipment Types**: Turbofan engines (based on NASA C-MAPSS dataset), pumps, compressors
- **Sensors**: Temperature, pressure, vibration, flow rate, power consumption, fuel flow
- **Noise**: Configurable Gaussian noise matching real sensor characteristics
- **Operating Conditions**: Simulates varying operational settings

### 2. **Degradation Patterns**

- **Linear**: Steady wear (bearing degradation, seal wear)
- **Exponential**: Accelerating failure (crack propagation)
- **Step**: Sudden performance drop (component failure)
- **Oscillating**: Periodic issues (imbalance, misalignment)
- **Combined**: Multiple concurrent degradation mechanisms

### 3. **Failure Modes**

**Turbofan Engine:**

- High Pressure Compressor (HPC) degradation
- Fan degradation
- Turbine blade degradation

**Pump:**

- Bearing wear
- Seal failure
- Impeller damage

**Compressor:**

- Valve degradation
- Motor bearing failure

### 4. **Kafka Integration**

- Publishes to `raw_sensor_data` topic
- Equipment ID as partition key (ensures ordering per equipment)
- GZIP compression for efficiency
- Configurable batch size and retry logic

## Installation

### Prerequisites

- Python 3.8+
- Apache Kafka (optional if using mock mode)

### Setup

```bash
cd data_generator
pip install -r requirements.txt
```

## Usage

### Basic Usage (Mock Mode - No Kafka Required)

```bash
python main.py --mock --num-equipment 5 --equipment-type turbofan_engine
```

### With Kafka

```bash
# Ensure Kafka is running on localhost:9092
python main.py --num-equipment 10 --equipment-type pump --sampling-interval 1.0
```

### Command Line Arguments

| Argument                | Type  | Default         | Description                             |
| ----------------------- | ----- | --------------- | --------------------------------------- |
| `--num-equipment`       | int   | 5               | Number of equipment to simulate         |
| `--equipment-type`      | str   | turbofan_engine | Type: turbofan_engine, pump, compressor |
| `--failure-probability` | float | 0.3             | Probability of failure injection (0-1)  |
| `--sampling-interval`   | float | 1.0             | Time between samples (seconds)          |
| `--max-cycles`          | int   | None            | Max cycles (None = infinite)            |
| `--mock`                | flag  | False           | Use mock publisher (no Kafka)           |
| `--config-dir`          | str   | None            | Custom config directory path            |

### Examples

**Simulate 10 pumps with high failure rate:**

```bash
python main.py --num-equipment 10 --equipment-type pump --failure-probability 0.5
```

**Fast simulation (10 samples/sec) for 1000 cycles:**

```bash
python main.py --sampling-interval 0.1 --max-cycles 1000
```

**Use custom configuration:**

```bash
python main.py --config-dir /path/to/custom/config
```

## Data Schema

### Sensor Data Message (Published to `raw_sensor_data`)

```json
{
  "equipment_id": "TURBOFAN_ENGINE_001",
  "equipment_type": "turbofan_engine",
  "timestamp": "2026-01-29T10:30:45.123456Z",
  "cycle": 1523,
  "operational_settings": {
    "operational_setting_1": 0.0023,
    "operational_setting_2": -0.0003,
    "operational_setting_3": 100.0
  },
  "sensor_readings": {
    "temperature_1": 518.67,
    "temperature_2": 642.15,
    "temperature_3": 1591.82,
    "pressure_1": 14.62,
    "vibration": 23.45,
    "fuel_flow": 8.42,
    ...
  },
  "metadata": {
    "location": "Factory_Floor_1",
    "model": "TURBOFAN_ENGINE_v1",
    "install_date": "2024-01-01T00:00:00Z",
    "failure_mode": "hpc_degradation",
    "rul_remaining": 87,
    "degradation_stage": "middle",
    "is_degraded": true
  }
}
```

**Key Fields:**

- `equipment_id`: Unique identifier (partition key for Kafka)
- `cycle`: Operational cycle number (monotonically increasing)
- `sensor_readings`: Dictionary of all sensor measurements
- `metadata.rul_remaining`: Ground truth Remaining Useful Life (cycles)
- `metadata.degradation_stage`: `healthy`, `early`, `middle`, `late`, `critical`

## Configuration

### Customizing Sensor Baselines

Edit `config/equipment_config.yaml`:

```yaml
pump:
  sensor_baseline:
    temperature:
      mean: 75.0
      std: 2.0
      min: 60.0
      max: 110.0
```

### Adding New Failure Modes

Edit `config/degradation_config.yaml`:

```yaml
failure_modes:
  pump:
    custom_failure:
      primary_sensors:
        - temperature
        - vibration
      degradation_pattern: exponential
      typical_rul_range: [50, 150]
      severity_multiplier: 1.5
```

### Kafka Connection

Edit `config/kafka_config.yaml`:

```yaml
kafka:
  bootstrap_servers:
    - "kafka-broker-1:9092"
    - "kafka-broker-2:9092"
  topics:
    raw_sensor_data: "raw_sensor_data"
```

## Monitoring

### Console Logs

Progress is logged every 100 cycles:

```
Cycle 100 | Active: 8 | Degraded: 3 | Failed: 2 | Total messages: 1000 | Rate: 10.2 msg/s
Publisher stats: {'messages_sent': 1000, 'errors_count': 0, 'is_connected': True}
```

### Graceful Shutdown

Press `Ctrl+C` to trigger graceful shutdown:

- Flushes pending Kafka messages
- Prints final statistics for each equipment
- Closes connections cleanly

## Integration with Predictive Maintenance System

### Data Flow

```
Data Generator → Kafka (raw_sensor_data) → Stream Processor → Feature Store → ML Pipeline
```

### Next Steps

1. **Stream Processor** (Module 3): Consumes from Kafka, performs feature engineering
2. **Feature Store** (Module 4): Stores time-series features
3. **ML Pipeline** (Modules 5-6): Trains models on historical data

### Ground Truth Labels

The `metadata.rul_remaining` field provides ground truth labels for:

- **Regression**: Predicting Remaining Useful Life (RUL)
- **Classification**: Healthy vs. Degraded vs. Failed
- **Anomaly Detection**: Training on healthy data

## Troubleshooting

### Kafka Connection Errors

```
ERROR - Failed to initialize Kafka producer: NoBrokersAvailable
```

**Solution**:

- Ensure Kafka is running: `docker ps | grep kafka`
- Check `bootstrap_servers` in `kafka_config.yaml`
- Use `--mock` flag for testing without Kafka

### Import Errors

```
ModuleNotFoundError: No module named 'numpy'
```

**Solution**: `pip install -r requirements.txt`

### Configuration Not Found

```
ERROR - Configuration file not found: config/equipment_config.yaml
```

**Solution**: Run from `data_generator/` directory or use `--config-dir`

## Performance Notes

- **Throughput**: ~100-1000 messages/sec (single producer)
- **Memory**: ~50-100 MB per 10 equipment
- **CPU**: Negligible (<5% single core)

For higher throughput, run multiple instances with different equipment IDs.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black .
flake8 .
```

## References

1. **NASA C-MAPSS Dataset**: Turbofan engine degradation data
   - A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" (PHM08)
2. **Kafka Documentation**: https://kafka.apache.org/documentation/
3. **Predictive Maintenance Architecture**: See main project README

## License

Part of the Predictive Maintenance System project.
