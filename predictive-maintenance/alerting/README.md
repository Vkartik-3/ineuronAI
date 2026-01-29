# Alert Engine

Real-time alerting system for critical equipment conditions in predictive maintenance.

## Features

- **Rule-Based Alerts**: Configurable alert rules for RUL, anomaly scores, health status, and sensor thresholds
- **Multiple Channels**: Email, Slack, webhooks, and database notifications
- **Alert Lifecycle**: Trigger, acknowledge, resolve workflow
- **Severity Levels**: Critical, warning, and info alerts
- **Cooldown Management**: Prevent alert fatigue with configurable cooldowns
- **Aggregation**: Group similar alerts to reduce noise
- **Suppression**: Maintenance windows and incident mode
- **History Tracking**: Store and query alert history

## Installation

```bash
cd alerting
pip install -r requirements.txt
```

## Configuration

Edit `config/alert_config.yaml`:

```yaml
rules:
  rul_critical:
    enabled: true
    condition: "rul <= 10"
    severity: "critical"
    message: "Equipment {equipment_id} has critical RUL: {rul} cycles"
    cooldown: 300

notifications:
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    recipients:
      critical: ["ops-team@example.com"]

  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channels:
      critical: "#alerts-critical"
```

## Quick Start

### Initialize Alert Manager

```python
from alerting.alert_manager import AlertManager

# Initialize with config
alert_mgr = AlertManager('alerting/config/alert_config.yaml')
```

### Process Predictions

```python
# Prediction data from inference service
prediction = {
    'equipment_id': 'EQ001',
    'rul': 8,  # Critical!
    'anomaly_score': 0.85,
    'health_status': 'critical',
    'temperature': 95.5,
    'vibration': 0.75,
    'timestamp': '2024-01-15T10:30:00Z'
}

# Evaluate rules and trigger alerts
alerts = alert_mgr.process_prediction(prediction)

print(f"Triggered {len(alerts)} alerts")
for alert in alerts:
    print(f"  - {alert.severity.value}: {alert.message}")
```

### Manage Alerts

```python
# Get active alerts
active = alert_mgr.get_active_alerts(equipment_id='EQ001')

# Acknowledge alert
alert_mgr.acknowledge_alert(alert_id='rul_critical_EQ001_1234567890', user='john@example.com')

# Resolve alert
alert_mgr.resolve_alert(alert_id='rul_critical_EQ001_1234567890')

# Get statistics
stats = alert_mgr.get_statistics()
print(f"Total alerts: {stats['total_alerts']}")
print(f"Active: {stats['active_alerts']}")
print(f"Critical: {stats['severity_counts']['critical']}")
```

## Alert Rules

### Built-in Rules

**RUL-based**:

- `rul_critical`: RUL ≤ 10 cycles
- `rul_warning`: RUL ≤ 30 cycles
- `rul_info`: RUL ≤ 50 cycles

**Anomaly-based**:

- `anomaly_critical`: Score ≥ 0.9
- `anomaly_warning`: Score ≥ 0.7

**Health Status**:

- `health_imminent_failure`: Imminent failure predicted
- `health_critical`: Critical health state

**Sensor-based**:

- `temperature_high`: Temperature > 95°C
- `vibration_high`: Vibration > 0.8g

**Trend-based**:

- `rapid_degradation`: RUL dropping > 5 cycles/hour

### Custom Rules

Add custom rules to `alert_config.yaml`:

```yaml
rules:
  my_custom_rule:
    enabled: true
    condition: "temperature > 90 and vibration > 0.6"
    severity: "warning"
    message: "Combined high temp and vibration on {equipment_id}"
    cooldown: 600
```

## Notification Channels

### Email

```yaml
notifications:
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_user: "alerts@example.com"
    smtp_password: "${SMTP_PASSWORD}"
    recipients:
      critical:
        - "ops-team@example.com"
        - "manager@example.com"
      warning:
        - "maintenance-team@example.com"
```

**Environment variable**:

```bash
export SMTP_PASSWORD="your_password"
```

### Slack

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channels:
      critical: "#alerts-critical"
      warning: "#alerts-warning"
    mention_users:
      critical:
        - "@ops-lead"
```

**Setup**:

1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Set environment variable:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### Webhook

```yaml
notifications:
  webhook:
    enabled: true
    url: "https://api.example.com/webhooks/alerts"
    method: "POST"
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
```

### Database

```yaml
notifications:
  database:
    enabled: true
    connection_string: "postgresql://user:pass@localhost:5432/alerts"
    table_name: "alert_history"
    retention_days: 90
```

## Integration with Inference Service

### Option 1: Direct Integration

```python
from alerting.alert_manager import get_alert_manager
from inference_service.api.main import app

alert_mgr = get_alert_manager()

@app.post("/predict/rul")
async def predict_rul(request):
    # Make prediction
    prediction = model.predict(...)

    # Trigger alerts
    alerts = alert_mgr.process_prediction({
        'equipment_id': request.equipment_id,
        'rul': prediction.rul,
        'health_status': prediction.health_status,
        **request.sensor_data
    })

    return {
        'prediction': prediction,
        'alerts': [a.to_dict() for a in alerts]
    }
```

### Option 2: Kafka Integration

```python
from kafka import KafkaConsumer
from alerting.alert_manager import get_alert_manager

# Subscribe to predictions topic
consumer = KafkaConsumer('predictions')
alert_mgr = get_alert_manager()

for message in consumer:
    prediction = json.loads(message.value)
    alerts = alert_mgr.process_prediction(prediction)
```

## Alert Lifecycle

```
┌─────────┐
│ Trigger │ ──> Alert created when rule condition met
└────┬────┘
     │
     v
┌────────────┐
│ Triggered  │ ──> Notifications sent
└────┬───────┘
     │
     v
┌──────────────┐
│ Acknowledged │ ──> User acknowledges alert
└────┬─────────┘
     │
     v
┌──────────┐
│ Resolved │ ──> Issue fixed or alert no longer valid
└──────────┘
```

## Alert Suppression

### Maintenance Windows

Suppress alerts during scheduled maintenance:

```yaml
suppression:
  enabled: true
  maintenance_windows:
    - equipment_id: "EQ001"
      start: "2024-01-15 10:00:00"
      end: "2024-01-15 12:00:00"
      reason: "Scheduled maintenance"
```

### Incident Mode

During major incidents, suppress low-priority alerts:

```yaml
suppression:
  incident_mode:
    enabled: true
    suppress_below_severity: "warning" # Only show critical alerts
```

## Alert Aggregation

Group similar alerts to reduce noise:

```yaml
aggregation:
  enabled: true
  window_seconds: 300 # 5 minutes
  group_by:
    - equipment_id
    - severity
  max_group_size: 10
```

## Testing

### Test Individual Rule

```python
from alerting.alert_manager import AlertManager

alert_mgr = AlertManager()

# Test rule with mock data
test_alert = alert_mgr.test_alert(
    rule_id='rul_critical',
    equipment_id='EQ001'
)

if test_alert:
    print(f"Rule triggered: {test_alert.message}")
```

### Run Tests

```bash
pytest tests/test_alert_rules.py
pytest tests/test_notifiers.py
```

## Monitoring

### Alert Statistics

```python
stats = alert_mgr.get_statistics()

print(f"""
Total Alerts: {stats['total_alerts']}
Active: {stats['active_alerts']}

By Severity:
  Critical: {stats['severity_counts']['critical']}
  Warning: {stats['severity_counts']['warning']}
  Info: {stats['severity_counts']['info']}

By Status:
  Triggered: {stats['status_counts']['triggered']}
  Acknowledged: {stats['status_counts']['acknowledged']}
  Resolved: {stats['status_counts']['resolved']}

Avg Acknowledgment Time: {stats['average_acknowledgment_time_seconds']:.1f}s
""")
```

### Health Check

```python
# Check if alert manager is working
try:
    stats = alert_mgr.get_statistics()
    print("✅ Alert engine healthy")
except Exception as e:
    print(f"❌ Alert engine error: {e}")
```

## Best Practices

1. **Set Appropriate Cooldowns**: Prevent alert fatigue by setting reasonable cooldown periods (300-1800s)

2. **Use Severity Levels Wisely**:

   - **Critical**: Immediate action required, potential equipment failure
   - **Warning**: Should be addressed soon, degradation detected
   - **Info**: Informational, schedule maintenance

3. **Configure Business Hours**: Route after-hours critical alerts appropriately

4. **Test Rules**: Use `test_alert()` to verify rules before deployment

5. **Monitor Alert Volume**: Track alert statistics to identify noisy rules

6. **Document Escalation**: Define clear escalation paths for critical alerts

7. **Regular Cleanup**: Configure database retention to manage storage

## Troubleshooting

**Alerts not triggering**:

- Check rule `enabled: true` in config
- Verify condition syntax
- Test with `test_alert()` method
- Check logs for evaluation errors

**Notifications not sending**:

- Verify credentials (SMTP password, Slack webhook)
- Check notifier `enabled: true`
- Test network connectivity
- Review logs for errors

**Too many alerts**:

- Increase cooldown periods
- Enable aggregation
- Adjust thresholds in rules
- Use suppression during maintenance

**Database issues**:

- Verify connection string
- Check database credentials
- Ensure table exists
- Monitor disk space

## API Reference

### AlertManager

```python
class AlertManager:
    def __init__(config_path: str)
    def process_prediction(data: Dict) -> List[Alert]
    def get_active_alerts(equipment_id: str = None, severity: str = None) -> List[Dict]
    def acknowledge_alert(alert_id: str, user: str) -> bool
    def resolve_alert(alert_id: str) -> bool
    def get_statistics() -> Dict
    def test_alert(rule_id: str, equipment_id: str) -> Alert
```

### Alert

```python
class Alert:
    alert_id: str
    rule_id: str
    equipment_id: str
    severity: AlertSeverity
    message: str
    data: Dict
    timestamp: datetime
    status: AlertStatus

    def acknowledge(user: str)
    def resolve()
    def to_dict() -> Dict
```

## Next Steps

Module 8 (Alert Engine) complete! ✅

**Final Phase 3 Module**:

- **Module 9**: Dashboard for visualization and monitoring

## References

- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/overview/)
- [PagerDuty Alert Lifecycle](https://www.pagerduty.com/)
- [Slack Webhooks](https://api.slack.com/messaging/webhooks)
