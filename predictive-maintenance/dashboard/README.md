# Real-time Dashboard

Interactive dashboards for predictive maintenance monitoring and visualization.

## Components

### Streamlit Dashboard

Interactive web application for equipment health monitoring.

**Features**:

- Equipment health grid with real-time status
- RUL distribution and trends
- Anomaly score visualization
- Alert history and management
- Sensor correlation plots
- Auto-refresh capability
- Filters by location, status, equipment type

### Grafana Dashboards

Time-series visualizations for historical analysis.

**Panels**:

- RUL gauge and trends
- Anomaly score charts
- Sensor data (temperature, vibration, pressure)
- Alert timeline
- Equipment comparison

## Installation

```bash
cd dashboard
pip install -r requirements.txt
```

## Quick Start

### Run Streamlit Dashboard

```bash
# From project root
streamlit run dashboard/streamlit_app/app.py

# Custom port
streamlit run dashboard/streamlit_app/app.py --server.port 8501
```

Access at: http://localhost:8501

### Configure Grafana

1. **Start Grafana**:

```bash
# Docker
docker run -d -p 3000:3000 grafana/grafana

# Or install from https://grafana.com/grafana/download
```

2. **Add Data Sources**:

   - InfluxDB for time-series sensor data
   - PostgreSQL for alerts and predictions

3. **Import Dashboard**:
   - Go to Dashboards → Import
   - Upload `grafana/predictive_maintenance_dashboard.json`

## Streamlit Dashboard Features

### Key Metrics

Dashboard displays:

- Total equipment count
- Critical/imminent failure count
- Warning count
- Healthy equipment count
- Average RUL across fleet

### Equipment Health Grid

Sortable table showing:

- Equipment ID
- Equipment type
- Location
- Current RUL
- Health status (color-coded)
- Anomaly score
- Temperature & vibration
- Days since last maintenance

### Visualizations

**RUL Distribution**:

- Histogram of RUL across all equipment
- Color-coded by health status
- Helps identify maintenance scheduling needs

**Health Status Breakdown**:

- Pie chart of equipment by status
- Quick overview of fleet health

**Anomaly Scores**:

- Scatter plot with anomaly thresholds
- Warning line at 0.7
- Critical line at 0.9

**Sensor Correlation**:

- Temperature vs vibration plot
- Bubble size indicates anomaly score
- Identify abnormal sensor patterns

### Alert Management

Recent alerts with:

- Timestamp
- Severity badge (critical/warning/info)
- Message
- Acknowledge button for triggered alerts
- Status indicator (triggered/acknowledged/resolved)

### Filters

Sidebar filters for:

- **Location**: Plant A, B, C
- **Health Status**: Healthy, Warning, Critical, Imminent Failure
- **Equipment Type**: Pump, Motor, Compressor, Turbine

### Auto-Refresh

- Toggle auto-refresh on/off
- Configurable refresh interval (10-120 seconds)
- Manual refresh button
- Last updated timestamp

## Configuration

Edit `config/dashboard_config.yaml`:

```yaml
dashboard:
  title: "Predictive Maintenance Dashboard"
  refresh_interval: 30 # seconds

data_sources:
  inference_api:
    url: "http://localhost:8000"

  database:
    type: "postgresql"
    host: "localhost"
    port: 5432
    database: "predictive_maintenance"
```

## Integration

### Connect to Inference API

```python
# In app.py
inference_api = APIClient('http://localhost:8000')

# Get predictions
predictions = inference_api.get('/predict/batch')
```

### Connect to Alert Engine

```python
# Get active alerts
alerts = requests.get('http://localhost:8001/alerts/active').json()

# Acknowledge alert
requests.post('http://localhost:8001/alerts/acknowledge', json={
    'alert_id': 'ALT0001',
    'user': 'john@example.com'
})
```

### Database Connection

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="predictive_maintenance",
    user="pm_user",
    password="password"
)

# Query alert history
query = "SELECT * FROM alert_history WHERE timestamp > NOW() - INTERVAL '24 hours'"
df = pd.read_sql(query, conn)
```

## Grafana Setup

### Data Source Configuration

**InfluxDB**:

```yaml
URL: http://localhost:8086
Organization: predictive_maintenance
Token: your_token
Default Bucket: sensor_data
```

**PostgreSQL**:

```yaml
Host: localhost:5432
Database: predictive_maintenance
User: pm_user
Password: your_password
SSL Mode: disable
```

### Dashboard Panels

**RUL Gauge**:

```flux
from(bucket: "sensor_data")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "predictions")
  |> filter(fn: (r) => r._field == "rul")
  |> last()
```

**RUL Trend**:

```flux
from(bucket: "sensor_data")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "predictions")
  |> filter(fn: (r) => r._field == "rul")
  |> aggregateWindow(every: 10m, fn: mean)
```

**Alert History**:

```sql
SELECT
  alert_id,
  equipment_id,
  severity,
  message,
  timestamp,
  status
FROM alert_history
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC
LIMIT 20
```

## Customization

### Add Custom Visualizations

```python
import plotly.express as px

# Custom chart
fig = px.scatter_3d(
    df,
    x='temperature',
    y='vibration',
    z='rul',
    color='health_status'
)
st.plotly_chart(fig)
```

### Custom Metrics

```python
# Calculate custom KPIs
maintenance_due = len(df[df['rul'] < 30])
avg_anomaly = df['anomaly_score'].mean()

st.metric("Maintenance Due", maintenance_due)
st.metric("Avg Anomaly", f"{avg_anomaly:.3f}")
```

### Theme Customization

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Deployment

### Streamlit Cloud

```bash
# Push to GitHub
git add .
git commit -m "Add dashboard"
git push

# Deploy on streamlit.io
# Select repository and branch
# Set secrets in dashboard settings
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY dashboard/requirements.txt .
RUN pip install -r requirements.txt

COPY dashboard/ .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t pm-dashboard .
docker run -p 8501:8501 pm-dashboard
```

### Grafana Docker Compose

```yaml
version: "3.8"

services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana:/etc/grafana/provisioning

volumes:
  grafana-storage:
```

## Screenshots

### Streamlit Dashboard

- Equipment health grid with color-coded status
- Interactive charts and filters
- Real-time alert monitoring

### Grafana Dashboards

- Time-series RUL trends
- Sensor data visualization
- Alert timeline

## Performance

**Streamlit**:

- Handles 20+ equipment monitoring
- Auto-refresh every 30 seconds
- Sub-second load time with caching

**Grafana**:

- Query optimization with aggregation
- 10-minute time windows for historical data
- Efficient InfluxDB queries

## Troubleshooting

**Streamlit not loading**:

- Check port 8501 is available
- Verify API endpoints are accessible
- Check configuration file path

**Grafana no data**:

- Verify data source connections
- Test queries in Explore tab
- Check InfluxDB bucket name
- Verify PostgreSQL permissions

**Slow performance**:

- Increase refresh interval
- Reduce query time range
- Add data aggregation
- Enable caching

## Best Practices

1. **Auto-Refresh**: Set appropriate intervals (30-60s) to balance freshness and load
2. **Filters**: Use filters to reduce data volume and improve performance
3. **Caching**: Cache expensive computations with `@st.cache_data`
4. **Error Handling**: Add try-except blocks for API calls
5. **Responsive Design**: Use columns and containers for mobile-friendly layouts

## API Endpoints

Dashboard expects these endpoints:

**Inference API**:

- `GET /health` - API health check
- `GET /models` - List available models
- `POST /predict/rul` - RUL prediction
- `POST /predict/batch` - Batch predictions

**Alert API**:

- `GET /alerts/active` - Get active alerts
- `POST /alerts/acknowledge` - Acknowledge alert
- `GET /alerts/statistics` - Alert statistics

## Next Steps

All Phase 3 modules complete! ✅

**Project Complete**:

- ✅ Phase 1: Data Foundation (Modules 1-3)
- ✅ Phase 2: ML Pipeline (Modules 4-6)
- ✅ Phase 3: Inference & Alerting (Modules 7-9)

Ready for production deployment!

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [InfluxDB Flux](https://docs.influxdata.com/flux/v0.x/)
