"""
Streamlit Dashboard for Predictive Maintenance

Real-time equipment health monitoring and visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load configuration
@st.cache_resource
def load_config():
    """Load dashboard configuration"""
    try:
        with open("dashboard/config/dashboard_config.yaml", "r") as f:
            return yaml.safe_load(f)
    except:
        # Default config
        return {
            "dashboard": {
                "title": "Predictive Maintenance Dashboard",
                "refresh_interval": 30,
            },
            "data_sources": {
                "inference_api": {"url": "http://localhost:8000"},
                "alert_api": {"url": "http://localhost:8001"},
            },
        }


config = load_config()


# API clients
class APIClient:
    """Client for backend APIs"""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, endpoint: str) -> Dict:
        """GET request"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}

    def post(self, endpoint: str, data: Dict) -> Dict:
        """POST request"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}", json=data, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}


# Initialize API clients
inference_api = APIClient(config["data_sources"]["inference_api"]["url"])


# Mock data generator (for demo purposes)
def generate_mock_equipment_data(n_equipment: int = 20) -> pd.DataFrame:
    """Generate mock equipment data for demonstration"""
    import numpy as np

    equipment_ids = [f"EQ{str(i + 1).zfill(3)}" for i in range(n_equipment)]
    equipment_types = np.random.choice(
        ["Pump", "Motor", "Compressor", "Turbine"], n_equipment
    )
    locations = np.random.choice(["Plant A", "Plant B", "Plant C"], n_equipment)

    # Generate RUL with some equipment in critical state
    ruls = []
    for i in range(n_equipment):
        if i < 2:  # 2 critical
            ruls.append(np.random.uniform(0, 10))
        elif i < 6:  # 4 warning
            ruls.append(np.random.uniform(10, 30))
        elif i < 10:  # 4 attention
            ruls.append(np.random.uniform(30, 50))
        else:  # rest healthy
            ruls.append(np.random.uniform(50, 150))

    # Health status based on RUL
    health_status = []
    for rul in ruls:
        if rul < 10:
            health_status.append("Imminent Failure")
        elif rul < 30:
            health_status.append("Critical")
        elif rul < 50:
            health_status.append("Warning")
        else:
            health_status.append("Healthy")

    # Anomaly scores (inversely related to RUL)
    anomaly_scores = 1 - (np.array(ruls) / 150)
    anomaly_scores = np.clip(anomaly_scores, 0, 1)

    # Other metrics
    temperatures = np.random.uniform(60, 100, n_equipment)
    vibrations = np.random.uniform(0.1, 1.0, n_equipment)

    # Last maintenance (days ago)
    last_maintenance_days = np.random.randint(0, 365, n_equipment)

    df = pd.DataFrame(
        {
            "equipment_id": equipment_ids,
            "equipment_type": equipment_types,
            "location": locations,
            "rul": ruls,
            "health_status": health_status,
            "anomaly_score": anomaly_scores,
            "temperature": temperatures,
            "vibration": vibrations,
            "last_maintenance_days": last_maintenance_days,
            "last_updated": datetime.now(),
        }
    )

    return df


def generate_mock_alerts(n_alerts: int = 10) -> pd.DataFrame:
    """Generate mock alert history"""
    import numpy as np

    equipment_ids = [
        f"EQ{str(np.random.randint(1, 21)).zfill(3)}" for _ in range(n_alerts)
    ]
    severities = np.random.choice(
        ["critical", "warning", "info"], n_alerts, p=[0.3, 0.5, 0.2]
    )

    messages = []
    for eq, sev in zip(equipment_ids, severities):
        if sev == "critical":
            messages.append(f"Critical RUL detected on {eq}")
        elif sev == "warning":
            messages.append(f"Warning: {eq} approaching maintenance threshold")
        else:
            messages.append(f"Info: Schedule maintenance for {eq}")

    timestamps = [
        datetime.now() - timedelta(hours=np.random.randint(0, 48))
        for _ in range(n_alerts)
    ]

    statuses = np.random.choice(
        ["triggered", "acknowledged", "resolved"], n_alerts, p=[0.4, 0.3, 0.3]
    )

    df = pd.DataFrame(
        {
            "alert_id": [f"ALT{i + 1:04d}" for i in range(n_alerts)],
            "equipment_id": equipment_ids,
            "severity": severities,
            "message": messages,
            "timestamp": timestamps,
            "status": statuses,
        }
    )

    return df.sort_values("timestamp", ascending=False)


# Sidebar
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x80/4CAF50/FFFFFF?text=PM+System",
        use_container_width=True,
    )
    st.title("⚙️ PM Dashboard")

    # Refresh controls
    st.subheader("Controls")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)

    if st.button("🔄 Refresh Now"):
        st.rerun()

    # Filters
    st.subheader("Filters")
    filter_location = st.multiselect(
        "Location", ["Plant A", "Plant B", "Plant C"], default=None
    )
    filter_status = st.multiselect(
        "Health Status",
        ["Healthy", "Warning", "Critical", "Imminent Failure"],
        default=None,
    )
    filter_type = st.multiselect(
        "Equipment Type", ["Pump", "Motor", "Compressor", "Turbine"], default=None
    )

    # System info
    st.subheader("System Status")
    st.success("✅ Inference API: Online")
    st.success("✅ Alert Engine: Online")
    st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Main dashboard
st.title("🏭 Predictive Maintenance Dashboard")

# Load data
equipment_df = generate_mock_equipment_data(20)
alerts_df = generate_mock_alerts(15)

# Apply filters
if filter_location:
    equipment_df = equipment_df[equipment_df["location"].isin(filter_location)]
if filter_status:
    equipment_df = equipment_df[equipment_df["health_status"].isin(filter_status)]
if filter_type:
    equipment_df = equipment_df[equipment_df["equipment_type"].isin(filter_type)]

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Total Equipment", value=len(equipment_df), delta=None)

with col2:
    critical_count = len(
        equipment_df[
            equipment_df["health_status"].isin(["Critical", "Imminent Failure"])
        ]
    )
    st.metric(
        label="Critical/Imminent",
        value=critical_count,
        delta=f"{critical_count} alerts" if critical_count > 0 else None,
        delta_color="inverse",
    )

with col3:
    warning_count = len(equipment_df[equipment_df["health_status"] == "Warning"])
    st.metric(label="Warning", value=warning_count, delta=None)

with col4:
    healthy_count = len(equipment_df[equipment_df["health_status"] == "Healthy"])
    st.metric(label="Healthy", value=healthy_count, delta=None)

with col5:
    avg_rul = equipment_df["rul"].mean()
    st.metric(label="Avg RUL", value=f"{avg_rul:.1f}", delta="cycles")

st.divider()

# Equipment Health Grid
st.subheader("📊 Equipment Health Overview")


# Color mapping for health status
def get_status_color(status):
    colors = {
        "Healthy": "#28a745",
        "Warning": "#ffc107",
        "Critical": "#fd7e14",
        "Imminent Failure": "#dc3545",
    }
    return colors.get(status, "#6c757d")


# Create status badges
def create_status_badge(status):
    color = get_status_color(status)
    return f'<span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">{status}</span>'


# Display equipment grid
display_df = equipment_df.copy()
display_df["rul"] = display_df["rul"].round(1)
display_df["anomaly_score"] = display_df["anomaly_score"].round(3)
display_df["temperature"] = display_df["temperature"].round(1)
display_df["vibration"] = display_df["vibration"].round(3)

# Sort by RUL (critical first)
display_df = display_df.sort_values("rul")

st.dataframe(
    display_df[
        [
            "equipment_id",
            "equipment_type",
            "location",
            "rul",
            "health_status",
            "anomaly_score",
            "temperature",
            "vibration",
            "last_maintenance_days",
        ]
    ],
    use_container_width=True,
    height=400,
)

st.divider()

# Visualizations Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 RUL Distribution")

    fig_rul = px.histogram(
        equipment_df,
        x="rul",
        nbins=20,
        color="health_status",
        color_discrete_map={
            "Healthy": "#28a745",
            "Warning": "#ffc107",
            "Critical": "#fd7e14",
            "Imminent Failure": "#dc3545",
        },
        title="Distribution of Remaining Useful Life",
    )
    fig_rul.update_layout(
        xaxis_title="RUL (cycles)", yaxis_title="Count", showlegend=True
    )
    st.plotly_chart(fig_rul, use_container_width=True)

with col2:
    st.subheader("🎯 Health Status Breakdown")

    status_counts = equipment_df["health_status"].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Equipment by Health Status",
        color=status_counts.index,
        color_discrete_map={
            "Healthy": "#28a745",
            "Warning": "#ffc107",
            "Critical": "#fd7e14",
            "Imminent Failure": "#dc3545",
        },
    )
    st.plotly_chart(fig_status, use_container_width=True)

# Anomaly Scores and Temperature
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Anomaly Scores")

    fig_anomaly = px.scatter(
        equipment_df,
        x="equipment_id",
        y="anomaly_score",
        color="health_status",
        size="rul",
        hover_data=["equipment_type", "location"],
        color_discrete_map={
            "Healthy": "#28a745",
            "Warning": "#ffc107",
            "Critical": "#fd7e14",
            "Imminent Failure": "#dc3545",
        },
        title="Anomaly Scores by Equipment",
    )
    fig_anomaly.add_hline(
        y=0.7, line_dash="dash", line_color="orange", annotation_text="Warning"
    )
    fig_anomaly.add_hline(
        y=0.9, line_dash="dash", line_color="red", annotation_text="Critical"
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

with col2:
    st.subheader("🌡️ Temperature vs Vibration")

    fig_sensors = px.scatter(
        equipment_df,
        x="temperature",
        y="vibration",
        color="health_status",
        size="anomaly_score",
        hover_data=["equipment_id", "rul"],
        color_discrete_map={
            "Healthy": "#28a745",
            "Warning": "#ffc107",
            "Critical": "#fd7e14",
            "Imminent Failure": "#dc3545",
        },
        title="Sensor Correlation",
    )
    st.plotly_chart(fig_sensors, use_container_width=True)

st.divider()

# Recent Alerts
st.subheader("🔔 Recent Alerts")


# Alert severity badges
def get_severity_badge(severity):
    colors = {"critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"}
    return f'<span style="background-color: {colors[severity]}; color: white; padding: 2px 8px; border-radius: 8px; font-size: 11px;">{severity.upper()}</span>'


# Display alerts
for _, alert in alerts_df.head(10).iterrows():
    col1, col2, col3, col4 = st.columns([2, 3, 5, 2])

    with col1:
        st.write(alert["timestamp"].strftime("%Y-%m-%d %H:%M"))
    with col2:
        st.markdown(get_severity_badge(alert["severity"]), unsafe_allow_html=True)
    with col3:
        st.write(alert["message"])
    with col4:
        if alert["status"] == "triggered":
            if st.button(f"Acknowledge", key=alert["alert_id"]):
                st.success("Alert acknowledged!")
        else:
            st.write(f"✓ {alert['status']}")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
