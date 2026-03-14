"""
Streamlit Dashboard for Predictive Maintenance

Real-time equipment health monitoring and visualization.
Queries TimescaleDB for live data with mock-data fallback.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@st.cache_resource
def load_config():
    """Load dashboard configuration"""
    paths = [
        "dashboard/config/dashboard_config.yaml",
        "config/dashboard_config.yaml",
        "/app/config/dashboard_config.yaml",
    ]
    for p in paths:
        try:
            with open(p, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    return {
        "dashboard": {
            "title": "Predictive Maintenance Dashboard",
            "refresh_interval": 30,
        },
        "data_sources": {
            "inference_api": {"url": "http://localhost:8000"},
            "alert_api": {"url": "http://localhost:8001"},
            "database": {
                "host": "timescaledb",
                "port": 5432,
                "database": "predictive_maintenance",
                "user": "pmuser",
                "password": "pmpassword",
            },
        },
    }


config = load_config()


# ---------------------------------------------------------------------------
# Database client (psycopg2 – real TimescaleDB queries)
# ---------------------------------------------------------------------------
class DatabaseClient:
    """Direct TimescaleDB queries for dashboard data."""

    def __init__(self, db_config: Dict[str, Any]):
        self._cfg = db_config
        self._conn = None

    # -- connection --------------------------------------------------------

    def _get_conn(self):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            return None

        if self._conn and not self._conn.closed:
            try:
                self._conn.cursor().execute("SELECT 1")
                return self._conn
            except Exception:
                self._conn = None

        host = os.environ.get("TIMESCALEDB_HOST", self._cfg.get("host", "timescaledb"))
        port = int(os.environ.get("TIMESCALEDB_PORT", self._cfg.get("port", 5432)))
        dbname = os.environ.get(
            "TIMESCALEDB_DATABASE", self._cfg.get("database", "predictive_maintenance")
        )
        user = os.environ.get("TIMESCALEDB_USER", self._cfg.get("user", "pmuser"))
        password = os.environ.get(
            "TIMESCALEDB_PASSWORD", self._cfg.get("password", "pmpassword")
        )

        try:
            self._conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                connect_timeout=5,
            )
            self._conn.autocommit = True
            return self._conn
        except Exception as exc:
            logger.warning("Could not connect to TimescaleDB: %s", exc)
            return None

    def _query(self, sql: str, params=None) -> Optional[pd.DataFrame]:
        conn = self._get_conn()
        if conn is None:
            return None
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception as exc:
            logger.warning("DB query failed: %s", exc)
            self._conn = None
            return None

    # -- public queries ----------------------------------------------------

    def get_equipment_health(self) -> Optional[pd.DataFrame]:
        """Latest prediction per equipment joined with equipment metadata."""
        sql = """
        SELECT
            e.equipment_id,
            e.equipment_type,
            e.location,
            p.rul_prediction    AS rul,
            p.health_status,
            p.anomaly_score,
            p.confidence,
            e.install_date,
            p.time              AS last_updated
        FROM equipment e
        LEFT JOIN LATERAL (
            SELECT *
              FROM predictions pp
             WHERE pp.equipment_id = e.equipment_id
             ORDER BY pp.time DESC
             LIMIT 1
        ) p ON true
        ORDER BY COALESCE(p.rul_prediction, 999) ASC
        """
        return self._query(sql)

    def get_alerts(self, limit: int = 50) -> Optional[pd.DataFrame]:
        """Recent alerts."""
        sql = """
        SELECT alert_id, equipment_id, rule_id, severity,
               message, status, triggered_at, acknowledged_by,
               acknowledged_at, resolved_at
          FROM alerts
         ORDER BY triggered_at DESC
         LIMIT %s
        """
        return self._query(sql, (limit,))

    def get_sensor_history(
        self, equipment_id: str, hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Sensor readings for one piece of equipment."""
        sql = """
        SELECT time, equipment_id, cycle, sensor_readings
          FROM sensor_readings
         WHERE equipment_id = %s
           AND time >= NOW() - INTERVAL '%s hours'
         ORDER BY time ASC
        """
        return self._query(sql, (equipment_id, hours))

    def get_prediction_history(
        self, equipment_id: str, hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Prediction history for one piece of equipment."""
        sql = """
        SELECT time, equipment_id, model_id, rul_prediction,
               health_status, anomaly_score, confidence
          FROM predictions
         WHERE equipment_id = %s
           AND time >= NOW() - INTERVAL '%s hours'
         ORDER BY time ASC
        """
        return self._query(sql, (equipment_id, hours))

    def get_alert_statistics(self) -> Optional[Dict[str, Any]]:
        """Aggregate alert stats."""
        sql = """
        SELECT
            COUNT(*)                                                     AS total,
            COUNT(*) FILTER (WHERE status IN ('triggered','acknowledged')) AS active,
            COUNT(*) FILTER (WHERE severity = 'critical')                 AS critical,
            COUNT(*) FILTER (WHERE severity = 'warning')                  AS warning,
            COUNT(*) FILTER (WHERE severity = 'info')                     AS info
          FROM alerts
        """
        df = self._query(sql)
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        return {
            "total": int(row["total"]),
            "active": int(row["active"]),
            "critical": int(row["critical"]),
            "warning": int(row["warning"]),
            "info": int(row["info"]),
        }

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


# ---------------------------------------------------------------------------
# REST API client (inference service, etc.)
# ---------------------------------------------------------------------------
class APIClient:
    """Client for backend REST APIs"""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, endpoint: str) -> Dict:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("API GET %s failed: %s", endpoint, e)
            return {}

    def post(self, endpoint: str, data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}", json=data, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("API POST %s failed: %s", endpoint, e)
            return {}


# ---------------------------------------------------------------------------
# Initialise clients
# ---------------------------------------------------------------------------
inference_api = APIClient(config["data_sources"]["inference_api"]["url"])
db_client = DatabaseClient(config["data_sources"].get("database", {}))


# ---------------------------------------------------------------------------
# Mock data generators (fallback when DB unavailable)
# ---------------------------------------------------------------------------
def generate_mock_equipment_data(n_equipment: int = 20) -> pd.DataFrame:
    """Generate mock equipment data — used only when DB is unreachable."""
    import numpy as np

    equipment_ids = [f"EQ{str(i + 1).zfill(3)}" for i in range(n_equipment)]
    equipment_types = np.random.choice(
        ["Pump", "Motor", "Compressor", "Turbine"], n_equipment
    )
    locations = np.random.choice(["Plant A", "Plant B", "Plant C"], n_equipment)

    ruls = []
    for i in range(n_equipment):
        if i < 2:
            ruls.append(np.random.uniform(0, 10))
        elif i < 6:
            ruls.append(np.random.uniform(10, 30))
        elif i < 10:
            ruls.append(np.random.uniform(30, 50))
        else:
            ruls.append(np.random.uniform(50, 150))

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

    anomaly_scores = 1 - (np.array(ruls) / 150)
    anomaly_scores = np.clip(anomaly_scores, 0, 1)

    return pd.DataFrame(
        {
            "equipment_id": equipment_ids,
            "equipment_type": equipment_types,
            "location": locations,
            "rul": ruls,
            "health_status": health_status,
            "anomaly_score": anomaly_scores,
            "last_updated": datetime.now(),
        }
    )


def generate_mock_alerts(n_alerts: int = 10) -> pd.DataFrame:
    """Generate mock alert history — used only when DB is unreachable."""
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


# ---------------------------------------------------------------------------
# Data loaders — try DB first, fall back to mock
# ---------------------------------------------------------------------------
def _derive_health_status(rul) -> str:
    """Map RUL to a human label."""
    if rul is None:
        return "Unknown"
    if rul < 10:
        return "Imminent Failure"
    if rul < 30:
        return "Critical"
    if rul < 50:
        return "Warning"
    return "Healthy"


def load_equipment_data() -> pd.DataFrame:
    """Load equipment health from TimescaleDB; fall back to mock data."""
    df = db_client.get_equipment_health()
    if df is not None and not df.empty:
        if "health_status" not in df.columns or df["health_status"].isna().all():
            df["health_status"] = df["rul"].apply(_derive_health_status)
        else:
            df["health_status"] = df["health_status"].fillna(
                df["rul"].apply(_derive_health_status)
            )
        df["rul"] = df["rul"].fillna(999)
        df["anomaly_score"] = df["anomaly_score"].fillna(0.0)
        df["equipment_type"] = df["equipment_type"].fillna("Unknown")
        df["location"] = df["location"].fillna("Unknown")
        return df

    st.warning("⚠️ Could not reach TimescaleDB — showing demo data")
    return generate_mock_equipment_data()


def load_alerts(limit: int = 50) -> pd.DataFrame:
    """Load alerts from TimescaleDB; fall back to mock data."""
    df = db_client.get_alerts(limit)
    if df is not None and not df.empty:
        if "triggered_at" in df.columns:
            df = df.rename(columns={"triggered_at": "timestamp"})
        return df.sort_values("timestamp", ascending=False)

    return generate_mock_alerts(limit)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ PM Dashboard")

    st.subheader("Controls")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)

    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.subheader("Filters")
    filter_location = st.multiselect(
        "Location", ["Plant A", "Plant B", "Plant C", "Unknown"], default=None
    )
    filter_status = st.multiselect(
        "Health Status",
        ["Healthy", "Warning", "Critical", "Imminent Failure", "Unknown"],
        default=None,
    )
    filter_type = st.multiselect(
        "Equipment Type",
        ["Pump", "Motor", "Compressor", "Turbine", "Unknown"],
        default=None,
    )

    # System connectivity
    st.subheader("System Status")
    db_ok = db_client._get_conn() is not None
    if db_ok:
        st.success("✅ TimescaleDB: Connected")
    else:
        st.error("❌ TimescaleDB: Offline")

    api_data = inference_api.get("/health")
    if api_data:
        st.success("✅ Inference API: Online")
    else:
        st.warning("⚠️ Inference API: Unreachable")

    st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Main dashboard — tabbed layout
# ---------------------------------------------------------------------------
st.title("🏭 Predictive Maintenance Dashboard")

tab_overview, tab_equipment, tab_alerts, tab_model, tab_system = st.tabs(
    [
        "📊 Overview",
        "🔍 Equipment Detail",
        "🔔 Alert Management",
        "🧠 Model Performance",
        "🖥️ System Health",
    ]
)

equipment_df = load_equipment_data()
alerts_df = load_alerts(50)

# Apply filters
if filter_location:
    equipment_df = equipment_df[equipment_df["location"].isin(filter_location)]
if filter_status:
    equipment_df = equipment_df[equipment_df["health_status"].isin(filter_status)]
if filter_type:
    equipment_df = equipment_df[equipment_df["equipment_type"].isin(filter_type)]

status_colors = {
    "Healthy": "#28a745",
    "Warning": "#ffc107",
    "Critical": "#fd7e14",
    "Imminent Failure": "#dc3545",
    "Unknown": "#6c757d",
}
severity_colors = {"critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"}

# ===================== TAB 1: OVERVIEW ====================================
with tab_overview:
    # ---- Key Metrics ---------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Equipment", value=len(equipment_df))

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
        st.metric(label="Warning", value=warning_count)

    with col4:
        healthy_count = len(equipment_df[equipment_df["health_status"] == "Healthy"])
        st.metric(label="Healthy", value=healthy_count)

    with col5:
        avg_rul = equipment_df["rul"].mean()
        st.metric(label="Avg RUL", value=f"{avg_rul:.1f}", delta="cycles")

    st.divider()

    # ---- Equipment Health Grid -----------------------------------------------
    st.subheader("📊 Equipment Health Overview")

    display_df = equipment_df.copy()
    display_df["rul"] = display_df["rul"].round(1)
    display_df["anomaly_score"] = display_df["anomaly_score"].round(3)

    display_cols = [
        c
        for c in [
            "equipment_id",
            "equipment_type",
            "location",
            "rul",
            "health_status",
            "anomaly_score",
            "last_updated",
        ]
        if c in display_df.columns
    ]

    display_df = display_df.sort_values("rul")
    st.dataframe(display_df[display_cols], use_container_width=True, height=400)

    st.divider()

    # ---- Visualisations ------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 RUL Distribution")
        fig_rul = px.histogram(
            equipment_df,
            x="rul",
            nbins=20,
            color="health_status",
            color_discrete_map=status_colors,
            title="Distribution of Remaining Useful Life",
        )
        fig_rul.update_layout(xaxis_title="RUL (cycles)", yaxis_title="Count")
        st.plotly_chart(fig_rul, use_container_width=True)

    with col2:
        st.subheader("🎯 Health Status Breakdown")
        status_counts = equipment_df["health_status"].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Equipment by Health Status",
            color=status_counts.index,
            color_discrete_map=status_colors,
        )
        st.plotly_chart(fig_status, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Anomaly Scores")
        fig_anomaly = px.scatter(
            equipment_df,
            x="equipment_id",
            y="anomaly_score",
            color="health_status",
            size="rul",
            hover_data=[
                c for c in ["equipment_type", "location"] if c in equipment_df.columns
            ],
            color_discrete_map=status_colors,
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
        st.subheader("🏢 Equipment by Location")
        if "location" in equipment_df.columns:
            loc_status = (
                equipment_df.groupby(["location", "health_status"])
                .size()
                .reset_index(name="count")
            )
            fig_loc = px.bar(
                loc_status,
                x="location",
                y="count",
                color="health_status",
                color_discrete_map=status_colors,
                title="Health Distribution per Location",
                barmode="stack",
            )
            st.plotly_chart(fig_loc, use_container_width=True)

# ===================== TAB 2: EQUIPMENT DETAIL ============================
with tab_equipment:
    selected_eq = st.selectbox(
        "Select Equipment", equipment_df["equipment_id"].tolist()
    )

    if selected_eq:
        eq_row = equipment_df[equipment_df["equipment_id"] == selected_eq]
        if not eq_row.empty:
            eq_row = eq_row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Equipment ID", selected_eq)
            c2.metric("Health", eq_row.get("health_status", "Unknown"))
            c3.metric("RUL", f"{eq_row.get('rul', 'N/A'):.1f} cycles")
            c4.metric("Anomaly Score", f"{eq_row.get('anomaly_score', 0):.3f}")

        st.divider()

        # RUL Trend Chart
        st.subheader("📈 RUL Trend")
        hours_range = st.slider("Time window (hours)", 1, 168, 24, key="rul_hours")
        pred_df = db_client.get_prediction_history(selected_eq, hours=hours_range)
        if pred_df is not None and not pred_df.empty:
            fig_rul_trend = go.Figure()
            fig_rul_trend.add_trace(
                go.Scatter(
                    x=pred_df["time"],
                    y=pred_df["rul_prediction"],
                    mode="lines+markers",
                    name="RUL Prediction",
                    line=dict(color="#1f77b4"),
                )
            )
            # Warning and critical thresholds
            fig_rul_trend.add_hline(
                y=30, line_dash="dash", line_color="orange", annotation_text="Warning"
            )
            fig_rul_trend.add_hline(
                y=10, line_dash="dash", line_color="red", annotation_text="Critical"
            )
            fig_rul_trend.update_layout(
                title=f"RUL Trend — {selected_eq}",
                xaxis_title="Time",
                yaxis_title="RUL (cycles)",
                height=400,
            )
            st.plotly_chart(fig_rul_trend, use_container_width=True)

            # Confidence band if available
            if "confidence" in pred_df.columns:
                st.caption(
                    f"Latest confidence: {pred_df['confidence'].iloc[-1]:.2%}"
                    if pred_df["confidence"].iloc[-1]
                    else ""
                )
        else:
            st.info("No prediction history. Data will appear once the pipeline runs.")

        # Sensor Time-Series
        st.subheader("📡 Sensor Readings")
        sensor_df = db_client.get_sensor_history(selected_eq, hours=hours_range)
        if sensor_df is not None and not sensor_df.empty:
            # sensor_readings is a JSONB column; expand a few sensors
            try:
                if "sensor_readings" in sensor_df.columns:
                    expanded = pd.json_normalize(sensor_df["sensor_readings"])
                    expanded["time"] = sensor_df["time"].values
                    # Show first 6 sensors
                    sensor_cols = [c for c in expanded.columns if c != "time"][:6]
                    for sc in sensor_cols:
                        expanded[sc] = pd.to_numeric(expanded[sc], errors="coerce")
                    fig_sensors = make_subplots(
                        rows=len(sensor_cols),
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=sensor_cols,
                    )
                    for i, col in enumerate(sensor_cols, 1):
                        fig_sensors.add_trace(
                            go.Scatter(
                                x=expanded["time"],
                                y=expanded[col],
                                mode="lines",
                                name=col,
                            ),
                            row=i,
                            col=1,
                        )
                    fig_sensors.update_layout(
                        height=200 * len(sensor_cols),
                        title_text=f"Sensor Readings — {selected_eq}",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_sensors, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render sensor data: {exc}")
        else:
            st.info("No sensor data available for this equipment.")

        # Alert History for Equipment
        st.subheader("🔔 Equipment Alert History")
        if not alerts_df.empty:
            eq_alerts = alerts_df[alerts_df["equipment_id"] == selected_eq]
            if not eq_alerts.empty:
                st.dataframe(eq_alerts, use_container_width=True)
            else:
                st.info("No alerts for this equipment.")
        else:
            st.info("No alerts data available.")

        # Maintenance Log
        st.subheader("🔧 Maintenance Log")
        maint_df = db_client._query(
            """
            SELECT log_id, equipment_id, maintenance_type, description,
                   performed_by, scheduled_date, completed_date, cost
              FROM maintenance_logs
             WHERE equipment_id = %s
             ORDER BY COALESCE(completed_date, scheduled_date) DESC
             LIMIT 20
            """,
            (selected_eq,),
        )
        if maint_df is not None and not maint_df.empty:
            st.dataframe(maint_df, use_container_width=True)
        else:
            st.info("No maintenance records for this equipment.")

# ===================== TAB 3: ALERT MANAGEMENT ===========================
with tab_alerts:
    alert_api_url = (
        config["data_sources"]
        .get("alert_api", {})
        .get("url", "http://alert-engine:8001")
    )
    alert_api = APIClient(alert_api_url)

    # Stats row
    alert_stats = db_client.get_alert_statistics()
    if alert_stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Alerts", alert_stats["total"])
        c2.metric("Active", alert_stats["active"])
        c3.metric("Critical", alert_stats["critical"])
        c4.metric("Warning", alert_stats["warning"])
        c5.metric("Info", alert_stats["info"])
        st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    alert_sev_filter = col_f1.selectbox(
        "Severity", ["All", "critical", "warning", "info"]
    )
    alert_status_filter = col_f2.selectbox(
        "Status", ["All", "triggered", "acknowledged", "resolved"]
    )
    alert_limit = col_f3.number_input("Limit", 10, 500, 50, step=10)

    filtered_alerts = alerts_df.copy()
    if alert_sev_filter != "All":
        filtered_alerts = filtered_alerts[
            filtered_alerts["severity"] == alert_sev_filter
        ]
    if alert_status_filter != "All":
        filtered_alerts = filtered_alerts[
            filtered_alerts["status"] == alert_status_filter
        ]
    filtered_alerts = filtered_alerts.head(alert_limit)

    if filtered_alerts.empty:
        st.info("No alerts matching filters.")
    else:
        for _, alert in filtered_alerts.iterrows():
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([2, 1.5, 4, 1.5, 2])
                ts = alert.get("timestamp") or alert.get("triggered_at")
                with c1:
                    if hasattr(ts, "strftime"):
                        st.write(ts.strftime("%Y-%m-%d %H:%M"))
                    else:
                        st.write(str(ts)[:16])
                with c2:
                    sev = alert["severity"]
                    color = severity_colors.get(sev, "#6c757d")
                    st.markdown(
                        f'<span style="background-color: {color}; color: white; '
                        f'padding: 2px 8px; border-radius: 8px; font-size: 11px;">'
                        f"{sev.upper()}</span>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.write(alert["message"])
                with c4:
                    st.write(alert.get("equipment_id", ""))
                with c5:
                    aid = alert["alert_id"]
                    if alert["status"] == "triggered":
                        if st.button("Ack", key=f"ack_{aid}"):
                            resp = alert_api.post(
                                f"/alerts/{aid}/acknowledge",
                                {"user": "dashboard_user"},
                            )
                            if resp:
                                st.success("Acknowledged!")
                                st.rerun()
                    elif alert["status"] == "acknowledged":
                        if st.button("Resolve", key=f"res_{aid}"):
                            resp = alert_api.post(f"/alerts/{aid}/resolve", {})
                            if resp:
                                st.success("Resolved!")
                                st.rerun()
                    else:
                        st.write(f"✓ {alert['status']}")

# ===================== TAB 4: MODEL PERFORMANCE ===========================
with tab_model:
    st.subheader("🧠 Model Performance")

    # Try to load model info from API
    model_data = inference_api.get("/models")
    if model_data:
        for m in model_data if isinstance(model_data, list) else []:
            with st.expander(
                f"**{m.get('name', 'Unknown')}** (v{m.get('version', '?')})"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Type", m.get("type", "?"))
                c2.metric("Loaded", "✅" if m.get("loaded") else "❌")
                c3.metric(
                    "Last Updated",
                    str(m.get("last_updated", "N/A"))[:19],
                )
                perf = m.get("performance_metrics")
                if perf:
                    st.json(perf)
    else:
        st.warning("Model data not available (inference API unreachable).")

    st.divider()

    # Model registry from DB
    st.subheader("📋 Model Registry")
    registry_df = db_client._query(
        """
        SELECT model_id, model_name, model_version, model_type,
               framework, status, rmse, mae, r2_score, created_at
          FROM model_registry
         ORDER BY created_at DESC
         LIMIT 20
        """
    )
    if registry_df is not None and not registry_df.empty:
        st.dataframe(registry_df, use_container_width=True)
    else:
        st.info("No model registry data available.")

    # Training runs
    st.subheader("📊 Training Runs")
    training_df = db_client._query(
        """
        SELECT run_id, model_id, dataset_info, training_duration_seconds,
               epochs_completed, final_train_loss, final_val_loss,
               best_metric_value, started_at, completed_at, status
          FROM training_runs
         ORDER BY started_at DESC
         LIMIT 20
        """
    )
    if training_df is not None and not training_df.empty:
        st.dataframe(training_df, use_container_width=True)
    else:
        st.info("No training run data available.")

    # Drift logs
    st.subheader("📉 Drift Detection History")
    drift_df = db_client._query(
        """
        SELECT log_id, model_id, drift_type, drift_score,
               affected_features, action_taken, detected_at
          FROM drift_logs
         ORDER BY detected_at DESC
         LIMIT 20
        """
    )
    if drift_df is not None and not drift_df.empty:
        st.dataframe(drift_df, use_container_width=True)
    else:
        st.info("No drift events recorded.")

# ===================== TAB 5: SYSTEM HEALTH ===============================
with tab_system:
    st.subheader("🖥️ System Health")

    # Service health
    api_health = inference_api.get("/health")
    if api_health:
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", api_health.get("status", "unknown"))
        c2.metric("Uptime", f"{api_health.get('uptime', 0):.0f}s")
        c3.metric("Version", api_health.get("version", "?"))

        # Dependencies
        deps = api_health.get("dependencies", {})
        if deps:
            st.subheader("Dependency Status")
            for name, info in deps.items():
                if isinstance(info, dict):
                    status = info.get("status", "unknown")
                    icon = (
                        "✅"
                        if status == "healthy"
                        else "⚠️"
                        if status == "degraded"
                        else "❌"
                    )
                    latency = info.get("latency_ms")
                    lat_str = f" ({latency:.1f}ms)" if latency else ""
                    st.write(f"{icon} **{name}**: {status}{lat_str}")

        # Models loaded
        models = api_health.get("models_loaded", {})
        if models:
            st.subheader("Model Status")
            for model_name, loaded in models.items():
                icon = "✅" if loaded else "❌"
                st.write(f"{icon} {model_name}: {'Loaded' if loaded else 'Not loaded'}")
    else:
        st.warning("Inference API not reachable.")

    st.divider()

    # Alert engine health
    st.subheader("Alert Engine")
    alert_health = APIClient(
        config["data_sources"]
        .get("alert_api", {})
        .get("url", "http://alert-engine:8001")
    ).get("/health")
    if alert_health:
        c1, c2 = st.columns(2)
        c1.metric("Status", alert_health.get("status", "unknown"))
        c2.metric("Kafka Consumer", alert_health.get("kafka_consumer", "unknown"))
        alert_deps = alert_health.get("dependencies", {})
        if alert_deps:
            for name, info in alert_deps.items():
                if isinstance(info, dict):
                    status = info.get("status", "unknown")
                    icon = "✅" if status in ("healthy", "running") else "❌"
                    st.write(f"{icon} **{name}**: {status}")
    else:
        st.warning("Alert engine not reachable.")

    st.divider()

    # Prometheus metrics summary (query Prometheus API if available)
    st.subheader("📊 Key Metrics (Prometheus)")
    prom_url = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")

    def _prom_query(query: str):
        try:
            resp = requests.get(
                f"{prom_url}/api/v1/query",
                params={"query": query},
                timeout=5,
            )
            data = resp.json()
            if data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                if results:
                    return float(results[0]["value"][1])
        except Exception:
            pass
        return None

    c1, c2, c3, c4 = st.columns(4)

    total_infer = _prom_query("sum(inference_requests_total)")
    c1.metric("Total Inferences", f"{int(total_infer)}" if total_infer else "N/A")

    avg_latency = _prom_query(
        "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))"
    )
    c2.metric("P95 Latency", f"{avg_latency * 1000:.0f}ms" if avg_latency else "N/A")

    processed = _prom_query("sum(sensor_data_processed_total)")
    c3.metric("Sensors Processed", f"{int(processed)}" if processed else "N/A")

    alerts_total = _prom_query("sum(alerts_triggered_total)")
    c4.metric("Alerts Triggered", f"{int(alerts_total)}" if alerts_total else "N/A")


# ---- Auto-refresh --------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
