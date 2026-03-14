"""
Alert Manager

Orchestrates alert evaluation, notification, and tracking.
"""

import logging
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import os
import threading

from rules.alert_rules import AlertRuleEngine, Alert, AlertSeverity
from notifiers.email_notifier import EmailNotifier
from notifiers.slack_notifier import SlackNotifier
from notifiers.webhook_notifier import WebhookNotifier
from notifiers.database_notifier import DatabaseNotifier


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Central alert management system

    Coordinates rule evaluation, notification dispatch, and alert tracking.
    """

    def __init__(self, config_path: str = "alerting/config/alert_config.yaml"):
        """
        Initialize alert manager

        Args:
            config_path: Path to alert configuration file
        """
        self.config = self._load_config(config_path)
        self.rule_engine = AlertRuleEngine(self.config)
        self.notifiers = self._initialize_notifiers()
        self.alert_count = 0

        # Alert aggregation: buffer alerts per equipment within a window
        agg_cfg = self.config.get("aggregation", {})
        self._agg_enabled = agg_cfg.get("enabled", True)
        self._agg_window_sec = agg_cfg.get("window_seconds", 300)  # 5 min
        self._agg_buffer: Dict[str, List[Alert]] = defaultdict(list)
        self._agg_timestamps: Dict[str, datetime] = {}
        self._agg_lock = threading.Lock()

        # Maintenance-mode suppression
        self._maintenance_equipment: set = (
            set()
        )  # equipment_ids currently in maintenance

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _initialize_notifiers(self) -> Dict[str, Any]:
        """Initialize all notification handlers"""
        notifiers = {}

        notifications_config = self.config.get("notifications", {})

        # Email notifier
        if notifications_config.get("email", {}).get("enabled", False):
            try:
                notifiers["email"] = EmailNotifier(notifications_config["email"])
                logger.info("Email notifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize email notifier: {e}")

        # Slack notifier
        if notifications_config.get("slack", {}).get("enabled", False):
            try:
                notifiers["slack"] = SlackNotifier(notifications_config["slack"])
                logger.info("Slack notifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Slack notifier: {e}")

        # Webhook notifier
        if notifications_config.get("webhook", {}).get("enabled", False):
            try:
                notifiers["webhook"] = WebhookNotifier(notifications_config["webhook"])
                logger.info("Webhook notifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize webhook notifier: {e}")

        # Database notifier
        if notifications_config.get("database", {}).get("enabled", False):
            try:
                notifiers["database"] = DatabaseNotifier(
                    notifications_config["database"]
                )
                logger.info("Database notifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database notifier: {e}")

        return notifiers

    def process_prediction(self, prediction_data: Dict[str, Any]) -> List[Alert]:
        """
        Process prediction data and trigger alerts.

        Alerts for equipment in maintenance mode are suppressed.
        When aggregation is enabled alerts are buffered per-equipment
        and a single aggregated notification is sent per window.
        """
        equipment_id = prediction_data.get("equipment_id", "")

        # Suppression: skip equipment in maintenance mode
        if equipment_id in self._maintenance_equipment:
            logger.info(
                "Alert suppressed — equipment %s is in maintenance mode", equipment_id
            )
            return []

        try:
            # Evaluate rules
            triggered_alerts = self.rule_engine.evaluate(prediction_data)

            if triggered_alerts:
                logger.info(
                    f"Triggered {len(triggered_alerts)} alerts for {equipment_id}"
                )

                if self._agg_enabled:
                    self._buffer_alerts(equipment_id, triggered_alerts)
                else:
                    self._send_notifications(triggered_alerts)

                self.alert_count += len(triggered_alerts)

            return triggered_alerts

        except Exception as e:
            logger.error(f"Failed to process prediction: {e}")
            return []

    # -- aggregation helpers ------------------------------------------------

    def _buffer_alerts(self, equipment_id: str, alerts: List[Alert]):
        """Buffer alerts per-equipment.  Flush when the window expires."""
        with self._agg_lock:
            now = datetime.utcnow()
            self._agg_buffer[equipment_id].extend(alerts)

            # First alert for this equipment in this window?
            if equipment_id not in self._agg_timestamps:
                self._agg_timestamps[equipment_id] = now
                return  # don't flush yet

            # Check if window has expired
            elapsed = (now - self._agg_timestamps[equipment_id]).total_seconds()
            if elapsed >= self._agg_window_sec:
                self._flush_buffer(equipment_id)

    def _flush_buffer(self, equipment_id: str):
        """Send an aggregated notification and clear the buffer."""
        buffered = self._agg_buffer.pop(equipment_id, [])
        self._agg_timestamps.pop(equipment_id, None)

        if not buffered:
            return

        if len(buffered) == 1:
            self._send_notifications(buffered)
        else:
            # Pick the highest-severity alert as representative
            severity_order = {
                AlertSeverity.CRITICAL: 4,
                AlertSeverity.WARNING: 3,
                AlertSeverity.INFO: 2,
            }
            buffered.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)
            representative = buffered[0]
            representative.message = (
                f"[Aggregated {len(buffered)} alerts] {representative.message}"
            )
            self._send_notifications([representative])
            logger.info(
                "Sent aggregated alert for %s (%d alerts combined)",
                equipment_id,
                len(buffered),
            )

    def flush_all_buffers(self):
        """Flush every equipment buffer (used during shutdown)."""
        with self._agg_lock:
            for eq_id in list(self._agg_buffer.keys()):
                self._flush_buffer(eq_id)

    # -- maintenance mode ---------------------------------------------------

    def set_maintenance_mode(self, equipment_id: str, enabled: bool = True):
        """Enable/disable maintenance mode for a piece of equipment."""
        if enabled:
            self._maintenance_equipment.add(equipment_id)
            logger.info("Maintenance mode ENABLED for %s", equipment_id)
        else:
            self._maintenance_equipment.discard(equipment_id)
            logger.info("Maintenance mode DISABLED for %s", equipment_id)

    def is_in_maintenance(self, equipment_id: str) -> bool:
        return equipment_id in self._maintenance_equipment

    def _send_notifications(self, alerts: List[Alert]) -> None:
        """Send alerts through all configured notifiers"""
        for alert in alerts:
            for notifier_name, notifier in self.notifiers.items():
                try:
                    notifier.send_alert(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {notifier_name}: {e}")

    def get_active_alerts(
        self, equipment_id: Optional[str] = None, severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts

        Args:
            equipment_id: Filter by equipment
            severity: Filter by severity

        Returns:
            List of active alert dictionaries
        """
        severity_enum = AlertSeverity(severity) if severity else None
        alerts = self.rule_engine.get_active_alerts(equipment_id, severity_enum)
        return [alert.to_dict() for alert in alerts]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert identifier
            user: User acknowledging the alert

        Returns:
            True if successful
        """
        return self.rule_engine.acknowledge_alert(alert_id, user)

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful
        """
        return self.rule_engine.resolve_alert(alert_id)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics

        Returns:
            Statistics dictionary
        """
        stats = self.rule_engine.get_alert_statistics()
        stats["total_processed"] = self.alert_count
        return stats

    def test_alert(self, rule_id: str, equipment_id: str) -> Optional[Alert]:
        """
        Test an alert rule with mock data

        Args:
            rule_id: Rule to test
            equipment_id: Equipment identifier

        Returns:
            Test alert if rule triggers
        """
        # Find rule
        rule = None
        for r in self.rule_engine.rules:
            if r.rule_id == rule_id:
                rule = r
                break

        if not rule:
            logger.error(f"Rule not found: {rule_id}")
            return None

        # Create mock data based on rule condition
        mock_data = {
            "equipment_id": equipment_id,
            "rul": 5,  # Critical RUL
            "anomaly_score": 0.95,  # High anomaly
            "health_status": "critical",
            "temperature": 100,
            "vibration": 0.9,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Evaluate rule
        if rule.evaluate(mock_data):
            alert = rule.create_alert(mock_data)
            logger.info(f"Test alert created: {alert.alert_id}")
            return alert
        else:
            logger.info(f"Rule {rule_id} did not trigger with mock data")
            return None


# Global alert manager instance
alert_manager = None


def get_alert_manager(
    config_path: str = "alerting/config/alert_config.yaml",
) -> AlertManager:
    """Get or create global alert manager instance"""
    global alert_manager
    if alert_manager is None:
        alert_manager = AlertManager(config_path)
    return alert_manager
