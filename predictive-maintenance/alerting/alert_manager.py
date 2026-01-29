"""
Alert Manager

Orchestrates alert evaluation, notification, and tracking.
"""

import logging
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from .rules.alert_rules import AlertRuleEngine, Alert, AlertSeverity
from .notifiers.email_notifier import EmailNotifier
from .notifiers.slack_notifier import SlackNotifier
from .notifiers.webhook_notifier import WebhookNotifier
from .notifiers.database_notifier import DatabaseNotifier


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
        Process prediction data and trigger alerts

        Args:
            prediction_data: Prediction results from inference service
                Expected fields:
                - equipment_id: str
                - rul: float (optional)
                - anomaly_score: float (optional)
                - health_status: str (optional)
                - temperature: float (optional)
                - vibration: float (optional)
                - other sensor data...

        Returns:
            List of triggered alerts
        """
        try:
            # Evaluate rules
            triggered_alerts = self.rule_engine.evaluate(prediction_data)

            if triggered_alerts:
                logger.info(
                    f"Triggered {len(triggered_alerts)} alerts for {prediction_data.get('equipment_id')}"
                )

                # Send notifications
                self._send_notifications(triggered_alerts)

                self.alert_count += len(triggered_alerts)

            return triggered_alerts

        except Exception as e:
            logger.error(f"Failed to process prediction: {e}")
            return []

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
