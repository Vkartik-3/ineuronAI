"""
Alert Rule Engine

Evaluates conditions and triggers alerts based on prediction data.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import re


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status"""

    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class Alert:
    """Alert data model"""

    def __init__(
        self,
        rule_id: str,
        equipment_id: str,
        severity: AlertSeverity,
        message: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        self.alert_id = f"{rule_id}_{equipment_id}_{int(datetime.utcnow().timestamp())}"
        self.rule_id = rule_id
        self.equipment_id = equipment_id
        self.severity = severity
        self.message = message
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.status = AlertStatus.TRIGGERED
        self.acknowledged_at = None
        self.acknowledged_by = None
        self.resolved_at = None

    def acknowledge(self, user: str) -> None:
        """Mark alert as acknowledged"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user

    def resolve(self) -> None:
        """Mark alert as resolved"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

    def suppress(self) -> None:
        """Suppress alert"""
        self.status = AlertStatus.SUPPRESSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "equipment_id": self.equipment_id,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class AlertRule:
    """Alert rule definition"""

    def __init__(
        self,
        rule_id: str,
        condition: str,
        severity: str,
        message: str,
        enabled: bool = True,
        cooldown: int = 300,
    ):
        self.rule_id = rule_id
        self.condition = condition
        self.severity = AlertSeverity(severity)
        self.message = message
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_triggered = {}  # equipment_id -> timestamp

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """
        Evaluate rule condition

        Args:
            data: Prediction data with equipment metrics

        Returns:
            True if condition is met
        """
        if not self.enabled:
            return False

        try:
            # Check cooldown
            equipment_id = data.get("equipment_id")
            if equipment_id in self.last_triggered:
                elapsed = (
                    datetime.utcnow() - self.last_triggered[equipment_id]
                ).total_seconds()
                if elapsed < self.cooldown:
                    return False

            # Evaluate condition
            # Replace variables in condition with actual values
            condition = self.condition
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, str):
                    condition = condition.replace(key, f"'{value}'")

            # Safely evaluate condition
            result = eval(condition, {"__builtins__": {}}, {})

            if result:
                self.last_triggered[equipment_id] = datetime.utcnow()

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate rule {self.rule_id}: {e}")
            return False

    def create_alert(self, data: Dict[str, Any]) -> Alert:
        """
        Create alert from rule

        Args:
            data: Prediction data

        Returns:
            Alert instance
        """
        # Format message with data
        message = self.message.format(**data)

        return Alert(
            rule_id=self.rule_id,
            equipment_id=data["equipment_id"],
            severity=self.severity,
            message=message,
            data=data,
        )


class AlertRuleEngine:
    """
    Alert rule engine for evaluating conditions and triggering alerts
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rule engine

        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        self.rules = self._load_rules()
        self.alert_history = []
        self.active_alerts = {}  # equipment_id -> List[Alert]

    def _load_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration"""
        rules = []

        for rule_id, rule_config in self.config.get("rules", {}).items():
            try:
                rule = AlertRule(
                    rule_id=rule_id,
                    condition=rule_config["condition"],
                    severity=rule_config["severity"],
                    message=rule_config["message"],
                    enabled=rule_config.get("enabled", True),
                    cooldown=rule_config.get("cooldown", 300),
                )
                rules.append(rule)
                logger.info(f"Loaded alert rule: {rule_id}")
            except Exception as e:
                logger.error(f"Failed to load rule {rule_id}: {e}")

        return rules

    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against prediction data

        Args:
            data: Prediction data from inference service

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self.rules:
            try:
                if rule.evaluate(data):
                    alert = rule.create_alert(data)

                    # Check suppression
                    if not self._is_suppressed(alert):
                        triggered_alerts.append(alert)
                        self._add_to_history(alert)
                        self._add_to_active(alert)

                        logger.info(
                            f"Alert triggered: {alert.rule_id} for {alert.equipment_id} "
                            f"({alert.severity.value})"
                        )
                    else:
                        logger.debug(
                            f"Alert suppressed: {alert.rule_id} for {alert.equipment_id}"
                        )

            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")

        return triggered_alerts

    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        suppression_config = self.config.get("suppression", {})

        if not suppression_config.get("enabled", False):
            return False

        # Check maintenance windows
        maintenance_windows = suppression_config.get("maintenance_windows", [])
        for window in maintenance_windows:
            if window["equipment_id"] == alert.equipment_id:
                start = datetime.fromisoformat(window["start"])
                end = datetime.fromisoformat(window["end"])
                if start <= datetime.utcnow() <= end:
                    alert.suppress()
                    return True

        # Check incident mode
        incident_mode = suppression_config.get("incident_mode", {})
        if incident_mode.get("enabled", False):
            suppress_below = incident_mode.get("suppress_below_severity", "warning")
            severity_order = ["info", "warning", "critical"]
            if severity_order.index(alert.severity.value) < severity_order.index(
                suppress_below
            ):
                alert.suppress()
                return True

        return False

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history"""
        self.alert_history.append(alert)

        # Trim history to last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def _add_to_active(self, alert: Alert) -> None:
        """Add alert to active alerts"""
        equipment_id = alert.equipment_id
        if equipment_id not in self.active_alerts:
            self.active_alerts[equipment_id] = []
        self.active_alerts[equipment_id].append(alert)

    def get_active_alerts(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get active alerts

        Args:
            equipment_id: Filter by equipment (optional)
            severity: Filter by severity (optional)

        Returns:
            List of active alerts
        """
        alerts = []

        if equipment_id:
            alerts = self.active_alerts.get(equipment_id, [])
        else:
            for eq_alerts in self.active_alerts.values():
                alerts.extend(eq_alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Filter only triggered/acknowledged alerts
        alerts = [
            a
            for a in alerts
            if a.status in [AlertStatus.TRIGGERED, AlertStatus.ACKNOWLEDGED]
        ]

        return alerts

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert identifier
            user: User acknowledging the alert

        Returns:
            True if successful
        """
        for eq_alerts in self.active_alerts.values():
            for alert in eq_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledge(user)
                    logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful
        """
        for eq_alerts in self.active_alerts.values():
            for alert in eq_alerts:
                if alert.alert_id == alert_id:
                    alert.resolve()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
        return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics

        Returns:
            Dictionary with alert metrics
        """
        total_alerts = len(self.alert_history)

        # Count by severity
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1

        # Count by status
        status_counts = {
            "triggered": 0,
            "acknowledged": 0,
            "resolved": 0,
            "suppressed": 0,
        }
        for alert in self.alert_history:
            status_counts[alert.status.value] += 1

        # Active alerts
        active_count = len(self.get_active_alerts())

        # Average acknowledgment time
        ack_times = []
        for alert in self.alert_history:
            if alert.acknowledged_at:
                ack_time = (alert.acknowledged_at - alert.timestamp).total_seconds()
                ack_times.append(ack_time)
        avg_ack_time = sum(ack_times) / len(ack_times) if ack_times else 0

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "severity_counts": severity_counts,
            "status_counts": status_counts,
            "average_acknowledgment_time_seconds": avg_ack_time,
        }
