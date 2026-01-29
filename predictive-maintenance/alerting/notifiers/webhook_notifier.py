"""
Webhook Notifier

Sends alerts to external systems via HTTP webhooks.
"""

import logging
import requests
from typing import Dict, Any, List
import json


logger = logging.getLogger(__name__)


class WebhookNotifier:
    """
    Generic webhook notification handler
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize webhook notifier

        Args:
            config: Webhook configuration from alert_config.yaml
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.url = config.get("url")
        self.method = config.get("method", "POST").upper()
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 10)

    def send_alert(self, alert: Any) -> bool:
        """
        Send alert via webhook

        Args:
            alert: Alert instance

        Returns:
            True if successful
        """
        if not self.enabled:
            logger.debug("Webhook notifications disabled")
            return False

        if not self.url:
            logger.error("Webhook URL not configured")
            return False

        try:
            # Prepare payload
            payload = self._create_payload(alert)

            # Send HTTP request
            if self.method == "POST":
                response = requests.post(
                    self.url, json=payload, headers=self.headers, timeout=self.timeout
                )
            elif self.method == "PUT":
                response = requests.put(
                    self.url, json=payload, headers=self.headers, timeout=self.timeout
                )
            else:
                logger.error(f"Unsupported HTTP method: {self.method}")
                return False

            # Check response
            if 200 <= response.status_code < 300:
                logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def _create_payload(self, alert: Any) -> Dict[str, Any]:
        """Create webhook payload"""
        return {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "equipment_id": alert.equipment_id,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "status": alert.status.value,
            "data": alert.data,
            "source": "predictive_maintenance_system",
        }

    def send_bulk_alerts(self, alerts: List[Any]) -> Dict[str, int]:
        """
        Send multiple alerts via webhook

        Args:
            alerts: List of Alert instances

        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0}

        for alert in alerts:
            if self.send_alert(alert):
                results["success"] += 1
            else:
                results["failed"] += 1

        return results
