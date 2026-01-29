"""
Slack Notifier

Sends alerts to Slack channels via webhooks.
"""

import logging
import requests
from typing import Dict, Any, List
from datetime import datetime


logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Slack notification handler using webhooks
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack notifier

        Args:
            config: Slack configuration from alert_config.yaml
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.webhook_url = config.get("webhook_url")
        self.channels = config.get("channels", {})
        self.mention_users = config.get("mention_users", {})
        self.timeout = config.get("timeout", 10)

    def send_alert(self, alert: Any) -> bool:
        """
        Send alert to Slack

        Args:
            alert: Alert instance

        Returns:
            True if successful
        """
        if not self.enabled:
            logger.debug("Slack notifications disabled")
            return False

        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        try:
            # Create Slack message
            payload = self._create_slack_message(alert)

            # Send to Slack
            response = requests.post(
                self.webhook_url, json=payload, timeout=self.timeout
            )

            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(
                    f"Slack API error: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _create_slack_message(self, alert: Any) -> Dict[str, Any]:
        """Create Slack message payload"""
        # Emoji and color based on severity
        severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
        severity_colors = {
            "critical": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }

        emoji = severity_emoji.get(alert.severity.value, "📢")
        color = severity_colors.get(alert.severity.value, "#6c757d")

        # Get channel and mentions
        channel = self.channels.get(alert.severity.value)
        mentions = self.mention_users.get(alert.severity.value, [])
        mention_text = " ".join(mentions) if mentions else ""

        # Create message blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.severity.value.upper()} Alert",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Equipment:*\n{alert.equipment_id}"},
                    {"type": "mrkdwn", "text": f"*Rule:*\n{alert.rule_id}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    },
                    {"type": "mrkdwn", "text": f"*Alert ID:*\n`{alert.alert_id}`"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Message:*\n{alert.message}"},
            },
        ]

        # Add details
        if alert.data:
            details_text = self._format_details(alert.data)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Details:*\n```{details_text}```",
                    },
                }
            )

        # Add action buttons
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Acknowledge"},
                        "style": "primary",
                        "value": alert.alert_id,
                        "action_id": "acknowledge_alert",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Dashboard"},
                        "url": f"http://dashboard.example.com/equipment/{alert.equipment_id}",
                        "action_id": "view_dashboard",
                    },
                ],
            }
        )

        # Construct payload
        payload = {
            "text": f"{emoji} {alert.severity.value.upper()}: {alert.message}",
            "blocks": blocks,
            "attachments": [{"color": color, "text": mention_text}],
        }

        if channel:
            payload["channel"] = channel

        return payload

    def _format_details(self, data: Dict[str, Any]) -> str:
        """Format alert data for Slack display"""
        lines = []
        for key, value in data.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def send_bulk_alerts(self, alerts: List[Any]) -> Dict[str, int]:
        """
        Send multiple alerts to Slack

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

    def send_summary(self, alerts: List[Any]) -> bool:
        """
        Send aggregated alert summary

        Args:
            alerts: List of alerts to summarize

        Returns:
            True if successful
        """
        if not self.enabled or not alerts:
            return False

        try:
            # Group by severity
            by_severity = {"critical": [], "warning": [], "info": []}
            for alert in alerts:
                by_severity[alert.severity.value].append(alert)

            # Create summary message
            summary_text = f"*Alert Summary* ({len(alerts)} total alerts)\n\n"
            summary_text += f"🚨 Critical: {len(by_severity['critical'])}\n"
            summary_text += f"⚠️ Warning: {len(by_severity['warning'])}\n"
            summary_text += f"ℹ️ Info: {len(by_severity['info'])}\n"

            payload = {
                "text": summary_text,
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": summary_text},
                    }
                ],
            }

            response = requests.post(
                self.webhook_url, json=payload, timeout=self.timeout
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send Slack summary: {e}")
            return False
