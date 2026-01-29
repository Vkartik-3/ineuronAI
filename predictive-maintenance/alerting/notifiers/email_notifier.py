"""
Email Notifier

Sends email alerts via SMTP.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class EmailNotifier:
    """
    Email notification handler using SMTP
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email notifier

        Args:
            config: Email configuration from alert_config.yaml
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.smtp_host = config.get("smtp_host")
        self.smtp_port = config.get("smtp_port", 587)
        self.smtp_user = config.get("smtp_user")
        self.smtp_password = config.get("smtp_password")
        self.from_address = config.get("from_address")
        self.recipients = config.get("recipients", {})

    def send_alert(self, alert: Any) -> bool:
        """
        Send alert via email

        Args:
            alert: Alert instance

        Returns:
            True if successful
        """
        if not self.enabled:
            logger.debug("Email notifications disabled")
            return False

        try:
            # Get recipients based on severity
            recipients = self.recipients.get(alert.severity.value, [])
            if not recipients:
                logger.warning(
                    f"No recipients configured for severity: {alert.severity.value}"
                )
                return False

            # Create message
            msg = self._create_email(alert, recipients)

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(
                f"Email alert sent for {alert.alert_id} to {len(recipients)} recipients"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _create_email(self, alert: Any, recipients: List[str]) -> MIMEMultipart:
        """Create email message"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value.upper()}] Alert: {alert.equipment_id}"
        msg["From"] = self.from_address
        msg["To"] = ", ".join(recipients)

        # Plain text body
        text_body = self._create_text_body(alert)
        text_part = MIMEText(text_body, "plain")
        msg.attach(text_part)

        # HTML body
        html_body = self._create_html_body(alert)
        html_part = MIMEText(html_body, "html")
        msg.attach(html_part)

        return msg

    def _create_text_body(self, alert: Any) -> str:
        """Create plain text email body"""
        return f"""
Predictive Maintenance Alert
============================

Severity: {alert.severity.value.upper()}
Equipment: {alert.equipment_id}
Rule: {alert.rule_id}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

Message:
{alert.message}

Details:
{self._format_data(alert.data)}

---
This is an automated alert from the Predictive Maintenance System.
Alert ID: {alert.alert_id}
"""

    def _create_html_body(self, alert: Any) -> str:
        """Create HTML email body"""
        severity_colors = {
            "critical": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }
        color = severity_colors.get(alert.severity.value, "#6c757d")

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ background-color: #f8f9fa; padding: 20px; margin-top: 20px; border-radius: 5px; }}
        .detail {{ margin: 10px 0; }}
        .label {{ font-weight: bold; }}
        .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🚨 Predictive Maintenance Alert</h2>
            <h3>{alert.severity.value.upper()}</h3>
        </div>

        <div class="content">
            <div class="detail">
                <span class="label">Equipment:</span> {alert.equipment_id}
            </div>
            <div class="detail">
                <span class="label">Rule:</span> {alert.rule_id}
            </div>
            <div class="detail">
                <span class="label">Time:</span> {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
            </div>

            <div style="margin-top: 20px; padding: 15px; background-color: white; border-left: 4px solid {color};">
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
            </div>

            <div style="margin-top: 20px;">
                <p><strong>Details:</strong></p>
                <pre style="background-color: white; padding: 10px; border-radius: 3px; overflow-x: auto;">
{self._format_data(alert.data)}
                </pre>
            </div>
        </div>

        <div class="footer">
            <p>This is an automated alert from the Predictive Maintenance System.</p>
            <p>Alert ID: {alert.alert_id}</p>
        </div>
    </div>
</body>
</html>
"""

    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format alert data for display"""
        lines = []
        for key, value in data.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def send_bulk_alerts(self, alerts: List[Any]) -> Dict[str, int]:
        """
        Send multiple alerts

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
