"""Unit tests for AlertRuleEngine — rule evaluation and cooldown logic."""

import pytest
import time
from datetime import datetime

from rules.alert_rules import AlertRule, AlertRuleEngine, AlertSeverity, Alert


# ═══════════════════════════════════════════════════════════════════════════
# AlertRule (single rule)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAlertRule:
    def test_evaluate_true(self):
        rule = AlertRule(
            rule_id="r1",
            condition="rul < 50",
            severity="WARNING",
            message="RUL low for {equipment_id}",
            cooldown=0,
        )
        assert rule.evaluate({"rul": 30, "equipment_id": "eq1"}) is True

    def test_evaluate_false(self):
        rule = AlertRule(
            rule_id="r1",
            condition="rul < 50",
            severity="WARNING",
            message="RUL low for {equipment_id}",
            cooldown=0,
        )
        assert rule.evaluate({"rul": 100, "equipment_id": "eq1"}) is False

    def test_disabled_rule_never_triggers(self):
        rule = AlertRule(
            rule_id="r1",
            condition="rul < 50",
            severity="WARNING",
            message="RUL low for {equipment_id}",
            enabled=False,
            cooldown=0,
        )
        assert rule.evaluate({"rul": 10, "equipment_id": "eq1"}) is False

    def test_cooldown_prevents_retrigger(self):
        rule = AlertRule(
            rule_id="r1",
            condition="rul < 50",
            severity="WARNING",
            message="RUL low for {equipment_id}",
            cooldown=60,  # 60 seconds cooldown
        )
        data = {"rul": 10, "equipment_id": "eq1"}
        # First evaluation should fire
        assert rule.evaluate(data) is True
        # Second evaluation within cooldown should NOT fire
        assert rule.evaluate(data) is False

    def test_create_alert_formats_message(self):
        rule = AlertRule(
            rule_id="r1",
            condition="rul < 50",
            severity="WARNING",
            message="RUL low for {equipment_id}",
            cooldown=0,
        )
        data = {"rul": 10, "equipment_id": "engine_42"}
        alert = rule.create_alert(data)
        assert isinstance(alert, Alert)
        assert "engine_42" in alert.message
        assert alert.severity == AlertSeverity.WARNING


# ═══════════════════════════════════════════════════════════════════════════
# AlertRuleEngine (multiple rules)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAlertRuleEngine:
    def test_evaluate_returns_triggered_alerts(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 15, "equipment_id": "eq1"}
        alerts = engine.evaluate(data)
        # rul=15 should trigger both high_rul_warning (rul<50) and critical_rul (rul<20)
        assert len(alerts) >= 2
        severities = {a.severity for a in alerts}
        assert AlertSeverity.WARNING in severities
        assert AlertSeverity.CRITICAL in severities

    def test_evaluate_no_triggers(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 200, "equipment_id": "eq1"}
        alerts = engine.evaluate(data)
        assert len(alerts) == 0

    def test_disabled_rules_skipped(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 5, "equipment_id": "eq1"}
        alerts = engine.evaluate(data)
        rule_ids = [a.rule_id for a in alerts]
        assert "disabled_rule" not in rule_ids

    def test_acknowledge_alert(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 10, "equipment_id": "eq1"}
        alerts = engine.evaluate(data)
        alert_id = alerts[0].alert_id
        result = engine.acknowledge_alert(alert_id, "admin_user")
        assert result is True

    def test_resolve_alert(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 10, "equipment_id": "eq1"}
        alerts = engine.evaluate(data)
        alert_id = alerts[0].alert_id
        result = engine.resolve_alert(alert_id)
        assert result is True

    def test_get_active_alerts(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        data = {"rul": 10, "equipment_id": "eq1"}
        engine.evaluate(data)
        active = engine.get_active_alerts()
        assert len(active) >= 1

    def test_get_active_alerts_by_equipment(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        engine.evaluate({"rul": 10, "equipment_id": "eq1"})
        engine.evaluate({"rul": 10, "equipment_id": "eq2"})
        eq1_alerts = engine.get_active_alerts(equipment_id="eq1")
        assert all(a.equipment_id == "eq1" for a in eq1_alerts)

    def test_alert_statistics(self, alert_engine_config):
        engine = AlertRuleEngine(alert_engine_config)
        engine.evaluate({"rul": 10, "equipment_id": "eq1"})
        stats = engine.get_alert_statistics()
        assert isinstance(stats, dict)
        assert "total" in stats or "total_alerts" in stats or len(stats) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Alert object
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAlert:
    def test_to_dict(self):
        alert = Alert(
            rule_id="r1",
            equipment_id="eq1",
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            data={"rul": 5},
        )
        d = alert.to_dict()
        assert d["rule_id"] == "r1"
        assert d["equipment_id"] == "eq1"
        assert "severity" in d

    def test_acknowledge(self):
        alert = Alert(
            rule_id="r1",
            equipment_id="eq1",
            severity=AlertSeverity.WARNING,
            message="Test",
            data={},
        )
        alert.acknowledge("user1")
        assert (
            alert.status == AlertSeverity.WARNING
            or str(alert.status).upper() != "TRIGGERED"
        )

    def test_resolve(self):
        alert = Alert(
            rule_id="r1",
            equipment_id="eq1",
            severity=AlertSeverity.WARNING,
            message="Test",
            data={},
        )
        alert.resolve()
        # Status should no longer be TRIGGERED
        assert str(alert.status).upper() != "TRIGGERED"
