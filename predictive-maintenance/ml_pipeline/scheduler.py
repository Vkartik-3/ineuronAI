"""
Retraining Scheduler Service.

Runs as a long-lived process that periodically:
  1. Checks for data/concept drift via DriftDetector.
  2. If drift is detected, triggers a full retraining cycle
     via RetrainingPipeline.

Uses APScheduler for cron-like execution so no external scheduler
(Airflow, crontab) is needed.

Environment variables:
    RETRAIN_CRON_HOUR      — Hour for the weekly check (default: 2)
    RETRAIN_CRON_DAY_OF_WEEK — Day of week, 0=Mon..6=Sun (default: 6)
    MLFLOW_TRACKING_URI    — MLflow server URL (default: http://mlflow:5000)
    SERVICE_NAME           — for structured logging (default: retrain-scheduler)
"""

import logging
import os
import signal
import sys
from pathlib import Path

# Try structured logging
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from shared.logging_config import configure_logging

    configure_logging(service_name="retrain-scheduler")
except ImportError:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# The retrain package lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain.retrain_pipeline import RetrainingPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
RETRAIN_CONFIG_PATH = os.environ.get(
    "RETRAIN_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "retrain" / "config" / "retrain_config.yaml"),
)
CRON_HOUR = int(os.environ.get("RETRAIN_CRON_HOUR", "2"))
CRON_DAY_OF_WEEK = os.environ.get(
    "RETRAIN_CRON_DAY_OF_WEEK", "sun"
)  # APScheduler style


def scheduled_retraining_check():
    """
    Wrapper invoked by the scheduler.

    Creates a fresh ``RetrainingPipeline`` each run so that config
    changes are picked up automatically.
    """
    logger.info("=== Scheduled retraining check starting ===")
    try:
        pipeline = RetrainingPipeline(config_path=RETRAIN_CONFIG_PATH)
        pipeline.run_scheduled_check()
        logger.info("=== Scheduled retraining check completed ===")
    except Exception:
        logger.exception("Scheduled retraining check failed")


def main():
    scheduler = BlockingScheduler()

    # Weekly drift-check + conditional retrain
    trigger = CronTrigger(
        day_of_week=CRON_DAY_OF_WEEK,
        hour=CRON_HOUR,
        minute=0,
    )
    scheduler.add_job(
        scheduled_retraining_check,
        trigger=trigger,
        id="weekly_drift_check",
        name="Weekly drift check & conditional retrain",
        max_instances=1,
        replace_existing=True,
    )
    logger.info(
        "Retraining scheduler configured — cron: day_of_week=%s, hour=%s",
        CRON_DAY_OF_WEEK,
        CRON_HOUR,
    )

    # Also run once immediately so the first check doesn't wait a week
    run_on_startup = os.environ.get("RETRAIN_RUN_ON_STARTUP", "false").lower() == "true"
    if run_on_startup:
        logger.info("Running initial retraining check on startup…")
        scheduled_retraining_check()

    # Graceful shutdown
    def _shutdown(signum, _frame):
        logger.info("Received signal %s — shutting down scheduler", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Scheduler starting — press Ctrl+C to stop")
    scheduler.start()


if __name__ == "__main__":
    main()
