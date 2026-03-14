"""
Alert Engine — main entry-point.

Runs two things concurrently:
  1. Kafka consumer (blocking loop in a background thread)
  2. FastAPI REST API on port 8001 (uvicorn in the main thread)
"""

import logging
import os
import signal
import sys
import threading

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.logging_config import configure_logging

    configure_logging(service_name="alert-engine")
except ImportError:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

import uvicorn

from alert_manager import AlertManager
from consumer import KafkaAlertConsumer
from api import create_alert_api

logger = logging.getLogger(__name__)


def main():
    config_path = os.environ.get("ALERT_CONFIG_PATH", "config/alert_config.yaml")
    logger.info("Starting alert engine — config=%s", config_path)

    manager = AlertManager(config_path=config_path)
    consumer = KafkaAlertConsumer(alert_manager=manager, config=manager.config)

    # Build FastAPI app
    app = create_alert_api(alert_manager=manager, kafka_consumer=consumer)

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down…", signum)
        consumer.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Run Kafka consumer in background daemon thread
    consumer_thread = threading.Thread(target=consumer.start, daemon=True)
    consumer_thread.start()
    logger.info("Kafka consumer thread started")

    # Run uvicorn (blocks the main thread)
    api_host = os.environ.get("ALERT_API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("ALERT_API_PORT", "8001"))
    uvicorn.run(app, host=api_host, port=api_port, log_level="info")


if __name__ == "__main__":
    main()
