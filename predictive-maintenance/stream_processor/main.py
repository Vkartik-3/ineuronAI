"""
Stream Processor Main Entry Point
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.logging_config import configure_logging
except ImportError:
    configure_logging = None

from pipeline import StreamProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    if configure_logging:
        configure_logging(service_name="stream-processor", log_level=log_level)
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("stream_processor.log"),
            ],
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Stream Processor for Predictive Maintenance"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/processor_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mock", action="store_true", help="Use mock components (no Kafka/TimescaleDB)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Stream Processor Starting")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Mock mode: {args.mock}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 60)

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        # Initialize processor
        processor = StreamProcessor(config_path=str(config_path), mock_mode=args.mock)

        # Graceful shutdown on SIGTERM / SIGINT
        def _shutdown(signum, frame):
            logger.info("Received signal %s — shutting down gracefully…", signum)
            processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # Start processing
        processor.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
