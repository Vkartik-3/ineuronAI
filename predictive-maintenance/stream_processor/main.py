"""
Stream Processor Main Entry Point
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import StreamProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
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
