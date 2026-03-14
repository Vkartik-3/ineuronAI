"""
Prometheus metrics for the Stream Processor.

Provides counters, histograms, and gauges for tracking sensor data processing,
feature computation latency, and Kafka consumer lag.

Exposes metrics via a lightweight HTTP server on a dedicated port (default 8002)
so the main processing loop isn't impacted.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
)

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
SENSOR_DATA_PROCESSED_TOTAL = Counter(
    "sensor_data_processed_total",
    "Total number of sensor messages processed",
    ["data_source"],  # "cmapss" | "synthetic"
)

SENSOR_DATA_ERRORS_TOTAL = Counter(
    "sensor_data_errors_total",
    "Total number of processing errors",
)

FEATURES_PUBLISHED_TOTAL = Counter(
    "features_published_total",
    "Total number of feature sets published to Kafka",
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
FEATURE_COMPUTATION_SECONDS = Histogram(
    "feature_computation_seconds",
    "Time in seconds to compute features for a single message",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

MESSAGE_PROCESSING_SECONDS = Histogram(
    "message_processing_seconds",
    "End-to-end processing time for a single sensor message",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
KAFKA_CONSUMER_LAG = Gauge(
    "kafka_consumer_lag",
    "Estimated Kafka consumer lag (messages behind)",
)

INPUT_QUEUE_SIZE = Gauge(
    "input_queue_size",
    "Current size of the internal input message queue",
)

PROCESSOR_RUNNING = Gauge(
    "processor_running",
    "Whether the stream processor is currently running (1=yes, 0=no)",
)

MESSAGES_PER_SECOND = Gauge(
    "messages_per_second",
    "Current throughput in messages per second",
)


def start_metrics_server(port: int = 8002):
    """Start a standalone HTTP server to serve /metrics on the given port."""
    start_http_server(port)
