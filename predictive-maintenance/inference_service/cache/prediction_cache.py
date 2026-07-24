"""
Redis Prediction Cache

Caches RUL prediction responses to serve repeated requests under 300 ms
without re-running the PyTorch MLP inference.

Cache key: SHA-256 hash of (equipment_id + serialised sensor sequence),
           ensuring unique keys per equipment × input combination.

Cache value: JSON-serialised prediction dict, stored with a configurable TTL
             (default 300 s / 5 minutes).

Gracefully degrades to pass-through when Redis is unavailable, so the
inference service remains functional even if the cache is down.

Resume context:
    "Delivered real-time engine health predictions under 300 ms by deploying
     a FastAPI service with GPU acceleration, Redis caching, and asynchronous
     processing."
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    logger.warning("redis-py not installed — prediction caching disabled")


class PredictionCache:
    """
    Redis-backed prediction cache for RUL inference results.

    Thread-safe (uses a single Redis connection with socket_keepalive).
    All operations are best-effort — exceptions are caught and logged so
    that a cache failure never breaks the inference endpoint.
    """

    _KEY_PREFIX = "rul:pred:"

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        ttl: int = 300,
        password: Optional[str] = None,
    ):
        """
        Args:
            host:     Redis hostname (default: 'redis' — Docker service name)
            port:     Redis port     (default: 6379)
            db:       Redis DB index (default: 0)
            ttl:      Cache TTL in seconds (default: 300 = 5 minutes)
            password: Optional Redis AUTH password
        """
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self._client: Optional[Any] = None

        if not _REDIS_AVAILABLE:
            logger.warning("Redis library not available — cache disabled")
            return

        try:
            self._client = _redis_lib.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=password,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
                socket_keepalive=True,
            )
            self._client.ping()
            logger.info(
                "PredictionCache connected to Redis at %s:%d (db=%d, ttl=%ds)",
                self.host, self.port, self.db, self.ttl,
            )
        except Exception as exc:
            logger.warning(
                "Redis not reachable at %s:%d — cache disabled: %s",
                self.host, self.port, exc,
            )
            self._client = None

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def make_key(
        self,
        equipment_id: str,
        sequence: List[Dict[str, float]],
        model_name: str = "",
        model_version: str = "",
    ) -> str:
        """
        Derive a deterministic cache key from the equipment ID, sensor sequence,
        active model name, and model version.

        Model name + version are included so that a hot-reload immediately
        produces different keys — stale predictions from the old model are never
        served after the new model is swapped in.

        Args:
            equipment_id:  Engine / equipment identifier
            sequence:      List of sensor reading dicts (the request payload)
            model_name:    Active model name (e.g. "mlp_rul_predictor")
            model_version: Active model version string (e.g. "v1.2.0")

        Returns:
            Redis key string, e.g. "rul:pred:abc123..."
        """
        # Canonicalise: sort list + sort each dict's keys for determinism
        canonical = json.dumps(
            {
                "equipment_id": equipment_id,
                "sequence": sequence,
                "model_name": model_name,
                "model_version": model_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"{self._KEY_PREFIX}{digest}"

    # ------------------------------------------------------------------
    # Get / set / delete
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached prediction.

        Args:
            key: Cache key (from make_key)

        Returns:
            Deserialised prediction dict, or None on cache miss / error.
        """
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Cache GET failed for key %s: %s", key, exc)
            return None

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store a prediction in the cache.

        Args:
            key:   Cache key (from make_key)
            value: Serialisable prediction dict
            ttl:   Override instance TTL (seconds)

        Returns:
            True on success, False on error.
        """
        if not self._client:
            return False
        try:
            effective_ttl = ttl if ttl is not None else self.ttl
            self._client.setex(key, effective_ttl, json.dumps(value, default=str))
            return True
        except Exception as exc:
            logger.debug("Cache SET failed for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Invalidate a single cache entry."""
        if not self._client:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as exc:
            logger.debug("Cache DELETE failed for key %s: %s", key, exc)
            return False

    def invalidate_equipment(self, equipment_id: str) -> int:
        """
        Delete all cached predictions for a given equipment ID.

        This is called after a model reload so stale predictions are not served.

        Returns:
            Number of keys deleted.
        """
        if not self._client:
            return 0
        try:
            # We don't store equipment_id in the key directly so we need to scan.
            # In production with many keys, consider maintaining a set per equipment_id.
            pattern = f"{self._KEY_PREFIX}*"
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                for k in keys:
                    try:
                        raw = self._client.get(k)
                        if raw and equipment_id in raw:
                            self._client.delete(k)
                            deleted += 1
                    except Exception:
                        pass
                if cursor == 0:
                    break
            logger.info("Invalidated %d cache entries for equipment %s", deleted, equipment_id)
            return deleted
        except Exception as exc:
            logger.debug("Cache invalidation failed for %s: %s", equipment_id, exc)
            return 0

    def flush(self) -> bool:
        """Flush ALL prediction cache entries (use with caution)."""
        if not self._client:
            return False
        try:
            pattern = f"{self._KEY_PREFIX}*"
            cursor, deleted_total = 0, 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=200)
                if keys:
                    self._client.delete(*keys)
                    deleted_total += len(keys)
                if cursor == 0:
                    break
            logger.info("Flushed %d prediction cache entries", deleted_total)
            return True
        except Exception as exc:
            logger.warning("Cache flush failed: %s", exc)
            return False

    def ping(self) -> bool:
        """Check Redis connectivity."""
        if not self._client:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PredictionCache":
        """
        Build a PredictionCache from the inference_config.yaml 'redis' block.

        Example config:
            redis:
              host: redis
              port: 6379
              db: 0
              ttl: 300
              password: null
        """
        redis_cfg = config.get("redis", {})
        return cls(
            host=os.environ.get("REDIS_HOST", redis_cfg.get("host", "redis")),
            port=int(os.environ.get("REDIS_PORT", redis_cfg.get("port", 6379))),
            db=int(redis_cfg.get("db", 0)),
            ttl=int(redis_cfg.get("ttl", 300)),
            password=os.environ.get("REDIS_PASSWORD", redis_cfg.get("password")),
        )
