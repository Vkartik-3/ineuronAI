"""
Feature Store Cache using Redis
"""

import redis
import json
import pickle
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FeatureCache:
    """Redis-based cache for features"""

    def __init__(self, config: Dict):
        """
        Initialize feature cache

        Args:
            config: Configuration dictionary
        """
        self.config = config
        redis_config = config.get("redis", {})

        self.host = redis_config.get("host", "localhost")
        self.port = redis_config.get("port", 6379)
        self.db = redis_config.get("db", 0)
        self.ttl = redis_config.get("ttl_seconds", 3600)

        try:
            self.client = redis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=False
            )
            self.client.ping()
            logger.info(f"Redis cache connected: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: config TTL)
        """
        if not self.client:
            return

        try:
            ttl = ttl or self.ttl
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting cache: {e}")

    def delete(self, key: str):
        """Delete key from cache"""
        if not self.client:
            return

        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")

    def clear(self):
        """Clear all cache"""
        if not self.client:
            return

        try:
            self.client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def cache_features(self, equipment_id: str, cycle: int, features: Dict):
        """
        Cache features for equipment at specific cycle

        Args:
            equipment_id: Equipment identifier
            cycle: Cycle number
            features: Feature dictionary
        """
        key = f"features:{equipment_id}:{cycle}"
        self.set(key, features)

    def get_cached_features(self, equipment_id: str, cycle: int) -> Optional[Dict]:
        """
        Get cached features

        Args:
            equipment_id: Equipment identifier
            cycle: Cycle number

        Returns:
            Feature dictionary or None
        """
        key = f"features:{equipment_id}:{cycle}"
        return self.get(key)

    def cache_dataframe(self, key: str, df: pd.DataFrame):
        """Cache entire DataFrame"""
        self.set(key, df)

    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame"""
        return self.get(key)
