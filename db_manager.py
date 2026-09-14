"""
TimescaleDB & PostgreSQL Database Manager for EvoQuant-AI.

Consolidates database operations, schema initialization, hypertable indexing,
news reasoning persistence, and historical analytics queries.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
from engine import CrossAssetPortfolioManager, PostgresPortfolioManager, logger

__all__ = [
    "DatabaseManager",
    "PostgresPortfolioManager",
    "CrossAssetPortfolioManager",
    "record_news_sentiment",
    "get_latest_news_sentiment",
    "logger",
]


class DatabaseManager(CrossAssetPortfolioManager):
    """
    Primary database manager inheriting canonical connection pooling,
    Darwinian persistence, and trade audit logging from CrossAssetPortfolioManager.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_news_sentiment(self, records: List[Dict[str, Any]]) -> bool:
        """
        Batched insert of structured LLM debate reasoning and sentiment into news_sentiment_log.
        Safe execution with graceful fallback on failure.
        """
        if not records:
            return True
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    params = [
                        (
                            r.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            str(r.get("symbol", "SPY")),
                            float(r.get("score", 0.0) or 0.0),
                            str(r.get("bull_thesis", "")),
                            str(r.get("bear_thesis", "")),
                            str(r.get("arbiter_reasoning", "")),
                            float(r.get("confidence", 1.0) or 1.0),
                        )
                        for r in records
                    ]
                    cur.executemany("""
                        INSERT INTO news_sentiment_log 
                        (timestamp, symbol, score, bull_thesis, bear_thesis, arbiter_reasoning, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, params)
                    conn.commit()
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to record news sentiment to TimescaleDB: {e}")
            return False

    def get_latest_news_sentiment(self, limit: int = 10) -> pd.DataFrame:
        """
        Fetch latest news debate entries ordered by timestamp descending.
        """
        try:
            query = """
                SELECT timestamp, symbol, score, bull_thesis, bear_thesis, arbiter_reasoning, confidence
                FROM news_sentiment_log
                ORDER BY timestamp DESC
                LIMIT %s;
            """
            df = self.fetch_dataframe(query, (limit,))
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.warning(f"⚠️ Failed to query news sentiment from TimescaleDB: {e}")
            return pd.DataFrame()


_default_db: Optional[DatabaseManager] = None


def get_db() -> Optional[DatabaseManager]:
    """Singleton/lazy accessor for module-level database interactions."""
    global _default_db
    if _default_db is None:
        try:
            _default_db = DatabaseManager()
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize default DatabaseManager: {e}")
            return None
    return _default_db


def record_news_sentiment(records: List[Dict[str, Any]]) -> bool:
    """Module-level helper to record news sentiment records."""
    try:
        db = get_db()
        if db is None:
            return False
        return db.record_news_sentiment(records)
    except Exception as e:
        logger.warning(f"⚠️ db_manager.record_news_sentiment fallback: {e}")
        return False


def get_latest_news_sentiment(limit: int = 10) -> pd.DataFrame:
    """Module-level helper to fetch latest news sentiment DataFrame."""
    try:
        db = get_db()
        if db is None:
            return pd.DataFrame()
        return db.get_latest_news_sentiment(limit)
    except Exception as e:
        logger.warning(f"⚠️ db_manager.get_latest_news_sentiment fallback: {e}")
        return pd.DataFrame()
