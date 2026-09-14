"""
Unit tests for News Debate Persistence & Dashboard Data Pipeline.

Tests:
1. NewsSentimentAgent.analyze_and_debate() persists to Redis (key: market:news_reasoning:latest,
   TTL 7200s, and history list).
2. analyze_and_debate() non-blocking resilience: survives Redis or PostgreSQL connection failures.
3. db_manager.record_news_sentiment() and get_latest_news_sentiment() handle batch inserts and queries.
4. dashboard.load_news_reasoning_data() reads Redis latest/history, falls back to TimescaleDB,
   and handles completely empty states gracefully.
5. render_news_debate_section() displays graceful awaiting fallback when no debate records exist.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import db_manager
from sentiment_agent import NewsSentimentAgent


class FakeRedis:
    """In-memory Redis mock supporting get, set, lpush, lrange, ltrim."""

    def __init__(self):
        self.store = {}
        self.lists = {}
        self.ttls = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self.store[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def lpush(self, key: str, value: str):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)
        return len(self.lists[key])

    def lrange(self, key: str, start: int, end: int):
        lst = self.lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start : end + 1]

    def ltrim(self, key: str, start: int, end: int):
        lst = self.lists.get(key, [])
        if end == -1:
            self.lists[key] = lst[start:]
        else:
            self.lists[key] = lst[start : end + 1]
        return True


def test_analyze_and_debate_persists_to_redis_and_db(monkeypatch):
    """Verify analyze_and_debate stores latest debate with 7200s TTL and pushes to history."""
    fake_redis = FakeRedis()
    agent = NewsSentimentAgent(api_key="test_key")
    monkeypatch.setattr(agent, "get_redis", lambda: fake_redis)

    recorded_db_payloads = []
    monkeypatch.setattr(
        db_manager,
        "record_news_sentiment",
        lambda records: recorded_db_payloads.extend(records) or True,
    )

    # Mock stage 1, 2, and 3 LLM calls
    async def fake_bull(*args, **kwargs):
        return "AI datacenter demand accelerates revenue growth."

    async def fake_bear(*args, **kwargs):
        return "Valuation multiple compression risk under higher yields."

    async def fake_arbitrate(*args, **kwargs):
        return {"NVDA": 0.65}

    monkeypatch.setattr(agent, "_generate_bull_theses", fake_bull)
    monkeypatch.setattr(agent, "_generate_bear_theses", fake_bear)
    monkeypatch.setattr(agent, "_arbitrate_debate", fake_arbitrate)

    async def _run():
        return await agent.analyze_and_debate("NVDA")

    result = asyncio.run(_run())

    # 1. Returned payload check
    assert result["symbol"] == "NVDA"
    assert result["score"] == 0.65
    assert "AI datacenter" in result["bull_thesis"]
    assert "Valuation multiple" in result["bear_thesis"]
    assert "Arbiter evaluated" in result["arbiter_reasoning"]
    assert result["confidence"] == 0.85

    # 2. Redis latest key + TTL 7200 check
    latest_raw = fake_redis.get("market:news_reasoning:latest")
    assert latest_raw is not None
    latest_parsed = json.loads(latest_raw)
    assert latest_parsed["symbol"] == "NVDA"
    assert latest_parsed["score"] == 0.65
    assert fake_redis.ttls.get("market:news_reasoning:latest") == 7200

    # 3. Redis history list check
    history = fake_redis.lrange("market:news_reasoning:history", 0, 10)
    assert len(history) == 1
    assert json.loads(history[0])["symbol"] == "NVDA"

    # 4. DB persistence check
    assert len(recorded_db_payloads) == 1
    assert recorded_db_payloads[0]["symbol"] == "NVDA"


def test_analyze_and_debate_non_blocking_on_redis_and_db_errors(monkeypatch):
    """
    Non-blocking news debate persistence:
    If Redis or TimescaleDB experiences a temporary disconnect, log warning and return
    in-memory debate payload without crashing the consumer.
    """
    agent = NewsSentimentAgent(api_key="test_key")

    class CrashingRedis:
        def set(self, *args, **kwargs):
            raise ConnectionError("Redis connection refused")

        def lpush(self, *args, **kwargs):
            raise ConnectionError("Redis connection refused")

    monkeypatch.setattr(agent, "get_redis", lambda: CrashingRedis())

    def crashing_db(*args, **kwargs):
        raise ConnectionError("TimescaleDB connection timeout")

    monkeypatch.setattr(db_manager, "record_news_sentiment", crashing_db)

    async def _run():
        return await agent.analyze_and_debate(
            symbol="AAPL",
            bull_thesis="Bull test",
            bear_thesis="Bear test",
            arbiter_reasoning="Arbiter test",
            score=0.2,
            confidence=0.9,
        )

    result = asyncio.run(_run())
    assert result["symbol"] == "AAPL"
    assert result["score"] == 0.2
    assert result["bull_thesis"] == "Bull test"
    assert result["bear_thesis"] == "Bear test"
    assert result["arbiter_reasoning"] == "Arbiter test"


def test_db_manager_record_and_get_news_sentiment(monkeypatch):
    """Verify db_manager records debate entries and queries them into a DataFrame."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("engine.ConnectionPool", return_value=mock_pool):
        from db_manager import DatabaseManager

        dm = DatabaseManager()
        dm.pool = mock_pool

        records = [
            {
                "timestamp": "2026-09-15T00:00:00Z",
                "symbol": "SPY",
                "score": 0.45,
                "bull_thesis": "Bull test",
                "bear_thesis": "Bear test",
                "arbiter_reasoning": "Reasoning test",
                "confidence": 0.85,
            }
        ]

        ok = dm.record_news_sentiment(records)
        assert ok is True
        assert mock_cur.executemany.called
        assert mock_conn.commit.called

        # Test query
        fake_df = pd.DataFrame(records)
        monkeypatch.setattr(dm, "fetch_dataframe", lambda query, params: fake_df)
        df_out = dm.get_latest_news_sentiment(limit=5)
        assert not df_out.empty
        assert "arbiter_reasoning" in df_out.columns
        assert df_out.iloc[0]["symbol"] == "SPY"


def test_dashboard_load_news_reasoning_data_pipeline(monkeypatch):
    """
    Verify dashboard.load_news_reasoning_data():
    - Retrieves from Redis when present
    - Falls back to TimescaleDB when Redis empty
    - Gracefully handles empty data
    """
    # Mock UI libraries with passthrough decorators
    st_mock = MagicMock()
    st_mock.cache_data = lambda *a, **kw: (lambda f: f)
    st_mock.cache_resource = lambda *a, **kw: (lambda f: f)
    st_mock.fragment = lambda *a, **kw: (lambda f: f)

    sys.modules["streamlit"] = st_mock
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.express"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()

    if "dashboard" in sys.modules:
        del sys.modules["dashboard"]

    import dashboard

    # 1. From Redis
    fake_redis = FakeRedis()
    sample_record = {
        "timestamp": "2026-09-15T01:00:00Z",
        "symbol": "QQQ",
        "score": 0.35,
        "bull_thesis": "Tech surge",
        "bear_thesis": "Rates higher",
        "arbiter_reasoning": "Slight bull edge",
        "confidence": 0.80,
    }
    fake_redis.set("market:news_reasoning:latest", json.dumps(sample_record))
    monkeypatch.setattr(dashboard, "get_redis_client", lambda: fake_redis)

    results = dashboard.load_news_reasoning_data()
    assert len(results) >= 1
    assert results[0]["symbol"] == "QQQ"
    assert results[0]["bull_thesis"] == "Tech surge"

    # 2. Redis empty -> TimescaleDB fallback
    empty_redis = FakeRedis()
    monkeypatch.setattr(dashboard, "get_redis_client", lambda: empty_redis)

    db_records = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-09-15 00:30:00", tz="UTC"),
                "symbol": "MSFT",
                "score": 0.50,
                "bull_thesis": "Cloud strength",
                "bear_thesis": "Capex drag",
                "arbiter_reasoning": "Cloud growth dominates",
                "confidence": 0.90,
            }
        ]
    )
    monkeypatch.setattr(db_manager, "get_latest_news_sentiment", lambda limit=10: db_records)

    results_db = dashboard.load_news_reasoning_data()
    assert len(results_db) == 1
    assert results_db[0]["symbol"] == "MSFT"
    assert results_db[0]["bear_thesis"] == "Capex drag"

    # 3. Both empty -> Graceful empty list
    monkeypatch.setattr(db_manager, "get_latest_news_sentiment", lambda limit=10: pd.DataFrame())
    results_empty = dashboard.load_news_reasoning_data()
    assert results_empty == []


def test_dashboard_fallback_card_when_no_debates(monkeypatch):
    """Verify render_news_debate_section shows awaiting card when no debates exist."""
    st_mock = MagicMock()
    st_mock.cache_data = lambda *a, **kw: (lambda f: f)
    st_mock.cache_resource = lambda *a, **kw: (lambda f: f)
    st_mock.fragment = lambda *a, **kw: (lambda f: f)

    sys.modules["streamlit"] = st_mock
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.express"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()

    if "dashboard" in sys.modules:
        del sys.modules["dashboard"]

    import dashboard

    monkeypatch.setattr(dashboard, "load_news_reasoning_data", lambda: [])
    dashboard.render_news_debate_section()

    st_mock.info.assert_called_with("🕒 Awaiting next 15-minute LLM market debate cycle...")
