"""
Integration/unit tests for the remediation work (Phases 1, 2, 3, 5, 6).

These are offline (no network, no DB, no broker) and exercise the new behavior:
  * true 15-minute aggregation and SPY publication,
  * Wilder RSI / 15m momentum correctness,
  * canonical allocator exposure caps and CVaR haircut,
  * shadow-mode + idempotent client order ids,
  * dividend guard short liability / flatten logic,
  * payload normalization (benchmark flag + headlines).
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest

import data_producer as dp
from broker import AsyncAlpacaBridge
from dividend_guard import DividendGuard
from portfolio_risk import SECTOR_MAP, canonical_allocate, cvar_haircut, classify_sector
from swarm_consumer import build_tickers_snapshot


class _FakeBar:
    def __init__(self, symbol, ts, o, h, l, c, v):
        self.symbol = symbol
        self.timestamp = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


def _reset_producer_state():
    dp._minute_buckets.clear()
    dp.history_15m.clear()
    dp._current_window_label = None


# ==========================================================================
# Phase 1: aggregation, SPY publication, indicators
# ==========================================================================
def test_producer_aggregates_true_15m_and_publishes_spy(monkeypatch):
    _reset_producer_state()
    published = []
    monkeypatch.setattr(dp, "_publish_payload", lambda p: published.append(p))

    base = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")

    async def _run():
        for i in range(5):
            ts = base + pd.Timedelta(minutes=i)
            await dp.on_bar(_FakeBar("AAPL", ts, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i))
            await dp.on_bar(_FakeBar("SPY", ts, 400 + i, 401 + i, 399 + i, 400.5 + i, 500 + i))
        # First bar of the next window triggers finalization of window A.
        nxt = base + pd.Timedelta(minutes=15)
        await dp.on_bar(_FakeBar("AAPL", nxt, 110, 111, 109, 110.5, 2000))

    asyncio.run(_run())

    assert len(published) == 1, f"Expected exactly one 15m payload, got {len(published)}"
    payload = published[0]
    assert "AAPL" in payload and "SPY" in payload, "SPY must be published for the macro guard"

    aapl = payload["AAPL"]
    assert aapl["open"] == 100.0
    assert aapl["high"] == 105.0
    assert aapl["low"] == 99.0
    assert abs(aapl["close"] - 104.5) < 1e-9
    assert aapl["volume"] == sum(1000 + i for i in range(5))
    assert aapl["timestamp"].startswith("2024-01-02T12:00")
    assert aapl["is_benchmark"] is False

    spy = payload["SPY"]
    assert spy["is_benchmark"] is True
    assert spy["close"] == pytest.approx(404.5)


def test_wilder_rsi_bounds_and_direction():
    up = pd.DataFrame({"close": np.linspace(100, 200, 60)})
    down = pd.DataFrame({"close": np.linspace(200, 100, 60)})

    rsi_up = dp.calculate_rsi14(up)
    rsi_down = dp.calculate_rsi14(down)
    assert 0.0 <= rsi_up <= 100.0 and 0.0 <= rsi_down <= 100.0
    assert rsi_up > 90.0
    assert rsi_down < 10.0


def test_momentum_15m_sign():
    rising = pd.DataFrame({"close": [100.0, 101.0]})
    falling = pd.DataFrame({"close": [101.0, 100.0]})
    assert dp.calculate_momentum_15m(rising) > 0
    assert dp.calculate_momentum_15m(falling) < 0


# ==========================================================================
# Phase 2: canonical allocator caps + CVaR
# ==========================================================================
def test_canonical_allocate_enforces_all_caps():
    tickers = [t for t in SECTOR_MAP.keys()][:60]
    convictions = {t: 1.0 for t in tickers}
    directions = {t: "BUY" for t in tickers}
    atr_map = {t: 1.0 for t in tickers}

    weights = canonical_allocate(
        convictions, atr_map, directions,
        max_position_cap=0.05,
        max_gross_exposure=1.0,
        max_net_exposure=0.60,
        max_sector_exposure=0.30,
        min_conviction=0.30,
        regime_scaler=1.0,
    )

    per_name = [abs(w) for w in weights.values() if w != 0]
    assert per_name and max(per_name) <= 0.05 + 1e-6
    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    assert gross <= 1.0 + 1e-6
    assert net <= 0.60 + 1e-6

    sector_gross: dict = {}
    for tk, w in weights.items():
        if w != 0:
            sector_gross[classify_sector(tk)] = sector_gross.get(classify_sector(tk), 0.0) + abs(w)
    for s, g in sector_gross.items():
        assert g <= 0.30 + 1e-6, f"Sector {s} breached cap with {g}"


def test_canonical_allocate_short_weights_negative():
    weights = canonical_allocate(
        {"XOM": 1.0, "CVX": 1.0}, {"XOM": 1.0, "CVX": 1.0},
        {"XOM": "SHORT", "CVX": "SHORT"},
        max_position_cap=0.05, max_net_exposure=0.60,
    )
    assert all(w <= 0 for w in weights.values())
    assert abs(sum(weights.values())) <= 0.60 + 1e-6


def test_cvar_haircut_reduces_under_tail_risk():
    benign = pd.Series(np.random.default_rng(0).normal(0.001, 0.002, 200))
    assert cvar_haircut(benign, budget=0.04) == 1.0

    tail = pd.Series([0.01] * 180 + [-0.30] * 20)
    assert cvar_haircut(tail, budget=0.04) < 1.0


# ==========================================================================
# Phase 3: shadow mode + idempotent order ids
# ==========================================================================
def test_broker_shadow_mode_never_touches_network():
    bridge = AsyncAlpacaBridge(api_key="k", secret_key="s", shadow_mode=True)
    assert bridge.is_active() is True
    res = asyncio.run(bridge.submit_order("AAPL", 10, "buy", agent_id="Agent_Alpha", window="2024-01-02T12:00"))
    assert res is not None and res["status"] == "shadow"
    assert res["qty"] == "10.0"


def test_broker_client_order_id_is_deterministic():
    a = AsyncAlpacaBridge._client_order_id("Agent_Alpha", "aapl", "buy", "w1")
    b = AsyncAlpacaBridge._client_order_id("Agent_Alpha", "AAPL", "BUY", "w1")
    c = AsyncAlpacaBridge._client_order_id("Agent_Beta", "AAPL", "buy", "w1")
    assert a == b
    assert a != c
    assert a.startswith("eq-")


# ==========================================================================
# Phase 5: dividend guard
# ==========================================================================
class _FakeDividendDB:
    def __init__(self, rows):
        self._rows = pd.DataFrame(rows)

    def fetch_upcoming_dividends(self):
        return self._rows


def test_dividend_guard_short_liability_and_flatten():
    today = date(2024, 5, 10)
    tomorrow = today + timedelta(days=1)
    db = _FakeDividendDB([
        {"ticker": "XOM", "ex_date": today, "amount_per_share": 0.95},
        {"ticker": "CVX", "ex_date": tomorrow, "amount_per_share": 1.63},
    ])
    guard = DividendGuard(db=db)

    # Shorting XOM on its ex-date is blocked; liability computed.
    assert guard.short_entry_allowed("XOM", -100, today=today) is False
    assert guard.short_dividend_liability("XOM", -100, today=today) == pytest.approx(95.0)
    # Long CVX should flatten ahead of tomorrow's ex-date.
    assert guard.should_flatten_long("CVX", today=today) is True
    # Borrow cost is positive and pro-rated.
    assert guard.borrow_cost(100, 100.0, days=1) > 0


# ==========================================================================
# Payload normalization
# ==========================================================================
def test_build_tickers_snapshot_marks_benchmark_and_coerces_fields():
    raw = {
        "AAPL": {"close": 190.0, "rsi": 55.0, "rsi14": 56.0, "atr": 2.0, "headlines": ""},
        "SPY": {"close": 500.0, "is_benchmark": True},
    }
    snap = build_tickers_snapshot(raw)
    assert snap["AAPL"]["close"] == 190.0
    assert snap["AAPL"]["rsi14"] == 56.0
    assert snap["SPY"]["is_benchmark"] is True
    # Missing fields default safely.
    assert snap["SPY"]["adv"] == 1000000.0


def test_news_fetcher_enrich_is_offline_and_populates_headlines():
    from news_fetcher import NewsFetcher

    fetcher = NewsFetcher(ttl_seconds=60)

    async def _run():
        async def fake_ticker(client, tk):
            return [f"{tk} beats estimates"]

        async def fake_macro(client):
            return ["Fed holds rates steady"]

        fetcher.fetch_ticker_headlines = fake_ticker  # type: ignore
        fetcher.fetch_macro_headlines = fake_macro  # type: ignore
        return await fetcher.enrich({"AAPL": {"close": 1.0}, "MSFT": {"close": 2.0}})

    out = asyncio.run(_run())
    assert "beats estimates" in out["AAPL"]["headlines"]
    assert "Macro:" in out["AAPL"]["headlines"]
    assert out["MSFT"]["headlines"]


# ==========================================================================
# Regression: DLQ TypeError fix, LLM fallback wiring, config centralization
# ==========================================================================
def test_dividend_guard_short_entry_allowed_accepts_optional_sizing():
    """short_entry_allowed must not raise when sizing/timestamp are omitted."""
    guard = DividendGuard(db=None)
    # No calendar -> shorting permitted; must not raise a missing-argument TypeError.
    assert guard.short_entry_allowed("AAPL") in (True, False)
    assert guard.short_entry_allowed("AAPL", shares=0.0) is True
    assert guard.short_entry_allowed("AAPL", shares=25.0) is True
    # current_time-only path is accepted and does not raise.
    assert guard.short_entry_allowed("AAPL", current_time=datetime.now(timezone.utc)) is True


def test_config_centralizes_gemini_settings_and_endpoint():
    import config

    assert config.GEMINI_MODEL, "GEMINI_MODEL must be resolved from the environment"
    assert config.GEMINI_FALLBACK_MODEL, "GEMINI_FALLBACK_MODEL must be resolved"
    assert config.settings.gemini_model == config.GEMINI_MODEL

    base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    full = config.gemini_chat_url(base)
    assert full.endswith("/chat/completions")
    # A fully-qualified endpoint must not be double-suffixed.
    assert config.gemini_chat_url(full) == full
    assert config.settings.gemini_chat_url.endswith("/chat/completions")


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "https://example.invalid"),
                response=self,
            )


def _make_fake_client(responses, seen_models):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None, **kwargs):
            seen_models.append((json or {}).get("model"))
            code, payload = responses.pop(0)
            return _FakeResponse(code, payload)

    return _FakeClient


def test_sentiment_llm_falls_back_on_500_and_404(monkeypatch):
    import sentiment_agent as sa

    for failure_code in (500, 404):
        agent = sa.NewsSentimentAgent(api_key="dummy")
        agent.model = f"primary-{failure_code}"
        agent.fallback_model = "fallback-model"
        agent.max_retries = 1
        responses = [
            (failure_code, {}),
            (200, {"choices": [{"message": {"content": '{"AAPL": 0.3}'}}]}),
        ]
        seen_models: list = []
        monkeypatch.setattr(
            sa.httpx, "AsyncClient", _make_fake_client(responses, seen_models)
        )
        out = asyncio.run(agent._call_gemini_text("prompt"))
        assert seen_models == [f"primary-{failure_code}", "fallback-model"]
        assert "AAPL" in out


def test_sentiment_llm_uses_90s_timeout_and_bearer_auth(monkeypatch):
    import sentiment_agent as sa

    agent = sa.NewsSentimentAgent(api_key="secret-token")
    captured = {}
    responses = [
        (200, {"choices": [{"message": {"content": '{"NVDA": -0.4}'}}]})
    ]

    class _Client:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None, **kwargs):
            captured["headers"] = headers
            captured["model"] = (json or {}).get("model")
            return _FakeResponse(responses[0][0], responses[0][1])

    monkeypatch.setattr(sa.httpx, "AsyncClient", _Client)
    out = asyncio.run(agent._call_gemini_text("prompt"))

    assert "NVDA" in out
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["model"] == agent.model
    timeout = captured["timeout"]
    assert float(timeout.read) >= 90.0
    assert float(timeout.connect) <= 15.0
