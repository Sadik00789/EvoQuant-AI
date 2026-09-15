"""
Market Data Producer — true 15-minute bar aggregation.

Phase 1 remediation:
  * Minute bars are accumulated into real 15-minute OHLCV buckets and only
    finalized once a window completes (no more mislabeling 1-minute data as 15m).
  * All indicators (Wilder RSI-14, MACD, ATR-14, 15m momentum, relative strength)
    are computed on the aggregated 15-minute series.
  * SPY and defensive instruments are published so the consumer's macro trend
    guard and volatility regime scaler can actually run.
  * A single immutable, timestamped payload is emitted per completed window.
  * A background backfill seeds 15m history so the guard is live immediately.
"""

from dotenv import load_dotenv

load_dotenv()

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import redis
import yfinance as yf
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar
from curl_cffi import requests as curl_requests
from psycopg_pool import ConnectionPool

from config import BENCHMARK_SYMBOLS, DEFENSIVE_SYMBOLS, Settings, settings
import metrics

TICK_INTERVAL_MINUTES = settings.tick_interval_minutes
HISTORY_WINDOW_BARS = settings.history_window_bars
REL_STRENGTH_LOOKBACK = settings.rel_strength_lookback_bars

REDIS_HOST = settings.redis_host
REDIS_PORT = settings.redis_port
broker = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password=(settings.redis_password or None))

# Full 150-stock liquid US Mega/Large-Cap universe.
UNIVERSE = [
    # Tech & Semiconductors (40)
    "NVDA", "AMD", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "AMZN", "NFLX", "INTC",
    "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "CSCO", "ACN", "IBM", "AMAT",
    "MU", "LRCX", "NOW", "PANW", "SNPS", "CDNS", "KLAC", "MCHP", "ADI", "ROP",
    "ASML", "ARM", "MRVL", "NXPI", "ON", "WDAY", "SNOW", "DDOG", "CRWD", "ZS",
    # Financials & Payments (23)
    "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "AXP", "PYPL",
    "BLK", "SCHW", "CB", "MMC", "PGR", "USB", "PNC", "TFC", "COF", "BNY",
    "MET", "AIG", "ALL",
    # Healthcare & Pharma (23)
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "LLY", "DHR",
    "BMY", "GILD", "CVS", "CI", "ISRG", "MDT", "SYK", "BSX", "ZBH", "HUM",
    "ELV", "HCA", "MCK",
    # Consumer & Retail (23)
    "PG", "HD", "DIS", "COST", "PEP", "KO", "WMT", "NKE", "MCD", "SBUX",
    "LOW", "TJX", "TGT", "EL", "BKNG", "YUM", "CMG", "MAR", "ABNB", "ORLY",
    "ROST", "KHC", "STZ",
    # Industrials & Aerospace (18)
    "HON", "UNP", "GE", "CAT", "BA", "DE", "LMT", "RTX", "ADP", "MMM",
    "UPS", "FDX", "NOC", "GD", "EMR", "ETN", "ITW", "CSX",
    # Energy, Utilities, Real Estate & Telecom (19)
    "XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO", "T", "VZ",
    "TMUS", "PLD", "AMT", "SPGI", "MDLZ", "PSX", "VLO", "OKE", "KMI",
    # Materials (4)
    "LIN", "APD", "SHW", "FCX",
]

# Symbols emitted for potential trading (benchmarks excluded from trade targets).
TRADEABLE_SYMBOLS = list(dict.fromkeys(UNIVERSE + DEFENSIVE_SYMBOLS))
# Full set maintained and published each window (includes benchmark SPY).
PUBLISH_SYMBOLS = list(dict.fromkeys(TRADEABLE_SYMBOLS + BENCHMARK_SYMBOLS))
ALL_SYMBOLS = list(PUBLISH_SYMBOLS)

# Canonical 20-ticker list retained for tests / fallback ordering.
TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "QCOM",
    "AVGO", "SPY", "QQQ", "IWM", "GLD",
    "SLV", "TLT", "COIN", "PLTR", "ARM",
]

# ---------------------------------------------------------------------------
# Aggregation state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
# symbol -> list of minute-bar dicts for the in-progress 15m window
_minute_buckets: Dict[str, List[Dict[str, float]]] = {}
# symbol -> aggregated 15m OHLCV DataFrame indexed by UTC timestamp
history_15m: Dict[str, pd.DataFrame] = {}
_current_window_label: Optional[pd.Timestamp] = None
_last_publish_ts: Optional[pd.Timestamp] = None
_last_publish_symbols: int = 0

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _ensure_history(symbol: str) -> pd.DataFrame:
    df = history_15m.get(symbol)
    if df is None:
        df = pd.DataFrame(columns=_OHLCV_COLUMNS)
        df.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        history_15m[symbol] = df
    return history_15m[symbol]


def _floor_window(ts: pd.Timestamp, minutes: int = TICK_INTERVAL_MINUTES) -> pd.Timestamp:
    """Floor a UTC timestamp to the start of its N-minute window."""
    return ts.floor(f"{minutes}min")


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _append_history(symbol: str, ts: pd.Timestamp, ohlcv: Dict[str, float]) -> None:
    """Append a finalized 15m bar, deduplicating by timestamp and trimming window."""
    df = _ensure_history(symbol)
    if ts in df.index:
        return
    row = pd.DataFrame([ohlcv], index=pd.DatetimeIndex([ts], name="timestamp"))
    df = pd.concat([df, row])
    df = df[~df.index.duplicated(keep="last")].sort_index().tail(HISTORY_WINDOW_BARS)
    history_15m[symbol] = df


# ---------------------------------------------------------------------------
# Indicator computation on the 15-minute series
# ---------------------------------------------------------------------------
def calculate_rsi14(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's RSI-14 on the 15m close series. Neutral 50.0 on insufficient data."""
    if df is None or len(df) < period + 1:
        return 50.0
    try:
        close = df["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        val = float(rsi.iloc[-1])
        if pd.isna(val):
            return 50.0
        return round(val, 2)
    except Exception:
        return 50.0


def calculate_momentum_15m(df: pd.DataFrame) -> float:
    """Percent change between the two most recent 15m closes, scaled to percent."""
    if df is None or len(df) < 2:
        return 0.0
    try:
        prev = float(df["close"].iloc[-2])
        curr = float(df["close"].iloc[-1])
        if prev <= 0:
            return 0.0
        return round(((curr - prev) / prev) * 100.0, 4)
    except Exception:
        return 0.0


def calculate_advanced_indicators(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> dict:
    """Compute RSI/MACD/ATR/relative-strength/ADV on 15m bars."""
    fallback = {"rsi": 50.0, "macd_hist": 0.0, "atr": 1.0, "rel_strength_spy": 0.0, "adv": 1000000.0}
    if df is None or len(df) < 15:
        return fallback
    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi_val = calculate_rsi14(df, 14)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()

        rel_strength = 0.0
        lookback = REL_STRENGTH_LOOKBACK
        if spy_df is not None and len(spy_df) >= lookback and len(df) >= lookback:
            stock_ret = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]
            spy_ret = (
                spy_df["close"].iloc[-1] - spy_df["close"].iloc[-lookback]
            ) / spy_df["close"].iloc[-lookback]
            rel_strength = round(float((stock_ret - spy_ret) * 100), 2)

        # ADV estimate: mean 15m volume over the retained window -> daily tradeable volume.
        # 26 x 15m bars per US session.
        avg_vol = float(df["volume"].mean()) if "volume" in df.columns else 1000.0
        adv = float(max(avg_vol, 1.0) * 26)

        def _last(series, default):
            try:
                v = float(series.iloc[-1])
                return default if pd.isna(v) else v
            except Exception:
                return default

        return {
            "rsi": rsi_val,
            "macd_hist": round(_last(macd_hist, 0.0), 4),
            "atr": round(_last(atr, 1.0), 2),
            "rel_strength_spy": rel_strength,
            "adv": max(adv, 100000.0),
        }
    except Exception:
        return fallback


def build_15m_stats(symbol: str) -> Dict[str, Any]:
    """Build a stats dict for a symbol from its aggregated 15m history."""
    empty = {
        "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
        "rsi": 50.0, "rsi14": 50.0, "momentum": 0.0, "momentum_15m": 0.0,
        "macd_hist": 0.0, "atr": 1.0, "rel_strength_spy": 0.0, "adv": 1000000.0,
    }
    df = history_15m.get(symbol)
    if df is None or df.empty:
        return empty
    try:
        last = df.iloc[-1]
        spy_data = history_15m.get("SPY")
        indicators = calculate_advanced_indicators(df, spy_data)
        rsi14 = calculate_rsi14(df, 14)
        mom = calculate_momentum_15m(df)
        return {
            "open": float(last.get("open", last.get("close", 0.0))),
            "high": float(last.get("high", 0.0)),
            "low": float(last.get("low", 0.0)),
            "close": float(last.get("close", 0.0)),
            "volume": float(last.get("volume", 0.0)),
            "rsi": indicators["rsi"],
            "rsi14": rsi14,
            "momentum": mom,
            "momentum_15m": mom,
            "macd_hist": indicators["macd_hist"],
            "atr": indicators["atr"],
            "rel_strength_spy": indicators["rel_strength_spy"],
            "adv": indicators["adv"],
            "is_benchmark": symbol in BENCHMARK_SYMBOLS,
        }
    except Exception:
        return empty


def _finalize_window(window_label: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """
    Aggregate the in-progress minute buckets into 15m bars for the given window
    and return the publish payload. Returns None if no usable data.
    """
    global _last_publish_ts, _last_publish_symbols

    if not _minute_buckets:
        return None

    finalized = 0
    for symbol, bars in list(_minute_buckets.items()):
        if not bars:
            continue
        opens = [b["open"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        vols = [b["volume"] for b in bars]
        _append_history(
            symbol,
            window_label,
            {
                "open": opens[0],
                "high": max(highs),
                "low": min(lows),
                "close": closes[-1],
                "volume": sum(vols),
            },
        )
        finalized += 1

    if finalized == 0:
        return None

    payload: Dict[str, Any] = {}
    window_iso = window_label.isoformat()
    for symbol in PUBLISH_SYMBOLS:
        df = history_15m.get(symbol)
        if df is None or df.empty:
            continue
        stats = build_15m_stats(symbol)
        if stats.get("close", 0.0) <= 0:
            continue
        stats["timestamp"] = window_iso
        stats["symbol"] = symbol
        payload[symbol] = stats

    _last_publish_ts = window_label
    _last_publish_symbols = len(payload)
    return payload


def _publish_payload(payload: Dict[str, Any]) -> None:
    try:
        message = json.dumps(payload)
        # Durable Redis Stream (primary transport consumed by the swarm group).
        try:
            broker.xadd(settings.redis_stream, {"payload": message}, maxlen=5000)
        except Exception as xe:
            print(f"⚠️ [PRODUCER] Stream XADD failed: {xe}")
        # Legacy pub/sub channel retained for backward-compatible subscribers.
        try:
            broker.publish("market_events", message)
        except Exception:
            pass
        metrics.increment("windows.published")
        metrics.set_gauge("last_window_symbols", len(payload))
        print(
            f"📡 [PRODUCER] 15m window {payload.get('SPY', {}).get('timestamp', '?')} "
            f"published for {len(payload)}/{len(PUBLISH_SYMBOLS)} symbols."
        )
    except Exception as e:
        print(f"⚠️ [PRODUCER] Publish failed: {e}")


async def on_bar(bar: Bar):
    """Accumulate minute bars and finalize the window on rollover."""
    global _current_window_label

    symbol = getattr(bar, "symbol", None)
    if not symbol or symbol not in PUBLISH_SYMBOLS:
        return

    ts = _utc(getattr(bar, "timestamp", datetime.now(timezone.utc)))
    label = _floor_window(ts, TICK_INTERVAL_MINUTES)

    payload_to_publish = None
    with _state_lock:
        if _current_window_label is None:
            _current_window_label = label
        elif label != _current_window_label:
            payload_to_publish = _finalize_window(_current_window_label)
            _minute_buckets.clear()
            _current_window_label = label

        _minute_buckets.setdefault(symbol, []).append(
            {
                "open": float(getattr(bar, "open", bar.close)),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

    if payload_to_publish:
        _publish_payload(payload_to_publish)


def _watchdog_loop():
    """Finalize a stale window if no bars arrive (e.g. at session close)."""
    global _current_window_label
    while True:
        time.sleep(20.0)
        payload_to_publish = None
        with _state_lock:
            if _current_window_label is None or not _minute_buckets:
                continue
            now = pd.Timestamp.now(tz="UTC")
            if (now - _current_window_label) > pd.Timedelta(minutes=TICK_INTERVAL_MINUTES + 1):
                payload_to_publish = _finalize_window(_current_window_label)
                _minute_buckets.clear()
                _current_window_label = None
        if payload_to_publish:
            _publish_payload(payload_to_publish)


# ---------------------------------------------------------------------------
# Historical backfill so indicators / macro guard are ready at startup
# ---------------------------------------------------------------------------
def _seed_from_dataframe(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        pdf = pd.DataFrame(
            {
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].astype(float),
            }
        )
        pdf.index = pd.DatetimeIndex(pdf.index)
        if pdf.index.tz is None:
            pdf.index = pdf.index.tz_localize("UTC")
        else:
            pdf.index = pdf.index.tz_convert("UTC")
        pdf.index.name = "timestamp"
        pdf = pdf.dropna()
        existing = _ensure_history(symbol)
        combined = pd.concat([existing, pdf])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index().tail(HISTORY_WINDOW_BARS)
        history_15m[symbol] = combined
    except Exception as e:
        print(f"⚠️ [BACKFILL] {symbol}: {e}")


def backfill_15m_history(symbols: List[str]):
    """Best-effort 15m history seed via yfinance. SPY/defensive are seeded first."""
    priority = [s for s in (BENCHMARK_SYMBOLS + DEFENSIVE_SYMBOLS) if s in symbols]
    rest = [s for s in symbols if s not in priority]
    for group in (priority, rest):
        if not group:
            continue
        try:
            data = yf.download(
                group,
                period="60d",
                interval="15m",
                group_by="ticker",
                progress=False,
                threads=True,
                auto_adjust=False,
            )
            if data is None or getattr(data, "empty", True):
                continue
            for symbol in group:
                try:
                    df = data[symbol] if len(group) > 1 else data
                    _seed_from_dataframe(symbol, df)
                except Exception:
                    continue
            print(f"✅ [BACKFILL] Seeded 15m history for {len(group)} symbols ({group[:3]}...).")
        except Exception as e:
            print(f"⚠️ [BACKFILL] Batch failed for {len(group)} symbols: {e}")


# ---------------------------------------------------------------------------
# Dividend calendar sync (fixed payment_date + refreshable)
# ---------------------------------------------------------------------------
def sync_dividend_calendar():
    POSTGRES_HOST = settings.postgres_host
    POSTGRES_PORT = settings.postgres_port
    POSTGRES_DB = settings.postgres_db
    POSTGRES_USER = settings.postgres_user
    POSTGRES_PASS = settings.postgres_password

    conninfo = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:"
        f"{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    print("📅 Syncing dividend calendar with TimescaleDB...")
    session = curl_requests.Session(impersonate="chrome120")
    try:
        with ConnectionPool(conninfo=conninfo, min_size=1, max_size=2) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    synced_count = 0
                    for tk in UNIVERSE:
                        try:
                            time.sleep(0.5)
                            ticker_obj = yf.Ticker(tk, session=session)
                            info = ticker_obj.info
                            ex_date_ts = info.get("exDividendDate") or info.get("ex_dividend_date")
                            pay_date_ts = (
                                info.get("dividendDate")
                                or info.get("dividend_date")
                                or ex_date_ts
                            )
                            div_rate = info.get("dividendRate", 0.0) or 0.0
                            if ex_date_ts and div_rate > 0:
                                ex_date_str = pd.to_datetime(ex_date_ts, unit="s").strftime("%Y-%m-%d")
                                pay_date_str = pd.to_datetime(pay_date_ts, unit="s").strftime("%Y-%m-%d")
                                quarterly_div = round(float(div_rate / 4.0), 4)
                                cur.execute(
                                    """
                                    INSERT INTO dividend_schedule
                                        (ticker, ex_date, payment_date, amount_per_share)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (ticker, ex_date) DO UPDATE
                                        SET payment_date = EXCLUDED.payment_date,
                                            amount_per_share = EXCLUDED.amount_per_share;
                                    """,
                                    (tk, ex_date_str, pay_date_str, quarterly_div),
                                )
                                synced_count += 1
                        except Exception:
                            continue
                    conn.commit()
                    print(f"✅ Dividend calendar synchronized ({synced_count} schedules).")
    except Exception as e:
        print(f"⚠️ Dividend calendar sync note (DB connection deferred): {e}")


if __name__ == "__main__":
    print(
        f"📡 Booting Alpaca WebSocket for {len(ALL_SYMBOLS)} assets "
        f"(true {TICK_INTERVAL_MINUTES}m aggregation)..."
    )

    threading.Thread(
        target=backfill_15m_history, args=(ALL_SYMBOLS,), daemon=True
    ).start()
    threading.Thread(target=sync_dividend_calendar, daemon=True).start()
    threading.Thread(target=_watchdog_loop, daemon=True).start()

    stream = StockDataStream(
        __import__("os").getenv("ALPACA_API_KEY"),
        __import__("os").getenv("ALPACA_SECRET_KEY"),
    )
    stream.subscribe_bars(on_bar, *ALL_SYMBOLS)
    stream.run()
