"""
Centralized, environment-driven configuration for EvoQuant-AI.

Phase 0 of the remediation plan: every hard-coded cap, threshold, timeframe and
feature flag is externalized here so behavior is tunable without code changes and
consistent across producer, consumer, risk engine and execution bridge.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


# ----------------------------------------------------------------------
# Google AI Studio (Gemini / Gemma) LLM configuration.
#
# Centralized here so every caller (sentiment debate, engine debate, genome
# mutation) reads the SAME model id and endpoint. The model id is sourced
# dynamically from the environment; a fallback model is used automatically
# whenever the primary model is unroutable (HTTP 500) or not found (HTTP 404).
# ----------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
GOOGLE_API_BASE_URL: str = os.getenv(
    "GOOGLE_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)

# HTTP budget for large batched debate prompts. Connect stays short so a dead
# endpoint fails fast; read is generous enough to survive long generations.
GEMINI_TIMEOUT_SECONDS: float = _env_float("GEMINI_TIMEOUT_SECONDS", 90.0)
GEMINI_CONNECT_TIMEOUT_SECONDS: float = _env_float("GEMINI_CONNECT_TIMEOUT_SECONDS", 15.0)

# Endpoint path appended to the OpenAI-compatible base.
_GEMINI_CHAT_PATH = "chat/completions"


def gemini_chat_url(base_url: str = None) -> str:
    """Build the full chat-completions URL from the configured base.

    Tolerates users supplying either the OpenAI-compatible base
    (`.../v1beta/openai/`) or the fully-qualified endpoint.
    """
    base = (base_url or GOOGLE_API_BASE_URL or "").strip()
    if not base:
        base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    base = base.rstrip("/")
    if base.endswith(_GEMINI_CHAT_PATH):
        return base
    return f"{base}/{_GEMINI_CHAT_PATH}"


# Symbols used purely as benchmarks / macro context and never traded directly.
BENCHMARK_SYMBOLS: List[str] = ["SPY"]

# Defensive instruments available to hedger personas and the macro guard.
DEFENSIVE_SYMBOLS: List[str] = ["QQQ", "IWM", "GLD", "TLT", "SLV"]


@dataclass
class Settings:
    # --- Data / timeframe ---
    tick_interval_minutes: int = 15
    history_window_bars: int = 300          # 15m bars retained per symbol
    rel_strength_lookback_bars: int = 12    # 12 x 15m = ~3 hours
    macro_sma_bars: int = 200               # SPY 200-period SMA

    # --- Risk / position sizing ---
    max_position_cap: float = 0.05
    target_volatility: float = 0.15
    max_gross_exposure: float = 1.00        # sum(|weights|)
    max_net_exposure: float = 0.60          # |sum(signed weights)|
    max_sector_exposure: float = 0.30
    cvar_alpha: float = 0.05
    cvar_budget: float = 0.04               # max portfolio CVaR as fraction of equity

    # --- Directional hard risk overlay ---
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.050
    cooldown_bars: int = 4
    cooldown_seconds: float = 3600.0
    session_drawdown_pct: float = 0.05

    # --- Execution ---
    shadow_mode: bool = False               # compute decisions, submit no orders
    order_poll_timeout_s: float = 20.0
    fee_bps: float = 0.0                    # per-side commission in basis points
    slippage_cap: float = 0.05

    # --- Evolution ---
    population_size: int = 5
    epoch_tick_threshold: int = 20

    # --- Transport / storage ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_stream: str = "market_events_stream"
    redis_group: str = "evoquant_swarm"
    redis_consumer: str = "consumer-1"
    redis_dlq: str = "market_events_dlq"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "evoquant_db"
    postgres_user: str = "evoquant"
    postgres_password: str = "evoquant_secret_pass"

    # --- Broker ---
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # --- Agent sub-accounts: agent_id -> {"key","secret","base_url"} ---
    alpaca_subaccounts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # --- Feature flags ---
    sentiment_enabled: bool = True
    headlines_enabled: bool = True
    regime_scaler_enabled: bool = True

    # --- LLM / Google AI Studio ---
    gemini_api_key: str = ""
    gemini_model: str = "gemma-4-31b-it"
    gemini_fallback_model: str = "gemini-2.5-flash"
    google_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_timeout_seconds: float = 90.0
    gemini_connect_timeout_seconds: float = 15.0

    @property
    def universe_is_configured(self) -> bool:
        return True

    @classmethod
    def from_env(cls) -> "Settings":
        subaccounts: Dict[str, Dict[str, str]] = {}
        raw = os.getenv("ALPACA_SUBACCOUNTS", "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    for agent_id, creds in parsed.items():
                        if isinstance(creds, dict):
                            subaccounts[str(agent_id)] = {
                                "key": str(creds.get("key", "")),
                                "secret": str(creds.get("secret", "")),
                                "base_url": str(creds.get("base_url", "")),
                            }
            except (ValueError, TypeError):
                subaccounts = {}

        return cls(
            tick_interval_minutes=_env_int("TICK_INTERVAL_MINUTES", 15),
            macro_sma_bars=_env_int("MACRO_SMA_BARS", 200),
            max_position_cap=_env_float("MAX_POSITION_CAP", 0.05),
            target_volatility=_env_float("TARGET_VOLATILITY", 0.15),
            max_gross_exposure=_env_float("MAX_GROSS_EXPOSURE", 1.00),
            max_net_exposure=_env_float("MAX_NET_EXPOSURE", 0.60),
            max_sector_exposure=_env_float("MAX_SECTOR_EXPOSURE", 0.30),
            cvar_alpha=_env_float("CVAR_ALPHA", 0.05),
            cvar_budget=_env_float("CVAR_BUDGET", 0.04),
            stop_loss_pct=_env_float("STOP_LOSS_PCT", 0.025),
            take_profit_pct=_env_float("TAKE_PROFIT_PCT", 0.050),
            cooldown_bars=_env_int("COOLDOWN_BARS", 4),
            cooldown_seconds=_env_float("COOLDOWN_SECONDS", 3600.0),
            session_drawdown_pct=_env_float("SESSION_DRAWDOWN_PCT", 0.05),
            shadow_mode=_env_bool("SHADOW_MODE", False),
            fee_bps=_env_float("FEE_BPS", 0.0),
            population_size=_env_int("POPULATION_SIZE", 5),
            epoch_tick_threshold=_env_int("EPOCH_TICK_THRESHOLD", 20),
            redis_host=_env_str("REDIS_HOST", "localhost"),
            redis_port=_env_int("REDIS_PORT", 6379),
            redis_password=_env_str("REDIS_PASSWORD", ""),
            postgres_host=_env_str("POSTGRES_HOST", "localhost"),
            postgres_port=_env_int("POSTGRES_PORT", 5432),
            postgres_db=_env_str("POSTGRES_DB", "evoquant_db"),
            postgres_user=_env_str("POSTGRES_USER", "evoquant"),
            postgres_password=_env_str("POSTGRES_PASSWORD", "evoquant_secret_pass"),
            alpaca_base_url=_env_str("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            alpaca_subaccounts=subaccounts,
            sentiment_enabled=_env_bool("SENTIMENT_ENABLED", True),
            headlines_enabled=_env_bool("HEADLINES_ENABLED", True),
            regime_scaler_enabled=_env_bool("REGIME_SCALER_ENABLED", True),
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=_env_str("GEMINI_MODEL", GEMINI_MODEL),
            gemini_fallback_model=_env_str("GEMINI_FALLBACK_MODEL", GEMINI_FALLBACK_MODEL),
            google_api_base_url=_env_str("GOOGLE_API_BASE_URL", GOOGLE_API_BASE_URL),
            gemini_timeout_seconds=_env_float("GEMINI_TIMEOUT_SECONDS", GEMINI_TIMEOUT_SECONDS),
            gemini_connect_timeout_seconds=_env_float(
                "GEMINI_CONNECT_TIMEOUT_SECONDS", GEMINI_CONNECT_TIMEOUT_SECONDS
            ),
        )

    @property
    def gemini_chat_url(self) -> str:
        return gemini_chat_url(self.google_api_base_url)


settings = Settings.from_env()
