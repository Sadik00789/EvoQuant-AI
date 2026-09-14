"""
Canonical portfolio risk allocator (Phase 2).

Single source of truth for turning qualitative signals into sized, capped
allocations. Consolidates the previously duplicated risk-parity logic
(engine.RiskParityOptimizer and AdvancedRiskEngine.calculate_risk_parity_allocations)
into one implementation used by the live consumer.

Enforces, in order:
  1. Conviction gate (|conviction| > threshold).
  2. Inverse-volatility risk parity (conviction / ATR), normalized.
  3. Per-name position cap.
  4. Regime scaler (volatility targeting / macro guard).
  5. Sector exposure cap.
  6. Aggregate gross exposure cap.
  7. Aggregate net exposure cap (directional balance).
  8. Optional portfolio CVaR budget haircut.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from engine import CrossAssetPortfolioManager

logger = logging.getLogger("PortfolioRisk")

# Ticker -> sector, covering the traded universe and defensive instruments.
SECTOR_MAP: Dict[str, str] = {}


def _register(sector: str, tickers):
    for t in tickers:
        SECTOR_MAP[t] = sector


_register("Tech", [
    "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "NFLX", "INTC", "CRM",
    "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "CSCO", "ACN", "IBM", "AMAT", "MU",
    "LRCX", "NOW", "PANW", "SNPS", "CDNS", "KLAC", "MCHP", "ADI", "ROP",
])
_register("Consumer", [
    "TSLA", "PG", "HD", "DIS", "COST", "PEP", "KO", "WMT", "NKE", "MCD", "SBUX",
    "LOW", "TJX", "TGT", "EL", "BKNG",
])
_register("Financials", [
    "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "AXP", "PYPL", "BLK", "SCHW",
    "CB", "MMC", "PGR",
])
_register("Healthcare", [
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "LLY", "DHR", "BMY",
    "GILD", "CVS", "CI", "ISRG",
])
_register("Industrials", ["HON", "UNP", "GE", "CAT", "BA", "DE", "LMT", "RTX", "ADP", "MMM"])
_register("Energy", ["XOM", "CVX", "COP", "SLB", "EOG"])
_register("Utilities", ["NEE", "DUK", "SO", "T"])
_register("Telecom", ["VZ", "TMUS"])
_register("RealEstate", ["PLD", "AMT"])
_register("Index", ["SPY", "QQQ", "IWM"])
_register("Commodities", ["GLD", "SLV"])
_register("Bonds", ["TLT"])


def classify_sector(ticker: str) -> str:
    return SECTOR_MAP.get(str(ticker).upper(), "Other")


def cvar_haircut(
    returns: pd.Series,
    budget: float = 0.04,
    alpha: float = 0.05,
) -> float:
    """
    Returns a multiplicative scale in (0, 1] such that the portfolio's
    conditional value-at-risk is brought within `budget`. No-op when within budget.
    """
    try:
        if returns is None or len(returns) < 10:
            return 1.0
        clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 10 or budget <= 0:
            return 1.0
        sorted_r = np.sort(clean.values)
        cutoff = max(1, int(np.floor(alpha * len(sorted_r))))
        cvar = float(-np.mean(sorted_r[:cutoff]))
        if cvar <= 0 or cvar <= budget:
            return 1.0
        return float(np.clip(budget / cvar, 0.25, 1.0))
    except Exception:
        return 1.0


def canonical_allocate(
    convictions: Dict[str, float],
    atr_map: Dict[str, float],
    directions: Dict[str, str],
    max_position_cap: float = 0.05,
    max_gross_exposure: float = 1.0,
    max_net_exposure: float = 0.6,
    max_sector_exposure: float = 0.30,
    min_conviction: float = 0.30,
    regime_scaler: float = 1.0,
    cvar_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Return signed target weights keyed by ticker.
    Long actions -> +weight, SHORT -> -weight, others -> 0.
    """
    if not convictions:
        return {}

    long_side = {"BUY"}
    short_side = {"SHORT"}

    raw: Dict[str, float] = {}
    for tk, conv in convictions.items():
        direction = str(directions.get(tk, "HOLD")).upper()
        if direction not in long_side and direction not in short_side:
            raw[tk] = 0.0
            continue
        try:
            c = float(conv)
        except Exception:
            c = 0.0
        if c < min_conviction:
            raw[tk] = 0.0
            continue
        atr = atr_map.get(tk, 1.0)
        try:
            atr = float(atr)
        except Exception:
            atr = 1.0
        if atr is None or np.isnan(atr) or atr <= 0:
            atr = 1.0
        raw[tk] = c / max(atr, 0.1)

    total_raw = sum(v for v in raw.values() if v > 0)
    if total_raw <= 0 or np.isnan(total_raw):
        return {tk: 0.0 for tk in convictions}

    scale = float(regime_scaler) * float(cvar_scale)

    weights: Dict[str, float] = {}
    for tk in convictions:
        r = raw.get(tk, 0.0)
        if r <= 0:
            weights[tk] = 0.0
            continue
        w = (r / total_raw) * scale
        w = min(w, max_position_cap)
        direction = str(directions.get(tk, "HOLD")).upper()
        weights[tk] = round(w if direction in long_side else -w, 6)

    # --- Sector cap ---
    sector_gross: Dict[str, float] = {}
    for tk, w in weights.items():
        if w == 0:
            continue
        sector_gross[classify_sector(tk)] = sector_gross.get(classify_sector(tk), 0.0) + abs(w)
    for sector, gross in list(sector_gross.items()):
        if gross > max_sector_exposure and gross > 0:
            factor = max_sector_exposure / gross
            for tk in weights:
                if weights[tk] != 0 and classify_sector(tk) == sector:
                    weights[tk] = round(weights[tk] * factor, 6)

    # --- Gross exposure cap ---
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross_exposure and gross > 0:
        factor = max_gross_exposure / gross
        for tk in weights:
            weights[tk] = round(weights[tk] * factor, 6)

    # --- Net exposure cap (reduce the dominant directional side) ---
    net = sum(weights.values())
    if abs(net) > max_net_exposure and net != 0:
        long_sum = sum(w for w in weights.values() if w > 0)
        short_sum = sum(-w for w in weights.values() if w < 0)
        if net > 0 and long_sum > 0:
            # Trim longs so net lands at the cap while preserving shorts.
            target_long = max_net_exposure + short_sum
            factor = max(0.0, min(1.0, target_long / long_sum))
            for tk in weights:
                if weights[tk] > 0:
                    weights[tk] = round(weights[tk] * factor, 6)
        elif net < 0 and short_sum > 0:
            target_short = max_net_exposure + long_sum
            factor = max(0.0, min(1.0, target_short / short_sum))
            for tk in weights:
                if weights[tk] < 0:
                    weights[tk] = round(weights[tk] * factor, 6)

    return weights
