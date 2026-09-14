"""
Swarm Orchestrator (Phases 1-5 integrated).

Pipeline per completed 15-minute window:
  1. Consume an immutable window payload from a Redis Stream (consumer group + ack + DLQ).
  2. Daily dividend payout/debit + dividend/short-cost guard.
  3. News-enrich the screened ticker batch, run the bull/bear/arbiter debate,
     populate the sentiment cache.
  4. Risk overlay: directional stop-loss / take-profit, post-stopout cooldowns,
     session circuit breaker.
  5. Deterministic conviction fusion per agent -> canonical portfolio-risk
     allocation (per-name, sector, gross, net, CVaR aware).
  6. Ledger update + per-agent broker reconciliation on isolated paper
     sub-accounts (SKIPPED entirely in SHADOW mode).
  7. Snapshot telemetry, leaderboard, and Darwinian culling with genome persistence.
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import math
import os
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import redis.asyncio as redis
import yfinance as yf

import metrics

from broker import BrokerRegistry
from config import BENCHMARK_SYMBOLS, settings
from dividend_guard import DividendGuard
from engine import (
    AgentSignalDecision,
    CrossAssetPortfolioManager,
    CrossAssetRiskDecision,
    DualModelTradingSwarm,
    QualitativeSignal,
    RiskParityOptimizer,
    logger,
)
from evolution_engine import AgentGenome, EvolutionarySwarmManager
from news_fetcher import NewsFetcher
from portfolio_risk import canonical_allocate, cvar_haircut
from risk_engine import AdvancedRiskEngine
from sentiment_agent import NewsSentimentAgent, select_top_20_candidates

EPOCH_TICK_THRESHOLD = settings.epoch_tick_threshold
STOP_LOSS_PCT = settings.stop_loss_pct
TAKE_PROFIT_PCT = settings.take_profit_pct
MAX_SINGLE_POS_CAP = settings.max_position_cap
COOLDOWN_BARS = settings.cooldown_bars
SESSION_DRAWDOWN_PCT = settings.session_drawdown_pct
SHADOW_MODE = settings.shadow_mode
MIN_TRADE_NOTIONAL = 50.0

risk_engine = AdvancedRiskEngine(
    target_volatility=settings.target_volatility,
    max_position_pct=MAX_SINGLE_POS_CAP,
    stop_loss_pct=STOP_LOSS_PCT,
    take_profit_pct=TAKE_PROFIT_PCT,
    cooldown_bars=COOLDOWN_BARS,
    session_drawdown_pct=SESSION_DRAWDOWN_PCT,
    cooldown_seconds=settings.cooldown_seconds,
)
sentiment_agent = NewsSentimentAgent()
news_fetcher = NewsFetcher()
broker_registry = BrokerRegistry()
dividend_guard = DividendGuard(db=None)

PERSONA_THRESHOLDS = {
    "Agent_Alpha": 0.20,
    "Agent_Beta": 0.40,
    "Agent_Gamma": 0.25,
    "Agent_Delta": 0.25,
    "Agent_Epsilon": 0.30,
}


# ==========================================================================
# State restoration
# ==========================================================================
def restore_agent_states_from_db(swarm_mgr, db):
    """Restore active population, cash, holdings, entry prices and evolved genomes."""
    try:
        engine = getattr(db, "engine", None)
        if engine is None:
            postgres_url = (
                f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
                f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
            )
            import sqlalchemy as _sa

            engine = _sa.create_engine(postgres_url)

        accounts_df = pd.read_sql(
            "SELECT agent_id, cash FROM agent_accounts WHERE cash > 0 ORDER BY updated_at DESC;",
            engine,
        )
        snapshots_df = pd.read_sql(
            "SELECT DISTINCT ON (agent_id) agent_id, cash, equity FROM agent_snapshots "
            "ORDER BY agent_id, timestamp DESC;",
            engine,
        )
        first_snapshots_df = pd.read_sql(
            "SELECT DISTINCT ON (agent_id) agent_id, equity FROM agent_snapshots "
            "ORDER BY agent_id, timestamp ASC;",
            engine,
        )
        first_snap_map = (
            dict(zip(first_snapshots_df["agent_id"], first_snapshots_df["equity"]))
            if not first_snapshots_df.empty
            else {}
        )
        holdings_df = pd.read_sql(
            "SELECT agent_id, ticker, amount, entry_price FROM agent_holdings WHERE amount != 0;",
            engine,
        )

        # Evolved genomes (guarded: mock/legacy DBs may not implement this)
        genomes: Dict[str, dict] = {}
        try:
            loaded = db.load_agent_genomes()
            if isinstance(loaded, dict):
                genomes = loaded
        except Exception:
            genomes = {}

        active_agent_ids: List[str] = []
        if not accounts_df.empty:
            active_agent_ids = accounts_df["agent_id"].tolist()
        if not snapshots_df.empty:
            active_from_snaps = snapshots_df[snapshots_df["equity"] > 0]["agent_id"].tolist()
            for ag in active_from_snaps:
                if ag not in active_agent_ids:
                    active_agent_ids.append(ag)

        existing_map = {a.agent_id: a for a in swarm_mgr.population}
        restored_pop: List[AgentGenome] = []

        for ag_id in active_agent_ids:
            cash_val = 100000.0
            if not accounts_df.empty and ag_id in accounts_df["agent_id"].values:
                cash_val = float(accounts_df[accounts_df["agent_id"] == ag_id].iloc[0]["cash"])
            elif not snapshots_df.empty and ag_id in snapshots_df["agent_id"].values:
                cash_val = float(snapshots_df[snapshots_df["agent_id"] == ag_id].iloc[0]["cash"])

            init_cap = float(first_snap_map.get(ag_id, cash_val))
            g = genomes.get(ag_id, {})

            if ag_id in existing_map:
                agent = existing_map[ag_id]
                agent.cash = cash_val
                agent.initial_capital = init_cap
            else:
                agent = AgentGenome(
                    agent_id=ag_id,
                    persona_prompt=str(g.get("persona_prompt") or "Evolved quantitative trading agent."),
                    lineage_root=str(g.get("lineage_root") or ""),
                    generation=int(g.get("generation", 1) or 1),
                    initial_capital=init_cap,
                    cash=cash_val,
                    holdings={},
                    entry_prices={},
                    equity_history=[cash_val],
                )

            # Overlay persisted evolved traits
            if g:
                try:
                    agent.persona_prompt = str(g.get("persona_prompt") or agent.persona_prompt)
                    agent.lineage_root = str(g.get("lineage_root") or agent.lineage_root)
                    agent.generation = int(g.get("generation", agent.generation) or agent.generation)
                    agent.sentiment_weight = float(g.get("sentiment_weight", agent.sentiment_weight) or agent.sentiment_weight)
                    agent.technical_weight = float(g.get("technical_weight", agent.technical_weight) or agent.technical_weight)
                    agent.stop_loss_pct = float(g.get("stop_loss_pct", agent.stop_loss_pct) or agent.stop_loss_pct)
                    agent.take_profit_pct = float(g.get("take_profit_pct", agent.take_profit_pct) or agent.take_profit_pct)
                    agent.tenure_ticks = int(g.get("tenure_ticks", agent.tenure_ticks) or 0)
                except Exception:
                    pass

            restored_pop.append(agent)

        # Backfill only missing (non-dead) slots up to the configured population.
        target_size = int(settings.population_size)
        if len(restored_pop) < target_size:
            baseline_pop = swarm_mgr._bootstrap_initial_population()
            restored_ids = {a.agent_id for a in restored_pop}
            dead_ids = set()
            if not snapshots_df.empty:
                dead_ids.update(snapshots_df[snapshots_df["equity"] <= 0]["agent_id"].tolist())
            for base_agent in baseline_pop:
                if len(restored_pop) >= target_size:
                    break
                if base_agent.agent_id not in restored_ids and base_agent.agent_id not in dead_ids:
                    restored_pop.append(base_agent)
                    restored_ids.add(base_agent.agent_id)

        swarm_mgr.population = restored_pop[:target_size]

        for agent in swarm_mgr.population:
            if not hasattr(agent, "entry_prices"):
                agent.entry_prices = {}
            if not snapshots_df.empty:
                agent_snap = snapshots_df[snapshots_df["agent_id"] == agent.agent_id]
                if not agent_snap.empty:
                    agent.cash = float(agent_snap.iloc[0]["cash"])
            if not holdings_df.empty:
                agent_pos = holdings_df[holdings_df["agent_id"] == agent.agent_id]
                if not agent_pos.empty:
                    agent.holdings = {row["ticker"]: float(row["amount"]) for _, row in agent_pos.iterrows()}
                    agent.entry_prices = {row["ticker"]: float(row["entry_price"]) for _, row in agent_pos.iterrows()}

            long_val = sum(q * agent.entry_prices.get(tk, 0.0) for tk, q in agent.holdings.items() if q > 0)
            short_liab = sum(abs(q) * agent.entry_prices.get(tk, 0.0) for tk, q in agent.holdings.items() if q < 0)
            restored_equity = round(agent.cash + long_val - short_liab, 2)
            agent.equity_history = [restored_equity]

            base_cap = agent.initial_capital if getattr(agent, "initial_capital", 0.0) > 0 else 100000.0
            pnl = ((restored_equity - base_cap) / base_cap) * 100
            db.log_snapshot(agent.agent_id, restored_equity, agent.cash, pnl)

        logger.info(
            f"✅ Restored {len(swarm_mgr.population)} agents "
            f"(genomes: {len(genomes)}, shadow={SHADOW_MODE})."
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not restore state from DB (starting with defaults): {e}")


# ==========================================================================
# Payload / screening helpers
# ==========================================================================
def build_tickers_snapshot(market_state: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize the producer payload into a consistent snapshot dict."""
    snap: Dict[str, Dict[str, Any]] = {}
    for tk, data in (market_state or {}).items():
        try:
            close = float(data.get("close", 0.0) or 0.0)
            sym = str(tk).upper()
            snap[sym] = {
                "open": float(data.get("open", close) or close),
                "high": float(data.get("high", close) or close),
                "low": float(data.get("low", close) or close),
                "close": close,
                "volume": float(data.get("volume", 0.0) or 0.0),
                "rsi": float(data.get("rsi", 50.0) or 50.0),
                "rsi14": float(data.get("rsi14", data.get("rsi", 50.0)) or 50.0),
                "momentum": float(data.get("momentum", 0.0) or 0.0),
                "momentum_15m": float(data.get("momentum_15m", data.get("momentum", 0.0)) or 0.0),
                "macd_hist": float(data.get("macd_hist", 0.0) or 0.0),
                "atr": float(data.get("atr", 1.0) or 1.0),
                "rel_strength_spy": float(data.get("rel_strength_spy", 0.0) or 0.0),
                "adv": float(data.get("adv", 1000000.0) or 1000000.0),
                "is_benchmark": bool(data.get("is_benchmark", sym in BENCHMARK_SYMBOLS)),
                "headlines": str(data.get("headlines", data.get("news", "-")) or "-")[:400],
            }
        except Exception:
            continue
    return snap


def select_top_20_for_debate(
    snapshots: Dict[str, Dict[str, Any]], top_n: int = 20, active_holdings: Any = None
) -> Dict[str, Dict[str, Any]]:
    """Retain open positions while keeping the LLM batch at `top_n`."""
    try:
        if active_holdings is None:
            return select_top_20_candidates(snapshots, set(), top_n=top_n)
        if isinstance(active_holdings, dict):
            held = {str(k) for k, v in active_holdings.items() if abs(float(v or 0.0)) > 0}
        elif isinstance(active_holdings, (set, list, tuple)):
            held = {str(s) for s in active_holdings}
        else:
            held = set()
        return select_top_20_candidates(snapshots, held, top_n=top_n)
    except Exception:
        return select_top_20_candidates(snapshots, set(), top_n=top_n)


def _persona_threshold(agent) -> float:
    base_thr = 0.30
    aid = str(getattr(agent, "agent_id", "")).lower()
    for key, thr in PERSONA_THRESHOLDS.items():
        if key.lower() in aid:
            base_thr = thr
            break
    if getattr(agent, "generation", 1) > 1:
        base_thr = min(0.45, base_thr + 0.05)
    return base_thr


def _agent_signal_map(agent, market_state: Dict[str, Dict[str, Any]], screened_set) -> Dict[str, QualitativeSignal]:
    """Deterministic cache-driven conviction map for one agent (zero LLM calls)."""
    base_thr = _persona_threshold(agent)
    try:
        sw = float(getattr(agent, "sentiment_weight", 0.55))
        tw = float(getattr(agent, "technical_weight", 0.45))
        if not (0.20 <= sw <= 0.80):
            sw = 0.55
        if not (0.20 <= tw <= 0.80):
            tw = 0.45
        s_sum = sw + tw
        if s_sum > 0:
            sw = sw / s_sum
            tw = 1.0 - sw
    except Exception:
        sw, tw = 0.55, 0.45

    signals: Dict[str, QualitativeSignal] = {}
    for ticker in market_state.keys():
        md = market_state.get(ticker, {}) or {}
        sentiment = float(sentiment_agent.cache.get_sentiment(ticker) or 0.0)
        rsi = float(md.get("rsi14", md.get("rsi", 50.0)) or 50.0)
        mom = float(md.get("momentum_15m", md.get("momentum", 0.0)) or 0.0)
        macd = float(md.get("macd_hist", 0.0) or 0.0)
        rel = float(md.get("rel_strength_spy", 0.0) or 0.0)

        tech = 0.0
        if rsi < 30:
            tech += 0.4
        elif rsi < 45:
            tech += 0.15
        elif rsi > 70:
            tech -= 0.4
        elif rsi > 60:
            tech -= 0.15
        if mom > 0.3:
            tech += 0.3
        elif mom > 0.05:
            tech += 0.1
        elif mom < -0.3:
            tech -= 0.3
        elif mom < -0.05:
            tech -= 0.1
        if macd > 0:
            tech += 0.15
        elif macd < 0:
            tech -= 0.15
        if rel > 1.0:
            tech += 0.1
        elif rel < -1.0:
            tech -= 0.1
        tech = max(-1.0, min(1.0, tech))

        is_screened = True if screened_set is None else (str(ticker).upper() in screened_set)
        combined = (sw * sentiment + tw * tech) if is_screened else tech
        conviction = round(min(1.0, abs(combined)), 4)
        pos_qty = float(agent.holdings.get(ticker, 0.0) or 0.0)

        if pos_qty > 0:
            if combined <= -base_thr and conviction > 0.2:
                action = "SELL"
            elif combined >= base_thr:
                action = "BUY"
            else:
                action = "HOLD"
                conviction = round(conviction * 0.5, 4)
        elif pos_qty < 0:
            if combined >= base_thr and conviction > 0.2:
                action = "COVER"
            elif combined <= -base_thr:
                action = "SHORT"
            else:
                action = "HOLD"
                conviction = round(conviction * 0.5, 4)
        else:
            if combined >= base_thr:
                action = "BUY"
            elif combined <= -base_thr:
                action = "SHORT"
            else:
                action = "HOLD"
                conviction = 0.0

        if action in ("BUY", "SHORT") and conviction <= 0.15:
            action = "HOLD"
            conviction = 0.0

        signals[ticker] = QualitativeSignal(ticker=ticker, action=action, conviction=float(conviction))
    return signals


def compute_deterministic_signals(
    population: List[Any],
    market_state: Dict[str, Dict[str, Any]],
    shared_thesis: Dict[str, Any],
    screened_tickers: Any = None,
) -> Dict[str, CrossAssetRiskDecision]:
    """Backward-compatible wrapper returning CrossAssetRiskDecision per agent."""
    decisions_map: Dict[str, CrossAssetRiskDecision] = {}
    screened_set = set(str(s).upper() for s in screened_tickers) if screened_tickers else None
    for agent in population:
        try:
            signals = _agent_signal_map(agent, market_state, screened_set)
            decision_input = AgentSignalDecision(
                signals=signals,
                macro_reasoning="Cache-fused news sentiment + RSI/MOM/MACD composite.",
            )
            decisions_map[agent.agent_id] = RiskParityOptimizer.optimize_allocations(decision_input, shared_thesis)
        except Exception as e:
            logger.warning(f"⚠️ Deterministic signal build failed for [{getattr(agent, 'agent_id', '?')}]: {e}")
            decisions_map[getattr(agent, "agent_id", "unknown")] = CrossAssetRiskDecision(
                decisions={}, macro_reasoning="Deterministic fallback: no signals."
            )
    return decisions_map


# ==========================================================================
# Ledger + execution
# ==========================================================================
def _apply_target_book(agent, target_weights: Dict[str, float], prices: Dict[str, float], db) -> List[tuple]:
    """Move the virtual book to the target signed weights, returning executed trades."""
    trades: List[tuple] = []
    try:
        equity = agent.calculate_equity(prices)
    except Exception:
        return trades
    if equity <= 0:
        return trades

    for tk, w in target_weights.items():
        px = prices.get(tk)
        if not px or px <= 0:
            continue
        want = (float(w) * equity) / float(px)
        have = float(agent.holdings.get(tk, 0.0) or 0.0)
        delta = want - have
        if abs(delta) < 1e-9:
            continue
        # Skip dust unless it fully closes an existing position.
        if abs(delta) * px < MIN_TRADE_NOTIONAL and not (have != 0.0 and abs(want) < 1e-9):
            continue

        # Position flip: SHORT to LONG
        if have < 0 and want > 0:
            cover_sh = abs(have)
            agent.fill_order(tk, "COVER", cover_sh, px)
            db.update_agent_cash(agent.agent_id, agent.cash)
            db.update_agent_holding(agent.agent_id, tk, agent.holdings.get(tk, 0.0), agent.entry_prices.get(tk, 0.0))
            db.log_trade(agent.agent_id, tk, "COVER", cover_sh, px, 0.0, reason="FLIP_COVER")
            trades.append((tk, "COVER", cover_sh, px))

            buy_sh = want
            buy_dollars = min(buy_sh * px, agent.available_cash)
            if buy_dollars >= MIN_TRADE_NOTIONAL:
                actual_buy_sh = buy_dollars / px
                agent.fill_order(tk, "BUY", actual_buy_sh, px)
                db.update_agent_cash(agent.agent_id, agent.cash)
                db.update_agent_holding(agent.agent_id, tk, agent.holdings.get(tk, 0.0), agent.entry_prices.get(tk, 0.0))
                db.log_trade(agent.agent_id, tk, "BUY", actual_buy_sh, px, abs(float(w)), reason="RISK_PARITY_ALLOCATION")
                trades.append((tk, "BUY", actual_buy_sh, px))
            continue

        # Position flip: LONG to SHORT
        if have > 0 and want < 0:
            sell_sh = have
            agent.fill_order(tk, "SELL", sell_sh, px)
            db.update_agent_cash(agent.agent_id, agent.cash)
            db.update_agent_holding(agent.agent_id, tk, agent.holdings.get(tk, 0.0), agent.entry_prices.get(tk, 0.0))
            db.log_trade(agent.agent_id, tk, "SELL", sell_sh, px, 0.0, reason="FLIP_SELL")
            trades.append((tk, "SELL", sell_sh, px))

            short_sh = abs(want)
            agent.fill_order(tk, "SHORT", short_sh, px)
            db.update_agent_cash(agent.agent_id, agent.cash)
            db.update_agent_holding(agent.agent_id, tk, agent.holdings.get(tk, 0.0), agent.entry_prices.get(tk, 0.0))
            db.log_trade(agent.agent_id, tk, "SHORT", short_sh, px, abs(float(w)), reason="RISK_PARITY_ALLOCATION")
            trades.append((tk, "SHORT", short_sh, px))
            continue

        if have >= 0 and delta > 0:
            action = "BUY"
        elif have > 0 and delta < 0:
            action = "SELL"
        elif have <= 0 and delta < 0:
            action = "SHORT"
        else:
            action = "COVER"

        # Constrain long allocation to available purchasing power
        if action == "BUY":
            allocated_dollars = delta * px
            max_buy_dollars = min(allocated_dollars, agent.available_cash)
            if max_buy_dollars < MIN_TRADE_NOTIONAL:
                continue
            delta = max_buy_dollars / px

        agent.fill_order(tk, action, abs(delta), px)

        db.update_agent_cash(agent.agent_id, agent.cash)
        db.update_agent_holding(agent.agent_id, tk, agent.holdings.get(tk, 0.0), agent.entry_prices.get(tk, 0.0))
        db.log_trade(agent.agent_id, tk, action, abs(delta), px, abs(float(w)), reason="RISK_PARITY_ALLOCATION")
        trades.append((tk, action, abs(delta), px))
    return trades


async def _reconcile_agent(agent, prices: Dict[str, float], window: str):
    """Align the physical paper sub-account with the virtual book."""
    try:
        desired = {
            tk: float(qty)
            for tk, qty in agent.holdings.items()
            if abs(float(qty or 0.0)) > 1e-9
        }
        bridge = broker_registry.get(agent.agent_id)
        await bridge.reconcile_agent(agent.agent_id, desired, window=window, prices=prices)
    except Exception as e:
        logger.warning(f"⚠️ Reconciliation failed for [{agent.agent_id}]: {e}")


async def liquidate_all_to_cash(
    population: List[Any], prices: Dict[str, float], market_state: Dict[str, Any], db, reason: str = "SESSION_BREAKER"
):
    """Emergency liquidation of all open exposure to cash."""
    for agent in population:
        if not hasattr(agent, "entry_prices"):
            agent.entry_prices = {}
        for tk, shares in list(agent.holdings.items()):
            if shares == 0 or tk not in prices:
                continue
            try:
                current_price = float(prices[tk])
                adv = float(market_state.get(tk, {}).get("adv", 1000000.0) or 1000000.0)
                action = "SELL" if shares > 0 else "COVER"
                exec_price = risk_engine.calculate_execution_price(current_price, abs(shares), adv, action)
                agent.fill_order(tk, action, abs(shares), exec_price)
                db.update_agent_cash(agent.agent_id, agent.cash)
                db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason=reason)
                logger.warning(f"  🛑 [{agent.agent_id}] {reason} LIQUIDATE {tk} ({action}) @ ${exec_price:.2f}")
                bad_side = "sell" if action == "SELL" else "buy"
                await broker_registry.get(agent.agent_id).submit_order(
                    tk, abs(shares), bad_side, agent.agent_id, window=reason, allow_fractional=(bad_side == "buy")
                )
            except Exception as e:
                logger.warning(f"⚠️ Liquidation failed [{agent.agent_id}] {tk}: {e}")


# ==========================================================================
# SPY macro history seed
# ==========================================================================
def _seed_spy_history() -> List[float]:
    try:
        df = yf.download("SPY", period="60d", interval="15m", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return []
        closes = df["Close"].dropna().astype(float).tolist()
        return [float(c) for c in closes[-settings.history_window_bars:]]
    except Exception as e:
        logger.debug(f"SPY history seed skipped: {e}")
        return []


# ==========================================================================
# Main consumer
# ==========================================================================
async def run_consumer():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment.")

    db = CrossAssetPortfolioManager()
    swarm_mgr = EvolutionarySwarmManager(api_key=api_key, population_size=int(settings.population_size))
    swarm = DualModelTradingSwarm(api_key=api_key)

    dividend_guard.db = db

    for agent in swarm_mgr.population:
        db.register_agent(agent.agent_id)
    restore_agent_states_from_db(swarm_mgr, db)

    state: Dict[str, Any] = {
        "tick": 0,
        "spy_prices": _seed_spy_history(),
        "last_dividend_date": "",
    }

    logger.info("🤖 EVOQUANT SWARM ONLINE (15m bars | streams | per-agent paper sub-accounts).")
    logger.info(
        f"🛡️ Risk overlay: SL -{STOP_LOSS_PCT*100:.1f}% | TP +{TAKE_PROFIT_PCT*100:.1f}% | "
        f"cap {MAX_SINGLE_POS_CAP*100:.1f}% | gross {settings.max_gross_exposure:.2f} | "
        f"net {settings.max_net_exposure:.2f} | session DD {SESSION_DRAWDOWN_PCT*100:.1f}%"
    )
    if SHADOW_MODE:
        logger.warning("👻 SHADOW MODE ACTIVE — decisions run, no broker orders will be submitted.")

    async def process_tick(market_state: Dict[str, Dict[str, Any]]):
        prices = {tk: float(d.get("close", 0.0) or 0.0) for tk, d in market_state.items()}
        tradeable = {tk: p for tk, p in prices.items() if p > 0 and tk not in BENCHMARK_SYMBOLS}
        state["tick"] += 1
        tick_counter = state["tick"]
        metrics.increment("windows.processed")
        metrics.set_gauge("last_window_tick", tick_counter)
        window = next(
            (str(v.get("timestamp")) for v in market_state.values() if isinstance(v, dict) and v.get("timestamp")),
            datetime.now(timezone.utc).isoformat(),
        )

        for agent in swarm_mgr.population:
            agent.tenure_ticks = getattr(agent, "tenure_ticks", 0) + 1
            agent.last_prices.update(prices)
            eq = agent.calculate_equity(prices)
            base_cap = agent.initial_capital if getattr(agent, "initial_capital", 0.0) > 0 else 100000.0
            pnl = ((eq - base_cap) / base_cap) * 100.0
            agent.equity = eq
            agent.pnl_pct = pnl
            if not agent.equity_history or agent.equity_history[-1] != eq:
                agent.equity_history.append(eq)
            db.log_snapshot(agent.agent_id, eq, agent.cash, pnl)

        logger.info(f"\n==================== 🔔 WINDOW #{tick_counter} ({window}) ====================")

        # --- Daily dividend processing ---
        today_str = window[:10]
        if today_str != state["last_dividend_date"]:
            try:
                db.process_daily_dividends(today_str)
                state["last_dividend_date"] = today_str
            except Exception as e:
                logger.warning(f"⚠️ Dividend processing note: {e}")

        try:
            current_timestamp = datetime.fromisoformat(window.replace("Z", "+00:00")).timestamp()
        except Exception:
            current_timestamp = _time.time()

        # --- Pre-screening holdings for orphan-free coverage ---
        pre_active_holdings = {
            str(tk).upper()
            for agent in swarm_mgr.population
            for tk, shares in (getattr(agent, "holdings", {}) or {}).items()
            if abs(float(shares or 0.0)) > 0
        }

        # --- News enrichment + batched adversarial debate ---
        tickers_snapshot = build_tickers_snapshot(market_state)
        top_20_snapshot: Dict[str, Any] = {}
        if settings.sentiment_enabled:
            try:
                top_20_snapshot = select_top_20_for_debate(tickers_snapshot, top_n=20, active_holdings=pre_active_holdings)
                if settings.headlines_enabled:
                    async with httpx.AsyncClient() as nclient:
                        top_20_snapshot = await news_fetcher.enrich(top_20_snapshot, client=nclient)
                debate_scores = await sentiment_agent.run_adversarial_batch(top_20_snapshot)
                logger.info(f"🧠 [Debate] cached {len(debate_scores)} news-aware scores.")
            except Exception as e:
                metrics.increment("debate.failures")
                logger.warning(f"⚠️ Debate failed on window #{tick_counter}: {e}")

        # --- SPY macro regime scaler ---
        if "SPY" in market_state:
            spy_close = float(market_state["SPY"].get("close", 0.0) or 0.0)
            if spy_close > 0:
                state["spy_prices"].append(spy_close)
                if len(state["spy_prices"]) > settings.history_window_bars:
                    state["spy_prices"].pop(0)
        spy_prices_series = pd.Series(state["spy_prices"]) if len(state["spy_prices"]) >= 5 else pd.Series()
        spy_returns = spy_prices_series.pct_change().dropna() if len(spy_prices_series) > 2 else pd.Series()
        if settings.regime_scaler_enabled:
            regime_scaler = risk_engine.calculate_regime_scaler(spy_returns, spy_prices=spy_prices_series)
        else:
            regime_scaler = 1.0

        # --- Phase A: directional hard risk guard ---
        for agent in swarm_mgr.population:
            if not hasattr(agent, "entry_prices"):
                agent.entry_prices = {}
            try:
                agent_sl = float(getattr(agent, "stop_loss_pct", STOP_LOSS_PCT))
                if not (0.015 <= agent_sl <= 0.045):
                    agent_sl = STOP_LOSS_PCT
            except Exception:
                agent_sl = STOP_LOSS_PCT
            try:
                agent_tp = float(getattr(agent, "take_profit_pct", TAKE_PROFIT_PCT))
                if not (0.030 <= agent_tp <= 0.090):
                    agent_tp = TAKE_PROFIT_PCT
            except Exception:
                agent_tp = TAKE_PROFIT_PCT

            for tk, shares in list(agent.holdings.items()):
                if shares == 0 or tk not in prices:
                    continue
                current_price = float(prices[tk])
                entry_price = float(agent.entry_prices.get(tk, current_price) or current_price)
                adv = float(market_state.get(tk, {}).get("adv", 1000000.0) or 1000000.0)
                direction = "SHORT" if shares < 0 else "LONG"
                exit_signal = risk_engine.check_stop_loss_take_profit(entry_price, current_price, direction, agent_sl, agent_tp)
                if exit_signal not in ("STOP_LOSS", "TAKE_PROFIT"):
                    continue

                action = "SELL" if shares > 0 else "COVER"
                exec_price = risk_engine.calculate_execution_price(current_price, abs(shares), adv, action)
                agent.fill_order(tk, action, abs(shares), exec_price)
                reason = "HARD_STOP_LOSS" if exit_signal == "STOP_LOSS" else "HARD_TAKE_PROFIT"
                db.update_agent_cash(agent.agent_id, agent.cash)
                db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason=reason)
                if exit_signal == "STOP_LOSS":
                    risk_engine.record_stopout(agent.agent_id, tk, tick_counter, COOLDOWN_BARS, current_timestamp=current_timestamp)
                    logger.warning(f"  🚨 [{agent.agent_id}] STOP-LOSS {tk} ({direction}->{action}) @ ${exec_price:.2f}")
                else:
                    logger.info(f"  🎯 [{agent.agent_id}] TAKE-PROFIT {tk} @ ${exec_price:.2f}")

                bridge = broker_registry.get(agent.agent_id)
                await bridge.submit_order(
                    tk, abs(shares), "sell" if action == "SELL" else "buy",
                    agent.agent_id, window=window, allow_fractional=(action != "SELL"),
                )

        risk_engine.prune_expired_cooldowns(tick_counter, current_timestamp=current_timestamp)

        all_active_holdings = list({
            str(tk).upper()
            for agent in swarm_mgr.population
            for tk, shares in agent.holdings.items() if shares != 0
        })
        shared_thesis = swarm.analyze_technical_state(market_state, active_holdings=all_active_holdings)

        # --- Phase B: session breaker ---
        swarm_equity_now = 0.0
        for agent in swarm_mgr.population:
            try:
                swarm_equity_now += float(
                    risk_engine.calculate_total_equity(agent.cash, agent.holdings, getattr(agent, "entry_prices", {}), prices)
                )
            except Exception:
                continue
        risk_engine.update_session_peak(swarm_equity_now, current_timestamp=current_timestamp)
        if risk_engine.check_session_drawdown(swarm_equity_now, current_timestamp=current_timestamp):
            logger.error(f"🚨 SESSION BREAKER: equity ${swarm_equity_now:,.2f} breached. Liquidating to cash.")
            await liquidate_all_to_cash(swarm_mgr.population, prices, market_state, db, reason="SESSION_BREAKER")
        trading_halted = risk_engine.is_trading_halted()

        # --- Phase C: allocate + execute per agent ---
        atr_map = {tk: float(d.get("atr", 1.0) or 1.0) for tk, d in market_state.items()}
        screened_set = set((top_20_snapshot or {}).keys())

        for agent in swarm_mgr.population:
            try:
                signals = _agent_signal_map(agent, market_state, screened_set)
            except Exception as e:
                logger.warning(f"⚠️ Signal build failed [{agent.agent_id}]: {e}")
                continue

            convictions = {tk: s.conviction for tk, s in signals.items()}
            directions = {tk: s.action for tk, s in signals.items()}

            # Dividend / short-cost adjustments.
            for tk in list(signals.keys()):
                qty = float(agent.holdings.get(tk, 0.0) or 0.0)
                # Pass the prospective sizing explicitly (keyword) so the guard
                # never depends on an implicit positional default.
                if directions.get(tk) == "SHORT" and qty >= 0 and not dividend_guard.short_entry_allowed(
                    tk, shares=abs(qty), current_time=datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
                ):
                    directions[tk] = "HOLD"
                    convictions[tk] = 0.0
                elif qty > 0 and dividend_guard.should_flatten_long(tk):
                    directions[tk] = "SELL"
                    convictions[tk] = max(convictions.get(tk, 0.0), 0.5)

            try:
                returns = pd.Series(agent.equity_history[-50:]).pct_change().dropna()
                cvar_scale = cvar_haircut(returns, budget=settings.cvar_budget, alpha=settings.cvar_alpha)
            except Exception:
                cvar_scale = 1.0

            target_weights = canonical_allocate(
                convictions=convictions,
                atr_map=atr_map,
                directions=directions,
                max_position_cap=MAX_SINGLE_POS_CAP,
                max_gross_exposure=settings.max_gross_exposure,
                max_net_exposure=settings.max_net_exposure,
                max_sector_exposure=settings.max_sector_exposure,
                min_conviction=_persona_threshold(agent),
                regime_scaler=regime_scaler,
                cvar_scale=cvar_scale,
            )

            # Update mark-to-market equity for telemetry.
            try:
                current_equity = float(agent.calculate_equity(prices))
            except Exception:
                current_equity = agent.cash
            agent.equity_history.append(current_equity)

            if trading_halted:
                logger.warning(f"  🛑 [{agent.agent_id}] Session breaker active — skipping new entries.")
                continue

            trades = _apply_target_book(agent, target_weights, prices, db)
            for tk, action, qty, px in trades:
                logger.info(f"    [{agent.agent_id}] {action} {tk} {qty:.2f}sh @ ${px:.2f}")
            if trades:
                metrics.increment("trades.applied", len(trades))

            await _reconcile_agent(agent, prices, window)
            metrics.increment("reconcile.runs")

        # --- Telemetry + leaderboard ---
        logger.info("\n🏆 --- COMPETING AGENT LEADERBOARD ---")
        sorted_swarm = sorted(swarm_mgr.population, key=lambda a: a.equity_history[-1] if a.equity_history else 0.0, reverse=True)
        for rank, agent in enumerate(sorted_swarm, 1):
            base_cap = agent.initial_capital if getattr(agent, "initial_capital", 0.0) > 0 else 100000.0
            pnl = ((agent.equity_history[-1] - base_cap) / base_cap) * 100 if base_cap else 0.0
            db.log_snapshot(agent.agent_id, agent.equity_history[-1], agent.cash, pnl)
            try:
                db.save_agent_genome(agent)
            except Exception:
                pass
            active_holdings = [
                f"{tk}: {'LONG' if shares > 0 else 'SHORT'} {abs(shares):.1f}sh"
                for tk, shares in agent.holdings.items() if shares != 0
            ]
            summary = ", ".join(active_holdings[:4]) if active_holdings else "100% Cash"
            logger.info(
                f"  #{rank} | {agent.agent_id:<28} | Equity: ${agent.equity_history[-1]:>11,.2f} "
                f"({pnl:+6.2f}%) | Cash: ${agent.cash:>11,.2f} | {summary}"
            )
        logger.info(metrics.summary_line())

        # --- Darwinian culling ---
        if tick_counter % EPOCH_TICK_THRESHOLD == 0:
            await swarm_mgr.run_culling_cycle(
                prices=prices, risk_engine=risk_engine, db=db, execution_bridge=None
            )
            for agent in swarm_mgr.population:
                try:
                    db.register_agent(agent.agent_id)
                    db.save_agent_genome(agent)
                except Exception:
                    pass

    # ---------------- Redis Streams loop ----------------
    REDIS_HOST = settings.redis_host
    REDIS_PORT = settings.redis_port
    stream = settings.redis_stream
    group = settings.redis_group
    consumer_name = settings.redis_consumer

    # Short poll interval (ms) instead of a long block: the loop wakes up often
    # enough to observe new 15-minute windows while the socket stays alive.
    POLL_BLOCK_MS = 2000
    POLL_BATCH = 10

    async def _route_to_dlq(client, msg_id, raw_fields, error):
        """Dead-letter a genuinely unrecoverable (malformed) stream message."""
        metrics.increment("dlq.routed")
        try:
            await client.xadd(
                settings.redis_dlq,
                {"error": str(error), "src_id": str(msg_id), "payload": str(raw_fields)[:4000]},
                maxlen=1000,
            )
        except Exception as dlq_err:
            logger.warning(f"⚠️ Failed to route {msg_id} to DLQ: {dlq_err}")

    async def _decode_payload(fields):
        """Extract the JSON payload string from a stream entry, bytes-safe."""
        raw = None
        if isinstance(fields, dict):
            raw = fields.get("payload")
            if raw is None:
                raw = fields.get(b"payload")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        return raw

    while True:
        try:
            # socket_timeout=None is the key fix: the default 60s read timeout
            # expired while idle-waiting between 15-minute candles, tearing the
            # connection down. Connect timeout + keepalive keep reconnects sane.
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=(settings.redis_password or None),
                decode_responses=True,
                socket_timeout=None,
                socket_connect_timeout=15.0,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            try:
                await r.xgroup_create(stream, group, id="$", mkstream=True)
            except Exception:
                pass  # BUSYGROUP: the consumer group already exists.

            logger.info(f"📥 Consuming Redis Stream '{stream}' as group '{group}' / '{consumer_name}'.")
            while True:
                resp = await r.xreadgroup(
                    group, consumer_name, {stream: ">"}, count=POLL_BATCH, block=POLL_BLOCK_MS
                )
                if not resp:
                    continue
                for _stream_name, messages in resp:
                    for msg_id, fields in messages:
                        # --- Parse stage: genuine unrecoverable failures -> DLQ ---
                        try:
                            raw = await _decode_payload(fields)
                            market_state = json.loads(raw) if raw else None
                            if not isinstance(market_state, dict) or not market_state:
                                raise ValueError("empty or non-object payload")
                        except Exception as parse_err:
                            logger.error(
                                f"❌ Malformed stream payload ({msg_id}): {parse_err}. Routing to DLQ."
                            )
                            await _route_to_dlq(r, msg_id, fields, parse_err)
                            await r.xack(stream, group, msg_id)
                            continue

                        # --- Processing stage: log + ack transient faults (no DLQ loop) ---
                        try:
                            await process_tick(market_state)
                            await r.xack(stream, group, msg_id)
                        except Exception as proc_err:
                            metrics.increment("windows.failed")
                            logger.error(
                                f"❌ Tick processing failed ({msg_id}): {proc_err}. "
                                f"Acknowledging to avoid a DLQ rejection loop."
                            )
                            await r.xack(stream, group, msg_id)
        except Exception as e:
            metrics.increment("redis.reconnects")
            logger.error(f"❌ Redis stream connection lost: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5.0)


if __name__ == "__main__":
    asyncio.run(run_consumer())
