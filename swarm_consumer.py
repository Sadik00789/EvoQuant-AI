import os
import json
import math
import asyncio
import httpx
import sqlalchemy
import redis.asyncio as redis
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv

from engine import (
    DualModelTradingSwarm,
    CrossAssetPortfolioManager,
    AlpacaExecutionBridge,
    RiskParityOptimizer,
    QualitativeSignal,
    AgentSignalDecision,
    CrossAssetRiskDecision,
    logger,
)
from evolution_engine import EvolutionarySwarmManager, AgentGenome
from risk_engine import AdvancedRiskEngine
from sentiment_agent import NewsSentimentAgent

load_dotenv()

EPOCH_TICK_THRESHOLD = 20  # Culling evaluation every 20 ticks (~5 hours)
STOP_LOSS_PCT = 0.025      # Hard stop loss at 2.5% (directional)
TAKE_PROFIT_PCT = 0.05     # Hard take profit at 5.0% (directional)
MAX_SINGLE_POS_CAP = 0.050  # Global Single Position Cap at 5.0%
COOLDOWN_BARS = 4          # Post-stopout lockout: 4 bars / 1 hour
SESSION_DRAWDOWN_PCT = 0.05  # Session circuit breaker: 5% from peak

# Initialize Risk, Sentiment, and Broker Execution Engines
risk_engine = AdvancedRiskEngine(
    target_volatility=0.15,
    max_position_pct=MAX_SINGLE_POS_CAP,
    stop_loss_pct=STOP_LOSS_PCT,
    take_profit_pct=TAKE_PROFIT_PCT,
    cooldown_bars=COOLDOWN_BARS,
    session_drawdown_pct=SESSION_DRAWDOWN_PCT,
)
sentiment_agent = NewsSentimentAgent()
broker_bridge = AlpacaExecutionBridge()


def restore_agent_states_from_db(swarm_mgr, db):
    """Restore active agent population, cash, holdings, and entry prices from PostgreSQL on container startup."""
    try:
        engine = getattr(db, 'engine', None)
        if engine is None:
            postgres_user = os.getenv("POSTGRES_USER", "evoquant")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "evoquant_secret_pass")
            postgres_host = os.getenv("POSTGRES_HOST", "timescaledb")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            postgres_db = os.getenv("POSTGRES_DB", "evoquant_db")
            postgres_url = f"postgresql+psycopg://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            engine = sqlalchemy.create_engine(postgres_url)

        # Restore active agents from database (where cash > 0 or equity > 0)
        accounts_df = pd.read_sql(
            "SELECT agent_id, cash FROM agent_accounts WHERE cash > 0 ORDER BY updated_at DESC;",
            engine
        )

        snapshots_df = pd.read_sql(
            "SELECT DISTINCT ON (agent_id) agent_id, cash, equity FROM agent_snapshots ORDER BY agent_id, timestamp DESC;",
            engine
        )

        first_snapshots_df = pd.read_sql(
            "SELECT DISTINCT ON (agent_id) agent_id, equity FROM agent_snapshots ORDER BY agent_id, timestamp ASC;",
            engine
        )
        first_snap_map = dict(zip(first_snapshots_df['agent_id'], first_snapshots_df['equity'])) if not first_snapshots_df.empty else {}

        holdings_df = pd.read_sql(
            "SELECT agent_id, ticker, amount, entry_price FROM agent_holdings WHERE amount != 0;",
            engine
        )

        active_agent_ids = []
        if not accounts_df.empty:
            active_agent_ids = accounts_df['agent_id'].tolist()
        if not snapshots_df.empty:
            active_from_snaps = snapshots_df[snapshots_df['equity'] > 0]['agent_id'].tolist()
            for ag in active_from_snaps:
                if ag not in active_agent_ids:
                    active_agent_ids.append(ag)

        existing_map = {a.agent_id: a for a in swarm_mgr.population}
        restored_pop = []

        for ag_id in active_agent_ids:
            cash_val = 100000.0
            if not accounts_df.empty and ag_id in accounts_df['agent_id'].values:
                cash_val = float(accounts_df[accounts_df['agent_id'] == ag_id].iloc[0]['cash'])
            elif not snapshots_df.empty and ag_id in snapshots_df['agent_id'].values:
                cash_val = float(snapshots_df[snapshots_df['agent_id'] == ag_id].iloc[0]['cash'])

            init_cap = float(first_snap_map.get(ag_id, cash_val))

            if ag_id in existing_map:
                agent = existing_map[ag_id]
                agent.cash = cash_val
                agent.initial_capital = init_cap
                restored_pop.append(agent)
            else:
                restored_pop.append(AgentGenome(
                    agent_id=ag_id,
                    persona_prompt="You are an evolved quantitative trading agent focusing on risk-adjusted equity growth.",
                    generation=2 if "Gen" in ag_id else 1,
                    initial_capital=init_cap,
                    cash=cash_val,
                    holdings={},
                    entry_prices={},
                    equity_history=[cash_val]
                ))

        # If fewer than 5 active agents exist, backfill only missing slots with default baseline personas
        if len(restored_pop) < 5:
            baseline_pop = swarm_mgr._bootstrap_initial_population()
            restored_ids = {a.agent_id for a in restored_pop}
            dead_ids = set()
            if not snapshots_df.empty:
                dead_ids.update(snapshots_df[snapshots_df['equity'] <= 0]['agent_id'].tolist())
            for base_agent in baseline_pop:
                if len(restored_pop) >= 5:
                    break
                if base_agent.agent_id not in restored_ids and base_agent.agent_id not in dead_ids:
                    restored_pop.append(base_agent)
                    restored_ids.add(base_agent.agent_id)

        swarm_mgr.population = restored_pop[:5]

        for agent in swarm_mgr.population:
            if not hasattr(agent, 'entry_prices'):
                agent.entry_prices = {}

            # Restore Cash
            if not snapshots_df.empty:
                agent_snap = snapshots_df[snapshots_df['agent_id'] == agent.agent_id]
                if not agent_snap.empty:
                    agent.cash = float(agent_snap.iloc[0]['cash'])

            # Restore Active Holdings & Entry Prices
            if not holdings_df.empty:
                agent_pos = holdings_df[holdings_df['agent_id'] == agent.agent_id]
                if not agent_pos.empty:
                    agent.holdings = {row['ticker']: float(row['amount']) for _, row in agent_pos.iterrows()}
                    agent.entry_prices = {row['ticker']: float(row['entry_price']) for _, row in agent_pos.iterrows()}

            # Recalculate Restored Equity on Startup using restored entry prices
            long_val = sum(qty * agent.entry_prices.get(tk, 0.0) for tk, qty in agent.holdings.items() if qty > 0)
            short_liability = sum(abs(qty) * agent.entry_prices.get(tk, 0.0) for tk, qty in agent.holdings.items() if qty < 0)
            restored_equity = round(agent.cash + long_val - short_liability, 2)
            agent.equity_history = [restored_equity]

            # Log instant startup snapshot so TimescaleDB updates immediately
            base_cap = agent.initial_capital if getattr(agent, 'initial_capital', 0.0) > 0 else 100000.0
            pnl = ((restored_equity - base_cap) / base_cap) * 100
            db.log_snapshot(agent.agent_id, restored_equity, agent.cash, pnl)

        logger.info("✅ Successfully restored agent cash, holdings, equity, and logged startup snapshots.")
    except Exception as e:
        logger.warning(f"⚠️ Could not restore state from DB (starting with defaults): {e}")


def submit_safe_broker_order(broker, ticker: str, shares: float, action: str):
    """Submits orders to Alpaca, ensuring integer quantities for SHORT/COVER to prevent HTTP 422 errors."""
    if not broker.is_active():
        return
    try:
        if action in ["SHORT", "COVER"]:
            int_shares = math.floor(abs(shares))
            if int_shares >= 1:
                broker.submit_market_order(ticker, int_shares, action)
            else:
                logger.warning(f"⚠️ Skipped Alpaca {action} order for {ticker}: {shares:.2f} shares < 1.0 integer share minimum.")
        else:
            broker.submit_market_order(ticker, abs(shares), action)
    except Exception as e:
        logger.error(f"❌ [ALPACA BROKER EXCEPTION] {action} {ticker}: {e}")


def build_tickers_snapshot(market_state: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build 20-ticker snapshot dict for adversarial batch.
    Each entry carries OHLCV + RSI14 + 15m Momentum (+ legacy ATR/MACD for arbiter context).
    """
    snap: Dict[str, Dict[str, Any]] = {}
    for tk, data in (market_state or {}).items():
        try:
            close = float(data.get("close", 0.0) or 0.0)
            snap[str(tk).upper()] = {
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
                "headlines": str(data.get("headlines", data.get("news", "-")) or "-")[:200],
            }
        except Exception:
            continue
    return snap


def compute_deterministic_signals(
    population: List[Any],
    market_state: Dict[str, Dict[str, Any]],
    shared_thesis: Dict[str, Any],
) -> Dict[str, CrossAssetRiskDecision]:
    """
    Deterministic cache-driven conviction adjustment (O(1) per ticker, zero LLM calls).
    Reads sentiment from SentimentCache.get_sentiment(ticker) and fuses with RSI/Momentum/MACD.
    Preserves per-agent heterogeneity via persona-threshold offsets.
    """
    decisions_map: Dict[str, CrossAssetRiskDecision] = {}
    # Persona thresholds: aggressive vs conservative entry gates
    persona_thresholds = {
        "Agent_Alpha": 0.20,
        "Agent_Beta": 0.40,
        "Agent_Gamma": 0.25,
        "Agent_Delta": 0.25,
        "Agent_Epsilon": 0.30,
    }
    for agent in population:
        try:
            base_thr = 0.30
            for key, thr in persona_thresholds.items():
                if key.lower() in str(agent.agent_id).lower():
                    base_thr = thr
                    break
            # Gen2 offspring slightly more selective
            if getattr(agent, "generation", 1) > 1:
                base_thr = min(0.45, base_thr + 0.05)

            signals: Dict[str, QualitativeSignal] = {}
            for ticker in shared_thesis.keys():
                if ticker not in market_state:
                    continue
                md = market_state.get(ticker, {}) or {}
                sentiment = float(sentiment_agent.cache.get_sentiment(ticker) or 0.0)
                rsi = float(md.get("rsi14", md.get("rsi", 50.0)) or 50.0)
                mom = float(md.get("momentum_15m", md.get("momentum", 0.0)) or 0.0)
                macd = float(md.get("macd_hist", 0.0) or 0.0)
                rel = float(md.get("rel_strength_spy", 0.0) or 0.0)

                # Technical composite in [-1, 1]
                tech = 0.0
                # RSI mean-reversion + momentum
                if rsi < 30:
                    tech += 0.4
                elif rsi < 45:
                    tech += 0.15
                elif rsi > 70:
                    tech -= 0.4
                elif rsi > 60:
                    tech -= 0.15
                # Momentum continuation
                if mom > 0.3:
                    tech += 0.3
                elif mom > 0.05:
                    tech += 0.1
                elif mom < -0.3:
                    tech -= 0.3
                elif mom < -0.05:
                    tech -= 0.1
                # MACD trend
                if macd > 0:
                    tech += 0.15
                elif macd < 0:
                    tech -= 0.15
                # Relative strength tilt
                if rel > 1.0:
                    tech += 0.1
                elif rel < -1.0:
                    tech -= 0.1
                tech = max(-1.0, min(1.0, tech))

                # Fuse: 55% debate sentiment + 45% technicals
                combined = 0.55 * sentiment + 0.45 * tech
                conviction = round(min(1.0, abs(combined)), 4)
                pos_qty = float(agent.holdings.get(ticker, 0.0) or 0.0)

                if pos_qty > 0:
                    # Long open: hold unless bearish conviction breaches gate
                    if combined <= -base_thr and conviction > 0.2:
                        action = "SELL"
                    elif combined >= base_thr:
                        action = "BUY"
                    else:
                        action = "HOLD"
                        conviction = round(conviction * 0.5, 4)
                elif pos_qty < 0:
                    # Short open
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

                # Suppress dust convictions
                if action in ("BUY", "SHORT") and conviction <= 0.15:
                    action = "HOLD"
                    conviction = 0.0

                signals[ticker] = QualitativeSignal(ticker=ticker, action=action, conviction=float(conviction))

            agent_decision_input = AgentSignalDecision(
                signals=signals,
                macro_reasoning=f"Cache-fused debate sentiment + RSI/MOM/MACD composite (thr={base_thr:.2f}).",
            )
            decision = RiskParityOptimizer.optimize_allocations(agent_decision_input, shared_thesis)
            decisions_map[agent.agent_id] = decision
        except Exception as e:
            logger.warning(f"⚠️ Deterministic signal build failed for [{getattr(agent, 'agent_id', '?')}]: {e}")
            decisions_map[getattr(agent, "agent_id", "unknown")] = CrossAssetRiskDecision(
                decisions={}, macro_reasoning="Deterministic fallback: no signals."
            )
    return decisions_map


async def liquidate_all_to_cash(
    population: List[Any], prices: Dict[str, float], market_state: Dict[str, Any], db, reason: str = "SESSION_BREAKER"
):
    """Emergency liquidation of all open exposure to cash (session breaker)."""
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
                if shares > 0:
                    agent.cash += shares * exec_price
                else:
                    agent.cash -= abs(shares) * exec_price
                agent.holdings[tk] = 0.0
                agent.entry_prices[tk] = 0.0
                db.update_agent_cash(agent.agent_id, agent.cash)
                db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason=reason)
                logger.warning(f"  🛑 [{agent.agent_id}] {reason} LIQUIDATE {tk} ({action}) @ ${exec_price:.2f}")
                submit_safe_broker_order(broker_bridge, tk, abs(shares), action)
            except Exception as e:
                logger.warning(f"⚠️ Liquidation failed [{agent.agent_id}] {tk}: {e}")


async def run_consumer():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment.")

    db = CrossAssetPortfolioManager()
    swarm_mgr = EvolutionarySwarmManager(api_key=api_key, population_size=5)
    swarm = DualModelTradingSwarm(api_key=api_key)

    for agent in swarm_mgr.population:
        db.register_agent(agent.agent_id)

    # RECOVER PORTFOLIO STATE & LOG STARTUP SNAPSHOT
    restore_agent_states_from_db(swarm_mgr, db)

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

    logger.info("🤖 QUANT-UPGRADED EVOLUTIONARY SWARM ONLINE (POSTGRESQL / TIMESCALEDB ACTIVE).")
    logger.info("⚡ ENGINE POWERED BY GEMMA 4-31B BATCHED ADVERSARIAL DEBATE (3 CALLS / 15M BAR).")
    logger.info("🛡️ Hard Risk Overlay Active: Directional Stop-Loss (-2.5%) | Take-Profit (+5.0%) | Cooldown 4 bars | Session Breaker 5%")
    logger.info("📰 Batched Bull/Bear Debate + SentimentCache (TTL 1200s) Active. 288 calls/day, 0.2 RPM.")
    if broker_bridge.is_active():
        logger.info("⚡ ALPACA PAPER TRADING BROKER BRIDGE ACTIVE.")

    tick_counter = 0
    spy_prices_history = []
    last_processed_date = ""

    # Resilient Outer Loop with Auto-Reconnect on Redis Disconnection
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe('market_events')

            async with httpx.AsyncClient() as client:
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        market_state = json.loads(message['data'])
                        prices = {tk: data["close"] for tk, data in market_state.items()}
                        tick_counter += 1

                        # Increment tenure ticks for active agents
                        for agent in swarm_mgr.population:
                            agent.tenure_ticks = getattr(agent, 'tenure_ticks', 0) + 1

                        logger.info(f"\n==================== 🔔 MARKET TICK #{tick_counter} ====================")

                        # Daily Ex-Dividend Payout / Debit Engine Trigger with Replay Timestamp Sync
                        tick_time_str = next((v.get("timestamp") for v in market_state.values() if isinstance(v, dict) and "timestamp" in v), None)
                        if tick_time_str:
                            today_date_str = str(tick_time_str)[:10]
                        else:
                            today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                        if today_date_str != last_processed_date:
                            try:
                                db.process_daily_dividends(today_date_str)
                                last_processed_date = today_date_str
                            except Exception as e:
                                logger.warning(f"⚠️ Daily dividend processing note: {e}")

                        # ---------------------------------------------------------
                        # PRE-TICK DEBATE EXECUTION (exactly 3 API calls per 15m bar)
                        # 1. run_adversarial_batch -> Bull + Bear parallel + Arbiter
                        # 2. Populate SentimentCache with 20 scores
                        # 3. Agents read via get_sentiment O(1)
                        # ---------------------------------------------------------
                        try:
                            tickers_snapshot = build_tickers_snapshot(market_state)
                            debate_scores = await sentiment_agent.run_adversarial_batch(tickers_snapshot)
                            logger.info(
                                f"🧠 [Batched Debate] Tick #{tick_counter}: cached {len(debate_scores)} scores "
                                f"(3 calls, 0.2 RPM). Sample: {dict(list(debate_scores.items())[:3])}"
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ Batched debate failed on tick #{tick_counter}: {e}")

                        # Track SPY price history for 200 SMA Macro Guard
                        if "SPY" in market_state:
                            spy_close = market_state["SPY"]["close"]
                            spy_prices_history.append(spy_close)
                            if len(spy_prices_history) > 250:
                                spy_prices_history.pop(0)

                        spy_series = pd.Series(spy_prices_history).pct_change().dropna() if len(spy_prices_history) > 2 else pd.Series()
                        spy_price_series = pd.Series(spy_prices_history) if len(spy_prices_history) >= 200 else None
                        regime_scaler = risk_engine.calculate_regime_scaler(spy_series, spy_prices=spy_price_series)

                        # -------------------------------------------------------------
                        # PHASE A: DIRECTIONAL HARD RISK GUARD (LONG & SHORT aware)
                        # Uses risk_engine.check_stop_loss_take_profit + cooldowns
                        # -------------------------------------------------------------
                        for agent in swarm_mgr.population:
                            if not hasattr(agent, 'entry_prices'):
                                agent.entry_prices = {}

                            for tk, shares in list(agent.holdings.items()):
                                if shares != 0 and tk in prices:
                                    current_price = float(prices[tk])
                                    entry_price = float(agent.entry_prices.get(tk, current_price) or current_price)
                                    adv = float(market_state.get(tk, {}).get("adv", 1000000.0) or 1000000.0)

                                    direction = "SHORT" if shares < 0 else "LONG"
                                    exit_signal = risk_engine.check_stop_loss_take_profit(
                                        entry_price, current_price, direction,
                                        STOP_LOSS_PCT, TAKE_PROFIT_PCT,
                                    )

                                    if exit_signal in ("STOP_LOSS", "TAKE_PROFIT"):
                                        action = "SELL" if shares > 0 else "COVER"
                                        exec_price = risk_engine.calculate_execution_price(current_price, abs(shares), adv, action)

                                        if shares > 0:
                                            agent.cash += shares * exec_price
                                        else:
                                            agent.cash -= abs(shares) * exec_price  # Pay cash to cover short liability

                                        agent.holdings[tk] = 0.0
                                        agent.entry_prices[tk] = 0.0

                                        reason = "HARD_STOP_LOSS" if exit_signal == "STOP_LOSS" else "HARD_TAKE_PROFIT"
                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                                        db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason=reason)
                                        if exit_signal == "STOP_LOSS":
                                            # Post-stopout cooldown: lock re-entry for 4 bars / 1 hour
                                            risk_engine.record_stopout(agent.agent_id, tk, tick_counter, COOLDOWN_BARS)
                                            logger.warning(f"  🚨 [{agent.agent_id}] HARD STOP-LOSS on {tk} ({direction}->{action}): Exec @ ${exec_price:.2f}")
                                        else:
                                            logger.info(f"  🎯 [{agent.agent_id}] HARD TAKE-PROFIT on {tk} ({direction}->{action}): Exec @ ${exec_price:.2f}")

                                        submit_safe_broker_order(broker_bridge, tk, abs(shares), action)

                        # Prune expired cooldowns each tick
                        try:
                            risk_engine.prune_expired_cooldowns(tick_counter)
                        except Exception:
                            pass

                        all_active_holdings = list({
                            tk for agent in swarm_mgr.population 
                            for tk, shares in agent.holdings.items() if shares != 0
                        })

                        # Shared technical thesis (zero LLM calls, pure pandas filter)
                        shared_thesis = swarm.analyze_technical_state(market_state, active_holdings=all_active_holdings)

                        # -------------------------------------------------------------
                        # PHASE B: DETERMINISTIC CACHE-DRIVEN STRATEGY EXECUTION
                        # Zero LLM calls — reads SentimentCache O(1) per ticker.
                        # Total per-bar LLM budget remains exactly 3 (debate only).
                        # -------------------------------------------------------------
                        decisions_map = compute_deterministic_signals(
                            population=swarm_mgr.population,
                            market_state=market_state,
                            shared_thesis=shared_thesis,
                        )

                        # Session equity tracking + circuit breaker (5% peak-to-trough)
                        try:
                            swarm_equity_now = 0.0
                            for agent in swarm_mgr.population:
                                lv = sum(
                                    qty * prices.get(tk, agent.entry_prices.get(tk, 0.0))
                                    for tk, qty in agent.holdings.items() if qty > 0
                                )
                                sl = sum(
                                    abs(qty) * prices.get(tk, agent.entry_prices.get(tk, 0.0))
                                    for tk, qty in agent.holdings.items() if qty < 0
                                )
                                swarm_equity_now += float(agent.cash + lv - sl)
                            risk_engine.update_session_peak(swarm_equity_now)
                            if risk_engine.check_session_drawdown(swarm_equity_now):
                                logger.error(
                                    f"🚨 SESSION BREAKER: swarm equity ${swarm_equity_now:,.2f} "
                                    f"≥5% below peak ${risk_engine.session_peak_equity:,.2f}. Liquidating to cash."
                                )
                                await liquidate_all_to_cash(
                                    swarm_mgr.population, prices, market_state, db, reason="SESSION_BREAKER"
                                )
                        except Exception as e:
                            logger.warning(f"⚠️ Session breaker check note: {e}")

                        trading_halted = risk_engine.is_trading_halted()

                        for agent in swarm_mgr.population:
                            # SAFE EQUITY CALCULATION: Fallback to entry price if current live tick price is missing
                            long_val = sum(
                                qty * prices.get(tk, agent.entry_prices.get(tk, 0.0)) 
                                for tk, qty in agent.holdings.items() if qty > 0
                            )
                            short_liability = sum(
                                abs(qty) * prices.get(tk, agent.entry_prices.get(tk, 0.0)) 
                                for tk, qty in agent.holdings.items() if qty < 0
                            )
                            current_equity = round(agent.cash + long_val - short_liability, 2)
                            agent.equity_history.append(current_equity)

                            decision = decisions_map.get(agent.agent_id)
                            if not decision or not decision.decisions:
                                continue

                            # If breaker tripped, skip all new entries (liquidation already done)
                            if trading_halted:
                                logger.warning(f"  🛑 [{agent.agent_id}] Session breaker active — skipping new entries.")
                                continue

                            for ticker, target in decision.decisions.items():
                                if ticker not in prices:
                                    continue

                                raw_price = prices[ticker]
                                adv = market_state.get(ticker, {}).get("adv", 1000000.0)

                                # Cooldown gate: reject re-entry for cooled-down tickers
                                if target.action in ("BUY", "SHORT") and risk_engine.is_cooled_down(
                                    agent.agent_id, ticker, tick_counter
                                ):
                                    logger.info(f"  🧊 [{agent.agent_id}] Cooldown reject: {ticker} locked until tick {risk_engine.cooldowns.get((agent.agent_id, ticker.upper()))}.")
                                    continue

                                raw_effective_alloc = target.allocation_pct * regime_scaler
                                effective_alloc = min(raw_effective_alloc, MAX_SINGLE_POS_CAP)
                                target_val = current_equity * effective_alloc
                                current_pos_qty = agent.holdings.get(ticker, 0.0)

                                # 1. BUY Execution (Long Entry / Scale Up with Short Proceeds Solvency Guard)
                                if target.action == "BUY":
                                    current_long_val = max(current_pos_qty, 0.0) * raw_price
                                    delta = target_val - current_long_val

                                    # Calculate true unencumbered cash strictly accounting for encumbered short margin
                                    short_liabilities = sum(
                                        abs(qty) * max(agent.entry_prices.get(tk, 0.0), prices.get(tk, agent.entry_prices.get(tk, 0.0)))
                                        for tk, qty in agent.holdings.items() if qty < 0
                                    )
                                    free_cash = max(0.0, agent.cash - short_liabilities)

                                    if delta > 50.0 and free_cash >= delta:
                                        approx_shares = delta / raw_price
                                        exec_price = risk_engine.calculate_execution_price(raw_price, approx_shares, adv, "BUY")
                                        shares = delta / exec_price

                                        old_shares = max(current_pos_qty, 0.0)
                                        old_entry = agent.entry_prices.get(ticker, exec_price)
                                        new_shares = old_shares + shares
                                        weighted_entry = ((old_shares * old_entry) + (shares * exec_price)) / new_shares

                                        agent.holdings[ticker] = new_shares
                                        agent.entry_prices[ticker] = weighted_entry
                                        agent.cash -= delta

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, ticker, new_shares, weighted_entry)
                                        db.log_trade(agent.agent_id, ticker, "BUY", shares, exec_price, effective_alloc, reason="RISK_PARITY_ALLOCATION")
                                        logger.info(f"    📈 [{agent.agent_id}] BOUGHT {ticker}: +{shares:.2f}sh @ ${exec_price:.2f} (Avg Cost: ${weighted_entry:.2f}) [Alloc: {effective_alloc*100:.1f}%]")

                                        submit_safe_broker_order(broker_bridge, ticker, shares, "BUY")

                                # 2. SELL Execution (Long Reduction / Exit)
                                elif target.action == "SELL" and current_pos_qty > 0:
                                    current_long_val = current_pos_qty * raw_price
                                    delta = target_val - current_long_val
                                    if delta < -50.0:
                                        sell_shares = min(abs(delta) / raw_price, current_pos_qty)
                                        exec_price = risk_engine.calculate_execution_price(raw_price, sell_shares, adv, "SELL")
                                        actual_cash_gained = sell_shares * exec_price

                                        agent.holdings[ticker] -= sell_shares
                                        agent.cash += actual_cash_gained

                                        if agent.holdings[ticker] <= 0.0001:
                                            agent.holdings[ticker] = 0.0
                                            agent.entry_prices[ticker] = 0.0

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, ticker, agent.holdings[ticker], agent.entry_prices.get(ticker, 0.0))
                                        db.log_trade(agent.agent_id, ticker, "SELL", sell_shares, exec_price, effective_alloc, reason="RISK_PARITY_REBALANCE")
                                        logger.info(f"    📉 [{agent.agent_id}] SOLD {ticker}: -{sell_shares:.2f}sh @ ${exec_price:.2f}")

                                        submit_safe_broker_order(broker_bridge, ticker, sell_shares, "SELL")

                                # 3. SHORT Execution (Short Entry / Scale Up)
                                elif target.action == "SHORT":
                                    current_short_val = abs(min(current_pos_qty, 0.0)) * raw_price
                                    short_delta = target_val - current_short_val
                                    if short_delta > 50.0:
                                        margin_info = risk_engine.evaluate_margin_health(agent.cash, agent.holdings, prices)
                                        if margin_info["free_margin"] >= short_delta:
                                            approx_shares = short_delta / raw_price
                                            exec_price = risk_engine.calculate_execution_price(raw_price, approx_shares, adv, "SHORT")
                                            actual_short_shares = short_delta / exec_price

                                            old_short_shares = abs(min(current_pos_qty, 0.0))
                                            old_entry = agent.entry_prices.get(ticker, exec_price)
                                            new_short_shares = old_short_shares + actual_short_shares
                                            weighted_entry = ((old_short_shares * old_entry) + (actual_short_shares * exec_price)) / new_short_shares

                                            agent.holdings[ticker] = -new_short_shares  # Negative quantity
                                            agent.entry_prices[ticker] = weighted_entry
                                            agent.cash += actual_short_shares * exec_price  # Add short sale proceeds

                                            db.update_agent_cash(agent.agent_id, agent.cash)
                                            db.update_agent_holding(agent.agent_id, ticker, -new_short_shares, weighted_entry)
                                            db.log_trade(agent.agent_id, ticker, "SHORT", actual_short_shares, exec_price, effective_alloc, reason="RISK_PARITY_SHORT")
                                            logger.info(f"    📉 [{agent.agent_id}] SHORTED {ticker}: -{actual_short_shares:.2f}sh @ ${exec_price:.2f} [Alloc: {effective_alloc*100:.1f}%]")

                                            submit_safe_broker_order(broker_bridge, ticker, actual_short_shares, "SHORT")

                                # 4. COVER Execution (Short Reduction / Exit)
                                elif target.action == "COVER" and current_pos_qty < 0:
                                    current_short_shares = abs(current_pos_qty)
                                    current_short_val = current_short_shares * raw_price
                                    short_delta = target_val - current_short_val
                                    cover_shares = min(abs(short_delta) / raw_price, current_short_shares) if short_delta < -50.0 else current_short_shares
                                    exec_price = risk_engine.calculate_execution_price(raw_price, cover_shares, adv, "COVER")
                                    cost = cover_shares * exec_price

                                    if agent.cash >= cost:
                                        agent.holdings[ticker] += cover_shares
                                        agent.cash -= cost

                                        if abs(agent.holdings[ticker]) <= 0.0001:
                                            agent.holdings[ticker] = 0.0
                                            agent.entry_prices[ticker] = 0.0

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, ticker, agent.holdings[ticker], agent.entry_prices.get(ticker, 0.0))
                                        db.log_trade(agent.agent_id, ticker, "COVER", cover_shares, exec_price, effective_alloc, reason="RISK_PARITY_COVER")
                                        logger.info(f"    📈 [{agent.agent_id}] COVERED {ticker}: +{cover_shares:.2f}sh @ ${exec_price:.2f}")

                                        submit_safe_broker_order(broker_bridge, ticker, cover_shares, "COVER")

                        # Print Competing Leaderboard & Log Snapshots
                        logger.info("\n🏆 --- COMPETING AGENT LEADERBOARD ---")
                        sorted_swarm = sorted(swarm_mgr.population, key=lambda a: a.equity_history[-1], reverse=True)
                        for rank, agent in enumerate(sorted_swarm, 1):
                            base_cap = agent.initial_capital if getattr(agent, 'initial_capital', 0.0) > 0 else 100000.0
                            pnl = ((agent.equity_history[-1] - base_cap) / base_cap) * 100
                            db.log_snapshot(agent.agent_id, agent.equity_history[-1], agent.cash, pnl)

                            active_holdings = [
                                f"{tk}: {'LONG' if shares > 0 else 'SHORT'} {abs(shares):.1f}sh" 
                                for tk, shares in agent.holdings.items() if shares != 0
                            ]
                            holdings_summary = ", ".join(active_holdings[:4]) if active_holdings else "100% Cash"

                            logger.info(
                                f"  #{rank} | {agent.agent_id:<22} | Equity: ${agent.equity_history[-1]:>10,.2f} "
                                f"({pnl:+6.2f}%) | Cash: ${agent.cash:>10,.2f} | Positions: [{holdings_summary}]"
                            )

                        # Darwinian Selection & Mutation
                        if tick_counter % EPOCH_TICK_THRESHOLD == 0:
                            await swarm_mgr.run_culling_cycle(
                                prices=prices, 
                                risk_engine=risk_engine, 
                                db=db, 
                                execution_bridge=broker_bridge
                            )

        except Exception as e:
            logger.error(f"❌ Redis subscriber connection lost: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5.0)

if __name__ == "__main__":
    asyncio.run(run_consumer())
