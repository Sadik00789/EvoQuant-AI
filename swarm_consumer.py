import os
import json
import asyncio
import httpx
import redis.asyncio as redis
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from engine import (
    DualModelTradingSwarm, 
    CrossAssetPortfolioManager, 
    AlpacaExecutionBridge, 
    logger
)
from evolution_engine import EvolutionarySwarmManager
from risk_engine import AdvancedRiskEngine
from sentiment_agent import NewsSentimentAgent

load_dotenv()

EPOCH_TICK_THRESHOLD = 20  # Culling evaluation every 20 ticks (~5 hours)
STOP_LOSS_PCT = -0.025     # Hard stop loss at -2.5%
TAKE_PROFIT_PCT = 0.050    # Hard take profit at +5.0%

# Initialize Risk, Sentiment, and Broker Execution Engines
risk_engine = AdvancedRiskEngine(target_volatility=0.15, max_position_pct=0.05)
sentiment_agent = NewsSentimentAgent()
broker_bridge = AlpacaExecutionBridge()

async def run_consumer():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment.")

    db = CrossAssetPortfolioManager()
    swarm_mgr = EvolutionarySwarmManager(api_key=groq_api_key, population_size=5)
    swarm = DualModelTradingSwarm(api_key=groq_api_key)

    for agent in swarm_mgr.population:
        db.register_agent(agent.agent_id)

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

    logger.info("🤖 QUANT-UPGRADED EVOLUTIONARY SWARM ONLINE (POSTGRESQL / TIMESCALEDB ACTIVE).")
    logger.info("🛡️ Hard Risk Overlay Active: Stop-Loss (-2.5%) | Take-Profit (+5.0%) [LONG & SHORT]")
    logger.info("📰 News RAG Sentiment & Dynamic Market Impact Slippage Engines Active.")
    if broker_bridge.is_active():
        logger.info("⚡ ALPACA PAPER TRADING BROKER BRIDGE ACTIVE.")

    tick_counter = 0
    macro_multiplier = 1.0
    spy_returns_history = []
    spy_prices_history = []
    last_processed_date = ""

    # Initial Macro Sentiment Fetch & Database Log
    try:
        macro_news = await sentiment_agent.analyze_macro_sentiment_async()
        macro_multiplier = macro_news.get("risk_multiplier", 1.0)
        db.log_macro_regime(
            sentiment_score=macro_news.get("sentiment_score", 0.0),
            risk_multiplier=macro_multiplier,
            reasoning=macro_news.get("summary_reasoning", "")
        )
        logger.info(f"📰 Initial News RAG Multiplier: {macro_multiplier:.2f}x | {macro_news.get('summary_reasoning', '')}")
    except Exception as e:
        logger.warning(f"⚠️ Initial news sentiment fetch failed: {e}")

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

                        logger.info(f"\n==================== 🔔 MARKET TICK #{tick_counter} ====================")

                        # Daily Ex-Dividend Payout / Debit Engine Trigger
                        today_date_str = datetime.utcnow().strftime("%Y-%m-%d")
                        if today_date_str != last_processed_date:
                            try:
                                db.process_daily_dividends(today_date_str)
                                last_processed_date = today_date_str
                            except Exception as e:
                                logger.warning(f"⚠️ Daily dividend processing note: {e}")

                        # Refresh news sentiment every 8 market ticks (~2 hours)
                        if tick_counter % 8 == 0:
                            try:
                                macro_news = await sentiment_agent.analyze_macro_sentiment_async()
                                macro_multiplier = macro_news.get("risk_multiplier", 1.0)
                                db.log_macro_regime(
                                    sentiment_score=macro_news.get("sentiment_score", 0.0),
                                    risk_multiplier=macro_multiplier,
                                    reasoning=macro_news.get("summary_reasoning", "")
                                )
                                logger.info(f"📰 Updated News RAG Multiplier: {macro_multiplier:.2f}x")
                            except Exception as e:
                                logger.warning(f"⚠️ Periodic news sentiment fetch failed: {e}")

                        # Track SPY price and return history for 200 SMA Macro Guard
                        if "SPY" in market_state:
                            spy_close = market_state["SPY"]["close"]
                            spy_prices_history.append(spy_close)
                            spy_returns_history.append(spy_close)

                            if len(spy_returns_history) > 30:
                                spy_returns_history.pop(0)

                        spy_series = pd.Series(spy_returns_history).pct_change().dropna() if len(spy_returns_history) > 2 else pd.Series()
                        spy_price_series = pd.Series(spy_prices_history) if len(spy_prices_history) >= 200 else None
                        regime_scaler = risk_engine.calculate_regime_scaler(spy_series, spy_prices=spy_price_series)

                        all_active_holdings = list({
                            tk for agent in swarm_mgr.population 
                            for tk, shares in agent.holdings.items() if shares != 0
                        })

                        shared_thesis = swarm.analyze_technical_state(market_state, active_holdings=all_active_holdings)

                        # -------------------------------------------------------------
                        # PHASE A: DUAL-SIDED HARD RISK GUARD CHECK (LONG & SHORT)
                        # -------------------------------------------------------------
                        for agent in swarm_mgr.population:
                            if not hasattr(agent, 'entry_prices'):
                                agent.entry_prices = {}

                            for tk, shares in list(agent.holdings.items()):
                                if shares != 0 and tk in prices:
                                    current_price = prices[tk]
                                    entry_price = agent.entry_prices.get(tk, current_price)
                                    adv = market_state.get(tk, {}).get("adv", 1000000.0)

                                    if shares > 0:
                                        pos_pnl = (current_price - entry_price) / entry_price
                                    else:
                                        pos_pnl = (entry_price - current_price) / entry_price  # Inverted PnL for Short

                                    # Hard Stop-Loss Liquidation
                                    if pos_pnl <= STOP_LOSS_PCT:
                                        action = "SELL" if shares > 0 else "COVER"
                                        exec_price = risk_engine.calculate_execution_price(current_price, abs(shares), adv, action)

                                        if shares > 0:
                                            agent.cash += shares * exec_price
                                        else:
                                            agent.cash -= abs(shares) * exec_price  # Pay cash to cover short liability

                                        agent.holdings[tk] = 0.0
                                        agent.entry_prices[tk] = 0.0

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                                        db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason="HARD_STOP_LOSS")
                                        logger.warning(f"  🚨 [{agent.agent_id}] HARD STOP-LOSS on {tk} ({action}): Exec @ ${exec_price:.2f} ({pos_pnl*100:.2f}%)")

                                        broker_bridge.submit_market_order(tk, abs(shares), action)

                                    # Hard Take-Profit Liquidation
                                    elif pos_pnl >= TAKE_PROFIT_PCT:
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
                                        db.log_trade(agent.agent_id, tk, action, abs(shares), exec_price, 0.0, reason="HARD_TAKE_PROFIT")
                                        logger.info(f"  🎯 [{agent.agent_id}] HARD TAKE-PROFIT on {tk} ({action}): Exec @ ${exec_price:.2f} (+{pos_pnl*100:.2f}%)")

                                        broker_bridge.submit_market_order(tk, abs(shares), action)

                        # -------------------------------------------------------------
                        # PHASE B: CONCURRENT ASYNC STRATEGY EXECUTION
                        # -------------------------------------------------------------
                        decisions_map = await swarm.execute_swarm_strategies_concurrently(
                            client=client,
                            shared_thesis=shared_thesis,
                            population=swarm_mgr.population,
                            prices=prices
                        )

                        for agent in swarm_mgr.population:
                            long_val = sum(qty * prices[tk] for tk, qty in agent.holdings.items() if qty > 0 and tk in prices)
                            short_liability = sum(abs(qty) * prices[tk] for tk, qty in agent.holdings.items() if qty < 0 and tk in prices)
                            current_equity = round(agent.cash + long_val - short_liability, 2)
                            agent.equity_history.append(current_equity)

                            decision = decisions_map.get(agent.agent_id)
                            if not decision or not decision.decisions:
                                continue

                            for ticker, target in decision.decisions.items():
                                if ticker not in prices:
                                    continue

                                raw_price = prices[ticker]
                                adv = market_state.get(ticker, {}).get("adv", 1000000.0)

                                effective_alloc = target.allocation_pct * macro_multiplier * regime_scaler
                                target_val = current_equity * effective_alloc
                                current_pos_qty = agent.holdings.get(ticker, 0.0)

                                # 1. BUY Execution (Long Entry / Scale Up)
                                if target.action == "BUY":
                                    current_long_val = max(current_pos_qty, 0.0) * raw_price
                                    delta = target_val - current_long_val
                                    if delta > 50.0 and agent.cash >= delta:
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
                                        logger.info(f"  📈 [{agent.agent_id}] BOUGHT {ticker}: +{shares:.2f}sh @ ${exec_price:.2f} (Avg Cost: ${weighted_entry:.2f})")

                                        broker_bridge.submit_market_order(ticker, shares, "BUY")

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
                                        logger.info(f"  📉 [{agent.agent_id}] SOLD {ticker}: -{sell_shares:.2f}sh @ ${exec_price:.2f}")

                                        broker_bridge.submit_market_order(ticker, sell_shares, "SELL")

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
                                            logger.info(f"  📉 [{agent.agent_id}] SHORTED {ticker}: -{actual_short_shares:.2f}sh @ ${exec_price:.2f}")

                                            broker_bridge.submit_market_order(ticker, actual_short_shares, "SHORT")

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
                                        logger.info(f"  📈 [{agent.agent_id}] COVERED {ticker}: +{cover_shares:.2f}sh @ ${exec_price:.2f}")

                                        broker_bridge.submit_market_order(ticker, cover_shares, "COVER")

                        # Print Competing Leaderboard & Log Snapshots
                        logger.info("\n🏆 --- COMPETING AGENT LEADERBOARD ---")
                        sorted_swarm = sorted(swarm_mgr.population, key=lambda a: a.equity_history[-1], reverse=True)
                        for rank, agent in enumerate(sorted_swarm, 1):
                            pnl = ((agent.equity_history[-1] - 100000.0) / 100000.0) * 100
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
                            await swarm_mgr.run_culling_cycle(prices=prices, risk_engine=risk_engine, db=db)

        except Exception as e:
            logger.error(f"❌ Redis subscriber connection lost: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5.0)

if __name__ == "__main__":
    asyncio.run(run_consumer())
