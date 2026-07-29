import os
import json
import asyncio
import httpx
import redis.asyncio as redis
import pandas as pd
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
    logger.info("🛡️ Hard Risk Overlay Active: Stop-Loss (-2.5%) | Take-Profit (+5.0%)")
    logger.info("📰 News RAG Sentiment & Dynamic Market Impact Slippage Engines Active.")
    if broker_bridge.is_active():
        logger.info("⚡ ALPACA PAPER TRADING BROKER BRIDGE ACTIVE.")

    tick_counter = 0
    macro_multiplier = 1.0
    spy_returns_history = []
    spy_prices_history = []

    # Initial Macro Sentiment Fetch & Database Log
    try:
        macro_news = sentiment_agent.analyze_macro_sentiment()
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

                        # Refresh news sentiment every 8 market ticks (~2 hours)
                        if tick_counter % 8 == 0:
                            try:
                                macro_news = sentiment_agent.analyze_macro_sentiment()
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
                            for tk, shares in agent.holdings.items() if shares > 0
                        })

                        shared_thesis = swarm.analyze_technical_state(market_state, active_holdings=all_active_holdings)

                        # -------------------------------------------------------------
                        # PHASE A: HARD DETERMINISTIC RISK GUARD CHECK
                        # -------------------------------------------------------------
                        for agent in swarm_mgr.population:
                            if not hasattr(agent, 'entry_prices'):
                                agent.entry_prices = {}

                            for tk, shares in list(agent.holdings.items()):
                                if shares > 0 and tk in prices:
                                    current_price = prices[tk]
                                    entry_price = agent.entry_prices.get(tk, current_price)
                                    pos_pnl = (current_price - entry_price) / entry_price
                                    adv = market_state.get(tk, {}).get("adv", 1000000.0)

                                    # Hard Stop-Loss Liquidation
                                    if pos_pnl <= STOP_LOSS_PCT:
                                        exec_price = risk_engine.calculate_execution_price(current_price, shares, adv, "SELL")
                                        cash_returned = shares * exec_price
                                        agent.cash += cash_returned
                                        agent.holdings[tk] = 0.0
                                        agent.entry_prices[tk] = 0.0

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                                        db.log_trade(agent.agent_id, tk, "SELL", shares, exec_price, 0.0, reason="HARD_STOP_LOSS")
                                        logger.warning(f"  🚨 [{agent.agent_id}] HARD STOP-LOSS TRIGGERED on {tk}: Liquidation @ ${exec_price:.2f} ({pos_pnl*100:.2f}%)")

                                        broker_bridge.submit_market_order(tk, shares, "SELL")

                                    # Hard Take-Profit Liquidation
                                    elif pos_pnl >= TAKE_PROFIT_PCT:
                                        exec_price = risk_engine.calculate_execution_price(current_price, shares, adv, "SELL")
                                        cash_returned = shares * exec_price
                                        agent.cash += cash_returned
                                        agent.holdings[tk] = 0.0
                                        agent.entry_prices[tk] = 0.0

                                        db.update_agent_cash(agent.agent_id, agent.cash)
                                        db.update_agent_holding(agent.agent_id, tk, 0.0, 0.0)
                                        db.log_trade(agent.agent_id, tk, "SELL", shares, exec_price, 0.0, reason="HARD_TAKE_PROFIT")
                                        logger.info(f"  🎯 [{agent.agent_id}] HARD TAKE-PROFIT TRIGGERED on {tk}: Liquidation @ ${exec_price:.2f} (+{pos_pnl*100:.2f}%)")

                                        broker_bridge.submit_market_order(tk, shares, "SELL")

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
                            asset_val = sum(agent.holdings.get(tk, 0.0) * prices[tk] for tk in agent.holdings if tk in prices)
                            current_equity = round(agent.cash + asset_val, 2)
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
                                current_val = agent.holdings.get(ticker, 0.0) * raw_price
                                delta = target_val - current_val

                                # BUY Execution with Weighted Average Entry Price
                                if delta > 50.0 and target.action == "BUY" and agent.cash >= delta:
                                    approx_shares = delta / raw_price
                                    exec_price = risk_engine.calculate_execution_price(raw_price, approx_shares, adv, "BUY")
                                    shares = delta / exec_price

                                    old_shares = agent.holdings.get(ticker, 0.0)
                                    old_entry = agent.entry_prices.get(ticker, exec_price)
                                    new_shares = old_shares + shares
                                    weighted_entry = ((old_shares * old_entry) + (shares * exec_price)) / new_shares

                                    agent.holdings[ticker] = new_shares
                                    agent.entry_prices[ticker] = weighted_entry
                                    agent.cash -= delta

                                    db.update_agent_cash(agent.agent_id, agent.cash)
                                    db.update_agent_holding(agent.agent_id, ticker, agent.holdings[ticker], weighted_entry)
                                    db.log_trade(agent.agent_id, ticker, "BUY", shares, exec_price, effective_alloc, reason="RISK_PARITY_ALLOCATION")
                                    logger.info(f"  📈 [{agent.agent_id}] BOUGHT {ticker}: +{shares:.2f}sh @ ${exec_price:.2f} (Avg Cost: ${weighted_entry:.2f})")

                                    broker_bridge.submit_market_order(ticker, shares, "BUY")

                                # SELL Execution Clamped to Available Holdings
                                current_qty = agent.holdings.get(ticker, 0.0)
                                if delta < -50.0 and target.action == "SELL" and current_qty > 0:
                                    sell_shares = min(abs(delta) / raw_price, current_qty)
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

                        # Print Competing Leaderboard & Log Snapshots
                        logger.info("\n🏆 --- COMPETING AGENT LEADERBOARD ---")
                        sorted_swarm = sorted(swarm_mgr.population, key=lambda a: a.equity_history[-1], reverse=True)
                        for rank, agent in enumerate(sorted_swarm, 1):
                            pnl = ((agent.equity_history[-1] - 100000.0) / 100000.0) * 100
                            db.log_snapshot(agent.agent_id, agent.equity_history[-1], agent.cash, pnl)

                            active_holdings = [f"{tk}: {shares:.1f}sh" for tk, shares in agent.holdings.items() if shares > 0]
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
