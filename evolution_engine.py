import os
import re
import json
import logging
import httpx
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger("EvolutionEngine")

@dataclass
class AgentGenome:
    agent_id: str
    persona_prompt: str
    generation: int = 1
    cash: float = 100000.0
    holdings: Dict[str, float] = field(default_factory=dict)
    entry_prices: Dict[str, float] = field(default_factory=dict)
    equity_history: List[float] = field(default_factory=lambda: [100000.0])

    def calculate_fitness(self) -> float:
        """
        Calculates risk-adjusted fitness score (Sharpe proxy) over rolling history window.
        Prevents division spikes when volatility is near zero.
        """
        if len(self.equity_history) < 2:
            return 0.0

        recent_history = self.equity_history[-50:]
        current_equity = recent_history[-1]
        pnl_pct = (current_equity - 100000.0) / 100000.0

        returns = [
            (recent_history[i] - recent_history[i-1]) / recent_history[i-1]
            for i in range(1, len(recent_history))
        ]
        if not returns:
            return round(pnl_pct, 4)

        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        fitness = pnl_pct / max(std_dev, 0.0001)
        return round(fitness, 4)

class EvolutionarySwarmManager:
    def __init__(self, api_key: str = None, population_size: int = 5):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.population_size = population_size
        self.current_generation = 1
        self.population: List[AgentGenome] = self._bootstrap_initial_population()

    def _bootstrap_initial_population(self) -> List[AgentGenome]:
        """Initializes 5 baseline agent genomes aligned with the Risk Parity Optimizer."""
        baseline_personas = [
            ("Agent_Alpha", "You are an Aggressive Growth Trader. Focus on high-momentum breakouts, outperforming SPY, and strong MACD expansion. Output high conviction (0.8-1.0) on top technical setups."),
            ("Agent_Beta", "You are a Conservative Risk Manager. Prioritize capital preservation, low volatility, and tight drawdown control. Issue buy/short signals with high conviction on clear setups."),
            ("Agent_Gamma", "You are a Mean-Reversion Trader. Exploit overbought (RSI>70) for short entries and oversold (RSI<30) extremes for buys."),
            ("Agent_Delta", "You are a Volatility Specialist. Exploit MACD trend divergences, shorting weak breakdowns and buying strong regime shifts."),
            ("Agent_Epsilon", "You are a Macro Balanced Indexer. Maintain broad multi-asset portfolio with long/short tactical overlays and moderate conviction scores.")
        ]
        
        return [
            AgentGenome(
                agent_id=name, 
                persona_prompt=prompt, 
                holdings={}, 
                entry_prices={}, 
                equity_history=[100000.0]
            )
            for name, prompt in baseline_personas
        ]

    async def run_culling_cycle(self, prices: dict = None, risk_engine = None, db = None, execution_bridge = None):
        """
        Executes True Darwinian Selection & Capital Transfer:
        1. Recalculates exact equity state using current asset prices.
        2. Ranks agents by risk-adjusted fitness score.
        3. Liquidates open positions of bottom agents and recovers their equity.
        4. Mutates winning parent into offspring using Gemma 4-31B.
        5. Reallocates 100% of recovered culled equity as inherited cash for offspring.
        """
        logger.info(f"🧬 --- EXECUTING DARWINIAN CULLING & INHERITANCE (GEN {self.current_generation}) ---")
        prices = prices or {}

        # 1. Update in-memory equity state
        if prices:
            for agent in self.population:
                asset_val = sum(agent.holdings.get(tk, 0.0) * prices[tk] for tk in agent.holdings if tk in prices)
                current_eq = round(agent.cash + asset_val, 2)
                if not agent.equity_history or agent.equity_history[-1] != current_eq:
                    agent.equity_history.append(current_eq)

        # 2. Rank agents by fitness
        self.population.sort(key=lambda x: x.calculate_fitness(), reverse=True)
        
        for idx, agent in enumerate(self.population):
            logger.info(
                f"   Rank #{idx+1} | {agent.agent_id:<25} | "
                f"Fitness: {agent.calculate_fitness():>8.4f} | "
                f"Equity: ${agent.equity_history[-1]:,.2f}"
            )

        survivors = self.population[:3]
        culled = self.population[3:]

        # 3. Mutate top performer into 2 offspring
        parent = survivors[0]
        offspring_1 = await self._mutate_genome(parent, "Higher Risk Sensitivity & Volatility Protection", 1)
        offspring_2 = await self._mutate_genome(parent, "Exploit Short-term Momentum Breakouts & Breakdown Shorts", 2)

        recipient_ids = [offspring_1.agent_id, offspring_2.agent_id]

        # 4. Liquidate culled agents & transfer capital
        total_recovered_equity = 0.0

        for dead in culled:
            dead_asset_val = sum(dead.holdings.get(tk, 0.0) * prices[tk] for tk in dead.holdings if tk in prices)
            dead_total_equity = max(0.0, dead.cash + dead_asset_val)
            total_recovered_equity += dead_total_equity

            # Execute database liquidation & capital transfer
            if db:
                db.cull_and_reallocate(
                    loser_agent_id=dead.agent_id,
                    recipient_agent_ids=recipient_ids,
                    current_prices=prices,
                    execution_bridge=execution_bridge
                )

            # Zero out memory state for culled agent
            dead.cash = 0.0
            dead.holdings = {}
            dead.entry_prices = {}
            logger.warning(f"  💀 CULLED & LIQUIDATED: {dead.agent_id} (Recovered Equity: ${dead_total_equity:,.2f})")

        # 5. Distribute inherited capital equally to offspring in memory
        share_per_offspring = round(total_recovered_equity / len(recipient_ids), 2) if recipient_ids else 0.0

        offspring_1.cash = share_per_offspring
        offspring_1.equity_history = [share_per_offspring]

        offspring_2.cash = share_per_offspring
        offspring_2.equity_history = [share_per_offspring]

        logger.info(f"🎁 [INHERITANCE] Offspring [{offspring_1.agent_id}] and [{offspring_2.agent_id}] inherited ${share_per_offspring:,.2f} starting cash each!")

        # 6. Update population state
        self.current_generation += 1
        self.population = survivors + [offspring_1, offspring_2]

        if db:
            for agent in self.population:
                db.register_agent(agent.agent_id)
        
        logger.info(f"🎉 Generation {self.current_generation} successfully spawned with 5 active agents!")

    async def _mutate_genome(self, parent: AgentGenome, mutation_trait: str, offspring_num: int) -> AgentGenome:
        """Queries Google AI Studio using Gemma 4-31B with rate-limit exception handling to mutate winning parent prompt."""
        prompt = f"""
You are an Evolutionary Prompt Engineer for trading algorithms.
A winning strategy prompt survived with high performance:
"{parent.persona_prompt}"

Create a slightly mutated version of this strategy prompt that incorporates the trait: "{mutation_trait}".
Ensure the prompt instructs the agent to evaluate technical theses and output trade actions (BUY, SELL, SHORT, COVER, or HOLD) with conviction scores (0.0 to 1.0).
Return ONLY a JSON object with key "new_prompt": {{"new_prompt": "string"}}
"""

        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Gemini API key is not set in environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemma-4-31b-it",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        mutated_prompt = parent.persona_prompt
        max_retries = 3
        backoff_factor = 2.0

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(url, json=payload, headers=headers, timeout=20.0)

                    if resp.status_code == 429:
                        sleep_time = backoff_factor ** (attempt + 1)
                        logger.warning(f"⚠️ [Rate Limit / 429] during genome mutation. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(sleep_time)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content).strip()
                    parsed = json.loads(cleaned)

                    if "new_prompt" in parsed and parsed["new_prompt"]:
                        mutated_prompt = parsed["new_prompt"]
                        logger.info("✅ Genome successfully mutated via [Google AI Studio - Gemma 4 31B]")
                        break

                except httpx.HTTPStatusError as hse:
                    logger.warning(f"⚠️ HTTP error {hse.response.status_code} during mutation: {hse.response.text}")
                    if hse.response.status_code in (400, 401, 403, 404):
                        break
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(backoff_factor ** (attempt + 1))
                except Exception as e:
                    logger.warning(f"⚠️ Mutation attempt failed: {e}")
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(backoff_factor ** (attempt + 1))

        base_parent_name = re.sub(r'^Gen\d+_', '', parent.agent_id)
        base_parent_name = re.sub(r'_v\d+$', '', base_parent_name)
        new_id = f"Gen{self.current_generation + 1}_{base_parent_name}_v{offspring_num}"
        
        logger.info(f"  👶 MUTATED OFFSPRING CREATED: {new_id}")
        
        return AgentGenome(
            agent_id=new_id,
            persona_prompt=mutated_prompt,
            generation=self.current_generation + 1,
            cash=0.0,  # Cash will be populated via inheritance
            holdings={},
            entry_prices={},
            equity_history=[0.0]
        )
