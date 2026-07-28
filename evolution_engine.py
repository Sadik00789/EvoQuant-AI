import os
import json
import logging
import httpx
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Dict

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
        Calculates risk-adjusted fitness score (Sharpe proxy).
        Prevents division spikes when volatility is near zero.
        """
        current_equity = self.equity_history[-1]
        pnl_pct = (current_equity - 100000.0) / 100000.0
        
        if len(self.equity_history) < 2:
            return round(pnl_pct, 4)

        returns = [
            (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            for i in range(1, len(self.equity_history))
        ]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        # Smooth minimum denominator to prevent extreme fitness spikes when std_dev is near 0
        fitness = pnl_pct / max(std_dev, 0.0001)
        return round(fitness, 4)

class EvolutionarySwarmManager:
    def __init__(self, api_key: str = None, population_size: int = 5):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.population_size = population_size
        self.current_generation = 1
        self.population: List[AgentGenome] = self._bootstrap_initial_population()

    def _bootstrap_initial_population(self) -> List[AgentGenome]:
        """Initializes 5 diverse baseline agent genomes aligned with the Risk Parity Optimizer."""
        baseline_personas = [
            ("Agent_Alpha", "You are an Aggressive Growth Trader. Focus on high-momentum breakouts, outperforming SPY, and strong MACD expansion. Output high conviction (0.8-1.0) on top technical setups."),
            ("Agent_Beta", "You are a Conservative Risk Manager. Prioritize capital preservation, low volatility, and tight drawdown control. Only issue buy signals with high conviction on low ATR defensive value assets."),
            ("Agent_Gamma", "You are a Mean-Reversion Trader. Exploit overbought (RSI>70) and oversold (RSI<30) extremes. Issue SELL or HOLD signals during range consolidation."),
            ("Agent_Delta", "You are a Volatility Specialist. Exploit MACD trend divergences and sudden market regime shifts with dynamic conviction scaling."),
            ("Agent_Epsilon", "You are a Macro Balanced Indexer. Maintain broad diversification across top market caps with steady, moderate conviction scores.")
        ]
        
        return [
            AgentGenome(agent_id=name, persona_prompt=prompt)
            for name, prompt in baseline_personas
        ]

    async def run_culling_cycle(self):
        """
        Executes Darwinian selection:
        1. Ranks agents by fitness score.
        2. Terminates the bottom 2 underperforming agents.
        3. Mutates top performer into 2 new offspring using multi-provider 70B models.
        """
        logger.info(f"🧬 --- EXECUTING EVOLUTIONARY CULLING (GEN {self.current_generation}) ---")

        # 1. Rank by fitness score
        self.population.sort(key=lambda x: x.calculate_fitness(), reverse=True)
        
        for idx, agent in enumerate(self.population):
            logger.info(f"  Rank #{idx+1} | {agent.agent_id:<25} | Fitness: {agent.calculate_fitness():>8.4f} | Equity: ${agent.equity_history[-1]:,.2f}")

        # 2. Kill the bottom 2 performers
        survivors = self.population[:3]
        culled = self.population[3:]

        for dead in culled:
            logger.warning(f"  💀 CULLED: {dead.agent_id} (Fitness: {dead.calculate_fitness()}) was terminated.")

        # 3. Generate 2 new mutated offspring from top performer
        parent = survivors[0]
        offspring_1 = await self._mutate_genome(parent, "Higher Risk Sensitivity & Volatility Protection", 1)
        offspring_2 = await self._mutate_genome(parent, "Exploit Short-term Momentum Breakouts", 2)

        self.current_generation += 1
        self.population = survivors + [offspring_1, offspring_2]
        
        logger.info(f"🎉 Generation {self.current_generation} successfully spawned with 5 active agents!")

    async def _mutate_genome(self, parent: AgentGenome, mutation_trait: str, offspring_num: int) -> AgentGenome:
        """Queries 70B models with fallback chain to semantically mutate winning parent prompt."""
        prompt = f"""
        You are an Evolutionary Prompt Engineer for trading algorithms.
        A winning strategy prompt survived with high performance:
        "{parent.persona_prompt}"

        Create a slightly mutated version of this strategy prompt that incorporates the trait: "{mutation_trait}".
        Ensure the prompt instructs the agent to evaluate technical theses and output trade actions (BUY/SELL/HOLD) with conviction scores (0.0 to 1.0).
        Return ONLY a JSON object with key "new_prompt": {{"new_prompt": "string"}}
        """

        providers = [
            {
                "name": "Groq",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "key": self.api_key or os.getenv("GROQ_API_KEY"),
                "model": "llama-3.3-70b-versatile"
            },
            {
                "name": "OpenRouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": os.getenv("OPENROUTER_API_KEY"),
                "model": "meta-llama/llama-3.3-70b-instruct"
            },
            {
                "name": "GitHub Models",
                "url": "https://models.inference.ai.azure.com/chat/completions",
                "key": os.getenv("GITHUB_TOKEN"),
                "model": "Llama-3.3-70B-Instruct"
            },
            {
                "name": "SambaNova",
                "url": "https://api.sambanova.ai/v1/chat/completions",
                "key": os.getenv("SAMBANOVA_API_KEY"),
                "model": "Meta-Llama-3.3-70B-Instruct"
            }
        ]

        mutated_prompt = parent.persona_prompt  # Fallback if all providers fail

        async with httpx.AsyncClient() as client:
            for p in providers:
                if not p["key"]:
                    continue

                headers = {
                    "Authorization": f"Bearer {p['key']}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": p["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3
                }

                try:
                    resp = await client.post(p["url"], json=payload, headers=headers, timeout=20.0)
                    if resp.status_code in (429, 404):
                        logger.warning(f"⚠️ [{p['name']}] Mutation failed ({resp.status_code}). Trying next provider...")
                        continue

                    resp.raise_for_status()
                    content = resp.json()['choices'][0]['message']['content']
                    cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()
                    parsed = json.loads(cleaned)
                    
                    if "new_prompt" in parsed:
                        mutated_prompt = parsed["new_prompt"]
                        logger.info(f"✅ Genome successfully mutated via [{p['name']}]")
                        break

                except Exception as e:
                    logger.warning(f"⚠️ Mutation attempt failed on [{p['name']}]: {e}")
                    continue

        # Clean parent base name to prevent recursive ID growth
        base_parent_name = re.sub(r'^Gen\d+_', '', parent.agent_id)
        base_parent_name = re.sub(r'_v\d+$', '', base_parent_name)
        new_id = f"Gen{self.current_generation + 1}_{base_parent_name}_v{offspring_num}"
        
        logger.info(f"  👶 MUTATED OFFSPRING CREATED: {new_id}")
        
        return AgentGenome(
            agent_id=new_id,
            persona_prompt=mutated_prompt,
            generation=self.current_generation + 1,
            cash=100000.0,
            holdings={},
            entry_prices={},
            equity_history=[100000.0]
        )