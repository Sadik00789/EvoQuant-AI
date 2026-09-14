import os
import re
import json
import logging
import uuid
import httpx
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import config

logger = logging.getLogger("EvolutionEngine")


def _derive_lineage_root(agent_id: str, persona_prompt: str = "") -> str:
    """Backfill lineage_root as Agent_Alpha/Beta/Gamma/Delta/Epsilon based on closest persona/id match."""
    try:
        aid = str(agent_id or "")
        low = aid.lower()
        for root in ("Agent_Alpha", "Agent_Beta", "Agent_Gamma", "Agent_Delta", "Agent_Epsilon"):
            if root.lower() in low:
                return root
        # GenX_Parent fragments: Gen2_Agent_Alpha_v1_abcd -> Agent_Alpha
        for root in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
            if root.lower() in low:
                return f"Agent_{root}"
        p = str(persona_prompt or "").lower()
        if "mean-reversion" in p or "mean reversion" in p or "contrarian" in p:
            return "Agent_Gamma"
        if "volatility" in p or "macd" in p:
            return "Agent_Delta"
        if "conservative" in p or "risk manager" in p:
            return "Agent_Beta"
        if "aggressive" in p or "growth" in p or "momentum" in p:
            return "Agent_Alpha"
        if "macro" in p or "balanced" in p or "index" in p:
            return "Agent_Epsilon"
    except Exception:
        pass
    return "Agent_Alpha"


def _mutate_trait(value: float, min_val: float, max_val: float, sigma: float = 0.02) -> float:
    """Gaussian perturbation with hard clipping: trait_new = clip(parent + N(0,0.02), min, max)."""
    try:
        return float(np.clip(float(value) + float(np.random.normal(0, sigma)), float(min_val), float(max_val)))
    except Exception:
        return float(np.clip(float(value), float(min_val), float(max_val)))


@dataclass
class AgentGenome:
    agent_id: str
    persona_prompt: str
    generation: int = 1
    initial_capital: float = 100000.0
    cash: float = 100000.0
    holdings: Dict[str, float] = field(default_factory=dict)
    entry_prices: Dict[str, float] = field(default_factory=dict)
    equity_history: List[float] = field(default_factory=lambda: [100000.0])
    tenure_ticks: int = 0
    lineage_root: str = "Agent_Alpha"
    is_elite: bool = False
    sentiment_weight: float = 0.55
    technical_weight: float = 0.45
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.050
    short_positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    margin_requirement: float = 0.50
    last_prices: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Backfill lineage for legacy agents constructed without explicit root
        try:
            if not getattr(self, "lineage_root", None):
                self.lineage_root = _derive_lineage_root(self.agent_id, self.persona_prompt)
            # Normalize bare roots like Alpha -> Agent_Alpha
            lr = str(self.lineage_root or "").strip()
            if lr in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
                self.lineage_root = f"Agent_{lr}"
                lr = self.lineage_root
            if not lr:
                self.lineage_root = _derive_lineage_root(self.agent_id, self.persona_prompt)
        except Exception:
            self.lineage_root = "Agent_Alpha"
        # Clamp evolvable traits into bounds and renormalize weights to sum 1.0
        try:
            self.sentiment_weight = float(np.clip(float(self.sentiment_weight), 0.20, 0.80))
            self.technical_weight = float(np.clip(float(self.technical_weight), 0.20, 0.80))
            s = float(self.sentiment_weight) + float(self.technical_weight)
            if s > 0:
                self.sentiment_weight = round(float(self.sentiment_weight) / s, 4)
                self.technical_weight = round(1.0 - float(self.sentiment_weight), 4)
            self.stop_loss_pct = float(np.clip(float(self.stop_loss_pct), 0.015, 0.045))
            self.take_profit_pct = float(np.clip(float(self.take_profit_pct), 0.030, 0.090))
        except Exception:
            pass

        # Auto-populate short positions from holdings if negative
        try:
            for tk, qty in list((self.holdings or {}).items()):
                if qty < 0 and tk not in self.short_positions:
                    ep = float(self.entry_prices.get(tk, 0.0) or 0.0)
                    self.short_positions[tk] = {"shares": abs(float(qty)), "entry_price": ep}
        except Exception:
            pass

    @property
    def available_cash(self) -> float:
        """True available purchasing power deducting collateralized short liability & margin hold."""
        try:
            short_liability = sum(
                pos['shares'] * self.last_prices.get(t, pos.get('entry_price', 0.0))
                for t, pos in self.short_positions.items()
            )
            margin_hold = short_liability * (1.0 + self.margin_requirement)
            return max(0.0, self.cash - margin_hold)
        except Exception:
            return max(0.0, self.cash)

    def calculate_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Institutional MTM equity: Cash + Longs - Shorts with zero price fallback guard."""
        if current_prices:
            for k, v in current_prices.items():
                if v and float(v) > 0:
                    self.last_prices[k] = float(v)

        long_value = 0.0
        short_liability = 0.0
        for ticker, shares in (self.holdings or {}).items():
            if shares == 0:
                continue
            pos_short = self.short_positions.get(ticker, {})
            fallback_price = float(pos_short.get('entry_price') or self.entry_prices.get(ticker, 0.0) or 0.0)
            price = float(self.last_prices.get(ticker, fallback_price) or fallback_price)
            if shares > 0:
                long_value += shares * price
            else:
                short_liability += abs(shares) * price

        return round(self.cash + long_value - short_liability, 2)

    def fill_order(self, ticker: str, action: str, shares: float, fill_price: float):
        """Execute order fill adhering to institutional margin and ledger rules."""
        act = action.upper()
        sh = abs(float(shares))
        px = float(fill_price)
        if px > 0:
            self.last_prices[ticker] = px

        if act == "BUY":
            self.cash -= sh * px
            curr_h = self.holdings.get(ticker, 0.0)
            old_entry = self.entry_prices.get(ticker, px)
            new_h = curr_h + sh
            if new_h > 0:
                self.entry_prices[ticker] = ((curr_h * old_entry) + (sh * px)) / new_h if curr_h > 0 else px
            self.holdings[ticker] = new_h
        elif act == "SELL":
            self.cash += sh * px
            new_h = self.holdings.get(ticker, 0.0) - sh
            self.holdings[ticker] = 0.0 if abs(new_h) < 1e-9 else new_h
            if abs(self.holdings[ticker]) < 1e-9:
                self.entry_prices[ticker] = 0.0
        elif act == "SHORT":
            self.cash += sh * px
            curr_pos = self.short_positions.get(ticker, {"shares": 0.0, "entry_price": px})
            tot_sh = curr_pos["shares"] + sh
            avg_ep = ((curr_pos["shares"] * curr_pos["entry_price"]) + (sh * px)) / tot_sh if tot_sh > 0 else px
            self.short_positions[ticker] = {"shares": tot_sh, "entry_price": avg_ep}
            self.entry_prices[ticker] = avg_ep
            self.holdings[ticker] = self.holdings.get(ticker, 0.0) - sh
        elif act == "COVER":
            self.cash -= sh * px
            curr_pos = self.short_positions.get(ticker, {"shares": sh, "entry_price": px})
            rem_sh = curr_pos["shares"] - sh
            if rem_sh > 1e-9:
                self.short_positions[ticker]["shares"] = rem_sh
            else:
                self.short_positions.pop(ticker, None)
            new_h = self.holdings.get(ticker, 0.0) + sh
            self.holdings[ticker] = 0.0 if abs(new_h) < 1e-9 else new_h
            if abs(self.holdings[ticker]) < 1e-9:
                self.entry_prices[ticker] = 0.0

    def max_drawdown(self) -> float:
        """Peak-to-trough max drawdown in [0,1)."""
        try:
            hist = [float(x) for x in (self.equity_history or []) if float(x) > 0]
            if len(hist) < 2:
                return 0.0
            peak = hist[0]
            mdd = 0.0
            for v in hist[1:]:
                if v > peak:
                    peak = v
                elif peak > 0:
                    dd = (peak - v) / peak
                    if dd > mdd:
                        mdd = dd
            return float(np.clip(mdd, 0.0, 0.99))
        except Exception:
            return 0.0

    def sortino_ratio(self) -> float:
        """Sortino proxy: mean(return) / downside_deviation with 0.0001 floor."""
        try:
            hist = [float(x) for x in (self.equity_history or [])[-50:] if float(x) > 0]
            if len(hist) < 2:
                base = float(self.initial_capital) if float(self.initial_capital) > 0 else 100000.0
                return (hist[-1] - base) / base if hist else 0.0
            rets = [(hist[i] - hist[i - 1]) / hist[i - 1] for i in range(1, len(hist)) if hist[i - 1] > 0]
            if not rets:
                return 0.0
            avg = sum(rets) / len(rets)
            downside = [r for r in rets if r < 0]
            if not downside:
                # No downside: reward consistency, scale by avg
                return round(avg / 0.0001 if avg != 0 else 0.0, 4)
            var = sum(x * x for x in downside) / len(rets)
            dd = max(var ** 0.5, 0.0001)
            return float(avg / dd)
        except Exception:
            return 0.0

    def calculate_fitness(self) -> float:
        """
        Strict fitness = Sortino_Ratio * (1 - Max_Drawdown).
        Falls back to relative PnL when history is degenerate. Preserves sign for elitism ranking.
        """
        if not self.equity_history:
            return 0.0
        recent = [eq for eq in self.equity_history[-50:] if eq is not None]
        recent = [float(x) for x in recent if isinstance(x, (int, float)) and float(x) > 0]
        if not recent:
            return 0.0
        base = float(self.initial_capital) if float(self.initial_capital) > 0 else 100000.0
        pnl_pct = (recent[-1] - base) / base if base > 0 else 0.0
        if len(recent) < 2:
            return round(float(pnl_pct), 4)
        try:
            sortino = float(self.sortino_ratio())
            # sortino_ratio already rounded in no-downside path; normalize otherwise
            mdd = float(self.max_drawdown())
            fitness = float(sortino) * (1.0 - float(mdd))
            # Guard NaN/inf, fallback to Sharpe-style PnL/vol
            if not np.isfinite(fitness):
                raise ValueError("non-finite fitness")
            # If sortino collapsed to 0 but PnL nonzero (flat vol), fallback to PnL to preserve ordering
            if fitness == 0.0 and pnl_pct != 0.0:
                rets = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent)) if recent[i - 1] > 0]
                if rets:
                    avg = sum(rets) / len(rets)
                    var = sum((r - avg) ** 2 for r in rets) / len(rets)
                    std = max(var ** 0.5, 0.0001)
                    fitness = float(pnl_pct) / float(std) * (1.0 - float(mdd))
            return round(float(fitness), 4)
        except Exception:
            return round(float(pnl_pct), 4)


class EvolutionarySwarmManager:
    def __init__(self, api_key: str = None, population_size: int = 5):
        self.api_key = api_key if api_key is not None else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.population_size = population_size
        self.current_generation = 1
        self.population: List[AgentGenome] = self._bootstrap_initial_population()

    def _bootstrap_initial_population(self) -> List[AgentGenome]:
        """Initializes 5 baseline agent genomes with distinct lineage roots and evolvable traits."""
        baseline_personas = [
            ("Agent_Alpha", "You are an Aggressive Growth Trader. Focus on high-momentum breakouts, outperforming SPY, and strong MACD expansion. Output high conviction (0.8-1.0) on top technical setups.", 0.55, 0.45, 0.025, 0.050),
            ("Agent_Beta", "You are a Conservative Risk Manager. Prioritize capital preservation, low volatility, and tight drawdown control. Issue buy/short signals with high conviction on clear setups.", 0.45, 0.55, 0.020, 0.040),
            ("Agent_Gamma", "You are a Mean-Reversion Trader. Exploit overbought (RSI>70) for short entries and oversold (RSI<30) extremes for buys.", 0.50, 0.50, 0.025, 0.055),
            ("Agent_Delta", "You are a Volatility Specialist. Exploit MACD trend divergences, shorting weak breakdowns and buying strong regime shifts.", 0.55, 0.45, 0.030, 0.060),
            ("Agent_Epsilon", "You are a Macro Balanced Indexer. Maintain broad multi-asset portfolio with long/short tactical overlays and moderate conviction scores.", 0.60, 0.40, 0.022, 0.045),
        ]
        return [
            AgentGenome(
                agent_id=name,
                persona_prompt=prompt,
                lineage_root=name,
                generation=1,
                initial_capital=100000.0,
                cash=100000.0,
                holdings={},
                entry_prices={},
                equity_history=[100000.0],
                tenure_ticks=1000,
                is_elite=False,
                sentiment_weight=sw,
                technical_weight=tw,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )
            for name, prompt, sw, tw, sl, tp in baseline_personas
        ]

    def _lineage_counts(self, agents: List[AgentGenome]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in agents:
            try:
                root = str(getattr(a, "lineage_root", "") or _derive_lineage_root(a.agent_id, a.persona_prompt))
            except Exception:
                root = "Agent_Alpha"
            counts[root] = counts.get(root, 0) + 1
        return counts

    def _would_violate_cap(self, agents: List[AgentGenome], candidate_root: str, cap: int = 2) -> bool:
        try:
            counts = self._lineage_counts(agents)
            return counts.get(str(candidate_root), 0) >= cap
        except Exception:
            return False

    def _create_immigrant(self, immigrant_num: int, cash: float = 0.0) -> AgentGenome:
        """Orthogonal immigrant injection when lineage cap blocks elite offspring."""
        if immigrant_num % 2 == 1:
            # Archetype A: Mean-Reverting Contrarian (fades RSI >75 / <25)
            persona = ("You are a Mean-Reverting Contrarian immigrant. Fade extreme RSI >75 with shorts and RSI <25 with buys. "
                       "Demand RSI confirmation, use tight 1.5pct stops, avoid chasing momentum, prefer reversals to VWAP.")
            root = "Immigrant_Contrarian"
            sw, tw, sl, tp = 0.35, 0.65, 0.015, 0.035
        else:
            # Archetype B: Low-Beta Volatility Hedger (favors GLD/TLT, tight stops)
            persona = ("You are a Low-Beta Volatility Hedger immigrant. Favor defensive GLD/TLT, low-beta quality, tight stops, "
                       "small size, hedge equity beta, prioritize capital preservation over growth.")
            root = "Immigrant_Hedger"
            sw, tw, sl, tp = 0.40, 0.60, 0.015, 0.030
        unique_suffix = uuid.uuid4().hex[:4]
        new_id = f"Gen{self.current_generation + 1}_Immigrant_v{immigrant_num}_{unique_suffix}"
        return AgentGenome(
            agent_id=new_id,
            persona_prompt=persona,
            lineage_root=root,
            generation=self.current_generation + 1,
            initial_capital=float(cash),
            cash=float(cash),
            holdings={},
            entry_prices={},
            equity_history=[float(cash)],
            tenure_ticks=0,
            is_elite=False,
            sentiment_weight=sw,
            technical_weight=tw,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )

    def _liquidate_agent_to_cash(self, agent: AgentGenome, prices: dict, db=None, execution_bridge=None, reason: str = "CULLING") -> float:
        """Clean position liquidation: market sell/cover at snapshot prices, credit cash, log trades. Returns recovered equity."""
        try:
            prices = prices or {}
            for tk, shares in list((agent.holdings or {}).items()):
                try:
                    qty = float(shares or 0.0)
                    if qty == 0:
                        continue
                    px = float(prices.get(tk, agent.entry_prices.get(tk, 0.0) or 0.0) or 0.0)
                    if px <= 0:
                        px = float(agent.entry_prices.get(tk, 1.0) or 1.0)
                    if qty > 0:
                        agent.cash = float(agent.cash) + qty * px
                        action = "SELL"
                    else:
                        agent.cash = float(agent.cash) - abs(qty) * px
                        action = "COVER"
                    if db is not None:
                        try:
                            db.log_trade(agent.agent_id, tk, action, abs(qty), px, 0.0, reason=reason)
                        except Exception:
                            pass
                    if execution_bridge is not None:
                        try:
                            if hasattr(execution_bridge, "is_active") and execution_bridge.is_active():
                                execution_bridge.submit_market_order(tk, abs(qty), action)
                        except Exception:
                            pass
                    agent.holdings[tk] = 0.0
                    try:
                        agent.entry_prices[tk] = 0.0
                    except Exception:
                        pass
                except Exception:
                    continue
            long_val = sum(qty * prices.get(tk, agent.entry_prices.get(tk, 0.0) or 0.0) for tk, qty in (agent.holdings or {}).items() if qty > 0)
            short_liab = sum(abs(qty) * prices.get(tk, agent.entry_prices.get(tk, 0.0) or 0.0) for tk, qty in (agent.holdings or {}).items() if qty < 0)
            return max(0.0, round(float(agent.cash) + float(long_val) - float(short_liab), 2))
        except Exception:
            try:
                return max(0.0, float(agent.cash))
            except Exception:
                return 0.0

    async def run_culling_cycle(self, prices: dict = None, risk_engine=None, db=None, execution_bridge=None):
        """
        Strict-elitism Darwinian selection with lineage capping and immigrant injection:
        1. Recalculates equity, ranks by fitness = Sortino*(1-MaxDD).
        2. #1 agent is elite: never liquidated, exempt from culling.
        3. Enforces max 2 agents per lineage_root; forces alternate parent if cap violated.
        4. Injects orthogonal immigrants when slots cannot be filled without violating cap.
        5. Clean liquidation + 100pct cash transfer to offspring (zero-sum).
        """
        logger.info(f"🧬 --- EXECUTING DARWINIAN CULLING & INHERITANCE (GEN {self.current_generation}) ---")
        prices = prices or {}
        # Backfill lineage for legacy agents
        for a in self.population:
            try:
                if not getattr(a, "lineage_root", None):
                    a.lineage_root = _derive_lineage_root(a.agent_id, a.persona_prompt)
                a.is_elite = False
            except Exception:
                pass
        # 1. Update in-memory equity state (mark-to-market symmetric)
        if prices:
            for agent in self.population:
                try:
                    long_val = sum(qty * prices.get(tk, agent.entry_prices.get(tk, 0.0)) for tk, qty in agent.holdings.items() if qty > 0)
                    short_liability = sum(abs(qty) * prices.get(tk, agent.entry_prices.get(tk, 0.0)) for tk, qty in agent.holdings.items() if qty < 0)
                    current_eq = round(agent.cash + long_val - short_liability, 2)
                    if not agent.equity_history or agent.equity_history[-1] != current_eq:
                        agent.equity_history.append(current_eq)
                except Exception:
                    continue
        # 2. Rank by fitness; mark strict elite
        ranked = sorted(self.population, key=lambda x: x.calculate_fitness(), reverse=True)
        elite = ranked[0] if ranked else None
        if elite is not None:
            elite.is_elite = True
            for a in ranked[1:]:
                a.is_elite = False
        # Tenure grace: immune infants, but elite always immune even if mature
        immune_agents = [a for a in ranked if a.tenure_ticks < 1000 and a.generation > 1]
        mature_agents = [a for a in ranked if a not in immune_agents]
        for idx, agent in enumerate(ranked):
            logger.info(
                f"   Rank #{idx+1} | {agent.agent_id:<32} | "
                f"Fitness: {agent.calculate_fitness():>8.4f} | "
                f"Equity: ${agent.equity_history[-1]:,.2f} | "
                f"Lineage: {getattr(agent, 'lineage_root', '?')} | "
                f"Tenure: {agent.tenure_ticks} ticks {'(Elite)' if getattr(agent, 'is_elite', False) else ('(Immune)' if agent in immune_agents else '')}"
            )
        # 3. Select culled: lowest fitness among mature non-elite first, never elite
        candidates = [a for a in list(reversed(mature_agents)) + list(reversed(immune_agents)) if not getattr(a, "is_elite", False)]
        # Fallback: if all mature are elite/immune edge, cull lowest non-elite overall
        if len(candidates) < 2:
            extra = [a for a in reversed(ranked) if not getattr(a, "is_elite", False) and a not in candidates]
            candidates += extra
        culled = candidates[:2]
        survivors = [a for a in self.population if a not in culled]
        survivors.sort(key=lambda x: x.calculate_fitness(), reverse=True)
        # 4. Parent selection with lineage cap (max 2 per root)
        elite_parent = survivors[0] if survivors else ranked[0]
        second_parent = None
        for cand in survivors[1:]:
            if not self._would_violate_cap([s for s in survivors if s is not elite_parent] + [elite_parent], getattr(cand, "lineage_root", "")):
                # Simulate post-birth counts: survivors + 2 offspring from elite lineage
                sim = list(survivors) + [elite_parent]
                # If both offspring inherit elite root, check cap
                elite_root = str(getattr(elite_parent, "lineage_root", ""))
                counts = self._lineage_counts(sim)
                # offspring would add 2 to elite_root
                if counts.get(elite_root, 0) + 1 > 2:
                    # Force alternate lineage for second parent
                    if str(getattr(cand, "lineage_root", "")) != elite_root:
                        second_parent = cand
                        break
                    continue
                second_parent = cand
                break
        if second_parent is None:
            for cand in survivors[1:]:
                if str(getattr(cand, "lineage_root", "")) != str(getattr(elite_parent, "lineage_root", "")):
                    second_parent = cand
                    break
        if second_parent is None and len(survivors) > 1:
            second_parent = survivors[1]
        elif second_parent is None:
            second_parent = elite_parent
        # 5. Spawn offspring, respecting cap via immigrants
        offspring_1 = await self._mutate_genome(elite_parent, "Higher Risk Sensitivity & Volatility Protection", 1)
        # Decide offspring_2: elite clone vs alternate parent vs immigrant
        elite_root = str(getattr(elite_parent, "lineage_root", ""))
        surv_roots = self._lineage_counts(survivors)
        # Projected count if offspring_2 also from elite
        proj_elite = surv_roots.get(elite_root, 0) + 2  # off1 + off2 both elite
        if proj_elite > 2:
            # Try alternate parent offspring
            alt_root = str(getattr(second_parent, "lineage_root", ""))
            proj_alt = surv_roots.get(alt_root, 0) + 1
            if alt_root != elite_root and proj_alt <= 2:
                offspring_2 = await self._mutate_genome(second_parent, "Exploit Short-term Momentum Breakouts & Breakdown Shorts", 2)
            else:
                # Orthogonal immigrant injection
                offspring_2 = self._create_immigrant(2, cash=0.0)
                logger.warning(f"  🧬 Lineage cap blocked elite clone; injected orthogonal {offspring_2.lineage_root} [{offspring_2.agent_id}]")
        else:
            offspring_2 = await self._mutate_genome(second_parent if second_parent is not elite_parent else elite_parent, "Exploit Short-term Momentum Breakouts & Breakdown Shorts", 2)
        # Final safety: if survivors + offspring violate cap, convert off2 to immigrant
        final_counts = self._lineage_counts(survivors + [offspring_1, offspring_2])
        if any(v > 2 for v in final_counts.values()):
            offspring_2 = self._create_immigrant(2, cash=0.0)
            logger.warning(f"  🧬 Post-hoc lineage enforcement; replaced with {offspring_2.lineage_root} [{offspring_2.agent_id}]")
        recipient_ids = [offspring_1.agent_id, offspring_2.agent_id]
        # 6. Clean liquidation + capital transfer (zero-sum)
        total_recovered_equity = 0.0
        for dead in culled:
            dead_total_equity = self._liquidate_agent_to_cash(dead, prices, db=None, execution_bridge=None, reason="CULLING")
            total_recovered_equity += dead_total_equity
            if db:
                try:
                    db.cull_and_reallocate(
                        loser_agent_id=dead.agent_id,
                        recipient_agent_ids=recipient_ids,
                        current_prices=prices,
                        execution_bridge=execution_bridge
                    )
                except Exception as e:
                    logger.warning(f"Cull DB note [{dead.agent_id}]: {e}")
            dead.cash = 0.0
            dead.holdings = {}
            dead.entry_prices = {}
            dead.equity_history.append(0.0)
            dead.is_elite = False
            logger.warning(f"  💀 CULLED & LIQUIDATED: {dead.agent_id} (Recovered Equity: ${dead_total_equity:,.2f})")
        share_per_offspring_1 = round(total_recovered_equity / 2.0, 2) if recipient_ids else 0.0
        share_per_offspring_2 = round(total_recovered_equity - share_per_offspring_1, 2) if recipient_ids else 0.0
        offspring_1.cash = share_per_offspring_1
        offspring_1.initial_capital = share_per_offspring_1
        offspring_1.equity_history = [share_per_offspring_1]
        offspring_1.tenure_ticks = 0
        offspring_1.is_elite = False
        offspring_2.cash = share_per_offspring_2
        offspring_2.initial_capital = share_per_offspring_2
        offspring_2.equity_history = [share_per_offspring_2]
        offspring_2.tenure_ticks = 0
        offspring_2.is_elite = False
        logger.info(f"🎁 [INHERITANCE] Offspring [{offspring_1.agent_id}] (${share_per_offspring_1:,.2f}) and [{offspring_2.agent_id}] (${share_per_offspring_2:,.2f}) successfully instantiated!")
        self.current_generation += 1
        self.population = survivors + [offspring_1, offspring_2]
        if db:
            for agent in self.population:
                try:
                    db.register_agent(agent.agent_id)
                except Exception:
                    pass
        logger.info(f"🎉 Generation {self.current_generation} successfully spawned with 5 active agents!")
        return [dead.agent_id for dead in culled], recipient_ids

    async def _mutate_genome(self, parent: AgentGenome, mutation_trait: str, offspring_num: int) -> AgentGenome:
        """Queries Google AI Studio using Gemma 4-31B with rate-limit and robust JSON parsing to mutate winning parent prompt."""
        prompt = f"""
You are an Evolutionary Prompt Engineer for quantitative trading systems.
A winning strategy prompt survived with superior performance:
"{parent.persona_prompt}"

Create a mutated version of this strategy prompt that incorporates the trait: "{mutation_trait}".
Ensure the prompt instructs the agent to evaluate technical theses and output trade actions (BUY, SELL, SHORT, COVER, or HOLD) with conviction scores (0.0 to 1.0).
Return ONLY a valid JSON object: {{"new_prompt": "string"}}
"""

        # Preserve legacy semantics: an explicitly-supplied empty string means
        # "offline"; only fall back to the environment when api_key is None.
        api_key = self.api_key if self.api_key is not None else (
            config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )

        mutated_prompt = parent.persona_prompt

        if not api_key:
            mutated_prompt = f"{parent.persona_prompt} (Mutated: {mutation_trait})"
        else:
            # Native Google auth is via the ?key= query param in the URL.
            headers = {"Content-Type": "application/json"}

            models = [config.GEMINI_MODEL]
            if config.GEMINI_FALLBACK_MODEL and config.GEMINI_FALLBACK_MODEL not in models:
                models.append(config.GEMINI_FALLBACK_MODEL)

            max_retries = 3
            backoff_factor = 2.0
            fallback_status = (400, 404, 500, 502, 503, 504)
            request_timeout = httpx.Timeout(
                timeout=float(config.GEMINI_TIMEOUT_SECONDS),
                connect=float(config.GEMINI_CONNECT_TIMEOUT_SECONDS),
            )

            async with httpx.AsyncClient(timeout=request_timeout) as client:
                for model in models:
                    url = config.gemini_generate_url(model, api_key)
                    gen_config: Dict[str, Any] = {
                        "maxOutputTokens": 2048,
                    }
                    if "gemma" in str(model).lower() or "thinking" in str(model).lower():
                        gen_config["thinkingConfig"] = {"includeThoughts": True}
                        gen_config["temperature"] = 1.0
                    else:
                        gen_config["temperature"] = 0.3

                    payload = {
                        "contents": [
                            {
                                "parts": [{"text": prompt}]
                            }
                        ],
                        "generationConfig": gen_config,
                    }
                    resolved = False
                    for attempt in range(max_retries):
                        try:
                            resp = await client.post(url, json=payload, headers=headers)

                            if resp.status_code == 429:
                                sleep_time = backoff_factor ** (attempt + 1)
                                logger.warning(f"⚠️ [Rate Limit / 429] during genome mutation on '{model}'. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                                if attempt == max_retries - 1:
                                    break
                                await asyncio.sleep(sleep_time)
                                continue

                            if resp.status_code in fallback_status:
                                if model != models[-1]:
                                    logger.warning(f"⚠️ Model '{model}' returned HTTP {resp.status_code} during mutation; retrying with fallback model.")
                                    break
                                elif resp.status_code in (500, 502, 503, 504):
                                    sleep_time = backoff_factor ** (attempt + 1)
                                    logger.warning(f"⚠️ Model '{model}' returned transient HTTP {resp.status_code} during mutation. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                                    if attempt == max_retries - 1:
                                        break
                                    await asyncio.sleep(sleep_time)
                                    continue
                                else:
                                    logger.warning(f"⚠️ Model '{model}' returned non-retryable HTTP {resp.status_code} during mutation.")
                                    break

                            resp.raise_for_status()
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if not candidates:
                                logger.warning(f"⚠️ No candidate content from '{model}' during mutation: {data}")
                                break
                            content = config.gemini_extract_text(candidates)

                            # Strip markdown formatting
                            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content).strip()

                            parsed = {}
                            try:
                                parsed = json.loads(cleaned)
                            except Exception:
                                # Robust fallback search for new_prompt inside the response
                                match = re.search(r'\{[\s\S]*"new_prompt"\s*:\s*"([^"]+)"[\s\S]*\}', content)
                                if match:
                                    parsed = {"new_prompt": match.group(1)}

                            if "new_prompt" in parsed and parsed["new_prompt"]:
                                mutated_prompt = parsed["new_prompt"]
                                logger.info(f"✅ Genome successfully mutated via Google AI Studio ('{model}').")
                                resolved = True
                                break
                            else:
                                logger.warning(f"⚠️ Empty or unparseable prompt response: {content[:100]}")

                        except httpx.HTTPStatusError as hse:
                            code = hse.response.status_code if hse.response is not None else 0
                            body = hse.response.text if hse.response is not None else str(hse)
                            logger.warning(f"⚠️ HTTP error {code} during mutation: {body}")
                            if code in fallback_status:
                                if model != models[-1]:
                                    break
                                elif code in (500, 502, 503, 504):
                                    sleep_time = backoff_factor ** (attempt + 1)
                                    logger.warning(f"⚠️ Transient HTTP {code} during mutation for '{model}'. Retrying in {sleep_time}s...")
                                    if attempt == max_retries - 1:
                                        break
                                    await asyncio.sleep(sleep_time)
                                    continue
                                else:
                                    break
                            if code in (401, 403):
                                break
                            if attempt == max_retries - 1:
                                break
                            await asyncio.sleep(backoff_factor ** (attempt + 1))
                        except Exception as e:
                            logger.warning(f"⚠️ Mutation attempt failed: {e}")
                            if attempt == max_retries - 1:
                                break
                            await asyncio.sleep(backoff_factor ** (attempt + 1))
                    if resolved:
                        break

        # Monotonic Unique ID Generation using current generation, parent tag, offspring index, and UUID hash
        base_parent_name = re.sub(r'^Gen\d+_', '', parent.agent_id)
        base_parent_name = re.sub(r'_v\d+.*$', '', base_parent_name)
        unique_suffix = uuid.uuid4().hex[:4]
        new_id = f"Gen{self.current_generation + 1}_{base_parent_name}_v{offspring_num}_{unique_suffix}"
        logger.info(f"  👶 MUTATED OFFSPRING CREATED: {new_id}")
        # Evolvable quantitative traits via Gaussian perturbations, renormalized to sum 1.0
        try:
            parent_sw = float(getattr(parent, "sentiment_weight", 0.55))
            parent_tw = float(getattr(parent, "technical_weight", 0.45))
            parent_sl = float(getattr(parent, "stop_loss_pct", 0.025))
            parent_tp = float(getattr(parent, "take_profit_pct", 0.050))
        except Exception:
            parent_sw, parent_tw, parent_sl, parent_tp = 0.55, 0.45, 0.025, 0.050
        child_sw = _mutate_trait(parent_sw, 0.20, 0.80, sigma=0.02)
        child_tw = _mutate_trait(parent_tw, 0.20, 0.80, sigma=0.02)
        s = child_sw + child_tw
        if s > 0:
            child_sw = round(child_sw / s, 4)
            child_tw = round(1.0 - child_sw, 4)
        child_sl = _mutate_trait(parent_sl, 0.015, 0.045, sigma=0.02)
        # Take-profit uses smaller sigma relative to wider range to avoid excessive jumps
        try:
            child_tp = float(np.clip(float(parent_tp) + float(np.random.normal(0, 0.02)), 0.030, 0.090))
        except Exception:
            child_tp = float(np.clip(float(parent_tp), 0.030, 0.090))
        try:
            parent_root = str(getattr(parent, "lineage_root", "") or _derive_lineage_root(parent.agent_id, parent.persona_prompt))
        except Exception:
            parent_root = "Agent_Alpha"
        return AgentGenome(
            agent_id=new_id,
            persona_prompt=mutated_prompt,
            lineage_root=parent_root,
            generation=self.current_generation + 1,
            initial_capital=0.0,
            cash=0.0,
            holdings={},
            entry_prices={},
            equity_history=[0.0],
            tenure_ticks=0,
            is_elite=False,
            sentiment_weight=round(float(child_sw), 4),
            technical_weight=round(float(child_tw), 4),
            stop_loss_pct=round(float(child_sl), 4),
            take_profit_pct=round(float(child_tp), 4),
        )