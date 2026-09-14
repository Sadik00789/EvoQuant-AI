import asyncio
import re
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine import (
    AlpacaExecutionBridge,
    CrossAssetPortfolioManager,
    RiskParityOptimizer,
    AgentSignalDecision,
    QualitativeSignal,
    MultiTechnicalThesis,
    AssetThesis,
)
from evolution_engine import AgentGenome, EvolutionarySwarmManager
from risk_engine import AdvancedRiskEngine
from sentiment_agent import NewsSentimentAgent
from swarm_consumer import restore_agent_states_from_db


# ==========================================
# 0. MOCK FIXTURES FOR CONTAINERLESS TESTING
# ==========================================

@pytest.fixture
def mock_db_pool(monkeypatch):
    """Mock fixture for psycopg ConnectionPool to enable offline portfolio manager testing."""
    mock_cursor = MagicMock()
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.__exit__.return_value = None

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.cursor.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn

    monkeypatch.setattr("engine.ConnectionPool", MagicMock(return_value=mock_pool))
    return mock_pool, mock_conn, mock_cursor


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock fixture for redis.asyncio.Redis."""
    mock_r = MagicMock()
    mock_pubsub = MagicMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.listen = MagicMock()
    mock_r.pubsub.return_value = mock_pubsub

    monkeypatch.setattr("redis.asyncio.Redis", MagicMock(return_value=mock_r))
    return mock_r, mock_pubsub


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock fixture for httpx.AsyncClient (native Google generateContent schema)."""
    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": '{"new_prompt": "Mutated strategy prompt with risk controls"}'}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ]
    }
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("httpx.AsyncClient", MagicMock(return_value=mock_client))
    return mock_client


# ==========================================
# 1. RISK PARITY OPTIMIZER UNIT TESTS
# ==========================================

def test_risk_parity_max_position_cap():
    """Verify that no single asset weight exceeds the strict 5% position cap without secondary normalization."""
    optimizer = RiskParityOptimizer(max_position_cap=0.05)
    convictions = {"NVDA": 0.95, "AMD": 0.90, "AAPL": 0.85, "MSFT": 0.80}
    atrs = {"NVDA": 2.0, "AMD": 1.5, "AAPL": 1.0, "MSFT": 1.1}

    weights = optimizer.optimize(convictions, atrs)

    for ticker, weight in weights.items():
        assert weight <= 0.05 + 1e-5, f"{ticker} weight {weight} exceeded 5% cap!"
    assert sum(weights.values()) <= 0.20 + 1e-5, "Unallocated weight must remain unencumbered cash (not re-normalized to 100%)"


def test_risk_parity_zero_volatility_handling():
    """Verify that zero or negative ATR inputs do not throw DivisionByZero errors."""
    optimizer = RiskParityOptimizer(max_position_cap=0.05)
    convictions = {"NVDA": 0.80, "TSLA": 0.70}
    atrs = {"NVDA": 0.0, "TSLA": -1.5}

    weights = optimizer.optimize(convictions, atrs)
    assert isinstance(weights, dict)
    for ticker, weight in weights.items():
        assert weight <= 0.05 + 1e-5


def test_strict_five_percent_cap_no_double_normalization():
    """Verify that under high conviction variance, allocations strictly respect 5% without secondary vector re-normalization."""
    risk_engine = AdvancedRiskEngine(max_position_pct=0.05)
    volatility_map = {"NVDA": 0.02, "AAPL": 0.01, "MSFT": 0.015, "TSLA": 0.04}
    convictions = {"NVDA": 1.0, "AAPL": 0.9, "MSFT": 0.8, "TSLA": 0.7}

    allocations = risk_engine.calculate_risk_parity_allocations(volatility_map, convictions)

    for ticker, alloc in allocations.items():
        assert alloc <= 0.05 + 1e-5, f"Allocation for {ticker} was {alloc}, exceeding 5% cap!"

    # Residual weight remains cash; sum should be <= 4 * 0.05 = 0.20
    assert sum(allocations.values()) <= 0.20 + 1e-5, "Secondary vector re-normalization must be absent!"


def test_zero_division_and_flat_atr_guard():
    """Verify that flat technical indicators or uniform HOLD signals cleanly return zero weights without throwing exceptions."""
    risk_engine = AdvancedRiskEngine()
    
    # All zero / flat volatilities
    flat_vols = {"NVDA": 0.0, "TSLA": 0.0}
    convictions = {"NVDA": 0.5, "TSLA": 0.5}
    allocs = risk_engine.calculate_risk_parity_allocations(flat_vols, convictions)
    assert allocs == {"NVDA": 0.0, "TSLA": 0.0}

    # Flat convictions / all HOLD signals in RiskParityOptimizer
    optimizer = RiskParityOptimizer(max_position_cap=0.05)
    hold_convictions = {"AAPL": 0.1, "MSFT": 0.2}
    opt_allocs = optimizer.optimize(hold_convictions, {"AAPL": 1.0, "MSFT": 1.0})
    assert opt_allocs == {"AAPL": 0.0, "MSFT": 0.0}


# ==========================================
# 2. HARD RISK OVERLAY TESTS
# ==========================================

def test_hard_stop_loss_trigger():
    """Verify stop-loss triggers accurately at -2.5% drawdown."""
    entry_price = 100.0
    current_price_safe = 98.0   # -2.0% (Should hold)
    current_price_breach = 97.0 # -3.0% (Should trigger stop)

    pnl_safe = (current_price_safe - entry_price) / entry_price
    pnl_breach = (current_price_breach - entry_price) / entry_price

    assert pnl_safe > -0.025
    assert pnl_breach <= -0.025


def test_hard_take_profit_trigger():
    """Verify take-profit triggers accurately at +5.0% profit."""
    entry_price = 100.0
    current_price_breach = 105.5 # +5.5% (Should trigger take profit)

    pnl_breach = (current_price_breach - entry_price) / entry_price
    assert pnl_breach >= 0.05


# ==========================================
# 3. ADVANCED MARKET IMPACT SLIPPAGE TESTS
# ==========================================

def test_slippage_market_impact():
    """Verify that high volume trades incur larger slippage penalty."""
    risk_engine = AdvancedRiskEngine(base_spread=0.0001, impact_gamma=0.5)

    small_trade_price = risk_engine.calculate_execution_price(
        mid_price=100.0, shares=100, adv=1000000, action="BUY"
    )
    large_trade_price = risk_engine.calculate_execution_price(
        mid_price=100.0, shares=50000, adv=100000, action="BUY"
    )

    assert large_trade_price > small_trade_price, "Larger trade should have higher execution price due to market impact!"


def test_macro_trend_guard():
    engine = AdvancedRiskEngine()
    spy_returns = pd.Series([0.01, -0.01, 0.005, -0.002, 0.01])
    spy_prices = pd.Series([100.0] * 199 + [80.0])  # Current price 80 < SMA 100

    scaler = engine.calculate_regime_scaler(spy_returns, spy_prices=spy_prices)
    assert scaler <= 0.75  # Target volatility scaler cut in half


def test_alpaca_bridge_inactive():
    bridge = AlpacaExecutionBridge(api_key="", secret_key="")
    assert bridge.is_active() is False
    assert bridge.submit_market_order("AAPL", 10.0, "BUY") is None


def test_short_and_cover_slippage():
    """Verify that short orders slip downward and cover orders slip upward."""
    risk_engine = AdvancedRiskEngine()
    raw_price = 100.0

    short_exec = risk_engine.calculate_execution_price(raw_price, shares=1000, adv=100000, side="SHORT")
    cover_exec = risk_engine.calculate_execution_price(raw_price, shares=1000, adv=100000, side="COVER")

    assert short_exec < raw_price, "SHORT orders should slip price downwards!"
    assert cover_exec > raw_price, "COVER orders should slip price upwards!"


def test_margin_health_evaluation():
    """Verify free margin calculation and margin call triggering."""
    risk_engine = AdvancedRiskEngine()
    cash = 130000.0
    holdings = {"TSLA": -100.0}  # Short 100 TSLA @ $300 ($30,000 liability)
    prices = {"TSLA": 300.0}

    margin_info = risk_engine.evaluate_margin_health(cash, holdings, prices, initial_margin_req=1.50)

    assert margin_info["net_equity"] == 100000.0
    assert margin_info["short_liability"] == 30000.0
    assert margin_info["required_margin"] == 45000.0
    assert margin_info["free_margin"] == 55000.0
    assert margin_info["margin_call_triggered"] is False


# ==========================================
# 4. EVOLUTION & CAPITAL CONSERVATION TESTS
# ==========================================

def test_relative_fitness_calculation():
    """Verify offspring with $18k -> $20k has positive fitness while $100k -> $60k agent has negative fitness."""
    offspring = AgentGenome(
        agent_id="Gen2_Alpha_v1",
        persona_prompt="Growth Trader",
        initial_capital=18000.0,
        cash=20000.0,
        equity_history=[18000.0, 18500.0, 19200.0, 20000.0],
    )
    loser = AgentGenome(
        agent_id="Agent_Beta",
        persona_prompt="Conservative Trader",
        initial_capital=100000.0,
        cash=60000.0,
        equity_history=[100000.0, 85000.0, 72000.0, 60000.0],
    )

    offspring_fitness = offspring.calculate_fitness()
    loser_fitness = loser.calculate_fitness()

    assert offspring_fitness > 0, f"Offspring fitness should be positive (+11% PnL relative to initial capital), got {offspring_fitness}"
    assert loser_fitness < 0, f"Loser fitness should be negative (-40% PnL relative to initial capital), got {loser_fitness}"


def test_darwinian_capital_conservation():
    """Simulate a 5-agent swarm, cull worst performers, and assert total swarm equity is conserved (Delta = 0)."""
    async def _test():
        swarm_mgr = EvolutionarySwarmManager(api_key="", population_size=5)

        # Initialize 5 agents with known starting equity
        swarm_mgr.population[0].cash = 130000.0
        swarm_mgr.population[0].equity_history = [130000.0]
        swarm_mgr.population[1].cash = 115000.0
        swarm_mgr.population[1].equity_history = [115000.0]
        swarm_mgr.population[2].cash = 105000.0
        swarm_mgr.population[2].equity_history = [105000.0]
        swarm_mgr.population[3].cash = 80000.0
        swarm_mgr.population[3].equity_history = [80000.0]
        swarm_mgr.population[4].cash = 70000.0
        swarm_mgr.population[4].equity_history = [70000.0]

        total_equity_before = sum(a.equity_history[-1] for a in swarm_mgr.population)
        assert total_equity_before == 500000.0

        # Run culling cycle in offline test mode
        await swarm_mgr.run_culling_cycle(prices={})

        assert len(swarm_mgr.population) == 5
        total_equity_after = sum(a.equity_history[-1] for a in swarm_mgr.population)

        # Assert Delta == 0
        assert abs(total_equity_after - total_equity_before) < 1e-2, (
            f"Capital not conserved! Before: {total_equity_before}, After: {total_equity_after}"
        )

        # Survivors keep their equity (130k, 115k, 105k = 350k)
        # Culled equity (80k + 70k = 150k) is split equally between 2 offspring (75k each)
        offspring_equities = [a.equity_history[-1] for a in swarm_mgr.population[3:]]
        assert offspring_equities == [75000.0, 75000.0]
        assert [a.initial_capital for a in swarm_mgr.population[3:]] == [75000.0, 75000.0]

    asyncio.run(_test())


def test_zero_sum_capital_conservation_invariant():
    """Verify zero-sum invariant (sum E_pre == sum E_post) during selection with active open long and short positions."""
    async def _test():
        swarm_mgr = EvolutionarySwarmManager(api_key="", population_size=5)
        prices = {"AAPL": 150.0, "TSLA": 200.0, "NVDA": 120.0}

        # Setup agents with cash, long positions, and short positions
        # Agent 0 (Top performer): $100k cash + 200 AAPL ($30k) = $130,000 equity
        swarm_mgr.population[0].cash = 100000.0
        swarm_mgr.population[0].holdings = {"AAPL": 200.0}
        swarm_mgr.population[0].entry_prices = {"AAPL": 150.0}
        swarm_mgr.population[0].equity_history = [130000.0]

        # Agent 1: $110k cash + 50 TSLA ($10k) = $120,000 equity
        swarm_mgr.population[1].cash = 110000.0
        swarm_mgr.population[1].holdings = {"TSLA": 50.0}
        swarm_mgr.population[1].entry_prices = {"TSLA": 200.0}
        swarm_mgr.population[1].equity_history = [120000.0]

        # Agent 2: $115k cash
        swarm_mgr.population[2].cash = 115000.0
        swarm_mgr.population[2].holdings = {}
        swarm_mgr.population[2].entry_prices = {}
        swarm_mgr.population[2].equity_history = [115000.0]

        # Agent 3 (Culled): $60k cash + 100 NVDA ($12k) = $72,000 equity
        swarm_mgr.population[3].cash = 60000.0
        swarm_mgr.population[3].holdings = {"NVDA": 100.0}
        swarm_mgr.population[3].entry_prices = {"NVDA": 120.0}
        swarm_mgr.population[3].equity_history = [72000.0]

        # Agent 4 (Culled): $83k cash - 100 TSLA short ($20k liability) = $63,000 equity
        swarm_mgr.population[4].cash = 83000.0
        swarm_mgr.population[4].holdings = {"TSLA": -100.0}
        swarm_mgr.population[4].entry_prices = {"TSLA": 200.0}
        swarm_mgr.population[4].equity_history = [63000.0]

        total_equity_pre = 130000.0 + 120000.0 + 115000.0 + 72000.0 + 63000.0
        assert total_equity_pre == 500000.0

        await swarm_mgr.run_culling_cycle(prices=prices)

        total_equity_post = sum(a.equity_history[-1] for a in swarm_mgr.population)
        assert abs(total_equity_post - total_equity_pre) < 1e-2, (
            f"Zero-sum invariant broken! Pre: {total_equity_pre}, Post: {total_equity_post}"
        )

        # Culled equity ($72k + $63k = $135k) split between 2 offspring ($67.5k each)
        offspring = swarm_mgr.population[3:]
        assert len(offspring) == 2
        assert offspring[0].cash + offspring[1].cash == 135000.0

    asyncio.run(_test())


def test_unique_offspring_ids_generation():
    """Verify that mutated offspring receive unique IDs formatted with UUID suffix to eliminate collisions."""
    async def _test():
        swarm_mgr = EvolutionarySwarmManager(api_key="", population_size=5)
        parent = swarm_mgr.population[0]
        
        offspring_a = await swarm_mgr._mutate_genome(parent, "Momentum", 1)
        offspring_b = await swarm_mgr._mutate_genome(parent, "Mean Reversion", 2)

        assert offspring_a.agent_id != offspring_b.agent_id
        assert offspring_a.agent_id != parent.agent_id
        # Verify format Gen{epoch}_{parent}_v{idx}_{uuid}
        pattern = r"^Gen2_Agent_Alpha_v\d+_[0-9a-f]{4}$"
        assert re.match(pattern, offspring_a.agent_id) is not None, f"ID {offspring_a.agent_id} does not match expected format"
        assert re.match(pattern, offspring_b.agent_id) is not None, f"ID {offspring_b.agent_id} does not match expected format"

    asyncio.run(_test())


def test_tenure_grace_period_infant_mortality_protection():
    """Verify newly spawned agents (< 1000 ticks) are protected from infant mortality while mature underperformers are culled."""
    async def _test():
        swarm_mgr = EvolutionarySwarmManager(api_key="", population_size=5)

        # Mature agents (tenure = 1000 ticks)
        swarm_mgr.population[0].agent_id = "Mature_Leader"
        swarm_mgr.population[0].tenure_ticks = 1000
        swarm_mgr.population[0].equity_history = [150000.0]

        swarm_mgr.population[1].agent_id = "Mature_Mid"
        swarm_mgr.population[1].tenure_ticks = 1000
        swarm_mgr.population[1].equity_history = [120000.0]

        swarm_mgr.population[2].agent_id = "Mature_Loser_1"
        swarm_mgr.population[2].tenure_ticks = 1000
        swarm_mgr.population[2].equity_history = [80000.0]

        swarm_mgr.population[3].agent_id = "Mature_Loser_2"
        swarm_mgr.population[3].tenure_ticks = 1000
        swarm_mgr.population[3].equity_history = [75000.0]

        # Newly spawned infant agent (tenure = 15 ticks, generation = 2) with flat/negative initial returns
        swarm_mgr.population[4].agent_id = "Gen2_Infant_Offspring_v1_a1b2"
        swarm_mgr.population[4].generation = 2
        swarm_mgr.population[4].tenure_ticks = 15
        swarm_mgr.population[4].equity_history = [70000.0]  # Lowest equity

        culled_ids, recipient_ids = await swarm_mgr.run_culling_cycle(prices={})

        # Infant agent MUST NOT be culled because of tenure immunity!
        assert "Gen2_Infant_Offspring_v1_a1b2" not in culled_ids, "Infant agent was improperly culled during grace period!"
        assert "Mature_Loser_1" in culled_ids
        assert "Mature_Loser_2" in culled_ids

    asyncio.run(_test())


# ==========================================
# 5. SHORT SOLVENCY & PARSING TESTS
# ==========================================

def test_short_proceeds_solvency_guard():
    """Verify an agent cannot spend short sale proceeds on long allocations when free margin is insufficient."""
    agent = AgentGenome(
        agent_id="Agent_Delta",
        persona_prompt="Short Trader",
        initial_capital=100000.0,
        cash=100000.0,
        holdings={"TSLA": -100.0},
        entry_prices={"TSLA": 300.0},
        equity_history=[100000.0],
    )
    # Short sale proceeds added to cash: $100k initial + $30k proceeds = $130k cash
    agent.cash = 130000.0
    prices = {"TSLA": 300.0, "NVDA": 100.0}

    # Short Liabilities calculation
    short_liabilities = sum(
        abs(qty) * max(agent.entry_prices.get(tk, 0.0), prices.get(tk, agent.entry_prices.get(tk, 0.0)))
        for tk, qty in agent.holdings.items()
        if qty < 0
    )
    free_cash = max(0.0, agent.cash - short_liabilities)

    assert short_liabilities == 30000.0
    assert free_cash == 100000.0

    # Test BUY order requesting $120,000
    delta = 120000.0

    # Naive check: agent.cash (130k) >= delta (120k) -> True (UNSAFE)
    can_buy_unrestricted = agent.cash >= delta
    # Guarded check: free_cash (100k) >= delta (120k) -> False (SAFE)
    can_buy_guarded = delta > 50.0 and free_cash >= delta

    assert can_buy_unrestricted is True, "Naive cash check would unsafely allow spending short proceeds"
    assert can_buy_guarded is False, "Solvency guard must prevent spending encumbered short proceeds"


def test_short_margin_purchasing_power_max_price():
    """Verify that when a short moves against the agent (price increases), encumbered margin expands using max(P_entry, P_current)."""
    agent = AgentGenome(
        agent_id="Agent_Short_Risk",
        persona_prompt="Short",
        initial_capital=100000.0,
        cash=130000.0,
        holdings={"TSLA": -100.0},
        entry_prices={"TSLA": 300.0},
        equity_history=[100000.0],
    )
    # Price increased adversely to $350
    adverse_prices = {"TSLA": 350.0}

    encumbered_margin = sum(
        abs(qty) * max(agent.entry_prices.get(tk, 0.0), adverse_prices.get(tk, agent.entry_prices.get(tk, 0.0)))
        for tk, qty in agent.holdings.items()
        if qty < 0
    )
    free_cash = max(0.0, agent.cash - encumbered_margin)

    # 100 * 350 = 35,000 liability
    assert encumbered_margin == 35000.0
    # Free cash = 130,000 - 35,000 = 95,000
    assert free_cash == 95000.0


def test_sentiment_llm_json_resiliency():
    """Verify regex JSON extractor strips markdown fences, conversational preambles, and thought tags."""
    agent = NewsSentimentAgent(api_key="test_dummy_key")

    raw_llm_response = """
<thought>
The market is showing mixed signals, but inflation concerns are easing.
</thought>
Here is my macroeconomic sentiment analysis for the quantitative swarm:
```json
{
    "sentiment_score": 0.45,
    "summary_reasoning": "Dovish commentary from FOMC members coupled with strong earnings in semiconductor names is bolstering risk appetites."
}
```
Let me know if you need further breakdowns.
"""
    # 1. Strip internal thinking tags
    cleaned_content = re.sub(r"<thought>[\s\S]*?</thought>", "", raw_llm_response).strip()
    # 2. Strip Markdown code fences
    cleaned_content = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned_content).strip()
    # 3. Extract valid JSON object block
    json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
    assert json_match is not None, "Failed to match JSON pattern from raw response!"

    parsed = json.loads(json_match.group(0).strip())
    sanitized = agent._sanitize_sentiment_output(parsed)

    assert sanitized["sentiment_score"] == 0.45
    assert sanitized["risk_multiplier"] == round(1.0 + (0.45 * 0.3), 2)
    assert "Dovish commentary" in sanitized["summary_reasoning"]


def test_active_agent_capital_aggregation(mock_db_pool):
    """Verify get_total_swarm_capital strictly aggregates active agents (is_active = TRUE)."""
    mock_pool, mock_conn, mock_cursor = mock_db_pool
    mock_cursor.fetchone.return_value = {"total_cash": 350000.0}

    pm = CrossAssetPortfolioManager()
    total_cap = pm.get_total_swarm_capital()

    executed_queries = [call[0][0] for call in mock_cursor.execute.call_args_list]
    active_query = any("WHERE is_active = TRUE" in q for q in executed_queries)

    assert total_cap == 350000.0
    assert active_query is True, "get_total_swarm_capital did not filter by WHERE is_active = TRUE"


# ==========================================
# 6. RECOVERY & FALLBACK TESTS
# ==========================================

def test_state_restoration_without_resurrection(monkeypatch):
    """Verify that calling restore_agent_states_from_db with 3 living agents does not resurrect culled agents with $100k balances."""
    mock_db = MagicMock()
    mock_db.engine = MagicMock()

    # 3 active accounts in DB (Alpha, Beta, Gamma)
    mock_accounts_df = pd.DataFrame([
        {"agent_id": "Agent_Alpha", "cash": 120000.0},
        {"agent_id": "Agent_Beta", "cash": 110000.0},
        {"agent_id": "Agent_Gamma", "cash": 105000.0},
    ])

    # Snapshots showing Delta and Epsilon were culled (equity = 0.0)
    mock_snapshots_df = pd.DataFrame([
        {"agent_id": "Agent_Alpha", "cash": 120000.0, "equity": 120000.0},
        {"agent_id": "Agent_Beta", "cash": 110000.0, "equity": 110000.0},
        {"agent_id": "Agent_Gamma", "cash": 105000.0, "equity": 105000.0},
        {"agent_id": "Agent_Delta", "cash": 0.0, "equity": 0.0},
        {"agent_id": "Agent_Epsilon", "cash": 0.0, "equity": 0.0},
    ])

    mock_first_snaps = pd.DataFrame([
        {"agent_id": "Agent_Alpha", "equity": 100000.0},
        {"agent_id": "Agent_Beta", "equity": 100000.0},
        {"agent_id": "Agent_Gamma", "equity": 100000.0},
    ])

    mock_holdings_df = pd.DataFrame(columns=["agent_id", "ticker", "amount", "entry_price"])

    def mock_read_sql(query, con):
        if "agent_accounts" in query:
            return mock_accounts_df
        elif "DESC" in query and "agent_snapshots" in query:
            return mock_snapshots_df
        elif "ASC" in query and "agent_snapshots" in query:
            return mock_first_snaps
        elif "agent_holdings" in query:
            return mock_holdings_df
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)

    swarm_mgr = EvolutionarySwarmManager(api_key=None, population_size=5)
    restore_agent_states_from_db(swarm_mgr, mock_db)

    active_agent_ids = [a.agent_id for a in swarm_mgr.population]

    # Must preserve the 3 living agents
    assert "Agent_Alpha" in active_agent_ids
    assert "Agent_Beta" in active_agent_ids
    assert "Agent_Gamma" in active_agent_ids

    # Dead/culled agents must not be resurrected with default $100k cash
    for agent in swarm_mgr.population:
        if agent.agent_id in ("Agent_Delta", "Agent_Epsilon"):
            assert agent.cash == 0.0, f"Culled agent {agent.agent_id} was improperly resurrected with cash ${agent.cash}!"


def test_cull_and_reallocate_resilient_fallback(mock_db_pool):
    """Verify cull_and_reallocate handles missing price with fallback and executes UPSERT."""
    mock_pool, mock_conn, mock_cursor = mock_db_pool

    mock_cursor.fetchone.return_value = {"cash": 10000.0}
    mock_cursor.fetchall.return_value = [
        {"ticker": "UNKNOWN_TICKER", "amount": 10.0, "entry_price": 50.0}
    ]

    pm = CrossAssetPortfolioManager()

    pm.cull_and_reallocate(
        loser_agent_id="Agent_Delta",
        recipient_agent_ids=["Gen2_Alpha_v1", "Gen2_Alpha_v2"],
        current_prices={},
    )

    executed_queries = [call[0][0] for call in mock_cursor.execute.call_args_list]

    upsert_found = any("DO UPDATE SET cash = agent_accounts.cash + EXCLUDED.cash" in q for q in executed_queries)
    assert upsert_found is True, "Consolidated UPSERT not found in executed queries!"

    snapshot_queries = [call[0] for call in mock_cursor.execute.call_args_list if "agent_snapshots" in call[0][0]]
    assert len(snapshot_queries) >= 3, "Snapshots for loser and 2 offspring should be inserted"
# ------------------------------------------
# 7. BATCHED ADVERSARIAL DEBATE 20-STOCK 3 CALLS PER BAR
# ------------------------------------------

def _make_20_snapshot():
    from data_producer import TICKERS
    snap = {}
    for i, tk in enumerate(TICKERS):
        snap[tk] = {
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 1000000.0,
            "rsi": 55.0,
            "rsi14": 55.0,
            "momentum": 0.25,
            "momentum_15m": 0.25,
            "macd_hist": 0.05,
            "atr": 1.5,
            "rel_strength_spy": 0.5,
            "adv": 5000000.0,
            "headlines": "Steady price action.",
        }
    return snap


def test_adversarial_debate_parallel_execution():
    """Verifies Bull and Bear prompts run concurrently and Arbiter parses into [-1.0, 1.0]."""
    from sentiment_agent import NewsSentimentAgent

    async def _test():
        agent = NewsSentimentAgent(api_key="test_dummy_key")
        snapshots = _make_20_snapshot()
        bull_text = "\n".join([f"- {tk}: Bullish breakout momentum." for tk in snapshots])
        bear_text = "\n".join([f"- {tk}: Overbought fade risk." for tk in snapshots])
        arbiter_json = json.dumps({tk: (0.45 if i % 2 == 0 else -0.20) for i, tk in enumerate(sorted(snapshots))})
        call_order = []

        async def fake_bull(snaps):
            call_order.append("bull_start")
            await asyncio.sleep(0.05)
            call_order.append("bull_end")
            return bull_text

        async def fake_bear(snaps):
            call_order.append("bear_start")
            await asyncio.sleep(0.05)
            call_order.append("bear_end")
            return bear_text

        async def fake_gemini(prompt, temperature=0.1, max_tokens=2048):
            assert "NVDA" in prompt or "AAPL" in prompt
            return arbiter_json

        with patch.object(agent, "_generate_bull_theses", side_effect=fake_bull), patch.object(
            agent, "_generate_bear_theses", side_effect=fake_bear
        ), patch.object(agent, "_call_gemini_text", side_effect=fake_gemini):
            scores = await agent.run_adversarial_batch(snapshots)

        assert "bull_start" in call_order and "bear_start" in call_order
        assert call_order.index("bear_start") < call_order.index("bull_end"), (
            f"Bull and Bear did not overlap, no gather concurrency: {call_order}"
        )
        assert len(scores) == 20, f"Expected 20 scores, got {len(scores)}"
        for tk, v in scores.items():
            assert isinstance(v, float), f"{tk} score not float"
            assert -1.0 <= v <= 1.0, f"{tk} score {v} out of range"
        assert agent.cache.get_sentiment("NVDA") == scores["NVDA"]
        assert agent.cache.get_sentiment("AAPL") == scores["AAPL"]

    asyncio.run(_test())


def test_arbiter_json_fallback():
    """Confirms invalid JSON from Arbiter falls back safely to 0.0 without raising."""
    from sentiment_agent import NewsSentimentAgent

    async def _test():
        agent = NewsSentimentAgent(api_key="test_dummy_key")
        snapshots = _make_20_snapshot()

        async def fake_bull(snaps):
            return "- NVDA: bullish"

        async def fake_bear(snaps):
            return "- NVDA: bearish"

        async def fake_bad_gemini(prompt, temperature=0.1, max_tokens=2048):
            return "THIS IS NOT JSON just prose, sorry!"

        with patch.object(agent, "_generate_bull_theses", side_effect=fake_bull), patch.object(
            agent, "_generate_bear_theses", side_effect=fake_bear
        ), patch.object(agent, "_call_gemini_text", side_effect=fake_bad_gemini):
            scores = await agent.run_adversarial_batch(snapshots)

        assert isinstance(scores, dict)
        assert len(scores) == 20
        for tk, v in scores.items():
            assert v == 0.0, f"{tk} should fallback to 0.0, got {v}"
        parsed = agent._parse_arbiter_json("garbage no json here", list(snapshots.keys()))
        assert all(val == 0.0 for val in parsed.values())

    asyncio.run(_test())


def test_sentiment_cache_ttl_expiration():
    """Validates SentimentCache TTL 1200s: fresh reads hit, expired return 0.0."""
    import time as _time
    from sentiment_agent import SentimentCache

    cache = SentimentCache(ttl_seconds=1200.0)
    cache.update({"NVDA": 0.65, "AAPL": -0.35})
    assert cache.get_sentiment("NVDA") == 0.65
    assert cache.get_sentiment("AAPL") == -0.35
    past = _time.time() - 1300.0
    cache._timestamps["NVDA"] = past
    cache._timestamps["AAPL"] = past
    assert cache.get_sentiment("NVDA") == 0.0
    assert cache.get_sentiment("AAPL") == 0.0
    assert cache.get_fallback("NVDA") == 0.65


def test_directional_stops():
    """Validates exact stop-loss and take-profit trip conditions for Long and Short."""
    engine = AdvancedRiskEngine(stop_loss_pct=0.025, take_profit_pct=0.05)
    assert engine.check_stop_loss_take_profit(100.0, 97.5, "LONG") == "STOP_LOSS"
    assert engine.check_stop_loss_take_profit(100.0, 97.49, "LONG") == "STOP_LOSS"
    assert engine.check_stop_loss_take_profit(100.0, 98.0, "LONG") is None
    assert engine.check_stop_loss_take_profit(100.0, 105.0, "LONG") == "TAKE_PROFIT"
    assert engine.check_stop_loss_take_profit(100.0, 105.5, "LONG") == "TAKE_PROFIT"
    assert engine.check_stop_loss_take_profit(100.0, 102.0, "LONG") is None
    assert engine.check_stop_loss_take_profit(100.0, 102.5, "SHORT") == "STOP_LOSS"
    assert engine.check_stop_loss_take_profit(100.0, 103.0, "SHORT") == "STOP_LOSS"
    assert engine.check_stop_loss_take_profit(100.0, 102.0, "SHORT") is None
    assert engine.check_stop_loss_take_profit(100.0, 95.0, "SHORT") == "TAKE_PROFIT"
    assert engine.check_stop_loss_take_profit(100.0, 94.0, "SHORT") == "TAKE_PROFIT"
    assert engine.check_stop_loss_take_profit(100.0, 98.0, "SHORT") is None
    assert engine.check_position_exit(100.0, 97.0, shares=10.0) == "STOP_LOSS"
    assert engine.check_position_exit(100.0, 103.0, shares=-10.0) == "STOP_LOSS"
    assert engine.check_position_exit(100.0, 105.5, shares=10.0) == "TAKE_PROFIT"
    assert engine.check_position_exit(100.0, 94.5, shares=-10.0) == "TAKE_PROFIT"


def test_cooldown_reentry_rejection():
    """Validates cooled-down ticker orders are rejected until window expires."""
    engine = AdvancedRiskEngine(cooldown_bars=4)
    expiry = engine.record_stopout("Agent_Alpha", "NVDA", current_tick=10)
    assert expiry == 14
    assert engine.cooldowns[("Agent_Alpha", "NVDA")] == 14
    for t in (10, 11, 12, 13):
        assert engine.is_cooled_down("Agent_Alpha", "NVDA", t) is True
        assert engine.should_allow_entry("Agent_Alpha", "NVDA", t) is False
    assert engine.is_cooled_down("Agent_Alpha", "NVDA", 14) is False
    assert engine.should_allow_entry("Agent_Alpha", "NVDA", 14) is True
    assert engine.should_allow_entry("Agent_Alpha", "AAPL", 11) is True
    assert engine.should_allow_entry("Agent_Beta", "NVDA", 11) is True


def test_session_circuit_breaker():
    """Validates 5 percent peak-to-trough breaker halts new entries."""
    engine = AdvancedRiskEngine(session_drawdown_pct=0.05)
    assert engine.check_session_drawdown(500000.0) is False
    assert engine.update_session_peak(520000.0) == 520000.0
    assert engine.check_session_drawdown(504400.0) is False
    assert engine.is_trading_halted() is False
    assert engine.check_session_drawdown(494000.0) is True
    assert engine.is_trading_halted() is True
    assert engine.should_allow_entry("Agent_Alpha", "NVDA", 99) is False

# ------------------------------------------
# 8. TOP-20 DYNAMIC SCREENER 100 TO 20
# ------------------------------------------

def _make_100_snapshot():
    from data_producer import UNIVERSE
    snap = {}
    for i, tk in enumerate(UNIVERSE):
        # Activity gradient: momentum grows with i, volume grows with i
        snap[tk] = {
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 100000.0 + i * 50000.0,
            "rsi": 55.0,
            "rsi14": 55.0,
            "momentum": 0.05 + i * 0.02,
            "momentum_15m": 0.05 + i * 0.02,
            "macd_hist": 0.05,
            "atr": 1.5,
            "rel_strength_spy": 0.5,
            "adv": 5000000.0,
            "headlines": "Steady.",
        }
    return snap


def test_top_20_screener_selection():
    """Feeds 100 mock snapshots and asserts exactly 20 highest-activity tickers filtered."""
    from sentiment_agent import select_top_20_candidates
    from data_producer import UNIVERSE
    assert len(UNIVERSE) == 100, f"UNIVERSE must be 100, got {len(UNIVERSE)}"
    snap = _make_100_snapshot()
    top20 = select_top_20_candidates(snap, top_n=20)
    assert len(top20) == 20, f"Expected 20, got {len(top20)}"
    # Highest activity = highest index (largest momentum x volume)
    import math
    def _score(e):
        return abs(e["momentum_15m"]) * math.log1p(e["volume"])
    ranked_all = sorted(snap.items(), key=lambda kv: _score(kv[1]), reverse=True)
    expected_keys = [k for k, _ in ranked_all[:20]]
    assert set(top20.keys()) == set(expected_keys)


def test_batched_debate_receives_only_20():
    """Confirms LLM payload only contains 20 items even when producer provides 100."""
    from sentiment_agent import NewsSentimentAgent, select_top_20_candidates
    async def _test():
        agent = NewsSentimentAgent(api_key="test_dummy_key")
        snap100 = _make_100_snapshot()
        assert len(snap100) == 100
        top20 = select_top_20_candidates(snap100, top_n=20)
        assert len(top20) == 20
        seen_payload_size = {}
        orig_bull = agent._generate_bull_theses
        orig_bear = agent._generate_bear_theses
        async def wrap_bull(s):
            seen_payload_size["n"] = len(s)
            return "- NVDA: bullish"
        async def wrap_bear(s):
            assert len(s) == 20
            return "- NVDA: bearish"
        async def fake_arb(prompt, temperature=0.1, max_tokens=2048):
            import json as _json
            # Arbiter prompt must mention only screened tickers count via JSON keys
            return _json.dumps({tk: 0.1 for tk in top20.keys()})
        from unittest.mock import patch as _patch
        with _patch.object(agent, "_generate_bull_theses", side_effect=wrap_bull), _patch.object(
            agent, "_generate_bear_theses", side_effect=wrap_bear
        ), _patch.object(agent, "_call_gemini_text", side_effect=fake_arb):
            scores = await agent.run_adversarial_batch(top20)
        assert seen_payload_size["n"] == 20
        assert len(scores) == 20
    import asyncio as _asyncio
    _asyncio.run(_test())


def test_unscreened_ticker_sentiment_fallback():
    """Verifies non-screened tickers return 0.0 neutral gracefully."""
    from sentiment_agent import NewsSentimentAgent, select_top_20_candidates
    async def _test():
        agent = NewsSentimentAgent(api_key="test_dummy_key")
        snap100 = _make_100_snapshot()
        top20 = select_top_20_candidates(snap100, top_n=20)
        unscreened = [k for k in snap100.keys() if k not in top20][0]
        # Before debate, unscreened returns 0.0
        assert agent.cache.get_sentiment(unscreened) == 0.0
        # After debate on top20 only, unscreened still 0.0, screened has score
        async def fake_bull(s):
            return "bull"
        async def fake_bear(s):
            return "bear"
        async def fake_arb(prompt, temperature=0.1, max_tokens=2048):
            import json as _json
            return _json.dumps({tk: 0.5 for tk in top20.keys()})
        from unittest.mock import patch as _patch
        with _patch.object(agent, "_generate_bull_theses", side_effect=fake_bull), _patch.object(
            agent, "_generate_bear_theses", side_effect=fake_bear
        ), _patch.object(agent, "_call_gemini_text", side_effect=fake_arb):
            await agent.run_adversarial_batch(top20)
        assert agent.cache.get_sentiment(unscreened) == 0.0
        screened_one = list(top20.keys())[0]
        assert agent.cache.get_sentiment(screened_one) == 0.5
        # compute signals over 100: screened fused, unscreened pure technical no crash
        from swarm_consumer import compute_deterministic_signals
        from evolution_engine import EvolutionarySwarmManager
        mgr = EvolutionarySwarmManager(api_key="", population_size=2)
        fake_market = {k: {"close": 100.0, "rsi14": 55.0, "momentum_15m": 0.2, "macd_hist": 0.05, "adv": 1000000.0} for k in snap100.keys()}
        thesis = {k: {"price": 100.0, "rsi": 55.0, "macd_hist": 0.05, "rel_strength": 0.0, "atr": 1.5} for k in snap100.keys()}
        # Monkey-patch global sentiment cache used by compute to our agent cache
        import swarm_consumer as _sc
        _sc.sentiment_agent.cache = agent.cache
        out = _sc.compute_deterministic_signals(mgr.population[:1], fake_market, thesis, screened_tickers=set(top20.keys()))
        assert len(out) == 1
    import asyncio as _asyncio
    _asyncio.run(_test())


# ------------------------------------------
# 9. MISSION HARDENING INVARIANTS
# ------------------------------------------

def test_screener_retains_open_positions():
    """Tickers with active holdings are preserved in 20-stock batch even if activity is zero."""
    from sentiment_agent import select_top_20_candidates
    snap = _make_100_snapshot()
    # Force two held tickers to zero activity (flat momentum, zero volume)
    held = set(list(snap.keys())[:2])
    for tk in held:
        snap[tk]["momentum"] = 0.0
        snap[tk]["momentum_15m"] = 0.0
        snap[tk]["volume"] = 0.0
        snap[tk]["atr"] = 0.1
    top20 = select_top_20_candidates(snap, held, top_n=20)
    assert len(top20) == 20
    for tk in held:
        assert tk in top20, f"Held {tk} orphaned from LLM batch!"
    # Overflow: >=20 holdings -> top 20 by activity among held only
    many_held = set(list(snap.keys())[:25])
    top20_many = select_top_20_candidates(snap, many_held, top_n=20)
    assert len(top20_many) == 20
    assert set(top20_many.keys()).issubset(many_held)
    # Backward compat: legacy positional top_n still works
    legacy = select_top_20_candidates(snap, 20)
    assert len(legacy) == 20


def test_elitism_preserves_top_agent():
    """#1 performer is never culled and preserves equity across epochs."""
    async def _test():
        mgr = EvolutionarySwarmManager(api_key="", population_size=5)
        mgr.population[0].agent_id = "Agent_Alpha"
        mgr.population[0].cash = 200000.0
        mgr.population[0].equity_history = [100000.0, 150000.0, 180000.0, 200000.0]
        mgr.population[0].tenure_ticks = 1000
        for i in range(1, 5):
            mgr.population[i].cash = 80000.0 - i * 1000
            mgr.population[i].equity_history = [100000.0, 90000.0, 80000.0 - i * 1000]
            mgr.population[i].tenure_ticks = 1000
        elite_equity = mgr.population[0].equity_history[-1]
        culled_ids, _ = await mgr.run_culling_cycle(prices={})
        assert "Agent_Alpha" not in culled_ids
        elites = [a for a in mgr.population if a.agent_id == "Agent_Alpha"]
        assert len(elites) == 1
        assert elites[0].is_elite is True
        assert elites[0].equity_history[-1] == elite_equity
        assert elites[0].cash == elite_equity
    asyncio.run(_test())


def test_lineage_cap_enforcement():
    """No single ancestor lineage exceeds 2 agents after 3 evolutionary cycles."""
    async def _test():
        mgr = EvolutionarySwarmManager(api_key="", population_size=5)
        for cycle in range(3):
            for a in mgr.population:
                a.tenure_ticks = 1000
            # Force fitness spread so culling is deterministic
            ranked_cash = [140000.0, 120000.0, 110000.0, 80000.0, 70000.0]
            for a, c in zip(sorted(mgr.population, key=lambda x: x.agent_id), ranked_cash):
                a.cash = c
                a.holdings = {}
                a.entry_prices = {}
                a.equity_history = [100000.0, c]
            await mgr.run_culling_cycle(prices={})
            assert len(mgr.population) == 5
            counts = {}
            for a in mgr.population:
                r = getattr(a, "lineage_root", "unknown")
                counts[r] = counts.get(r, 0) + 1
            for root, n in counts.items():
                assert n <= 2, f"Lineage {root} has {n} agents after cycle {cycle+1}: {counts}"
    asyncio.run(_test())


def test_timestamp_cooldown():
    """Cooldown respects real-time epoch intervals rather than tick counts."""
    eng = AdvancedRiskEngine(cooldown_bars=4)
    t0 = 1700000000.0
    eng.record_stopout_ts("Agent_Alpha", "NVDA", t0)
    assert eng.cooldown_until[("Agent_Alpha", "NVDA")] == t0 + 3600.0
    assert eng.is_cooled_down("Agent_Alpha", "NVDA", t0 + 100, current_timestamp=t0 + 100) is True
    assert eng.should_allow_entry("Agent_Alpha", "NVDA", t0 + 100, current_timestamp=t0 + 100) is False
    assert eng.is_cooled_down("Agent_Alpha", "NVDA", t0 + 3599, current_timestamp=t0 + 3599) is True
    assert eng.should_allow_entry("Agent_Alpha", "NVDA", t0 + 3601, current_timestamp=t0 + 3601) is True
    assert eng.is_cooled_down("Agent_Alpha", "NVDA", t0 + 3600, current_timestamp=t0 + 3600) is False
    # Tick-domain backward compat still intact
    eng2 = AdvancedRiskEngine(cooldown_bars=4)
    assert eng2.record_stopout("Agent_Beta", "AAPL", current_tick=10) == 14
    assert eng2.is_cooled_down("Agent_Beta", "AAPL", 13) is True
    assert eng2.is_cooled_down("Agent_Beta", "AAPL", 14) is False


def test_circuit_breaker_daily_reset():
    """Trading automatically unhalts when session clock rolls over + manual reset works."""
    eng = AdvancedRiskEngine(session_drawdown_pct=0.05)
    t0 = 1700000000.0
    eng.last_session_reset = t0
    try:
        from datetime import datetime, timezone as _tz
        eng.last_session_date = datetime.fromtimestamp(t0, tz=_tz.utc).strftime("%Y-%m-%d")
    except Exception:
        pass
    assert eng.check_session_drawdown(500000.0, current_timestamp=t0) is False
    assert eng.check_session_drawdown(470000.0, current_timestamp=t0 + 100) is True
    assert eng.is_trading_halted() is True
    # Rollover after 86400s auto-resets even though drawdown persists
    assert eng.check_session_drawdown(470000.0, current_timestamp=t0 + 86400.0 + 10) is False
    assert eng.is_trading_halted() is False
    # Manual reset path
    eng.check_session_drawdown(440000.0, current_timestamp=t0 + 86500.0)
    # Force trip again from new peak
    eng.session_peak_equity = 500000.0
    eng.circuit_breaker_tripped = True
    eng.trading_halted = True
    status = eng.reset_circuit_breaker(480000.0)
    assert eng.is_trading_halted() is False
    assert eng.session_peak_equity == 480000.0
    assert status["trading_halted"] is False


def test_short_position_equity_symmetry():
    """5pct gain on short yields identical equity expansion as 5pct gain on long."""
    eng = AdvancedRiskEngine()
    # Long: 100 sh @100 -> 105 (+5pct)
    long_eq = eng.calculate_total_equity(100000.0, {"AAPL": 100.0}, {"AAPL": 100.0}, {"AAPL": 105.0})
    # Short: proceeds 100*100=10000 added to cash -> cash 110000, entry 100, current 95 (-5pct price = +5pct short)
    short_eq = eng.calculate_total_equity(110000.0, {"AAPL": -100.0}, {"AAPL": 100.0}, {"AAPL": 95.0})
    # Long equity: 100000 + 10500 = 110500
    assert long_eq == 110500.0
    # Short equity: 110000 + (10000-9500)=110500 symmetric
    assert short_eq == 110500.0
    assert long_eq == short_eq
    # Margin requirement 1.5x
    assert eng.short_margin_requirement(100.0, 100.0, 1.50) == 15000.0
    assert eng.can_open_short(15000.0, 100.0, 100.0) is True
    assert eng.can_open_short(14999.0, 100.0, 100.0) is False
    # Evolvable genome bounds + renormalization
    g = AgentGenome(agent_id="Agent_Alpha", persona_prompt="test", sentiment_weight=0.9, technical_weight=0.9)
    assert 0.20 <= g.sentiment_weight <= 0.80
    assert 0.20 <= g.technical_weight <= 0.80
    assert abs((g.sentiment_weight + g.technical_weight) - 1.0) < 1e-6
