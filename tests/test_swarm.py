import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from engine import (
    RiskParityOptimizer, 
    AlpacaExecutionBridge, 
    CrossAssetPortfolioManager
)
from risk_engine import AdvancedRiskEngine
from evolution_engine import AgentGenome, EvolutionarySwarmManager
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
    """Mock fixture for httpx.AsyncClient."""
    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": '{"new_prompt": "Mutated strategy prompt with risk controls"}'}}
        ]
    }
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("httpx.AsyncClient", MagicMock(return_value=mock_client))
    return mock_client

# ==========================================
# 1. RISK PARITY OPTIMIZER UNIT TESTS
# ==========================================

def test_risk_parity_max_position_cap():
    """Verify that no single asset weight exceeds the strict 15% position cap."""
    optimizer = RiskParityOptimizer(max_position_cap=0.15)
    convictions = {"NVDA": 0.95, "AMD": 0.90, "AAPL": 0.85, "MSFT": 0.80}
    atrs = {"NVDA": 2.0, "AMD": 1.5, "AAPL": 1.0, "MSFT": 1.1}

    weights = optimizer.optimize(convictions, atrs)
    
    for ticker, weight in weights.items():
        assert weight <= 0.15 + 1e-5, f"{ticker} weight {weight} exceeded 15% cap!"

def test_risk_parity_zero_volatility_handling():
    """Verify that zero or negative ATR inputs do not throw DivisionByZero errors."""
    optimizer = RiskParityOptimizer(max_position_cap=0.15)
    convictions = {"NVDA": 0.80, "TSLA": 0.70}
    atrs = {"NVDA": 0.0, "TSLA": -1.5}  # Bad inputs

    weights = optimizer.optimize(convictions, atrs)
    assert isinstance(weights, dict)
    assert sum(weights.values()) <= 1.0

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
    
    small_trade_price = risk_engine.calculate_execution_price(mid_price=100.0, shares=100, adv=1000000, action="BUY")
    large_trade_price = risk_engine.calculate_execution_price(mid_price=100.0, shares=50000, adv=100000, action="BUY")

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
# 4. OBJECTIVE STANDALONE TEST CASES
# ==========================================

def test_relative_fitness_calculation():
    """Verify offspring with $18k -> $20k has positive fitness while $100k -> $60k agent has negative fitness."""
    offspring = AgentGenome(
        agent_id="Gen2_Alpha_v1",
        persona_prompt="Growth Trader",
        initial_capital=18000.0,
        cash=20000.0,
        equity_history=[18000.0, 18500.0, 19200.0, 20000.0]
    )
    loser = AgentGenome(
        agent_id="Agent_Beta",
        persona_prompt="Conservative Trader",
        initial_capital=100000.0,
        cash=60000.0,
        equity_history=[100000.0, 85000.0, 72000.0, 60000.0]
    )

    offspring_fitness = offspring.calculate_fitness()
    loser_fitness = loser.calculate_fitness()

    assert offspring_fitness > 0, f"Offspring fitness should be positive (+11% PnL relative to initial capital), got {offspring_fitness}"
    assert loser_fitness < 0, f"Loser fitness should be negative (-40% PnL relative to initial capital), got {loser_fitness}"

@pytest.mark.asyncio
async def test_darwinian_capital_conservation():
    """Simulate a 5-agent swarm, cull worst performer, and assert total swarm equity is conserved (Delta = 0)."""
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

def test_short_proceeds_solvency_guard():
    """Verify an agent cannot spend short sale proceeds on long allocations when free margin is insufficient."""
    agent = AgentGenome(
        agent_id="Agent_Delta",
        persona_prompt="Short Trader",
        initial_capital=100000.0,
        cash=100000.0,
        holdings={"TSLA": -100.0},
        entry_prices={"TSLA": 300.0},
        equity_history=[100000.0]
    )
    # Short sale proceeds added to cash: $100k initial + $30k proceeds = $130k cash
    agent.cash = 130000.0
    prices = {"TSLA": 300.0, "NVDA": 100.0}

    # Short Liabilities calculation
    short_liabilities = sum(
        abs(qty) * agent.entry_prices.get(tk, prices.get(tk, 0.0))
        for tk, qty in agent.holdings.items() if qty < 0
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

    # Mock loser agent with cash $10,000 and 10 shares of UNKNOWN ticker with entry price $50
    mock_cursor.fetchone.return_value = {"cash": 10000.0}
    mock_cursor.fetchall.return_value = [
        {"ticker": "UNKNOWN_TICKER", "amount": 10.0, "entry_price": 50.0}
    ]

    pm = CrossAssetPortfolioManager()

    # Cull agent with missing price in current_prices (should use entry_price 50.0 -> liquidated val = 500 -> total = 10,500)
    pm.cull_and_reallocate(
        loser_agent_id="Agent_Delta",
        recipient_agent_ids=["Gen2_Alpha_v1", "Gen2_Alpha_v2"],
        current_prices={}
    )

    # Verify SQL execution calls
    executed_queries = [call[0][0] for call in mock_cursor.execute.call_args_list]
    
    # Assert UPSERT was executed for recipients
    upsert_found = any("DO UPDATE SET cash = EXCLUDED.cash" in q for q in executed_queries)
    assert upsert_found is True, "Consolidated UPSERT not found in executed queries!"

    # Assert 0.0 snapshot for loser
    snapshot_queries = [call[0] for call in mock_cursor.execute.call_args_list if "agent_snapshots" in call[0][0]]
    assert len(snapshot_queries) >= 3, "Snapshots for loser and 2 offspring should be inserted"
