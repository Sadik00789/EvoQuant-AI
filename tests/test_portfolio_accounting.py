"""
Unit tests for Institutional Margin Accounting, Purchasing Power Locks,
Zero-Price Fallbacks, and Live MTM Equity Calculations.

Tests:
1. Cash receipt on short entry: self.cash credits shares * P_fill, preventing artificial equity drops.
2. Margin collateral lock: available_cash strictly locks short_liability * (1.0 + margin_req).
3. Long target book allocation constraint: BUYs are capped at agent.available_cash.
4. Zero price fallback guard: missing prices fall back to entry_price instead of 0.0.
5. Live MTM equity trajectory: short price appreciation reduces equity, depreciation increases equity.
6. Cover order settlement: realized PnL correctly settles into ledger cash and frees collateral.
7. AgentGenome backwards compatibility: positional / minimal instantiation preserved.
8. CrossAssetPortfolioManager parity: database/portfolio manager mirrors margin calculations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from evolution_engine import AgentGenome
from engine import CrossAssetPortfolioManager
from swarm_consumer import _apply_target_book


def test_agent_genome_backwards_compatibility():
    """Verify AgentGenome can be instantiated with only required positional arguments."""
    agent = AgentGenome(agent_id="legacy_test", persona_prompt="conservative value")
    assert agent.agent_id == "legacy_test"
    assert agent.cash == 100000.0
    assert agent.initial_capital == 100000.0
    assert agent.margin_requirement == 0.50
    assert agent.short_positions == {}
    assert agent.last_prices == {}
    assert agent.available_cash == 100000.0
    assert agent.calculate_equity() == 100000.0


def test_short_entry_credits_cash_and_preserves_equity():
    """
    Enforce Cash Receipt on Short Entry:
    self.cash += shares * P_fill
    short_liability = sum(shares * P_current)
    Total Equity = self.cash + long_value - short_liability

    Avoids trap: entering short must NOT suffer an immediate 10% equity loss!
    """
    agent = AgentGenome(agent_id="short_trader", persona_prompt="short seller", cash=100000.0)

    # Short 100 shares of TSLA at $100 ($10,000 notional)
    agent.fill_order("TSLA", "SHORT", 100.0, 100.0)

    # 1. Cash must be credited
    assert agent.cash == 110000.0

    # 2. Holdings should be negative
    assert agent.holdings["TSLA"] == -100.0
    assert agent.short_positions["TSLA"]["shares"] == 100.0
    assert agent.short_positions["TSLA"]["entry_price"] == 100.0

    # 3. Total equity immediately after entry must equal initial capital (no 10% instant loss)
    equity = agent.calculate_equity({"TSLA": 100.0})
    assert equity == 100000.0


def test_lock_margin_collateral_in_purchasing_power():
    """
    Lock Margin Collateral in Purchasing Power:
    margin_hold = short_liability * (1.0 + self.margin_requirement)
    available_cash = max(0.0, self.cash - margin_hold)

    Avoids trap: agent must NOT have phantom $110,000 purchasing power.
    On a $10,000 short with 50% margin requirement:
    margin_hold = $10,000 * 1.50 = $15,000
    available_cash = $110,000 - $15,000 = $95,000
    """
    agent = AgentGenome(
        agent_id="margin_guard_trader",
        persona_prompt="momentum short",
        cash=100000.0,
        margin_requirement=0.50,
    )

    agent.fill_order("SPY", "SHORT", 100.0, 100.0)

    assert agent.cash == 110000.0
    # Purchasing power must be $95,000 (never $110,000)
    assert agent.available_cash == 95000.0


def test_zero_price_fallback_guard_prevents_equity_wipes():
    """
    Zero Price Fallback Guard:
    In calculate_equity and available_cash, never default missing prices to 0.0:
    current_price = self.last_prices.get(ticker, pos.get('entry_price', 0.0))

    Avoids trap: omitted ticker from market bar must not wipe long value or zero short liability.
    """
    agent = AgentGenome(agent_id="fallback_trader", persona_prompt="balanced", cash=100000.0)

    # 1. Test short position fallback
    agent.fill_order("NVDA", "SHORT", 100.0, 150.0)
    assert agent.cash == 115000.0

    # Incoming market bar completely omits NVDA or passes empty dict
    equity_missing_tick = agent.calculate_equity({})
    assert equity_missing_tick == 100000.0
    assert agent.available_cash == 115000.0 - (15000.0 * 1.50)

    # Incoming market bar has 0.0 price for NVDA
    equity_zero_tick = agent.calculate_equity({"NVDA": 0.0})
    assert equity_zero_tick == 100000.0

    # 2. Test long position fallback
    agent_long = AgentGenome(agent_id="long_fallback", persona_prompt="long only", cash=100000.0)
    agent_long.fill_order("AAPL", "BUY", 100.0, 200.0)
    assert agent_long.cash == 80000.0

    # Omitted ticker: equity must not drop to $80,000
    equity_long_omitted = agent_long.calculate_equity({})
    assert equity_long_omitted == 100000.0

    # Explicit 0.0 price: equity must not drop to $80,000
    equity_long_zero = agent_long.calculate_equity({"AAPL": 0.0})
    assert equity_long_zero == 100000.0


def test_target_book_constrains_longs_to_available_cash():
    """
    Constraint: When evaluating target books in _apply_target_book,
    long allocations must be constrained by min(allocated_target, agent.available_cash).
    """
    agent = AgentGenome(
        agent_id="constrained_trader",
        persona_prompt="quant",
        cash=100000.0,
        margin_requirement=0.50,
    )

    # Enter a large short: 500 shares @ $100 = $50,000 notional short
    agent.fill_order("SHORT_STOCK", "SHORT", 500.0, 100.0)
    # cash = 100k + 50k = 150k
    # margin_hold = 50k * 1.5 = 75k
    # available_cash = 150k - 75k = 75,000
    assert agent.available_cash == 75000.0

    # Mock DB interactions
    fake_db = MagicMock()

    prices = {"SHORT_STOCK": 100.0, "LONG_STOCK": 100.0}

    # Agent target book requests $100,000 allocation in LONG_STOCK (weight = 1.0 on $100k equity)
    # Allocated dollars would normally be $100,000 (1000 shares), but available_cash is only $75,000.
    target_weights = {"LONG_STOCK": 1.0, "SHORT_STOCK": -0.5}

    trades = _apply_target_book(agent, target_weights, prices, fake_db)

    # Verify LONG_STOCK buy order was executed for at most $75,000 (750 shares @ $100)
    long_trades = [t for t in trades if t[0] == "LONG_STOCK" and t[1] == "BUY"]
    assert len(long_trades) == 1
    tk, action, shares, px = long_trades[0]
    assert shares <= 750.0 + 1e-6
    assert shares * px <= 75000.0 + 1e-4


def test_mark_to_market_short_price_fluctuations_and_cover():
    """
    Verify live MTM equity calculation on price movements and final short cover.
    - Short price up -> equity down
    - Short price down -> equity up
    - Short cover settles PnL and restores full collateral
    """
    agent = AgentGenome(agent_id="mtm_short", persona_prompt="short", cash=100000.0)
    agent.fill_order("XYZ", "SHORT", 100.0, 100.0)  # Cash = $110,000

    # Price jumps to $120 (+20% against short)
    equity_adverse = agent.calculate_equity({"XYZ": 120.0})
    # Equity = $110,000 - (100 * 120) = $98,000 (-$2,000)
    assert equity_adverse == 98000.0
    # Available cash = 110,000 - (12,000 * 1.5) = $92,000
    assert agent.available_cash == 92000.0

    # Price drops to $80 (-20% favorable to short)
    equity_favorable = agent.calculate_equity({"XYZ": 80.0})
    # Equity = $110,000 - (100 * 80) = $102,000 (+$2,000)
    assert equity_favorable == 102000.0
    # Available cash = 110,000 - (8,000 * 1.5) = $98,000
    assert agent.available_cash == 98000.0

    # Cover short at $80
    agent.fill_order("XYZ", "COVER", 100.0, 80.0)
    # Cash debits 100 * 80 = $8,000 -> Cash = 110,000 - 8,000 = $102,000
    assert agent.cash == 102000.0
    assert agent.holdings["XYZ"] == 0.0
    assert "XYZ" not in agent.short_positions
    assert agent.available_cash == 102000.0
    assert agent.calculate_equity({"XYZ": 80.0}) == 102000.0


def test_cross_asset_portfolio_manager_margin_parity():
    """Verify CrossAssetPortfolioManager provides matching margin accounting methods."""
    mock_pool = MagicMock()
    with patch("engine.ConnectionPool", return_value=mock_pool):
        pm = CrossAssetPortfolioManager()
        pm.pool = mock_pool
        pm.cash = 100000.0
        pm.margin_requirement = 0.50

        # Short order fill
        pm.fill_order("QQQ", "SHORT", 100.0, 100.0)
        assert pm.cash == 110000.0
        assert pm.available_cash == 95000.0
        assert pm.calculate_equity({"QQQ": 100.0}) == 100000.0

        # Zero price fallback guard
        assert pm.calculate_equity({}) == 100000.0
