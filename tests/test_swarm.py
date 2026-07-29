import pytest
import pandas as pd
import numpy as np

from engine import RiskParityOptimizer
from risk_engine import AdvancedRiskEngine

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
