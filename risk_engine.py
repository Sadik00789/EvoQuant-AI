import math
import pandas as pd
import numpy as np

class AdvancedRiskEngine:
    def __init__(self, target_volatility: float = 0.15, base_spread: float = 0.0001, impact_gamma: float = 0.5):
        self.target_volatility = target_volatility  # 15% Annual Target Volatility
        self.base_spread = base_spread              # 0.01% Base Spread
        self.impact_gamma = impact_gamma            # Market impact coefficient

    def calculate_regime_scaler(self, spy_returns: pd.Series) -> float:
        """
        Dynamic Volatility Regime Scaling.
        Calculates annualized volatility of SPY and returns exposure scaling factor [0.1, 1.0].
        """
        if len(spy_returns) < 14:
            return 1.0
        
        # Annualized Volatility calculation
        annual_vol = float(spy_returns.tail(20).std() * math.sqrt(252))
        if annual_vol <= 0 or np.isnan(annual_vol):
            return 1.0

        # Scale capital exposure downwards if market volatility spikes above target
        scaling_factor = self.target_volatility / annual_vol
        return float(np.clip(scaling_factor, 0.20, 1.0))

    def calculate_execution_price(self, mid_price: float, shares: float, adv: float, action: str) -> float:
        """
        Advanced Market Impact Slippage Model.
        Adjusts execution price dynamically based on order volume relative to Average Daily Volume (ADV).
        """
        if adv <= 0 or mid_price <= 0:
            slippage = self.base_spread
        else:
            order_ratio = shares / adv
            slippage = self.base_spread + self.impact_gamma * (order_ratio ** 2)

        slippage = float(np.clip(slippage, 0.0001, 0.05))  # Cap slippage at 5% max

        if action.upper() == "BUY":
            return mid_price * (1.0 + slippage)
        else:
            return mid_price * (1.0 - slippage)

if __name__ == "__main__":
    risk_engine = AdvancedRiskEngine()
    
    # Test Slippage
    price = 200.0
    buy_exec = risk_engine.calculate_execution_price(mid_price=price, shares=5000, adv=100000, action="BUY")
    print(f"Mid Price: ${price:.2f} | Buy Execution Price (with Market Impact): ${buy_exec:.4f}")
    
    # Test Volatility Regime Scaling
    fake_spy_returns = pd.Series(np.random.normal(0.0, 0.02, 30))  # High vol market
    scaler = risk_engine.calculate_regime_scaler(fake_spy_returns)
    print(f"High Volatility Market Capital Exposure Cap: {scaler * 100:.1f}%")