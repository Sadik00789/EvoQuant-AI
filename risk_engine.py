import math
import pandas as pd
import numpy as np

class AdvancedRiskEngine:
    def __init__(self, target_volatility: float = 0.15, base_spread: float = 0.0001, impact_gamma: float = 0.5):
        self.target_volatility = target_volatility  # 15% Annual Target Volatility
        self.base_spread = base_spread              # 0.01% Base Spread
        self.impact_gamma = impact_gamma            # Market impact coefficient

    def calculate_regime_scaler(self, spy_returns: pd.Series, spy_prices: pd.Series = None) -> float:
        """
        Computes the market regime multiplier.
        Dampens portfolio leverage during high volatility or when SPY breaks below
        its 200-period Simple Moving Average (Macro Trend Guard).
        """
        if len(spy_returns) < 5:
            return 1.0

        annualized_vol = spy_returns.std() * np.sqrt(252 * 26)  # ~15m interval scaling
        if annualized_vol <= 0:
            scaler = 1.0
        else:
            scaler = self.target_volatility / max(annualized_vol, 0.05)

        # Macro Trend Guard: Cut risk target by 50% if SPY trades below its 200-period SMA
        if spy_prices is not None and len(spy_prices) >= 200:
            sma_200 = spy_prices.rolling(window=200).mean().iloc[-1]
            current_spy = spy_prices.iloc[-1]
            if current_spy < sma_200:
                scaler *= 0.5
                logger.warning(
                    f"📉 MACRO TREND GUARD TRIGGERED: SPY (${current_spy:.2f}) < 200 SMA (${sma_200:.2f}). "
                    f"Scaling risk target to {scaler:.2f}x"
                )

        return float(np.clip(scaler, 0.25, 1.5))

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
