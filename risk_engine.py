import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("AdvancedRiskEngine")

class AdvancedRiskEngine:
    """
    Institutional Quantitative Risk Management Engine.
    Features:
    1. Downside Deviation (Semi-Variance) & Inverse-Volatility Risk Parity Allocations.
    2. Parametric Conditional Value-at-Risk (CVaR / Expected Shortfall).
    3. Square-Root Market Impact Slippage Model (BUY, SELL, SHORT, COVER) based on ADV.
    4. Volatility Regime Scaler with NaN Volatility Protection & Macro Trend Guarding (SPY 200 SMA).
    5. Short Margin Collateral & Free Margin Health Evaluator.
    """
    def __init__(self, target_volatility: float = 0.15, max_position_pct: float = 0.05):
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct

    def calculate_downside_volatility(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculates Downside Deviation (Semi-Variance).
        Penalizes negative returns while ignoring upside volatility to optimize
        allocations for Sortino-ratio maximization.
        
        Formula:
        $$SD_{down} = \\sqrt{\\frac{1}{N} \\sum_{t=1}^{N} \\min(0, R_t - R_{target})^2}$$
        """
        if returns is None or len(returns) < 2:
            return 0.0001
        
        cleaned_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cleaned_returns) < 2:
            return 0.0001

        downside_returns = cleaned_returns[cleaned_returns < target_return]
        if len(downside_returns) == 0:
            return 0.0001
            
        downside_variance = np.mean(downside_returns ** 2)
        return float(max(np.sqrt(downside_variance), 0.0001))

    def calculate_cvar(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """
        Calculates Conditional Value at Risk (CVaR / Expected Shortfall) at the (1 - alpha) confidence level.
        Evaluates the expected magnitude of tail losses beyond the VaR threshold.
        
        Formula:
        $$CVaR_{\\alpha}(X) = \\mathbb{E}[X \\mid X \\le VaR_{\\alpha}(X)]$$
        """
        if returns is None or len(returns) < 10:
            return 0.02

        cleaned_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cleaned_returns) < 10:
            return 0.02
            
        sorted_returns = np.sort(cleaned_returns.values)
        cutoff_index = int(np.floor(alpha * len(sorted_returns)))
        if cutoff_index == 0:
            cutoff_index = 1
            
        tail_losses = sorted_returns[:cutoff_index]
        cvar = -np.mean(tail_losses)
        return float(max(cvar, 0.001))

    def calculate_risk_parity_allocations(
        self, 
        volatility_map: Dict[str, float], 
        convictions: Dict[str, float], 
        max_cap: float = 0.05
    ) -> Dict[str, float]:
        """
        Computes Inverse-Volatility Risk Parity allocations weighted by agent conviction scores.
        Enforces strict position caps (default 5%) across all assets.
        """
        if not volatility_map:
            return {}

        inv_vols = {tk: 1.0 / max(vol, 0.0001) for tk, vol in volatility_map.items() if vol is not None}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0:
            return {tk: 0.0 for tk in volatility_map}

        raw_weights = {tk: inv_vols[tk] / total_inv_vol for tk in inv_vols}

        scaled_allocations = {}
        for tk, weight in raw_weights.items():
            conv = convictions.get(tk, 0.5)
            alloc = weight * conv
            clamped_alloc = min(alloc, max_cap)
            scaled_allocations[tk] = round(clamped_alloc, 4)

        return scaled_allocations

    def calculate_regime_scaler(self, spy_returns: pd.Series, spy_prices: Optional[pd.Series] = None) -> float:
        """
        Computes the market regime multiplier safely against NaN returns and flat volatility periods.
        Dampens portfolio leverage during high volatility or when SPY breaks below
        its 200-period Simple Moving Average (Macro Trend Guard).
        """
        if spy_returns is None or len(spy_returns) < 5 or spy_returns.empty:
            return 1.0

        cleaned_returns = spy_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cleaned_returns) < 5:
            return 1.0

        vol = cleaned_returns.std()
        # NaN / Flat Volatility Trap Guard
        if pd.isna(vol) or vol <= 0:
            return 1.0

        annualized_vol = vol * np.sqrt(252 * 26)  # ~15m interval scaling
        scaler = self.target_volatility / max(annualized_vol, 0.05)

        # Macro Trend Guard: Cut risk by 50% if SPY trades below its 200-period SMA
        if spy_prices is not None and len(spy_prices) >= 200:
            cleaned_prices = spy_prices.replace([np.inf, -np.inf], np.nan).dropna()
            if len(cleaned_prices) >= 200:
                sma_200 = cleaned_prices.rolling(window=200).mean().iloc[-1]
                current_spy = cleaned_prices.iloc[-1]
                if not pd.isna(sma_200) and current_spy < sma_200:
                    scaler *= 0.5
                    logger.warning(
                        f"📉 MACRO TREND GUARD TRIGGERED: SPY (${current_spy:.2f}) < 200 SMA (${sma_200:.2f}). "
                        f"Scaling risk target to {scaler:.2f}x"
                    )

        return float(np.clip(scaler, 0.25, 1.5))

    def calculate_execution_price(self, raw_price: float, shares: float, adv: float, side: str) -> float:
        """
        Applies Square-Root Market Impact Slippage Model across Long and Short actions based on ADV.
        
        Formula:
        $$P_{exec} = P_{raw} \\cdot \\left(1 \\pm \\gamma \\cdot \\sqrt{\\frac{\\text{Order Shares}}{\\text{ADV}}}\\right)$$
        """
        if raw_price <= 0 or adv <= 0 or shares <= 0:
            return max(raw_price, 0.01)

        participation_rate = shares / max(adv, 1.0)
        slippage_pct = 0.10 * np.sqrt(participation_rate)  # 10% market impact factor
        slippage_pct = min(slippage_pct, 0.05)  # Cap maximum slippage friction at 5%

        action = side.upper()
        if action in ("BUY", "COVER"):
            return round(raw_price * (1.0 + slippage_pct), 4)
        elif action in ("SELL", "SHORT"):
            return round(raw_price * (1.0 - slippage_pct), 4)
            
        return round(raw_price, 4)

    def evaluate_margin_health(
        self, 
        cash: float, 
        holdings: Dict[str, float], 
        prices: Dict[str, float], 
        initial_margin_req: float = 1.50
    ) -> Dict[str, Any]:
        """
        Calculates Net Equity, Long Valuation, Short Liability, and Free Margin.
        Triggers margin call flag if Free Margin falls below zero.
        """
        long_val = sum(qty * prices.get(tk, 0.0) for tk, qty in holdings.items() if qty > 0)
        short_liability = sum(abs(qty) * prices.get(tk, 0.0) for tk, qty in holdings.items() if qty < 0)

        net_equity = cash + long_val - short_liability
        required_margin = short_liability * initial_margin_req
        free_margin = net_equity - required_margin

        return {
            "net_equity": round(net_equity, 2),
            "long_val": round(long_val, 2),
            "short_liability": round(short_liability, 2),
            "required_margin": round(required_margin, 2),
            "free_margin": round(free_margin, 2),
            "margin_call_triggered": free_margin < 0.0
        }
