import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Optional, Tuple

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
    6. Directional Stop-Loss / Take-Profit gates (Long vs Short aware).
    7. Post-Stopout Symbol Cooldowns (4 bars / 1 hour lockout).
    8. Session Circuit Breaker (5% peak-to-trough drawdown -> halt + liquidate).
    """
    def __init__(
        self, 
        target_volatility: float = 0.15, 
        max_position_pct: float = 0.05, 
        base_spread: float = 0.0001, 
        impact_gamma: float = 0.5,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.05,
        cooldown_bars: int = 4,
        session_drawdown_pct: float = 0.05,
        **kwargs
    ):
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct
        self.base_spread = base_spread
        self.impact_gamma = impact_gamma
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.cooldown_bars = int(cooldown_bars)
        self.session_drawdown_pct = float(session_drawdown_pct)

        # Post-stopout cooldowns: (agent_id, ticker) -> expiry_tick (inclusive lockout)
        self.cooldowns: Dict[Tuple[str, str], int] = {}
        # Session breaker state
        self.session_peak_equity: Optional[float] = None
        self.circuit_breaker_tripped: bool = False
        self.trading_halted: bool = False

    # ------------------------------------------------------------------
    # Directional Stop-Loss / Take-Profit
    # ------------------------------------------------------------------
    def check_stop_loss_take_profit(
        self,
        entry_price: float,
        current_price: float,
        direction: str = "LONG",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> Optional[str]:
        """
        Direction-aware exit gate.
        LONG:
          SL if current <= entry * (1 - sl)
          TP if current >= entry * (1 + tp)
        SHORT:
          SL if current >= entry * (1 + sl)
          TP if current <= entry * (1 - tp)
        Returns 'STOP_LOSS' | 'TAKE_PROFIT' | None.
        """
        try:
            if entry_price is None or current_price is None:
                return None
            entry = float(entry_price)
            curr = float(current_price)
            if entry <= 0 or curr <= 0:
                return None
            sl = float(stop_loss_pct) if stop_loss_pct is not None else self.stop_loss_pct
            tp = float(take_profit_pct) if take_profit_pct is not None else self.take_profit_pct
            d = str(direction).upper()
            is_short = d in ("SHORT", "SELL_SHORT", "-1", "S")
            # Normalize LONG aliases
            if not is_short and d not in ("LONG", "BUY", "L", "1", "LONG_POSITION"):
                # Default to LONG for unknown but treat SHORT explicitly
                is_short = False
            if not is_short:
                # Long
                if curr <= entry * (1.0 - sl):
                    return "STOP_LOSS"
                if curr >= entry * (1.0 + tp):
                    return "TAKE_PROFIT"
                return None
            else:
                # Short
                if curr >= entry * (1.0 + sl):
                    return "STOP_LOSS"
                if curr <= entry * (1.0 - tp):
                    return "TAKE_PROFIT"
                return None
        except Exception:
            return None

    def check_position_exit(
        self,
        entry_price: float,
        current_price: float,
        shares: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> Optional[str]:
        """Convenience wrapper inferring direction from signed shares."""
        direction = "SHORT" if float(shares) < 0 else "LONG"
        return self.check_stop_loss_take_profit(
            entry_price, current_price, direction, stop_loss_pct, take_profit_pct
        )

    # ------------------------------------------------------------------
    # Post-Stopout Symbol Cooldowns
    # ------------------------------------------------------------------
    def record_stopout(self, agent_id: str, ticker: str, current_tick: int, bars: Optional[int] = None) -> int:
        """Assign cooldowns[(agent_id, T)] = current_tick + 4 (1 hour lockout). Returns expiry tick."""
        b = int(bars) if bars is not None else self.cooldown_bars
        expiry = int(current_tick) + b
        self.cooldowns[(str(agent_id), str(ticker).upper())] = expiry
        logger.warning(f"🧊 Cooldown set: [{agent_id}] {ticker} locked until tick {expiry} (current {current_tick}).")
        return expiry

    def is_cooled_down(self, agent_id: str, ticker: str, current_tick: int) -> bool:
        """True if (agent, ticker) is still inside cooldown window and re-entry must be rejected."""
        expiry = self.cooldowns.get((str(agent_id), str(ticker).upper()))
        if expiry is None:
            return False
        if int(current_tick) < int(expiry):
            return True
        # Expired -> prune
        try:
            del self.cooldowns[(str(agent_id), str(ticker).upper())]
        except KeyError:
            pass
        return False

    def should_allow_entry(self, agent_id: str, ticker: str, current_tick: int) -> bool:
        """Combined gate: rejects if circuit breaker halted OR symbol on cooldown."""
        if self.trading_halted or self.circuit_breaker_tripped:
            return False
        if self.is_cooled_down(agent_id, ticker, current_tick):
            return False
        return True

    def prune_expired_cooldowns(self, current_tick: int) -> None:
        expired = [k for k, v in self.cooldowns.items() if int(current_tick) >= int(v)]
        for k in expired:
            del self.cooldowns[k]

    # ------------------------------------------------------------------
    # Session Circuit Breaker (5% peak-to-trough)
    # ------------------------------------------------------------------
    def update_session_peak(self, current_equity: float) -> float:
        """Track running session peak equity. Returns current peak."""
        try:
            eq = float(current_equity)
        except Exception:
            return float(self.session_peak_equity or 0.0)
        if self.session_peak_equity is None or eq > self.session_peak_equity:
            self.session_peak_equity = eq
        return float(self.session_peak_equity)

    def check_session_drawdown(self, current_equity: float) -> bool:
        """
        If swarm equity drops >= 5% from session peak, trip breaker:
        lock swarm from new positions (trading_halted=True).
        Returns True if breaker is tripped (either newly or previously).
        """
        if self.circuit_breaker_tripped:
            return True
        try:
            eq = float(current_equity)
        except Exception:
            return False
        # Initialize peak on first call
        if self.session_peak_equity is None:
            self.session_peak_equity = eq
            return False
        if eq > self.session_peak_equity:
            self.session_peak_equity = eq
            return False
        if self.session_peak_equity <= 0:
            return False
        drawdown = (self.session_peak_equity - eq) / self.session_peak_equity
        if drawdown >= self.session_drawdown_pct:
            self.circuit_breaker_tripped = True
            self.trading_halted = True
            logger.error(
                f"🚨 SESSION CIRCUIT BREAKER TRIPPED: equity ${eq:,.2f} "
                f"is {drawdown*100:.2f}% below peak ${self.session_peak_equity:,.2f}. "
                f"Halting new entries, liquidating to cash."
            )
            return True
        return False

    def is_trading_halted(self) -> bool:
        return bool(self.trading_halted or self.circuit_breaker_tripped)

    def reset_session_breaker(self, new_peak: Optional[float] = None) -> None:
        """Manual reset (e.g., new session). Clears halt flag."""
        self.circuit_breaker_tripped = False
        self.trading_halted = False
        if new_peak is not None:
            self.session_peak_equity = float(new_peak)

    def get_session_status(self) -> Dict[str, Any]:
        return {
            "session_peak_equity": self.session_peak_equity,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "trading_halted": self.is_trading_halted(),
            "active_cooldowns": len(self.cooldowns),
        }

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
        max_cap: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Computes Inverse-Volatility Risk Parity allocations weighted by agent conviction scores.
        Enforces strict position caps (default 5.0%) across all individual assets.
        Residual unallocated weight strictly remains unencumbered cash (NO secondary re-normalization).
        """
        if not volatility_map:
            return {}

        effective_cap = max_cap if max_cap is not None else self.max_position_pct

        # Check for zero or negative or invalid volatilities (Flat ATR / Zero Volatility guard)
        valid_vols = {
            tk: float(vol) 
            for tk, vol in volatility_map.items() 
            if vol is not None and not np.isnan(vol) and float(vol) > 0.0
        }
        if not valid_vols:
            return {tk: 0.0 for tk in volatility_map}

        inv_vols = {tk: 1.0 / vol for tk, vol in valid_vols.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0 or np.isnan(total_inv_vol):
            return {tk: 0.0 for tk in volatility_map}

        raw_weights = {tk: inv_vols[tk] / total_inv_vol for tk in inv_vols}

        scaled_allocations = {}
        for tk in volatility_map:
            if tk not in raw_weights:
                scaled_allocations[tk] = 0.0
                continue
            weight = raw_weights[tk]
            conv = convictions.get(tk, 0.5) if convictions else 0.5
            if conv is None or np.isnan(conv) or conv <= 0.0:
                scaled_allocations[tk] = 0.0
                continue
            alloc = weight * conv
            
            # Strict 5% cap without secondary vector re-normalization
            clamped_alloc = min(alloc, effective_cap)
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

    def calculate_execution_price(
        self, 
        raw_price: float = 0.0, 
        shares: float = 0.0, 
        adv: float = 1.0, 
        side: str = "BUY", 
        mid_price: Optional[float] = None, 
        action: Optional[str] = None, 
        **kwargs
    ) -> float:
        """
        Applies Square-Root Market Impact Slippage Model across Long and Short actions based on ADV.
        
        Formula:
        $$P_{exec} = P_{raw} \\cdot \\left(1 \\pm \\gamma \\cdot \\sqrt{\\frac{\\text{Order Shares}}{\\text{ADV}}}\\right)$$
        """
        if mid_price is not None:
            raw_price = mid_price
        if action is not None:
            side = action

        if raw_price <= 0 or adv <= 0 or shares <= 0:
            return max(raw_price, 0.01)

        participation_rate = shares / max(adv, 1.0)
        slippage_pct = 0.10 * np.sqrt(participation_rate)  # 10% market impact factor
        slippage_pct = min(slippage_pct, 0.05)  # Cap maximum slippage friction at 5%

        act = side.upper()
        if act in ("BUY", "COVER"):
            return round(raw_price * (1.0 + slippage_pct), 4)
        elif act in ("SELL", "SHORT"):
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
