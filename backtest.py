"""
Point-in-time event backtester mirroring the live decision stack (Phase 6).

Unlike the previous long-only, fixed-cadence simulation, this version:
  * trades both longs and shorts,
  * uses the same canonical portfolio-risk allocator as live (per-name, sector,
    gross, net exposure caps),
  * applies a volatility regime scaler with a SPY 200-SMA macro guard,
  * models per-side transaction costs (slippage + fees),
  * enforces directional stop-loss / take-profit,
  * slices data strictly at or before each timestamp (no look-ahead),
  * reports walk-forward fold statistics.

It is intentionally deterministic and LLM-free so it can validate the
technical/risk core that the live system actually executes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_risk import canonical_allocate

UNIVERSE = [
    "NVDA", "AMD", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "AMZN", "NFLX", "INTC",
    "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "CSCO", "ACN", "IBM", "AMAT",
    "MU", "LRCX", "NOW", "PANW", "SNPS", "CDNS", "KLAC", "MCHP", "ADI", "ROP",
    "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "AXP", "PYPL",
    "BLK", "SCHW", "CB", "MMC", "PGR",
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "LLY", "DHR",
    "BMY", "GILD", "CVS", "CI", "ISRG",
    "PG", "HD", "DIS", "COST", "PEP", "KO", "WMT", "NKE", "MCD", "SBUX",
    "LOW", "TJX", "TGT", "EL", "BKNG",
    "HON", "UNP", "GE", "CAT", "BA", "DE", "LMT", "RTX", "ADP", "MMM",
    "XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO", "T", "VZ",
    "TMUS", "PLD", "AMT", "SPGI", "MDLZ", "SPY",
]


class PointInTimeBacktester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        start_date: str = "2020-01-01",
        end_date: str = "2026-01-01",
        slippage: float = 0.0005,
        fee_bps: float = 1.0,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.050,
        target_volatility: float = 0.15,
        rebalance_every: int = 1,
        conviction_threshold: float = 0.30,
        max_position_cap: float = 0.05,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 0.6,
        max_sector_exposure: float = 0.30,
    ):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.slippage = slippage
        self.fee_rate = fee_bps / 10000.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.target_volatility = target_volatility
        self.rebalance_every = max(1, rebalance_every)
        self.conviction_threshold = conviction_threshold
        self.max_position_cap = max_position_cap
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_sector_exposure = max_sector_exposure
        self.data: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    def fetch_historical_data(self):
        print(f"📥 Downloading daily data for {len(UNIVERSE)} assets ({self.start_date} → {self.end_date})...")
        raw = yf.download(UNIVERSE, start=self.start_date, end=self.end_date, interval="1d", progress=False, auto_adjust=True)
        for tk in UNIVERSE:
            try:
                df = pd.DataFrame({
                    "open": raw["Open"][tk],
                    "high": raw["High"][tk],
                    "low": raw["Low"][tk],
                    "close": raw["Close"][tk],
                    "volume": raw["Volume"][tk],
                }).dropna()
                self.data[tk] = df
            except KeyError:
                continue
        print("✅ Download complete.")

    @staticmethod
    def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    def _technical_state(self, tk: str, spy: pd.DataFrame, t: pd.Timestamp) -> Dict[str, float]:
        df = self.data.get(tk)
        if df is None or t not in df.index:
            return {}
        sub = df.loc[:t]
        spy_sub = spy.loc[:t] if spy is not None else None
        if len(sub) < 30:
            return {}
        close = sub["close"]
        rsi = self._wilder_rsi(close).iloc[-1]
        prev_close = close.shift(1)
        tr = pd.concat([
            sub["high"] - sub["low"],
            (sub["high"] - prev_close).abs(),
            (sub["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean().iloc[-1])
        mom = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0.0
        rel = 0.0
        if spy_sub is not None and len(spy_sub) >= 12 and len(close) >= 12:
            sret = (close.iloc[-1] - close.iloc[-12]) / close.iloc[-12]
            bret = (spy_sub["close"].iloc[-1] - spy_sub["close"].iloc[-12]) / spy_sub["close"].iloc[-12]
            rel = float((sret - bret) * 100)
        return {
            "close": float(close.iloc[-1]),
            "rsi": 50.0 if pd.isna(rsi) else float(rsi),
            "atr": max(atr, 0.01),
            "momentum": mom,
            "rel_strength_spy": rel,
        }

    @staticmethod
    def _regime_scaler(spy: pd.DataFrame, t: pd.Timestamp, target_vol: float) -> float:
        try:
            sub = spy.loc[:t]
            if len(sub) < 20:
                return 1.0
            rets = sub["close"].pct_change().dropna()
            vol = rets.std() * math.sqrt(252)
            if not np.isfinite(vol) or vol <= 0:
                return 1.0
            scaler = target_vol / max(vol, 0.05)
            if len(sub) >= 200:
                sma = sub["close"].rolling(200).mean().iloc[-1]
                if not pd.isna(sma) and sub["close"].iloc[-1] < sma:
                    scaler *= 0.5
            return float(np.clip(scaler, 0.25, 1.5))
        except Exception:
            return 1.0

    def _conviction(self, state: Dict[str, float]) -> Tuple[str, float]:
        rsi = state["rsi"]
        mom = state["momentum"]
        rel = state["rel_strength_spy"]
        tech = 0.0
        if rsi < 30:
            tech += 0.4
        elif rsi < 45:
            tech += 0.15
        elif rsi > 70:
            tech -= 0.4
        elif rsi > 60:
            tech -= 0.15
        if mom > 0.5:
            tech += 0.3
        elif mom > 0.05:
            tech += 0.1
        elif mom < -0.5:
            tech -= 0.3
        elif mom < -0.05:
            tech -= 0.1
        if rel > 1.0:
            tech += 0.1
        elif rel < -1.0:
            tech -= 0.1
        tech = max(-1.0, min(1.0, tech))
        if tech >= self.conviction_threshold:
            return "BUY", abs(tech)
        if tech <= -self.conviction_threshold:
            return "SHORT", abs(tech)
        return "HOLD", 0.0

    # ------------------------------------------------------------------
    def run(self):
        self.fetch_historical_data()
        spy = self.data.get("SPY")
        if spy is None or spy.empty:
            print("❌ SPY data unavailable; aborting.")
            return
        tradeable = [tk for tk in UNIVERSE if tk in self.data and tk != "SPY"]
        timestamps = list(spy.index[30:])

        cash = self.initial_capital
        holdings: Dict[str, float] = {tk: 0.0 for tk in tradeable}
        entry_prices: Dict[str, float] = {tk: 0.0 for tk in tradeable}
        equity_curve: List[float] = []
        trading_days: List[pd.Timestamp] = []
        turnover = 0.0

        print(f"🚀 Running point-in-time backtest across {len(tradeable)} names...")

        for t_idx, t in enumerate(timestamps):
            prices: Dict[str, float] = {}
            for tk in tradeable:
                if t in self.data[tk].index:
                    prices[tk] = float(self.data[tk].loc[t, "close"])
                else:
                    prices[tk] = entry_prices.get(tk, 0.0)

            long_val = sum(holdings[tk] * prices.get(tk, 0.0) for tk in tradeable if holdings[tk] > 0)
            short_liab = sum(abs(holdings[tk]) * prices.get(tk, 0.0) for tk in tradeable if holdings[tk] < 0)
            short_entry_val = sum(abs(holdings[tk]) * entry_prices.get(tk, 0.0) for tk in tradeable if holdings[tk] < 0)
            current_equity = cash + long_val - short_liab + short_entry_val
            equity_curve.append(current_equity)
            trading_days.append(t)

            # Directional stops.
            for tk, shares in list(holdings.items()):
                if shares == 0 or prices.get(tk, 0.0) <= 0 or entry_prices.get(tk, 0.0) <= 0:
                    continue
                px = prices[tk]
                e = entry_prices[tk]
                if shares > 0:
                    move = (px - e) / e
                    hit = move <= -self.stop_loss_pct or move >= self.take_profit_pct
                else:
                    move = (e - px) / e
                    hit = move <= -self.stop_loss_pct or move >= self.take_profit_pct
                if hit:
                    notional = abs(shares) * px
                    cost = notional * (self.slippage + self.fee_rate)
                    cash += (shares * px) - cost if shares > 0 else (shares * px) - cost
                    turnover += notional
                    holdings[tk] = 0.0
                    entry_prices[tk] = 0.0

            # Rebalance.
            if t_idx % self.rebalance_every == 0:
                convictions: Dict[str, float] = {}
                directions: Dict[str, str] = {}
                atr_map: Dict[str, float] = {}
                for tk in tradeable:
                    st = self._technical_state(tk, spy, t)
                    if not st:
                        continue
                    action, conv = self._conviction(st)
                    directions[tk] = action
                    convictions[tk] = conv
                    atr_map[tk] = st["atr"]

                regime = self._regime_scaler(spy, t, self.target_volatility)
                target_weights = canonical_allocate(
                    convictions, atr_map, directions,
                    max_position_cap=self.max_position_cap,
                    max_gross_exposure=self.max_gross_exposure,
                    max_net_exposure=self.max_net_exposure,
                    max_sector_exposure=self.max_sector_exposure,
                    min_conviction=self.conviction_threshold,
                    regime_scaler=regime,
                )

                for tk, w in target_weights.items():
                    px = prices.get(tk, 0.0)
                    if px <= 0:
                        continue
                    want = (w * current_equity) / px
                    have = holdings.get(tk, 0.0)
                    delta = want - have
                    if abs(delta * px) < 25.0:
                        continue
                    notional = abs(delta) * px
                    cost = notional * (self.slippage + self.fee_rate)
                    cash += (have - want) * px - cost
                    turnover += notional
                    if abs(want) < 1e-9:
                        entry_prices[tk] = 0.0
                    elif abs(have) < 1e-9 or ((have > 0) != (want > 0)):
                        entry_prices[tk] = px
                    elif abs(want) > abs(have):
                        entry_prices[tk] = ((abs(have) * entry_prices.get(tk, px)) + abs(delta) * px) / abs(want)
                    holdings[tk] = 0.0 if abs(want) < 1e-9 else want

        eq = pd.Series(equity_curve, index=pd.DatetimeIndex(trading_days))
        metrics = self._metrics(eq, turnover)
        self._print_metrics(metrics)
        self._walk_forward(eq)
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(eq: pd.Series, turnover: float) -> Dict[str, float]:
        if eq.empty or len(eq) < 2:
            return {}
        rets = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        years = max(len(eq) / 252.0, 1e-9)
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
        sharpe = (rets.mean() / rets.std()) * math.sqrt(252) if rets.std() > 0 else 0.0
        downside = rets[rets < 0]
        sortino = (rets.mean() / downside.std()) * math.sqrt(252) if len(downside) > 1 and downside.std() > 0 else 0.0
        mdd = ((eq.cummax() - eq) / eq.cummax()).max()
        return {
            "final_equity": float(eq.iloc[-1]),
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(mdd),
            "turnover": float(turnover),
        }

    def _print_metrics(self, m: Dict[str, float]):
        if not m:
            print("No metrics produced.")
            return
        print("\n================ 📊 BACKTEST PERFORMANCE SUMMARY ================")
        print(f"Final Equity:      ${m['final_equity']:,.2f}")
        print(f"Total Return:      {m['total_return']*100:+.2f}%")
        print(f"CAGR:              {m['cagr']*100:+.2f}%")
        print(f"Sharpe:            {m['sharpe']:.2f}")
        print(f"Sortino:           {m['sortino']:.2f}")
        print(f"Max Drawdown:      {m['max_drawdown']*100:.2f}%")
        print(f"Turnover (notional): ${m['turnover']:,.0f}")
        print("==================================================================")

    def _walk_forward(self, eq: pd.Series, folds: int = 4):
        if eq.empty or len(eq) < folds * 20:
            return
        print("\n================ 🔁 WALK-FORWARD FOLDS ================")
        size = len(eq) // folds
        for i in range(folds):
            seg = eq.iloc[i * size:(i + 1) * size]
            if len(seg) < 2:
                continue
            m = self._metrics(seg, turnover=0.0)
            print(
                f"Fold {i+1} [{seg.index[0].date()} → {seg.index[-1].date()}]: "
                f"ret {m['total_return']*100:+.2f}% | Sharpe {m['sharpe']:.2f} | MaxDD {m['max_drawdown']*100:.2f}%"
            )
        print("=======================================================")


if __name__ == "__main__":
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    print(f"📅 Backtest window: {start_date} → {end_date}")
    PointInTimeBacktester(start_date=start_date, end_date=end_date).run()
