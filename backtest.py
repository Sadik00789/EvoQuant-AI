import math
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple

# Import your existing Risk Parity engine components
from engine import RiskParityOptimizer

UNIVERSE = [
    "NVDA", "AMD", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "AMZN",
    "JPM", "V", "PG", "JNJ", "XOM", "COST", "KO", "WMT", "SPY"
]

class EventDrivenBacktester:
    def __init__(self, initial_capital: float = 100000.0, start_date: str = "2024-01-01", end_date: str = "2026-01-01", slippage: float = 0.0002):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.slippage = slippage
        self.optimizer = RiskParityOptimizer(max_position_cap=0.15)
        self.data: Dict[str, pd.DataFrame] = {}

    def fetch_historical_data(self):
        print(f"📥 Downloading historical price data ({self.start_date} to {self.end_date})...")
        raw_data = yf.download(UNIVERSE, start=self.start_date, end=self.end_date, interval="1d", progress=False)
        
        # Safely unpack MultiIndex columns per ticker
        for tk in UNIVERSE:
            try:
                df = pd.DataFrame({
                    'open': raw_data['Open'][tk],
                    'high': raw_data['High'][tk],
                    'low': raw_data['Low'][tk],
                    'close': raw_data['Close'][tk],
                    'volume': raw_data['Volume'][tk]
                }).dropna()
                self.data[tk] = df
            except KeyError:
                print(f"⚠️ Warning: Could not download data for {tk}")
        print("✅ Market data download complete.")

    def calculate_technical_state(self, tk_data: pd.DataFrame, current_time: pd.Timestamp) -> dict:
        """Point-in-Time Technical Analysis using strictly label-based slicing [:current_time]."""
        sub_df = tk_data.loc[:current_time]
        if len(sub_df) < 26:
            return {}

        close = sub_df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, 1e-6))
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # ATR 14 calculation
        prev_close = close.shift(1)
        tr1 = sub_df['high'] - sub_df['low']
        tr2 = (sub_df['high'] - prev_close).abs()
        tr3 = (sub_df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        return {
            "close": float(close.iloc[-1]),
            "rsi": round(rsi, 2),
            "atr": max(round(atr, 2), 0.01)
        }

    def run(self):
        self.fetch_historical_data()
        tradeable_universe = [tk for tk in UNIVERSE if tk in self.data and tk != "SPY"]
        timestamps = self.data["SPY"].index[30:]  # Skip warm-up period
        
        cash = self.initial_capital
        holdings: Dict[str, float] = {tk: 0.0 for tk in tradeable_universe}
        entry_prices: Dict[str, float] = {tk: 0.0 for tk in tradeable_universe}
        equity_curve: List[float] = []

        print("🚀 Executing Point-in-Time Backtest Simulation...")

        for t_idx, current_time in enumerate(timestamps):
            # 1. Update Portfolio Valuations
            current_prices = {}
            for tk in holdings:
                if current_time in self.data[tk].index:
                    current_prices[tk] = float(self.data[tk].loc[current_time, 'close'])
                else:
                    current_prices[tk] = entry_prices[tk]

            total_stock_value = sum(holdings[tk] * current_prices.get(tk, 0.0) for tk in holdings)
            current_equity = cash + total_stock_value
            equity_curve.append(current_equity)

            # 2. Hard Risk Overlays (Stop-Loss -2.5% | Take-Profit +5.0%)
            for tk, shares in list(holdings.items()):
                if shares > 0 and tk in current_prices and entry_prices[tk] > 0:
                    price = current_prices[tk]
                    pnl_pct = (price - entry_prices[tk]) / entry_prices[tk]
                    
                    if pnl_pct <= -0.025 or pnl_pct >= 0.05:
                        # Liquidate position with slippage friction
                        cash += shares * price * (1 - self.slippage)
                        holdings[tk] = 0.0
                        entry_prices[tk] = 0.0

            # 3. Simulate Technical Indicator Rebalancing Every 5 Days
            if t_idx % 5 == 0:
                convictions = {}
                atrs = {}

                for tk in holdings:
                    if current_time in self.data[tk].index:
                        metrics = self.calculate_technical_state(self.data[tk], current_time)
                        if metrics:
                            atrs[tk] = metrics["atr"]
                            if metrics["rsi"] < 35:
                                convictions[tk] = 0.85
                            elif metrics["rsi"] > 65:
                                convictions[tk] = 0.10
                            else:
                                convictions[tk] = 0.45

                # 4. Run Risk Parity Optimizer
                target_weights = self.optimizer.optimize(convictions, atrs)

                # Separate into Sells first (free up cash), then Buys
                sells = []
                buys = []

                for tk, target_w in target_weights.items():
                    if tk not in holdings or current_prices.get(tk, 0) <= 0:
                        continue
                    price = current_prices[tk]
                    target_alloc_dollars = current_equity * target_w
                    current_pos_dollars = holdings[tk] * price
                    diff_dollars = target_alloc_dollars - current_pos_dollars

                    if diff_dollars < 0:
                        sells.append((tk, abs(diff_dollars), price))
                    elif diff_dollars > 0:
                        buys.append((tk, diff_dollars, price))

                # Step 4a: Execute Sells First
                for tk, diff_dollars, price in sells:
                    sell_shares = diff_dollars / price
                    actual_sell_shares = min(holdings[tk], sell_shares)
                    holdings[tk] -= actual_sell_shares
                    cash += (actual_sell_shares * price) * (1 - self.slippage)
                    if holdings[tk] <= 1e-6:
                        holdings[tk] = 0.0
                        entry_prices[tk] = 0.0

                # Step 4b: Execute Buys Second
                for tk, diff_dollars, price in buys:
                    alloc_dollars = min(diff_dollars, cash)
                    if alloc_dollars > 0:
                        bought_dollars_after_slippage = alloc_dollars * (1 - self.slippage)
                        new_shares = bought_dollars_after_slippage / price
                        total_shares = holdings[tk] + new_shares
                        
                        entry_prices[tk] = ((holdings[tk] * entry_prices[tk]) + bought_dollars_after_slippage) / total_shares if total_shares > 0 else price
                        holdings[tk] = total_shares
                        cash -= alloc_dollars

        # 5. Compute Final Performance Statistics
        eq_series = pd.Series(equity_curve)
        returns = eq_series.pct_change().dropna()
        total_return = (eq_series.iloc[-1] - self.initial_capital) / self.initial_capital
        cagr = ((eq_series.iloc[-1] / self.initial_capital) ** (252 / len(eq_series))) - 1
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((eq_series.cummax() - eq_series) / eq_series.cummax()).max()

        print("\n================ 📊 BACKTEST PERFORMANCE SUMMARY ================")
        print(f"Initial Capital:         ${self.initial_capital:,.2f}")
        print(f"Final Equity:            ${eq_series.iloc[-1]:,.2f}")
        print(f"Total Cumulative Return: {total_return * 100:+.2f}%")
        print(f"CAGR (Annualized):       {cagr * 100:+.2f}%")
        print(f"Sharpe Ratio:            {sharpe:.2f}")
        print(f"Max Drawdown:            {max_drawdown * 100:.2f}%")
        print("==================================================================")

if __name__ == "__main__":
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    print(f"📅 Running Backtest Window: {start_date} ➔ {end_date}")

    tester = EventDrivenBacktester(
        initial_capital=100000.0,
        start_date=start_date,
        end_date=end_date
    )
    tester.run()
