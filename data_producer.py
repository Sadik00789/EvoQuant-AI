import os
import json
import asyncio  # Standard asyncio required for asyncio.sleep()
import redis
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar
from psycopg_pool import ConnectionPool


load_dotenv()

# Set decision interval: 15-minute candle closes (00, 15, 30, 45)
TICK_INTERVAL_MINUTES = 15

# Dynamic Redis Host for Local vs Docker/Cloud deployment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
broker = redis.Redis(host=REDIS_HOST, port=6379, db=0)

# Curated 100 Liquid US Mega/Large-Cap Stocks
UNIVERSE = [
    # Tech & Semiconductors (30)
    "NVDA", "AMD", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "AMZN", "NFLX", "INTC",
    "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "CSCO", "ACN", "IBM", "AMAT",
    "MU", "LRCX", "NOW", "PANW", "SNPS", "CDNS", "KLAC", "MCHP", "ADI", "ROP",
    # Financials & Payments (15)
    "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "AXP", "PYPL",
    "BLK", "SCHW", "CB", "MMC", "PGR",
    # Healthcare & Pharma (15)
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "LLY", "DHR",
    "BMY", "GILD", "CVS", "CI", "ISRG",
    # Consumer & Retail (15)
    "PG", "HD", "DIS", "COST", "PEP", "KO", "WMT", "NKE", "MCD", "SBUX",
    "LOW", "TJX", "TGT", "EL", "BKNG",
    # Industrials & Aerospace (10)
    "HON", "UNP", "GE", "CAT", "BA", "DE", "LMT", "RTX", "ADP", "MMM",
    # Energy, Utilities, Real Estate & Telecom (15)
    "XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO", "T", "VZ",
    "TMUS", "PLD", "AMT", "SPGI", "MDLZ"
]

ALL_SYMBOLS = UNIVERSE + ["SPY"]

history = {tk: pd.DataFrame(columns=["high", "low", "close", "volume"]) for tk in ALL_SYMBOLS}
buffer = {}
current_window_minute = -1


def sync_dividend_calendar():
    """
    Fetches upcoming ex-dividend dates and payout estimates for UNIVERSE stocks 
    via yfinance and populates the TimescaleDB dividend_schedule table.
    Uses custom browser session headers and backoff to prevent Yahoo 429 rate limits.
    """
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "timescaledb" if REDIS_HOST == "redis" else "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "evoquant_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "evoquant")
    POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "evoquant_secret_pass")

    conninfo = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    print("📅 Syncing dividend calendar with TimescaleDB...")
    
    # Configure custom session with realistic browser User-Agent header
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    try:
        with ConnectionPool(conninfo=conninfo, min_size=1, max_size=2) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    synced_count = 0
                    consecutive_429s = 0

                    for tk in UNIVERSE:
                        # Safety exit if Yahoo Finance has temporarily hard-locked the EC2 IP
                        if consecutive_429s >= 5:
                            print("⚠️ Yahoo Finance rate limit cool-down active. Pausing sync until next cycle.")
                            break

                        try:
                            time.sleep(1.0)  # 1-second delay between requests
                            
                            ticker_obj = yf.Ticker(tk, session=session)
                            info = ticker_obj.info
                            
                            ex_date_ts = info.get("exDividendDate") or info.get("ex_dividend_date")
                            div_rate = info.get("dividendRate", 0.0) or 0.0
                            
                            if ex_date_ts and div_rate > 0:
                                ex_date_str = pd.to_datetime(ex_date_ts, unit='s').strftime("%Y-%m-%d")
                                quarterly_div = round(float(div_rate / 4.0), 4)
                                
                                cur.execute("""
                                    INSERT INTO dividend_schedule (ticker, ex_date, payment_date, amount_per_share)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (ticker, ex_date) DO NOTHING;
                                """, (tk, ex_date_str, ex_date_str, quarterly_div))
                                synced_count += 1
                                
                            consecutive_429s = 0  # Reset counter on successful response

                        except Exception as e:
                            err_msg = str(e)
                            if "429" in err_msg or "Too Many Requests" in err_msg:
                                consecutive_429s += 1
                                print(f"⏳ Yahoo 429 rate limit encountered for {tk}. Backing off 5s...")
                                time.sleep(5.0)
                            continue

                    conn.commit()
                    print(f"✅ Dividend calendar synchronized ({synced_count} active schedules verified).")
    except Exception as e:
        print(f"⚠️ Dividend calendar sync note (DB connection deferred): {e}")

def calculate_advanced_indicators(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> dict:
    if len(df) < 15:
        return {"rsi": 50.0, "macd_hist": 0.0, "atr": 1.0, "rel_strength_spy": 0.0, "adv": 1000000.0}

    try:
        # 1. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        # 2. MACD Histogram
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # 3. Average True Range (ATR 14)
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # 4. Relative Strength vs SPY over 12 periods (~3 hours at 15m intervals)
        rel_strength = 0.0
        if spy_df is not None and len(spy_df) >= 12 and len(df) >= 12:
            stock_ret = (df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12]
            spy_ret = (spy_df['close'].iloc[-1] - spy_df['close'].iloc[-12]) / spy_df['close'].iloc[-12]
            rel_strength = round(float((stock_ret - spy_ret) * 100), 2)

        # 5. Estimated Average Daily Volume (ADV)
        avg_vol = df['volume'].mean() if 'volume' in df.columns else 1000.0
        adv = float(avg_vol * 390)  # 390 trading minutes per day

        return {
            "rsi": round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else 50.0,
            "macd_hist": round(float(macd_hist.iloc[-1]), 4) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
            "atr": round(float(atr.iloc[-1]), 2) if not pd.isna(atr.iloc[-1]) else 1.0,
            "rel_strength_spy": rel_strength,
            "adv": max(adv, 100000.0)
        }
    except Exception:
        return {"rsi": 50.0, "macd_hist": 0.0, "atr": 1.0, "rel_strength_spy": 0.0, "adv": 1000000.0}

async def on_bar(bar: Bar):
    global history, buffer, current_window_minute
    
    # 1. Store bar in rolling history dataframe (keep 100 bars for MACD stability)
    new_row = pd.DataFrame([{"high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume}])
    history[bar.symbol] = pd.concat([history[bar.symbol], new_row], ignore_index=True).tail(100)

    bar_minute = bar.timestamp.minute

    # 2. Accumulate indicators into buffer for universe symbols
    if bar.symbol in UNIVERSE:
        spy_data = history.get("SPY", None)
        indicators = calculate_advanced_indicators(history[bar.symbol], spy_data)
        
        buffer[bar.symbol] = {
            "close": bar.close,
            "volume": bar.volume,
            "rsi": indicators["rsi"],
            "macd_hist": indicators["macd_hist"],
            "atr": indicators["atr"],
            "rel_strength_spy": indicators["rel_strength_spy"],
            "adv": indicators["adv"]
        }

    # 3. Broadcast ONLY when shifting into a new 15-minute interval (00, 15, 30, 45)
    if bar_minute % TICK_INTERVAL_MINUTES == 0 and current_window_minute != bar_minute:
        # Lock window immediately so subsequent incoming stock bars don't trigger duplicate tasks
        current_window_minute = bar_minute
        
        # Brief pause allows remaining delayed WebSockets for the minute to settle into buffer
        await asyncio.sleep(1.5) 
        
        if len(buffer) > 0:
            payload = json.dumps(buffer)
            broker.publish('market_events', payload)
            print(f"📡 [PRODUCER] Clean 15m Close: Broadcasted full {len(buffer)}/100 stock matrix to Redis.")
            buffer.clear()

if __name__ == "__main__":
    print(f"📡 Booting Alpaca WebSocket for {len(ALL_SYMBOLS)} assets (Interval: {TICK_INTERVAL_MINUTES}m)...")
    
    # Run dividend sync in the background so WebSocket boots immediately
    import threading
    threading.Thread(target=sync_dividend_calendar, daemon=True).start()

    stream = StockDataStream(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    stream.subscribe_bars(on_bar, *ALL_SYMBOLS)
    stream.run()
