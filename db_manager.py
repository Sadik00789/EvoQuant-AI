import os
import logging
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PostgresManager")

class PostgresPortfolioManager:
    """
    Production-Grade TimescaleDB / PostgreSQL Ledger Manager.
    Uses thread-safe connection pooling (Psycopg 3) and TimescaleDB Hypertables
    for scalable time-series telemetry storage. Supports signed position accounting
    (Long/Short) and automated dividend schedule & audit tracking.
    """
    def __init__(
        self,
        host: str = None,
        port: int = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        initial_capital: float = 100000.0
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(port or os.getenv("POSTGRES_PORT", 5432))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "evoquant_db")
        self.user = user or os.getenv("POSTGRES_USER", "evoquant")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "evoquant_secret_pass")
        self.initial_capital = initial_capital

        conninfo = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        
        # Initialize thread-safe connection pool
        self.pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_pool_size,
            max_size=max_pool_size,
            kwargs={"row_factory": dict_row}
        )
        self._init_db()

    def _init_db(self):
        """Initializes tables, indexes, and converts snapshots to a TimescaleDB Hypertable."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Enable TimescaleDB extension
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                except Exception as e:
                    logger.warning(f"TimescaleDB extension load note: {e}")

                # 2. Accounts Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_accounts (
                        agent_id VARCHAR(64) PRIMARY KEY,
                        cash DOUBLE PRECISION NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 3. Holdings Table (Supports positive for Longs, negative for Shorts)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_holdings (
                        agent_id VARCHAR(64),
                        ticker VARCHAR(16),
                        amount DOUBLE PRECISION NOT NULL,
                        entry_price DOUBLE PRECISION DEFAULT 0.0,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (agent_id, ticker)
                    );
                """)

                # 4. Audit Trade Logs
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_logs (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        agent_id VARCHAR(64) NOT NULL,
                        ticker VARCHAR(16) NOT NULL,
                        action VARCHAR(8) NOT NULL,
                        shares DOUBLE PRECISION NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        allocation_pct DOUBLE PRECISION NOT NULL,
                        reason VARCHAR(64) DEFAULT 'ALLOCATION'
                    );
                """)

                # 5. Dividend Schedule Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dividend_schedule (
                        id BIGSERIAL PRIMARY KEY,
                        ticker VARCHAR(16) NOT NULL,
                        ex_date DATE NOT NULL,
                        payment_date DATE NOT NULL,
                        amount_per_share DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_ticker_ex_date UNIQUE (ticker, ex_date)
                    );
                """)

                # 6. Dividend Audit Logs Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dividend_logs (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        agent_id VARCHAR(64) NOT NULL,
                        ticker VARCHAR(16) NOT NULL,
                        action VARCHAR(16) NOT NULL,
                        shares DOUBLE PRECISION NOT NULL,
                        amount_per_share DOUBLE PRECISION NOT NULL,
                        total_amount DOUBLE PRECISION NOT NULL
                    );
                """)

                # 7. Macro Sentiment Regime Log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS macro_regime (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        sentiment_score DOUBLE PRECISION,
                        risk_multiplier DOUBLE PRECISION,
                        summary_reasoning TEXT
                    );
                """)

                # 8. Snapshots Table (Time-Series Data)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_snapshots (
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        agent_id VARCHAR(64) NOT NULL,
                        equity DOUBLE PRECISION NOT NULL,
                        cash DOUBLE PRECISION NOT NULL,
                        pnl_pct DOUBLE PRECISION NOT NULL
                    );
                """)

                # Convert snapshots to TimescaleDB Hypertable (chunked by time)
                try:
                    cur.execute("""
                        SELECT create_hypertable('agent_snapshots', 'timestamp', if_not_exists => TRUE);
                    """)
                    logger.info("✅ TimescaleDB Hypertable active for [agent_snapshots]")
                except Exception as e:
                    logger.warning(f"Hypertable notice (using standard PG table fallback): {e}")

                # Create analytical indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_agent_time 
                    ON agent_snapshots (agent_id, timestamp DESC);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_div_schedule_ex_date 
                    ON dividend_schedule (ex_date);
                """)
                
                conn.commit()

    def register_agent(self, agent_id: str):
        """Registers agent account if it doesn't already exist."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_accounts (agent_id, cash)
                    VALUES (%s, %s)
                    ON CONFLICT (agent_id) DO NOTHING;
                """, (agent_id, self.initial_capital))
                conn.commit()

    def get_agent_cash(self, agent_id: str) -> float:
        """Fetches current cash balance for an agent."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT cash FROM agent_accounts WHERE agent_id = %s;", (agent_id,))
                row = cur.fetchone()
                return float(row['cash']) if row else self.initial_capital

    def update_agent_cash(self, agent_id: str, new_cash: float):
        """Atomic UPSERT for agent cash."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_accounts (agent_id, cash, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (agent_id) 
                    DO UPDATE SET cash = EXCLUDED.cash, updated_at = CURRENT_TIMESTAMP;
                """, (agent_id, new_cash))
                conn.commit()

    def get_agent_holdings(self, agent_id: str) -> Dict[str, float]:
        """Fetches active non-zero holdings (Long > 0, Short < 0) for an agent."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, amount FROM agent_holdings 
                    WHERE agent_id = %s AND amount != 0;
                """, (agent_id,))
                rows = cur.fetchall()
                return {row['ticker']: float(row['amount']) for row in rows}

    def update_agent_holding(self, agent_id: str, ticker: str, amount: float, entry_price: float = 0.0):
        """Atomic UPSERT for stock holdings (handles positive long and negative short quantities)."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_holdings (agent_id, ticker, amount, entry_price, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (agent_id, ticker)
                    DO UPDATE SET 
                        amount = EXCLUDED.amount, 
                        entry_price = CASE WHEN EXCLUDED.entry_price > 0 THEN EXCLUDED.entry_price ELSE agent_holdings.entry_price END,
                        updated_at = CURRENT_TIMESTAMP;
                """, (agent_id, ticker, amount, entry_price))
                conn.commit()

    def process_daily_dividends(self, current_date_str: str):
        """
        Scans dividend_schedule for events matching current_date_str (Ex-Dividend Date).
        Credits long holders and debits short holders automatically.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, amount_per_share 
                    FROM dividend_schedule 
                    WHERE ex_date = %s;
                """, (current_date_str,))
                events = cur.fetchall()

                for event in events:
                    ticker = event['ticker']
                    div_per_share = float(event['amount_per_share'])

                    cur.execute("""
                        SELECT agent_id, amount FROM agent_holdings 
                        WHERE ticker = %s AND amount != 0;
                    """, (ticker,))
                    active_positions = cur.fetchall()

                    for pos in active_positions:
                        agent_id = pos['agent_id']
                        shares = float(pos['amount'])
                        total_payout = abs(shares) * div_per_share

                        if shares > 0:
                            # LONG POSITION: Credit Cash
                            action = "DIVIDEND_CREDIT"
                            cur.execute("UPDATE agent_accounts SET cash = cash + %s WHERE agent_id = %s;", (total_payout, agent_id))
                        else:
                            # SHORT POSITION: Debit Cash (Short Obligation)
                            action = "DIVIDEND_DEBIT"
                            total_payout = -total_payout
                            cur.execute("UPDATE agent_accounts SET cash = cash - %s WHERE agent_id = %s;", (abs(total_payout), agent_id))

                        cur.execute("""
                            INSERT INTO dividend_logs (agent_id, ticker, action, shares, amount_per_share, total_amount)
                            VALUES (%s, %s, %s, %s, %s, %s);
                        """, (agent_id, ticker, action, abs(shares), div_per_share, total_payout))

                        logger.info(f"💰 [{action}] {agent_id} | {ticker} | Shares: {shares:.2f} | Payout: ${total_payout:+.2f}")

                conn.commit()

    def fetch_dividend_logs(self, agent_id: str = None, limit: int = 50) -> pd.DataFrame:
        """Fetches recent dividend credits and debits for Streamlit UI analytics."""
        query = "SELECT timestamp, agent_id, ticker, action, shares, amount_per_share, total_amount FROM dividend_logs"
        if agent_id:
            query += " WHERE agent_id = %s ORDER BY timestamp DESC LIMIT %s;"
            return self.fetch_dataframe(query, (agent_id, limit))
        query += " ORDER BY timestamp DESC LIMIT %s;"
        return self.fetch_dataframe(query, (limit,))

    def fetch_upcoming_dividends(self) -> pd.DataFrame:
        """Fetches upcoming ex-dividend dates from dividend_schedule."""
        query = """
            SELECT ticker, ex_date, payment_date, amount_per_share 
            FROM dividend_schedule 
            WHERE ex_date >= CURRENT_DATE 
            ORDER BY ex_date ASC;
        """
        return self.fetch_dataframe(query)

    def log_trade(self, agent_id: str, ticker: str, action: str, shares: float, price: float, pct: float, reason: str = 'ALLOCATION'):
        """Logs trade execution event into audit ledger."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_logs (agent_id, ticker, action, shares, price, allocation_pct, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (agent_id, ticker, action, shares, price, pct, reason))
                conn.commit()

    def log_snapshot(self, agent_id: str, equity: float, cash: float, pnl_pct: float):
        """Inserts telemetry snapshot into TimescaleDB hypertable."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_snapshots (agent_id, equity, cash, pnl_pct)
                    VALUES (%s, %s, %s, %s);
                """, (agent_id, equity, cash, pnl_pct))
                conn.commit()

    def log_macro_regime(self, sentiment_score: float, risk_multiplier: float, reasoning: str):
        """Logs macro news sentiment and risk multiplier."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO macro_regime (sentiment_score, risk_multiplier, summary_reasoning)
                    VALUES (%s, %s, %s);
                """, (sentiment_score, risk_multiplier, reasoning))
                conn.commit()

    def fetch_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Executes query and returns results as a Pandas DataFrame."""
        with self.pool.connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def close(self):
        """Gracefully closes connection pool."""
        self.pool.close()
