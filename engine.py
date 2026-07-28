import os
import json
import logging
import asyncio
import httpx
import pandas as pd
from typing import Literal, Dict, Union, Any
from pydantic import BaseModel, Field, ValidationError
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SwarmEngine")

# ==========================================
# 1. PYDANTIC DATA SCHEMAS
# ==========================================

class AssetThesis(BaseModel):
    ticker: str
    market_regime: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    primary_driver: str
    atr: float = 1.0

class MultiTechnicalThesis(BaseModel):
    analyses: Dict[str, AssetThesis]
    correlation_risk: str

class QualitativeSignal(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    conviction: float = Field(..., ge=0.0, le=1.0)

class AgentSignalDecision(BaseModel):
    signals: Dict[str, QualitativeSignal]
    macro_reasoning: str

class PortfolioAllocation(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    allocation_pct: float

class CrossAssetRiskDecision(BaseModel):
    decisions: Dict[str, PortfolioAllocation]
    macro_reasoning: str

# ==========================================
# 2. CONVEX RISK PARITY OPTIMIZER
# ==========================================

class RiskParityOptimizer:
    """
    Mathematical Portfolio Construction.
    Converts qualitative LLM conviction scores into volatility-adjusted weights
    using Inverse-Volatility Risk Parity.
    """
    def __init__(self, max_position_cap: float = 0.15):
        self.max_position_cap = max_position_cap

    # 1. Method for backtest.py and unit testing (Dictionary Schema)
    def optimize(self, convictions: dict, atrs: dict) -> dict:
        if not convictions:
            return {}

        valid_trades = {k: v for k, v in convictions.items() if v > 0.3}
        if not valid_trades:
            return {}

        raw_weights = {}
        for tk, conv in valid_trades.items():
            atr = atrs.get(tk, 1.0)
            if atr <= 0 or atr is None:
                atr = 1.0
            raw_weights[tk] = conv / max(atr, 0.1)

        total_raw = sum(raw_weights.values())
        if total_raw == 0:
            return {}

        return {
            k: round(float(min(self.max_position_cap, v / total_raw)), 4)
            for k, v in raw_weights.items()
        }

    # 2. Dual-Compatible Method for Live Swarm Consumer
    @staticmethod
    def optimize_allocations(
        signals: AgentSignalDecision, 
        theses: Union[dict, MultiTechnicalThesis], 
        max_position_cap: float = 0.15
    ) -> CrossAssetRiskDecision:
        raw_decisions = {}
        buy_candidates = {}

        for ticker, sig in signals.signals.items():
            if sig.action == "BUY" and sig.conviction > 0.3:
                # Safely extract ATR whether theses is a raw dict or Pydantic object
                atr = 1.0
                if isinstance(theses, dict) and ticker in theses:
                    item = theses[ticker]
                    atr = item.get("atr", 1.0) if isinstance(item, dict) else getattr(item, "atr", 1.0)
                elif hasattr(theses, "analyses") and ticker in theses.analyses:
                    atr = theses.analyses[ticker].atr

                if atr <= 0 or atr is None:
                    atr = 1.0

                # Risk Parity Score: Higher conviction + lower volatility = higher weight
                risk_score = sig.conviction / max(atr, 0.1)
                buy_candidates[ticker] = risk_score
            else:
                raw_decisions[ticker] = PortfolioAllocation(ticker=ticker, action=sig.action, allocation_pct=0.0)

        # Normalize weights across buy candidates
        total_risk_score = sum(buy_candidates.values())
        if total_risk_score > 0:
            for ticker, score in buy_candidates.items():
                unclamped_weight = score / total_risk_score
                clamped_weight = round(float(min(max_position_cap, unclamped_weight)), 4)
                raw_decisions[ticker] = PortfolioAllocation(ticker=ticker, action="BUY", allocation_pct=clamped_weight)

        return CrossAssetRiskDecision(decisions=raw_decisions, macro_reasoning=signals.macro_reasoning)

# ==========================================
# 3. HYBRID AI SWARM ENGINE
# ==========================================

class DualModelTradingSwarm:
    def __init__(self, api_key: str = None):
        self.primary_api_key = api_key or os.getenv("GROQ_API_KEY")

    def analyze_technical_state(self, market_state: dict, active_holdings: list = None) -> dict:
        filtered_thesis = {}
        active_holdings = active_holdings or []

        # 1. Rank universe by momentum/divergence magnitude
        ranked_stocks = sorted(
            market_state.items(),
            key=lambda x: abs(x[1].get("rel_strength_spy", 0.0)) + abs(x[1].get("rsi", 50.0) - 50.0),
            reverse=True
        )

        # 2. Capture Top 10 market movers
        top_candidates = [tk for tk, _ in ranked_stocks[:10]]

        # 3. Always include currently held positions so agents can rebalance or exit
        combined_tickers = set(top_candidates + active_holdings)

        for ticker in combined_tickers:
            if ticker in market_state:
                metrics = market_state[ticker]
                filtered_thesis[ticker] = {
                    "price": metrics.get("close"),
                    "rsi": metrics.get("rsi"),
                    "macd_hist": metrics.get("macd_hist"),
                    "rel_strength": metrics.get("rel_strength_spy"),
                    "atr": metrics.get("atr")
                }

        return filtered_thesis

    async def execute_agent_strategy(
        self, 
        client: httpx.AsyncClient, 
        thesis: Union[dict, MultiTechnicalThesis], 
        portfolio_state: dict, 
        custom_persona: str
    ) -> CrossAssetRiskDecision:
        """Executes 70B strategy across a multi-provider free 70B fallback chain."""
        
        # Dual-compatible thesis parsing
        if isinstance(thesis, dict):
            compact_theses = {
                tk: f"P:{data.get('price')}|RSI:{data.get('rsi')}|MACD:{data.get('macd_hist')}|RS:{data.get('rel_strength')}|ATR:{data.get('atr')}"
                for tk, data in thesis.items()
            }
        elif hasattr(thesis, "analyses"):
            compact_theses = {
                tk: f"{a.market_regime}|Conf:{a.confidence_score}|ATR:{a.atr}"
                for tk, a in thesis.analyses.items()
            }
        else:
            compact_theses = {}

        strategy_input = json.dumps({"theses": compact_theses, "liquidity": portfolio_state})
        
        sys_prompt = (
            f"{custom_persona}\n"
            "Analyze theses and return trade actions (BUY/SELL/HOLD) with conviction (0.0 to 1.0).\n"
            'Format MUST be JSON: {"signals": {"TICKER": {"ticker": "TICKER", "action": "BUY", "conviction": 0.8}}, "macro_reasoning": "reason"}'
        )
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": strategy_input}
        ]

        # Priority Chain of Free 70B/72B Providers
        providers = [
            {
                "name": "Groq",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "key": self.primary_api_key or os.getenv("GROQ_API_KEY"),
                "model": "llama-3.3-70b-versatile"
            },
            {
                "name": "OpenRouter (70B)",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": os.getenv("OPENROUTER_API_KEY"),
                "model": "meta-llama/llama-3.3-70b-instruct",
                "headers": {
                    "HTTP-Referer": "https://github.com/EvoQuant-AI",
                    "X-Title": "EvoQuant Trading Swarm"
                }
            },
            {
                "name": "OpenRouter (70B Free Tag)",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": os.getenv("OPENROUTER_API_KEY"),
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "headers": {
                    "HTTP-Referer": "https://github.com/EvoQuant-AI",
                    "X-Title": "EvoQuant Trading Swarm"
                }
            },
            {
                "name": "GitHub Models",
                "url": "https://models.inference.ai.azure.com/chat/completions",
                "key": os.getenv("GITHUB_TOKEN"),
                "model": "Llama-3.3-70B-Instruct"
            },
            {
                "name": "SambaNova",
                "url": "https://api.sambanova.ai/v1/chat/completions",
                "key": os.getenv("SAMBANOVA_API_KEY"),
                "model": "Meta-Llama-3.3-70B-Instruct"
            }
        ]

        for p in providers:
            if not p["key"]:
                continue

            # Build default headers and merge custom provider headers
            headers = {
                "Authorization": f"Bearer {p['key']}",
                "Content-Type": "application/json"
            }
            if "headers" in p:
                headers.update(p["headers"])

            payload = {
                "model": p["model"],
                "messages": list(base_messages),
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }

            try:
                resp = await client.post(p["url"], json=payload, headers=headers, timeout=12.0)
                
                # Handle Rate Limit
                if resp.status_code == 429:
                    logger.warning(f"⚠️ [{p['name']}] Rate limit hit (429). Routing to next free provider...")
                    continue

                # Handle OpenRouter 404 Endpoint Exhaustion cleanly
                if resp.status_code == 404:
                    err_msg = resp.json().get("error", {}).get("message", "Upstream capacity busy")
                    logger.warning(f"⚠️ [{p['name']}] 404 ({err_msg}). Routing to next free provider...")
                    continue

                resp.raise_for_status()
                qualitative_signals = AgentSignalDecision.model_validate_json(
                    resp.json()['choices'][0]['message']['content']
                )
                
                logger.info(f"✅ Strategy evaluated successfully via [{p['name']}]")
                return RiskParityOptimizer.optimize_allocations(qualitative_signals, thesis)

            except ValidationError:
                logger.warning(f"Validation error on [{p['name']}]. Trying next provider...")
                continue
            except Exception as e:
                logger.warning(f"⚠️ [{p['name']}] Provider failed: {e}. Trying next provider...")
                continue

        raise RuntimeError("All free 70B providers failed or rate-limited.")

# ==========================================
# 4. FULL MULTI-AGENT POSTGRESQL / TIMESCALEDB LEDGER
# ==========================================

class CrossAssetPortfolioManager:
    """
    Production-Grade TimescaleDB / PostgreSQL Ledger Manager.
    Uses thread-safe connection pooling (Psycopg 3) and TimescaleDB Hypertables
    for scalable time-series telemetry storage.
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
        initial_capital: float = 100000.0,
        db_path: str = None  # Backward-compatibility parameter ignored
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

    def _get_connection(self):
        """Backward-compatible connection accessor context manager."""
        return self.pool.connection()

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

                # 3. Holdings Table
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

                # 5. Macro Sentiment Regime Log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS macro_regime (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        sentiment_score DOUBLE PRECISION,
                        risk_multiplier DOUBLE PRECISION,
                        summary_reasoning TEXT
                    );
                """)

                # 6. Snapshots Table (Time-Series Data)
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

                # Create analytical index on snapshots
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_agent_time 
                    ON agent_snapshots (agent_id, timestamp DESC);
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
        """Fetches active non-zero holdings for an agent."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, amount FROM agent_holdings 
                    WHERE agent_id = %s AND amount > 0;
                """, (agent_id,))
                rows = cur.fetchall()
                return {row['ticker']: float(row['amount']) for row in rows}

    def update_agent_holding(self, agent_id: str, ticker: str, amount: float, entry_price: float = 0.0):
        """Atomic UPSERT for stock holdings."""
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