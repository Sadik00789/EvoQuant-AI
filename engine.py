import os
import re
import json
import logging
import asyncio
import httpx
import pandas as pd
import requests
from typing import Literal, Dict, Union, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SwarmEngine")

# Helper utility to strip markdown fences from LLM responses
def clean_llm_json_string(raw_content: str) -> str:
    """
    Strips conversational preambles/postambles and markdown fences 
    to extract the exact raw JSON payload.
    """
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
    if match:
        raw_content = match.group(1).strip()
    
    start_idx = raw_content.find('{')
    end_idx = raw_content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return raw_content[start_idx:end_idx + 1].strip()
    
    return raw_content.strip()

# ==========================================
# 1. EXECUTION BRIDGE & PYDANTIC SCHEMAS
# ==========================================

class AlpacaExecutionBridge:
    """Direct REST API Execution Bridge for Alpaca Markets Paper/Live Trading."""
    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "Content-Type": "application/json"
        }

    def is_active(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def submit_market_order(self, symbol: str, qty: float, action: str) -> Optional[Dict[str, Any]]:
        if not self.is_active() or qty <= 0:
            return None

        side_map = {
            "BUY": "buy",
            "COVER": "buy",
            "SELL": "sell",
            "SHORT": "sell"
        }
        side = side_map.get(action.upper(), action.lower())

        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol.upper(),
            "qty": str(round(qty, 4)),
            "side": side,
            "type": "market",
            "time_in_force": "gtc"
        }
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=10.0)
            resp.raise_for_status()
            order_data = resp.json()
            logger.info(f"⚡ [ALPACA BROKER EXECUTED] {action.upper()} {qty:.2f} {symbol} | Order ID: {order_data.get('id')}")
            return order_data
        except Exception as e:
            logger.error(f"❌ [ALPACA BROKER ERROR] Failed to submit order for {symbol} ({action}): {e}")
            return None

class AssetThesis(BaseModel):
    ticker: str
    market_regime: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    primary_driver: str
    atr: float = 1.0

class MultiTechnicalThesis(BaseModel):
    analyses: Dict[str, AssetThesis]
    correlation_risk: str

# --- Adversarial Debate Schemas ---
class DebateArgument(BaseModel):
    ticker: str
    thesis_summary: str
    key_factors: List[str]
    strength_score: float = Field(..., ge=0.0, le=1.0)

class AgentDebateCase(BaseModel):
    arguments: Dict[str, DebateArgument]
    overall_perspective: str = Field(default="Balanced market perspective across selected assets.")

class QualitativeSignal(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "SHORT", "COVER", "HOLD"]
    conviction: float = Field(..., ge=0.0, le=1.0)

class AgentSignalDecision(BaseModel):
    signals: Dict[str, QualitativeSignal]
    macro_reasoning: str = Field(default="Synthesized multi-asset debate logic.")

class PortfolioAllocation(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "SHORT", "COVER", "HOLD"]
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
    using Inverse-Volatility Risk Parity. Enforces a strict 5% max position cap.
    """
    def __init__(self, max_position_cap: float = 0.05):
        self.max_position_cap = max_position_cap

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

    @staticmethod
    def optimize_allocations(
        signals: AgentSignalDecision, 
        theses: Union[dict, MultiTechnicalThesis], 
        max_position_cap: float = 0.05
    ) -> CrossAssetRiskDecision:
        raw_decisions = {}
        active_candidates = {}

        for ticker, sig in signals.signals.items():
            if sig.action in ("BUY", "SHORT") and sig.conviction > 0.3:
                atr = 1.0
                if isinstance(theses, dict) and ticker in theses:
                    item = theses[ticker]
                    atr = item.get("atr", 1.0) if isinstance(item, dict) else getattr(item, "atr", 1.0)
                elif hasattr(theses, "analyses") and ticker in theses.analyses:
                    atr = theses.analyses[ticker].atr

                if atr <= 0 or atr is None:
                    atr = 1.0

                risk_score = sig.conviction / max(atr, 0.1)
                active_candidates[ticker] = (sig.action, risk_score)
            else:
                raw_decisions[ticker] = PortfolioAllocation(ticker=ticker, action=sig.action, allocation_pct=0.0)

        total_risk_score = sum(score for _, score in active_candidates.values())
        if total_risk_score > 0:
            for ticker, (action, score) in active_candidates.items():
                unclamped_weight = score / total_risk_score
                clamped_weight = round(float(min(max_position_cap, unclamped_weight)), 4)
                raw_decisions[ticker] = PortfolioAllocation(ticker=ticker, action=action, allocation_pct=clamped_weight)

        return CrossAssetRiskDecision(decisions=raw_decisions, macro_reasoning=signals.macro_reasoning)

# ==========================================
# 3. GEMMA 4-31B SWARM ENGINE WITH DEBATE LOOP
# ==========================================

class DualModelTradingSwarm:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def analyze_technical_state(self, market_state: dict, active_holdings: list = None) -> dict:
        filtered_thesis = {}
        active_holdings = active_holdings or []

        ranked_stocks = sorted(
            market_state.items(),
            key=lambda x: abs(x[1].get("rel_strength_spy", 0.0)) + abs(x[1].get("rsi", 50.0) - 50.0),
            reverse=True
        )

        top_candidates = [tk for tk, _ in ranked_stocks[:12]]
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

    async def _call_gemma_provider(
        self,
        client: httpx.AsyncClient,
        sys_prompt: str,
        user_input: str,
        model_class: Any,
        role_tag: str = "LLM"
    ) -> Any:
        """Executes LLM calls via Google AI Studio using Gemma 4-31B with built-in rate-limit exception handling & backoff."""
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Gemini API key is not set in environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemma-4-31b-it",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        max_retries = 3
        backoff_factor = 2.0

        for attempt in range(max_retries):
            try:
                # 60.0 second timeout to handle heavy Gemma 4-31B structured JSON inference
                resp = await client.post(url, json=payload, headers=headers, timeout=60.0)

                if resp.status_code == 429:
                    sleep_time = backoff_factor ** (attempt + 1)
                    logger.warning(f"⚠️ [Rate Limit / 429] encountered for [{role_tag}] on Gemma 4 31B. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(sleep_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw_content = data['choices'][0]['message']['content']
                cleaned_content = clean_llm_json_string(raw_content)

                validated_output = model_class.model_validate_json(cleaned_content)
                logger.info(f"✅ [{role_tag}] Evaluated successfully via [Google AI Studio - Gemma 4 31B]")
                return validated_output

            except httpx.HTTPStatusError as hse:
                logger.warning(f"⚠️ HTTP status error {hse.response.status_code} for [{role_tag}]: {hse.response.text}")
                if hse.response.status_code in (400, 401, 403, 404):
                    raise
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(backoff_factor ** (attempt + 1))
            except ValidationError as ve:
                logger.warning(f"⚠️ Validation error on [{role_tag}]: {ve}. Retrying...")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                err_msg = str(e) if str(e) else type(e).__name__
                logger.warning(f"⚠️ [{role_tag}] Provider failed: {err_msg}. Retrying in {backoff_factor ** (attempt + 1)}s...")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(backoff_factor ** (attempt + 1))

        raise RuntimeError(f"Gemma 4 31B failed or rate-limited for [{role_tag}] after {max_retries} attempts.")

    async def _generate_bull_case(
        self, client: httpx.AsyncClient, strategy_input: str, persona: str
    ) -> AgentDebateCase:
        sys_prompt = (
            f"{persona}\n"
            "ROLE: Bull Researcher Agent.\n"
            "TASK: Build the strongest possible BULLISH thesis for each asset. Focus on asymmetric upside, "
            "MACD bullish momentum, oversold RSI bounces, key support levels, and positive relative strength.\n"
            "MUST INCLUDE BOTH `arguments` and `overall_perspective` in JSON output.\n"
            'Format MUST be JSON matching schema: {"arguments": {"TICKER": {"ticker": "TICKER", "thesis_summary": "summary", "key_factors": ["f1"], "strength_score": 0.85}}, "overall_perspective": "perspective"}'
        )
        return await self._call_gemma_provider(client, sys_prompt, strategy_input, AgentDebateCase, "BullResearcher")

    async def _generate_bear_case(
        self, client: httpx.AsyncClient, strategy_input: str, persona: str
    ) -> AgentDebateCase:
        sys_prompt = (
            f"{persona}\n"
            "ROLE: Bear Researcher Agent (Adversary).\n"
            "TASK: Hunt for flaws, Bull Traps, weak breakout volume, RSI bearish divergence, overhead resistance, "
            "and broader macro tail-risk for each asset. Build an aggressive BEARISH counter-case.\n"
            "MUST INCLUDE BOTH `arguments` and `overall_perspective` in JSON output.\n"
            'Format MUST be JSON matching schema: {"arguments": {"TICKER": {"ticker": "TICKER", "thesis_summary": "summary", "key_factors": ["f1"], "strength_score": 0.75}}, "overall_perspective": "perspective"}'
        )
        return await self._call_gemma_provider(client, sys_prompt, strategy_input, AgentDebateCase, "BearResearcher")

    async def execute_agent_strategy(
        self, 
        client: httpx.AsyncClient, 
        thesis: Union[dict, MultiTechnicalThesis], 
        portfolio_state: dict, 
        custom_persona: str
    ) -> CrossAssetRiskDecision:
        """
        Executes Adversarial Debate Loop:
        1. Runs Bull & Bear Research agents sequentially to prevent provider throttling.
        2. Passes both cases to Synthesizer/Judge agent to filter out confirmation bias.
        3. Optimizes allocations using Inverse-Volatility Risk Parity.
        """
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

        # --- STEP 1: SEQUENTIAL ADVERSARIAL DEBATE GENERATION ---
        try:
            bull_res = await self._generate_bull_case(client, strategy_input, custom_persona)
            bull_case_data = bull_res.model_dump_json()
        except Exception as e:
            logger.warning(f"⚠️ Bull Researcher failed: {e}. Falling back to default bullish case.")
            bull_case_data = "Bullish case unavailable due to rate limit or error."

        await asyncio.sleep(1.0)  # Pacing pause between LLM calls

        try:
            bear_res = await self._generate_bear_case(client, strategy_input, custom_persona)
            bear_case_data = bear_res.model_dump_json()
        except Exception as e:
            logger.warning(f"⚠️ Bear Researcher failed: {e}. Falling back to default bearish case.")
            bear_case_data = "Bearish case unavailable due to rate limit or error."

        await asyncio.sleep(1.0)  # Pacing pause between LLM calls

        # --- STEP 2: SYNTHESIZER / PORTFOLIO MANAGER JUDGMENT ---
        synth_input = json.dumps({
            "market_data_and_liquidity": strategy_input,
            "bull_researcher_case": bull_case_data,
            "bear_researcher_case": bear_case_data
        })

        synth_sys_prompt = (
            f"{custom_persona}\n"
            "ROLE: Chief Investment Officer / Impartial Judge.\n"
            "TASK: Evaluate the Bull Case and Bear Case side-by-side against market data. Detect bull traps, "
            "cross-examine arguments, weigh data points, and issue final trade actions with conviction scores (0.0 to 1.0).\n"
            "Allowed Actions: BUY (long), SELL (close long), SHORT (open short), COVER (close short), HOLD.\n"
            "DECISION RULES FOR SHORTING:\n"
            "- If Bear Case argument strength > Bull Case strength OR asset RSI > 68 OR relative strength vs SPY < -1.5, issue SHORT with conviction > 0.5.\n"
            "- If holding a SHORT position and stock rebounds significantly, issue COVER.\n"
            "MUST INCLUDE BOTH `signals` and `macro_reasoning` in JSON output.\n"
            'Format MUST be JSON showing both BUY and SHORT setups: '
            '{"signals": {"AAPL": {"ticker": "AAPL", "action": "BUY", "conviction": 0.8}, "NVDA": {"ticker": "NVDA", "action": "SHORT", "conviction": 0.85}}, "macro_reasoning": "Synthesized reasoning"}'
        )

        try:
            qualitative_signals = await self._call_gemma_provider(
                client, synth_sys_prompt, synth_input, AgentSignalDecision, "SynthesizerJudge"
            )
            return RiskParityOptimizer.optimize_allocations(qualitative_signals, thesis)
        except Exception as e:
            logger.error(f"❌ Synthesizer Judge failed: {e}. Executing hold-safe default.")
            return CrossAssetRiskDecision(decisions={}, macro_reasoning="Adversarial debate failed to reach consensus.")

    async def execute_swarm_strategies_concurrently(
        self,
        client: httpx.AsyncClient,
        shared_thesis: Dict[str, Any],
        population: List[Any],
        prices: Dict[str, float]
    ) -> Dict[str, CrossAssetRiskDecision]:
        """
        Evaluates agent strategies sequentially with rate-limit pacing to stay 
        strictly within Google AI Studio Free Tier limits (16k TPM / 30 RPM).
        """
        decisions_map = {}

        for agent in population:
            asset_val = sum(agent.holdings.get(tk, 0.0) * prices[tk] for tk in agent.holdings if tk in prices)
            current_equity = round(agent.cash + asset_val, 2)
            port_state = {
                "cash": round(agent.cash, 2),
                "holdings": {k: round(v, 4) for k, v in agent.holdings.items() if v != 0},
                "equity": current_equity
            }

            try:
                decision = await self.execute_agent_strategy(client, shared_thesis, port_state, agent.persona_prompt)
                decisions_map[agent.agent_id] = decision
            except Exception as e:
                logger.error(f"❌ Execution failed for [{agent.agent_id}]: {e}")
                decisions_map[agent.agent_id] = CrossAssetRiskDecision(decisions={}, macro_reasoning="Execution error")

            # 2.5-second pacing delay between agents to prevent TPM spikes
            await asyncio.sleep(2.5)

        return decisions_map

# ==========================================
# 4. FULL MULTI-AGENT POSTGRESQL / TIMESCALEDB LEDGER
# ==========================================

class CrossAssetPortfolioManager:
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
        max_position_cap: float = 0.05
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(port or os.getenv("POSTGRES_PORT", 5432))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "evoquant_db")
        self.user = user or os.getenv("POSTGRES_USER", "evoquant")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "evoquant_secret_pass")
        self.initial_capital = initial_capital
        self.max_position_cap = max_position_cap

        conninfo = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        
        self.pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_pool_size,
            max_size=max_pool_size,
            kwargs={"row_factory": dict_row}
        )
        self._init_db()

    def _init_db(self):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                except Exception as e:
                    logger.warning(f"TimescaleDB extension load note: {e}")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_accounts (
                        agent_id VARCHAR(64) PRIMARY KEY,
                        cash DOUBLE PRECISION NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

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

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS macro_regime (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        sentiment_score DOUBLE PRECISION,
                        risk_multiplier DOUBLE PRECISION,
                        summary_reasoning TEXT
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_snapshots (
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        agent_id VARCHAR(64) NOT NULL,
                        equity DOUBLE PRECISION NOT NULL,
                        cash DOUBLE PRECISION NOT NULL,
                        pnl_pct DOUBLE PRECISION NOT NULL
                    );
                """)

                try:
                    cur.execute("""
                        SELECT create_hypertable('agent_snapshots', 'timestamp', if_not_exists => TRUE);
                    """)
                    logger.info("✅ TimescaleDB Hypertable active for [agent_snapshots]")
                except Exception as e:
                    logger.warning(f"Hypertable notice: {e}")

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
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_accounts (agent_id, cash)
                    VALUES (%s, %s)
                    ON CONFLICT (agent_id) DO NOTHING;
                """, (agent_id, self.initial_capital))
                conn.commit()

    def get_agent_cash(self, agent_id: str) -> float:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT cash FROM agent_accounts WHERE agent_id = %s;", (agent_id,))
                row = cur.fetchone()
                return float(row['cash']) if row else self.initial_capital

    def update_agent_cash(self, agent_id: str, new_cash: float):
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
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, amount FROM agent_holdings 
                    WHERE agent_id = %s AND amount != 0;
                """, (agent_id,))
                rows = cur.fetchall()
                return {row['ticker']: float(row['amount']) for row in rows}

    def update_agent_holding(self, agent_id: str, ticker: str, amount: float, entry_price: float = 0.0):
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
                            action = "DIVIDEND_CREDIT"
                            cur.execute("UPDATE agent_accounts SET cash = cash + %s WHERE agent_id = %s;", (total_payout, agent_id))
                        else:
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
        query = "SELECT timestamp, agent_id, ticker, action, shares, amount_per_share, total_amount FROM dividend_logs"
        if agent_id:
            query += " WHERE agent_id = %s ORDER BY timestamp DESC LIMIT %s;"
            return self.fetch_dataframe(query, (agent_id, limit))
        query += " ORDER BY timestamp DESC LIMIT %s;"
        return self.fetch_dataframe(query, (limit,))

    def fetch_upcoming_dividends(self) -> pd.DataFrame:
        query = """
            SELECT ticker, ex_date, payment_date, amount_per_share 
            FROM dividend_schedule 
            WHERE ex_date >= CURRENT_DATE 
            ORDER BY ex_date ASC;
        """
        return self.fetch_dataframe(query)

    def log_trade(self, agent_id: str, ticker: str, action: str, shares: float, price: float, pct: float, reason: str = 'ALLOCATION'):
        clamped_pct = min(pct, self.max_position_cap)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_logs (agent_id, ticker, action, shares, price, allocation_pct, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (agent_id, ticker, action, shares, price, clamped_pct, reason))
                conn.commit()

    def log_snapshot(self, agent_id: str, equity: float, cash: float, pnl_pct: float):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_snapshots (agent_id, equity, cash, pnl_pct)
                    VALUES (%s, %s, %s, %s);
                """, (agent_id, equity, cash, pnl_pct))
                conn.commit()

    def log_macro_regime(self, sentiment_score: float, risk_multiplier: float, reasoning: str):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO macro_regime (sentiment_score, risk_multiplier, summary_reasoning)
                    VALUES (%s, %s, %s);
                """, (sentiment_score, risk_multiplier, reasoning))
                conn.commit()

    def fetch_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        with self.pool.connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def close(self):
        self.pool.close()

# Alias for backward compatibility across modules
PostgresPortfolioManager = CrossAssetPortfolioManager
