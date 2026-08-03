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

def clean_llm_json_string(raw_content: str) -> str:
    """
    Strips conversational preambles, Gemma thinking tags (<thought>...</thought>),
    and markdown fences to extract the exact raw JSON payload.
    """
    raw_content = re.sub(r"<thought>[\s\S]*?</thought>", "", raw_content).strip()

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

    def get_physical_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Queries Alpaca's actual physical position for a ticker to prevent order collisions."""
        if not self.is_active():
            return None
        url = f"{self.base_url}/v2/positions/{symbol.upper()}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def submit_market_order(self, symbol: str, qty: float, action: str) -> Optional[Dict[str, Any]]:
        if not self.is_active() or qty <= 0:
            return None

        action_upper = action.upper()

        # Pre-flight physical position check to prevent 403/422 collisions on shared account
        pos = self.get_physical_position(symbol)
        physical_qty = float(pos.get("qty", 0.0)) if pos else 0.0

        if action_upper == "SELL" and physical_qty <= 0:
            logger.warning(f"⚠️ [BROKER SKIPPED] {action_upper} {symbol}: Physical account holds no long position.")
            return None
        elif action_upper == "COVER" and physical_qty >= 0:
            logger.warning(f"⚠️ [BROKER SKIPPED] {action_upper} {symbol}: Physical account holds no short position.")
            return None

        # Quantize whole integer shares for short-side actions
        if action_upper in ("SHORT", "COVER"):
            rounded_qty = int(qty)
        else:
            rounded_qty = round(qty, 4)

        if rounded_qty <= 0:
            logger.warning(f"⚠️ [BROKER SKIPPED] {action_upper} {symbol}: Quantity rounded down to 0.")
            return None

        side_map = {
            "BUY": "buy",
            "COVER": "buy",
            "SELL": "sell",
            "SHORT": "sell"
        }
        side = side_map.get(action_upper, action_upper.lower())

        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol.upper(),
            "qty": str(rounded_qty),
            "side": side,
            "type": "market",
            "time_in_force": "day"
        }
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=10.0)
            resp.raise_for_status()
            order_data = resp.json()
            logger.info(f"⚡ [ALPACA BROKER EXECUTED] {action_upper} {rounded_qty} {symbol} | Order ID: {order_data.get('id')}")
            return order_data
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response is not None else "Unknown"
            logger.warning(f"⚠️ [ALPACA BROKER REJECTED] {action_upper} {rounded_qty} {symbol} (HTTP {status_code}): Shared account inventory lock or position constraint.")
            return None
        except Exception as e:
            logger.error(f"❌ [ALPACA BROKER ERROR] Failed to submit order for {symbol} ({action_upper}): {e}")
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

        # Order-preserving deduplication: prioritizes active holdings first so open positions are never dropped
        combined_tickers = list(dict.fromkeys(active_holdings + top_candidates))[:12]

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
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }

        max_retries = 3
        backoff_factor = 2.0

        for attempt in range(max_retries):
            try:
                resp = await client.post(url, json=payload, headers=headers, timeout=120.0)

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
            "TASK: Build a concise BULLISH thesis for each asset in the input payload.\n"
            "INSTRUCTIONS:\n"
            "- Keep `thesis_summary` to 1 short sentence per ticker.\n"
            "- Limit `key_factors` to maximum 2 brief bullet strings.\n"
            "- Do NOT use literal 'TICKER'. Use exact stock symbols from input (e.g., 'AAPL', 'NVDA').\n"
            'Example output: {"arguments": {"AAPL": {"ticker": "AAPL", "thesis_summary": "Bullish momentum above 200 SMA", "key_factors": ["MACD golden cross", "Strong RS"], "strength_score": 0.85}}, "overall_perspective": "Positive market outlook"}'
        )

        input_data = json.loads(strategy_input)
        theses = input_data.get("theses", {})
        ticker_items = list(theses.items())
        batch_size = 6

        merged_arguments = {}
        overall_perspectives = []

        for i in range(0, len(ticker_items), batch_size):
            chunk_theses = dict(ticker_items[i:i + batch_size])
            chunk_input = json.dumps({"theses": chunk_theses})

            res: AgentDebateCase = await self._call_gemma_provider(
                client, sys_prompt, chunk_input, AgentDebateCase, f"BullResearcher_Batch_{i//batch_size + 1}"
            )
            merged_arguments.update(res.arguments)
            if res.overall_perspective:
                overall_perspectives.append(res.overall_perspective)

            if i + batch_size < len(ticker_items):
                await asyncio.sleep(1.0)

        combined_perspective = " | ".join(overall_perspectives) if overall_perspectives else "Constructive stance across candidate pool."
        return AgentDebateCase(arguments=merged_arguments, overall_perspective=combined_perspective)

    async def _generate_bear_case(
        self, client: httpx.AsyncClient, strategy_input: str, persona: str
    ) -> AgentDebateCase:
        sys_prompt = (
            f"{persona}\n"
            "ROLE: Bear Researcher Agent (Adversary).\n"
            "TASK: Build a concise BEARISH thesis for each asset in the input payload.\n"
            "INSTRUCTIONS:\n"
            "- Keep `thesis_summary` to 1 short sentence per ticker.\n"
            "- Limit `key_factors` to maximum 2 brief bullet strings.\n"
            "- Do NOT use literal 'TICKER'. Use exact stock symbols from input (e.g., 'AAPL', 'NVDA').\n"
            'Example output: {"arguments": {"NVDA": {"ticker": "NVDA", "thesis_summary": "RSI overbought divergence near resistance", "key_factors": ["RSI > 70", "Volume fade"], "strength_score": 0.75}}, "overall_perspective": "Cautious posture recommended"}'
        )

        input_data = json.loads(strategy_input)
        theses = input_data.get("theses", {})
        ticker_items = list(theses.items())
        batch_size = 6

        merged_arguments = {}
        overall_perspectives = []

        for i in range(0, len(ticker_items), batch_size):
            chunk_theses = dict(ticker_items[i:i + batch_size])
            chunk_input = json.dumps({"theses": chunk_theses})

            res: AgentDebateCase = await self._call_gemma_provider(
                client, sys_prompt, chunk_input, AgentDebateCase, f"BearResearcher_Batch_{i//batch_size + 1}"
            )
            merged_arguments.update(res.arguments)
            if res.overall_perspective:
                overall_perspectives.append(res.overall_perspective)

            if i + batch_size < len(ticker_items):
                await asyncio.sleep(1.0)

        combined_perspective = " | ".join(overall_perspectives) if overall_perspectives else "Defensive stance across candidate pool."
        return AgentDebateCase(arguments=merged_arguments, overall_perspective=combined_perspective)

    async def execute_swarm_strategies_concurrently(
        self,
        client: httpx.AsyncClient,
        shared_thesis: Dict[str, Any],
        population: List[Any],
        prices: Dict[str, float]
    ) -> Dict[str, CrossAssetRiskDecision]:
        decisions_map = {}

        compact_theses = {
            tk: f"P:{data.get('price')}|RSI:{data.get('rsi')}|MACD:{data.get('macd_hist')}|RS:{data.get('rel_strength')}|ATR:{data.get('atr')}"
            for tk, data in shared_thesis.items()
        }
        strategy_input = json.dumps({"theses": compact_theses})

        logger.info("🧠 [Swarm Intelligence] Generating Parallel Bull & Bear Market Debates...")

        # 1. Run Bull and Bear Research concurrently instead of sequentially
        bull_task = self._generate_bull_case(client, strategy_input, "Global Technical Analyst")
        bear_task = self._generate_bear_case(client, strategy_input, "Global Risk Analyst")

        bull_res, bear_res = await asyncio.gather(bull_task, bear_task, return_exceptions=True)

        bull_case_data = bull_res.model_dump_json() if isinstance(bull_res, AgentDebateCase) else "Bullish case unavailable."
        bear_case_data = bear_res.model_dump_json() if isinstance(bear_res, AgentDebateCase) else "Bearish case unavailable."

        # 2. Evaluate Agent Judges with a Semaphore (Concurrency = 2) to optimize throughput while respecting Gemini quotas
        semaphore = asyncio.Semaphore(2)

        async def evaluate_agent(agent):
            async with semaphore:
                asset_val = sum(agent.holdings.get(tk, 0.0) * prices[tk] for tk in agent.holdings if tk in prices)
                current_equity = round(agent.cash + asset_val, 2)
                port_state = {
                    "cash": round(agent.cash, 2),
                    "holdings": {k: round(v, 4) for k, v in agent.holdings.items() if v != 0},
                    "equity": current_equity
                }

                synth_input = json.dumps({
                    "market_data_and_liquidity": json.dumps({"theses": compact_theses, "liquidity": port_state}),
                    "bull_researcher_case": bull_case_data,
                    "bear_researcher_case": bear_case_data
                })

                synth_sys_prompt = (
                    f"{agent.persona_prompt}\n"
                    "ROLE: Chief Investment Officer / Impartial Judge.\n"
                    "TASK: Evaluate Bull Case and Bear Case against market data. Issue final trade actions with conviction scores (0.0 to 1.0).\n"
                    "Allowed Actions: BUY (long), SELL (close long), SHORT (open short), COVER (close short), HOLD.\n"
                    "CRITICAL: Replace ticker placeholders with actual stock symbols from input data (e.g., 'AAPL', 'NVDA'). Do NOT output literal 'TICKER'.\n"
                    'Example output: {"signals": {"AAPL": {"ticker": "AAPL", "action": "BUY", "conviction": 0.8}, "NVDA": {"ticker": "NVDA", "action": "SHORT", "conviction": 0.85}}, "macro_reasoning": "Balanced multi-asset thesis"}'
                )

                try:
                    qualitative_signals = await self._call_gemma_provider(
                        client, synth_sys_prompt, synth_input, AgentSignalDecision, f"SynthesizerJudge_{agent.agent_id}"
                    )
                    decision = RiskParityOptimizer.optimize_allocations(qualitative_signals, shared_thesis)
                except Exception as e:
                    logger.error(f"❌ Synthesizer Judge failed for [{agent.agent_id}]: {e}")
                    decision = CrossAssetRiskDecision(decisions={}, macro_reasoning="Execution error")

                await asyncio.sleep(0.5)
                return agent.agent_id, decision

        # Execute all agents concurrently
        judge_results = await asyncio.gather(*[evaluate_agent(agent) for agent in population])

        for agent_id, decision in judge_results:
            decisions_map[agent_id] = decision

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

    def cull_and_reallocate(
        self, 
        loser_agent_id: str, 
        recipient_agent_ids: List[str], 
        current_prices: Dict[str, float], 
        execution_bridge: Optional[Any] = None
    ):
        """
        True Darwinian Culling: Liquidates an underperforming agent's positions,
        calculates its remaining equity, distributes it equally among offspring,
        zeros out the loser agent, and logs updated snapshots.
        """
        if not recipient_agent_ids:
            logger.warning("⚠️ No recipient agents specified for capital reallocation.")
            return

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Fetch loser's current cash balance
                cur.execute("SELECT cash FROM agent_accounts WHERE agent_id = %s;", (loser_agent_id,))
                row = cur.fetchone()
                if not row:
                    logger.error(f"❌ Agent [{loser_agent_id}] not found in database.")
                    return
                loser_cash = float(row['cash'])

                # 2. Fetch loser's active holdings
                cur.execute("""
                    SELECT ticker, amount FROM agent_holdings 
                    WHERE agent_id = %s AND amount != 0;
                """, (loser_agent_id,))
                holdings = cur.fetchall()

                liquidated_value = 0.0

                # 3. Liquidate holdings into virtual cash & execute broker orders
                for item in holdings:
                    ticker = item['ticker']
                    amount = float(item['amount'])
                    price = current_prices.get(ticker, 0.0)

                    if amount != 0 and price > 0:
                        position_val = amount * price
                        liquidated_value += position_val

                        # Execute physical broker liquidation if active
                        if execution_bridge and execution_bridge.is_active():
                            action = "SELL" if amount > 0 else "COVER"
                            execution_bridge.submit_market_order(ticker, abs(amount), action)

                        # Log trade event
                        action_str = "LIQUIDATE_LONG" if amount > 0 else "LIQUIDATE_SHORT"
                        self.log_trade(loser_agent_id, ticker, action_str, abs(amount), price, 0.0, reason="CULLING")

                    # Zero out loser's holding record
                    cur.execute("""
                        UPDATE agent_holdings 
                        SET amount = 0.0, updated_at = CURRENT_TIMESTAMP 
                        WHERE agent_id = %s AND ticker = %s;
                    """, (loser_agent_id, ticker))

                total_recovered_equity = max(0.0, loser_cash + liquidated_value)
                logger.info(f"💀 [DARWINIAN CULLING] Liquidated [{loser_agent_id}]. Total Recovered Equity: ${total_recovered_equity:,.2f}")

                # 4. Zero out the loser's account cash & log $0.00 snapshot immediately
                cur.execute("""
                    UPDATE agent_accounts 
                    SET cash = 0.0, updated_at = CURRENT_TIMESTAMP 
                    WHERE agent_id = %s;
                """, (loser_agent_id,))

                cur.execute("""
                    INSERT INTO agent_snapshots (agent_id, equity, cash, pnl_pct)
                    VALUES (%s, 0.0, 0.0, -100.0);
                """, (loser_agent_id,))

                if total_recovered_equity <= 0:
                    logger.warning(f"⚠️ [{loser_agent_id}] has no positive equity to transfer.")
                    conn.commit()
                    return

                # 5. Distribute recovered equity directly to offspring recipients (overwriting any pre-existing or default balances)
                share_per_offspring = round(total_recovered_equity / len(recipient_agent_ids), 2)

                for recipient_id in recipient_agent_ids:
                    cur.execute("""
                        INSERT INTO agent_accounts (agent_id, cash, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (agent_id) 
                        DO UPDATE SET cash = EXCLUDED.cash, updated_at = CURRENT_TIMESTAMP;
                    """, (recipient_id, share_per_offspring))

                    cur.execute("""
                        INSERT INTO agent_snapshots (agent_id, equity, cash, pnl_pct)
                        VALUES (%s, %s, %s, 0.0);
                    """, (recipient_id, share_per_offspring, share_per_offspring))
                    
                    logger.info(f"🎁 [INHERITANCE] Assigned exact inherited cash ${share_per_offspring:,.2f} from [{loser_agent_id}] ➔ [{recipient_id}]")

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

PostgresPortfolioManager = CrossAssetPortfolioManager
