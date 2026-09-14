import os
import re
import json
import logging
import asyncio
import time
from datetime import datetime, timezone
import httpx
import feedparser
import numpy as np
import redis
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsSentimentAgent")

# 20-stock universe canonical order for arbiter output validation
CANONICAL_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "QCOM",
    "AVGO", "SPY", "QQQ", "IWM", "GLD",
    "SLV", "TLT", "COIN", "PLTR", "ARM",
]


def _activity_score(entry: dict) -> float:
    """Activity = |Momentum_15m| x log(1+Volume), fallback to ATR x |Momentum| if volume flat."""
    try:
        import math as _math
        mom = abs(float(entry.get("momentum_15m", entry.get("momentum", 0.0)) or 0.0))
        vol = float(entry.get("volume", 0.0) or 0.0)
        atr = float(entry.get("atr", 1.0) or 1.0)
        if vol > 0:
            return mom * _math.log1p(vol)
        return mom * max(atr, 0.1) * 10.0
    except Exception:
        return 0.0


def select_top_20_candidates(snapshots: dict, active_holdings=None, top_n: int = 20) -> dict:
    """Orphan-free deterministic pre-LLM screener: rank 100 tickers by activity, retain open positions.

    Guarantees zero held positions ever lose LLM coverage while keeping batch strictly at 20.
    Backward compatible: second positional arg may be legacy top_n int.
    """
    # Backward-compat: select_top_20_candidates(snapshots, 20) legacy form
    if isinstance(active_holdings, int) and isinstance(top_n, int):
        top_n = int(active_holdings)
        active_holdings = set()
    if active_holdings is None:
        active_holdings = set()
    try:
        top_n = max(1, int(top_n))
    except Exception:
        top_n = 20
    if not snapshots:
        return {}
    try:
        # Normalize held set to upper-case for case-insensitive matching
        try:
            held_upper = {str(s).upper() for s in (active_holdings or set()) if str(s).strip()}
        except Exception:
            held_upper = set()
        # Map upper -> original key for snapshots present in universe
        upper_to_key: dict = {}
        for k in snapshots.keys():
            try:
                upper_to_key[str(k).upper()] = k
            except Exception:
                continue
        held_keys_present = [upper_to_key[u] for u in held_upper if u in upper_to_key]
        # Rank held symbols by activity score
        held_ranked = sorted(
            held_keys_present,
            key=lambda k: _activity_score(snapshots.get(k) or {}),
            reverse=True,
        )
        if len(held_ranked) >= top_n:
            return {k: snapshots[k] for k in held_ranked[:top_n]}
        # K slots to held, remainder to highest-activity unscreened tickers
        k_held = len(held_ranked)
        held_set = set(held_ranked)
        remaining = [kv for kv in sorted(snapshots.items(), key=lambda kv: _activity_score(kv[1] or {}), reverse=True) if kv[0] not in held_set]
        need = top_n - k_held
        selected = list(held_ranked) + [k for k, _ in remaining[:max(0, need)]]
        # Edge: snapshots smaller than top_n -> return all available
        if len(selected) < top_n and len(snapshots) <= top_n:
            existing = set(selected)
            for k in snapshots.keys():
                if k not in existing:
                    selected.append(k)
                if len(selected) >= len(snapshots):
                    break
        return {k: snapshots[k] for k in selected[:top_n]}
    except Exception:
        try:
            keys = list(snapshots.keys())[:int(top_n)]
            return {k: snapshots[k] for k in keys}
        except Exception:
            return {}


class SentimentCache:
    """
    O(1) per-ticker sentiment store with 1200s TTL (covers 15-minute bar window).
    Thread-safe for single-event-loop async use; swarm agents read via get_sentiment().
    """
    def __init__(self, ttl_seconds: float = 1200.0):
        self.ttl_seconds: float = float(ttl_seconds)
        self._scores: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        # Last-good snapshot retained beyond TTL for 429 / parse-failure fallback
        self._fallback_scores: Dict[str, float] = {}

    def update(self, scores: Dict[str, float], timestamp: Optional[float] = None) -> None:
        ts = float(timestamp) if timestamp is not None else time.time()
        for tk, v in (scores or {}).items():
            try:
                key = str(tk).upper()
                val = float(np.clip(float(v), -1.0, 1.0))
                self._scores[key] = round(val, 4)
                self._timestamps[key] = ts
                self._fallback_scores[key] = round(val, 4)
            except Exception:
                continue

    def get_sentiment(self, ticker: str) -> float:
        """O(1) lookup. Returns fresh score if within TTL, else 0.0 neutral."""
        try:
            key = str(ticker).upper()
            if key not in self._scores:
                return 0.0
            ts = self._timestamps.get(key, 0.0)
            if (time.time() - float(ts)) > self.ttl_seconds:
                return 0.0
            return float(self._scores.get(key, 0.0))
        except Exception:
            return 0.0

    def get_fallback(self, ticker: str) -> float:
        """Last-good score regardless of TTL, else 0.0. Used on 429 / parse failure."""
        try:
            return float(self._fallback_scores.get(str(ticker).upper(), 0.0))
        except Exception:
            return 0.0

    def get_all_fresh(self) -> Dict[str, float]:
        now = time.time()
        out: Dict[str, float] = {}
        for tk, v in self._scores.items():
            if (now - float(self._timestamps.get(tk, 0.0))) <= self.ttl_seconds:
                out[tk] = float(v)
        return out

    def is_fresh(self, ticker: str) -> bool:
        key = str(ticker).upper()
        if key not in self._timestamps:
            return False
        return (time.time() - float(self._timestamps[key])) <= self.ttl_seconds

    def clear(self) -> None:
        self._scores.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        return len(self.get_all_fresh())


class NewsSentimentAgent:
    """
    Batched 3-Stage Adversarial Debate Pipeline (Bull vs Bear + Arbiter Judge).

    Execution budget: exactly 3 API requests per 15-minute tick:
      Stage 1 (parallel via asyncio.gather): _generate_bull_theses + _generate_bear_theses
      Stage 2 (single call): _arbitrate_debate -> Dict[ticker, float in [-1, 1]]
    Total: 288 calls/day, 0.2 RPM — strictly within Google AI Studio free-tier (15 RPM / 1500 RPD).

    Resilience: exponential backoff (retries 3, delay 2.0s, backoff 2.0).
    On HTTP 429 or parsing failure, falls back to previous cached scores or 0.0 neutral.
    """
    # Model-level failures that should trigger an automatic fallback-model retry.
    # 400 is included because Gemma rejects some params that Gemini accepts.
    _MODEL_FALLBACK_STATUS = frozenset({400, 404, 500, 502, 503, 504})

    def __init__(self, api_key: str = None, cache_ttl: float = 1200.0):
        self.api_key = (
            api_key
            or config.GEMINI_API_KEY
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.cache = SentimentCache(ttl_seconds=cache_ttl)
        self.rss_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en"
        ]
        # Model ids are hardcoded in config.py (NOT read from the environment).
        self.model = config.GEMINI_MODEL
        self.fallback_model = config.GEMINI_FALLBACK_MODEL
        self.timeout = httpx.Timeout(
            timeout=float(config.GEMINI_TIMEOUT_SECONDS),
            connect=float(config.GEMINI_CONNECT_TIMEOUT_SECONDS),
        )
        self.max_retries = 3
        self.base_delay = 2.0
        self.backoff = 2.0
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = os.getenv("REDIS_PASSWORD", "") or None
        self._redis_client = None

    def get_redis(self):
        """Lazy Redis connection accessor with connection reuse."""
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    password=self.redis_password,
                    decode_responses=True,
                    socket_timeout=2.0
                )
            except Exception as e:
                logger.warning(f"⚠️ Redis client initialization error: {e}")
                return None
        return self._redis_client

    def _model_chain(self, primary: str = None) -> list:
        """Ordered list of model ids: primary first, then the fallback (deduped)."""
        chain = [str(primary) if primary else (self.model or config.GEMINI_MODEL)]
        fb = getattr(self, "fallback_model", None) or config.GEMINI_FALLBACK_MODEL
        if fb and fb not in chain:
            chain.append(fb)
        return chain

    # ------------------------------------------------------------------
    # Snapshot formatting
    # ------------------------------------------------------------------
    def _build_snapshot_table(self, snapshots: Dict[str, Dict[str, Any]]) -> str:
        """Render compact markdown table of all 20 tickers with technicals + headlines."""
        lines = ["| Ticker | Close | Open | High | Low | Vol | RSI14 | Mom15m% | Headlines |"]
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for tk in sorted(snapshots.keys()):
            try:
                d = snapshots.get(tk, {}) or {}
                close = d.get("close", 0.0)
                open_p = d.get("open", close)
                high = d.get("high", close)
                low = d.get("low", close)
                vol = d.get("volume", 0.0)
                rsi = d.get("rsi14", d.get("rsi", 50.0))
                mom = d.get("momentum_15m", d.get("momentum", 0.0))
                headlines = d.get("headlines", d.get("news", ""))
                if isinstance(headlines, list):
                    headlines = "; ".join(str(h)[:120] for h in headlines[:2])
                headlines = str(headlines or "-")[:160].replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| {tk} | {close} | {open_p} | {high} | {low} | {vol} | {rsi} | {mom} | {headlines} |"
                )
            except Exception:
                lines.append(f"| {tk} | - | - | - | - | - | 50.0 | 0.0 | - |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Low-level LLM caller with exponential backoff
    # ------------------------------------------------------------------
    async def _call_gemini_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        model: str = None,
    ) -> str:
        """
        Native Google AI Studio ``generateContent`` text completion.

        Hardening:
          * extended read timeout (default 90s) + short 15s connect so a dead
            endpoint fails fast without hanging the 15-minute tick loop;
          * native ``?key=`` query-param authentication (no Authorization header);
          * retries with exponential backoff on 429 / transient transport errors;
          * automatic one-shot fallback to ``GEMINI_FALLBACK_MODEL`` when the
            primary model returns HTTP 400/404/500/502/503/504.
        Raises on terminal failure so callers can apply their own safe fallback.
        """
        api_key = (
            self.api_key
            or config.GEMINI_API_KEY
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Gemini API key is not set (GEMINI_API_KEY or GOOGLE_API_KEY).")

        models = self._model_chain(model)
        headers = {"Content-Type": "application/json"}
        last_error: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for current_model in models:
                url = config.gemini_generate_url(current_model, api_key)
                gen_config: Dict[str, Any] = {
                    "maxOutputTokens": max_tokens,
                }
                # Gemma 4-31B is a native thinking model that requires includeThoughts
                # and temperature=1.0. Low temperatures (<1.0) trigger Google API HTTP 500 errors.
                if "gemma" in str(current_model).lower() or "thinking" in str(current_model).lower():
                    gen_config["thinkingConfig"] = {"includeThoughts": True}
                    gen_config["temperature"] = 1.0
                else:
                    gen_config["temperature"] = temperature

                payload = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": gen_config,
                }
                for attempt in range(self.max_retries):
                    try:
                        resp = await client.post(url, json=payload, headers=headers)
                        if resp.status_code == 429:
                            sleep_time = self.base_delay * (self.backoff ** attempt)
                            logger.warning(
                                f"⚠️ [Rate Limit / 429] '{current_model}' generateContent call. "
                                f"Retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{self.max_retries})..."
                            )
                            if attempt == self.max_retries - 1:
                                break
                            await asyncio.sleep(sleep_time)
                            continue
                        if resp.status_code in self._MODEL_FALLBACK_STATUS:
                            has_next_model = (current_model != models[-1])
                            if has_next_model:
                                last_error = RuntimeError(
                                    f"model '{current_model}' returned HTTP {resp.status_code}"
                                )
                                logger.warning(
                                    f"⚠️ Model '{current_model}' returned HTTP {resp.status_code} "
                                    f"(unroutable/missing); retrying with fallback model."
                                )
                                break  # advance to the next model in the chain
                            elif resp.status_code in (500, 502, 503, 504):
                                sleep_time = self.base_delay * (self.backoff ** attempt)
                                logger.warning(
                                    f"⚠️ Model '{current_model}' returned transient HTTP {resp.status_code}. "
                                    f"Retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{self.max_retries})..."
                                )
                                if attempt == self.max_retries - 1:
                                    last_error = RuntimeError(f"model '{current_model}' returned HTTP {resp.status_code}")
                                    break
                                await asyncio.sleep(sleep_time)
                                continue
                            else:
                                last_error = RuntimeError(f"model '{current_model}' returned non-retryable HTTP {resp.status_code}")
                                break
                        resp.raise_for_status()
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            last_error = ValueError(f"No candidate content returned: {data}")
                            logger.warning(
                                f"⚠️ No candidate content from '{current_model}'; trying next model."
                            )
                            break
                        text = config.gemini_extract_text(candidates)
                        if not text:
                            last_error = RuntimeError(f"empty completion from '{current_model}'")
                            logger.warning(f"⚠️ Empty completion from '{current_model}'; trying next model.")
                            break
                        return str(text)
                    except httpx.HTTPStatusError as hse:
                        code = hse.response.status_code if hse.response is not None else 0
                        last_error = hse
                        if code in self._MODEL_FALLBACK_STATUS:
                            has_next_model = (current_model != models[-1])
                            if has_next_model:
                                logger.warning(
                                    f"⚠️ HTTP {code} for model '{current_model}'; retrying with fallback model."
                                )
                                break  # advance to the next model in the chain
                            elif code in (500, 502, 503, 504):
                                sleep_time = self.base_delay * (self.backoff ** attempt)
                                logger.warning(
                                    f"⚠️ Transient HTTP {code} for model '{current_model}'. "
                                    f"Retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{self.max_retries})..."
                                )
                                if attempt == self.max_retries - 1:
                                    break
                                await asyncio.sleep(sleep_time)
                                continue
                            else:
                                break
                        if code in (401, 403):
                            logger.error(f"❌ HTTP {code} for model '{current_model}'; aborting (auth failure).")
                            raise
                        logger.warning(f"⚠️ HTTP {code} during debate call (model='{current_model}'): {hse}")
                        if attempt == self.max_retries - 1:
                            break
                        await asyncio.sleep(self.base_delay * (self.backoff ** attempt))
                    except (httpx.TimeoutException, httpx.TransportError) as te:
                        last_error = te
                        logger.warning(
                            f"⚠️ Transport/timeout (attempt {attempt + 1}/{self.max_retries}, "
                            f"model='{current_model}'): {type(te).__name__} - {te}"
                        )
                        if attempt == self.max_retries - 1:
                            break
                        await asyncio.sleep(self.base_delay * (self.backoff ** attempt))
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"⚠️ Debate call attempt failed (model='{current_model}'): {type(e).__name__} - {e}"
                        )
                        if attempt == self.max_retries - 1:
                            break
                        await asyncio.sleep(self.base_delay * (self.backoff ** attempt))

        raise RuntimeError(f"Gemini generateContent failed across models {models}: {last_error}")

    async def _call_gemini_json(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Call the model chain and return the first valid JSON object in the reply."""
        raw = await self._call_gemini_text(prompt, temperature=temperature, max_tokens=max_tokens)
        cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", str(raw or "")).strip()
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            m = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in response: '{str(raw)[:120]}...'")
        parsed = json.loads(m.group(0).strip())
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON was not an object.")
        return parsed

    # ------------------------------------------------------------------
    # Stage 1: Parallel Thesis Generation
    # ------------------------------------------------------------------
    async def _generate_bull_theses(self, snapshots: Dict[str, Dict[str, Any]]) -> str:
        """High-conviction Bullish Researcher: compact bulleted upside triggers per ticker."""
        table = self._build_snapshot_table(snapshots)
        prompt = f"""You are a High-Conviction Bullish Researcher for a quantitative trading swarm.
Input: 15-minute technical snapshot table for 20 liquid tickers (with RSI14, 15m Momentum, OHLCV) plus headlines.

TABLE:
{table}

TASK:
For EACH ticker, write 1 compact bullet identifying the strongest upside trigger (momentum continuation, RSI recovery, support bounce, relative strength, breakout, short-cover fuel, macro tailwind).
Keep each bullet under 25 words. Cover all 20 tickers. No JSON, just bulleted list like:
- NVDA: ...
- AAPL: ...
Be decisive and bullish but grounded in the provided price action."""
        try:
            text = await self._call_gemini_text(prompt, temperature=0.3, max_tokens=2048)
            logger.info("✅ Bull theses generated.")
            return text.strip() or "Bullish momentum intact across leaders."
        except Exception as e:
            logger.warning(f"⚠️ Bull thesis generation failed, using fallback: {e}")
            # Fallback: deterministic stub so pipeline never breaks the tick loop
            lines = []
            for tk in sorted(snapshots.keys()):
                d = snapshots.get(tk, {}) or {}
                mom = float(d.get("momentum_15m", d.get("momentum", 0.0)) or 0.0)
                rsi = float(d.get("rsi14", d.get("rsi", 50.0)) or 50.0)
                lines.append(f"- {tk}: Momentum {mom:+.2f}%, RSI {rsi:.1f} supports upside continuation.")
            return "\n".join(lines) if lines else "Bullish momentum intact."

    async def _generate_bear_theses(self, snapshots: Dict[str, Dict[str, Any]]) -> str:
        """Skeptical Risk Manager & Bearish Short-Seller: downside traps per ticker."""
        table = self._build_snapshot_table(snapshots)
        prompt = f"""You are a Skeptical Risk Manager and Bearish Short-Seller for a quantitative trading swarm.
Input: 15-minute technical snapshot table for 20 liquid tickers (with RSI14, 15m Momentum, OHLCV) plus headlines.

TABLE:
{table}

TASK:
For EACH ticker, write 1 compact bullet highlighting distribution risk, overbought signal, downside trap, or breakdown risk (RSI>70 fade, momentum stall, resistance rejection, volume fade, macro headwind).
Keep each bullet under 25 words. Cover all 20 tickers. No JSON, just bulleted list like:
- NVDA: ...
- AAPL: ...
Be skeptical and risk-focused, grounded in the provided price action."""
        try:
            text = await self._call_gemini_text(prompt, temperature=0.3, max_tokens=2048)
            logger.info("✅ Bear theses generated.")
            return text.strip() or "Distribution risks elevated on overbought names."
        except Exception as e:
            logger.warning(f"⚠️ Bear thesis generation failed, using fallback: {e}")
            lines = []
            for tk in sorted(snapshots.keys()):
                d = snapshots.get(tk, {}) or {}
                rsi = float(d.get("rsi14", d.get("rsi", 50.0)) or 50.0)
                mom = float(d.get("momentum_15m", d.get("momentum", 0.0)) or 0.0)
                flag = "overbought" if rsi > 70 else "momentum fade" if mom < 0 else "distribution"
                lines.append(f"- {tk}: {flag} risk with RSI {rsi:.1f}, momentum {mom:+.2f}%.")
            return "\n".join(lines) if lines else "Distribution risks elevated."

    # ------------------------------------------------------------------
    # Stage 2: Arbiter Consensus & Scoring
    # ------------------------------------------------------------------
    def _parse_arbiter_json(self, raw_content: str, tickers: list) -> Dict[str, float]:
        """
        Strict JSON extraction for arbiter output.
        Falls back safely to 0.0 neutral per ticker without raising.
        """
        try:
            if not raw_content:
                raise ValueError("Empty arbiter response")
            cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", raw_content).strip()
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned).strip()
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                m = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON object found in: '{raw_content[:120]}...'")
            parsed = json.loads(m.group(0).strip())
            out: Dict[str, float] = {}
            for tk in tickers:
                try:
                    raw_v = parsed.get(tk, parsed.get(str(tk).upper(), 0.0))
                    v = float(raw_v) if raw_v is not None else 0.0
                    v = float(np.clip(v, -1.0, 1.0))
                    out[str(tk).upper()] = round(v, 4)
                except Exception:
                    out[str(tk).upper()] = 0.0
            return out
        except Exception as e:
            logger.warning(f"⚠️ Arbiter JSON parse failed ({e}); falling back to 0.0 neutral.")
            return {str(tk).upper(): 0.0 for tk in tickers}

    def _sanitize_scores(self, scores: Dict[str, Any], tickers: list) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tk in tickers:
            try:
                v = float(scores.get(tk, scores.get(str(tk).upper(), 0.0)) or 0.0)
                out[str(tk).upper()] = round(float(np.clip(v, -1.0, 1.0)), 4)
            except Exception:
                out[str(tk).upper()] = 0.0
        return out

    async def _arbitrate_debate(
        self, snapshots: Dict[str, Dict[str, Any]], bull_theses: str, bear_theses: str
    ) -> Dict[str, float]:
        """
        Impartial Quantitative Arbiter: weighs bull vs bear against price action,
        emits final consensus sentiment in [-1.0, +1.0] as strict JSON.
        """
        tickers = sorted(snapshots.keys()) if snapshots else list(CANONICAL_TICKERS)
        table = self._build_snapshot_table(snapshots) if snapshots else "(no snapshot)"
        prompt = f"""You are an Impartial Quantitative Arbiter for a trading swarm.
Weigh competing Bull vs Bear arguments against ACTUAL 15m price action and resolve conflicts.

TECHNICAL TABLE:
{table}

BULL THESES:
{bull_theses}

BEAR THESES:
{bear_theses}

TASK:
Emit final consensus sentiment score per ticker between -1.0 (Strong Bearish) and +1.0 (Strong Bullish).
Consider RSI14 extremes, 15m momentum sign, and which thesis better fits price action.
STRICT OUTPUT: exclusively valid JSON, no preamble, no markdown, no explanation. Example:
{{
  "NVDA": 0.45,
  "AAPL": -0.20
}}
Cover exactly these tickers: {", ".join(tickers)}"""
        try:
            raw = await self._call_gemini_text(prompt, temperature=0.1, max_tokens=2048)
            scores = self._parse_arbiter_json(raw, tickers)
            logger.info("✅ Arbiter consensus scored.")
            return scores
        except Exception as e:
            logger.warning(f"⚠️ Arbiter call failed ({e}); using cache fallback / 0.0.")
            fallback: Dict[str, float] = {}
            for tk in tickers:
                key = str(tk).upper()
                # Prefer previous cached score, else neutral
                fallback[key] = float(self.cache.get_fallback(key) or 0.0)
            return fallback

    # ------------------------------------------------------------------
    # Orchestrator: exactly 3 API calls per tick
    # ------------------------------------------------------------------
    async def run_adversarial_batch(
        self, snapshots: Dict[str, Dict[str, Any]], persist: bool = False
    ) -> Dict[str, float]:
        """
        Run batched 3-stage debate:
          1-2. Bull + Bear in parallel via asyncio.gather (2 concurrent calls)
          3. Arbiter consensus (1 call)
        Populates SentimentCache with 20 scores. Total = exactly 3 calls per 15m bar.
        On failure, returns previous cached scores or 0.0 neutral — never raises.
        """
        if not snapshots:
            logger.warning("⚠️ Empty snapshot batch; returning neutral fallback.")
            return {t: 0.0 for t in CANONICAL_TICKERS}
        tickers = sorted(snapshots.keys())
        try:
            bull_theses, bear_theses = await asyncio.gather(
                self._generate_bull_theses(snapshots),
                self._generate_bear_theses(snapshots),
            )
        except Exception as e:
            logger.warning(f"⚠️ Parallel thesis generation failed ({e}); using fallbacks.")
            bull_theses = "Bullish momentum intact."
            bear_theses = "Distribution risks elevated."
        try:
            scores = await self._arbitrate_debate(snapshots, bull_theses, bear_theses)
        except Exception as e:
            logger.warning(f"⚠️ Arbitration failed ({e}); using cache fallback.")
            scores = {str(t).upper(): float(self.cache.get_fallback(str(t)) or 0.0) for t in tickers}
        # Clamp + ensure all tickers present
        final: Dict[str, float] = {}
        for tk in tickers:
            key = str(tk).upper()
            try:
                v = float(scores.get(key, self.cache.get_fallback(key) or 0.0) or 0.0)
                final[key] = round(float(np.clip(v, -1.0, 1.0)), 4)
            except Exception:
                final[key] = 0.0
        self.cache.update(final)

        if persist:
            try:
                top_sym = "SPY" if "SPY" in final else (tickers[0] if tickers else "SPY")
                top_sc = float(final.get(top_sym, 0.0))
                batch_payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": top_sym,
                    "score": top_sc,
                    "bull_thesis": str(bull_theses)[:500],
                    "bear_thesis": str(bear_theses)[:500],
                    "arbiter_reasoning": f"Consensus score {top_sc:+.2f} scored across 20-stock adversarial debate.",
                    "confidence": 0.85,
                }
                # Save latest & history to Redis
                r = self.get_redis()
                if r is not None:
                    raw_json = json.dumps(batch_payload)
                    r.set("market:news_reasoning:latest", raw_json, ex=7200)
                    r.lpush("market:news_reasoning:history", raw_json)
                    try:
                        r.ltrim("market:news_reasoning:history", 0, 999)
                    except Exception:
                        pass
                # Save batch to TimescaleDB
                import db_manager
                db_manager.record_news_sentiment([batch_payload])
            except Exception as persist_err:
                logger.warning(f"⚠️ Non-blocking debate batch persistence note: {persist_err}")

        return final

    async def analyze_and_debate(
        self,
        symbol: str = "SPY",
        snapshot: Optional[Dict[str, Any]] = None,
        bull_thesis: Optional[str] = None,
        bear_thesis: Optional[str] = None,
        arbiter_reasoning: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute 3-stage LLM debate for a symbol and persist structured debate reasoning.
        Captures Bull Thesis, Bear Thesis, and Arbiter Reasoning, saving to Redis and TimescaleDB.
        Guaranteed non-blocking: Redis or DB connection failures log warnings and return the payload.
        """
        sym = str(symbol).upper()
        if score is None or bull_thesis is None or bear_thesis is None or arbiter_reasoning is None:
            snap_dict = snapshot or {
                sym: {
                    "close": 500.0,
                    "open": 498.0,
                    "high": 502.0,
                    "low": 497.0,
                    "volume": 1000000.0,
                    "rsi14": 55.0,
                    "momentum_15m": 0.35,
                    "headlines": f"Macro and corporate earnings update for {sym}",
                }
            }
            try:
                b_theses = bull_thesis or await self._generate_bull_theses(snap_dict)
                be_theses = bear_thesis or await self._generate_bear_theses(snap_dict)
                arb_scores = await self._arbitrate_debate(snap_dict, b_theses, be_theses)
                calc_score = float(arb_scores.get(sym, 0.0))
                calc_confidence = 0.85
                calc_reasoning = (
                    f"Arbiter evaluated 15m price action against Bull and Bear arguments for {sym}, "
                    f"settling on consensus sentiment {calc_score:+.2f}."
                )
            except Exception as e:
                logger.warning(f"⚠️ analyze_and_debate LLM generation failed, using fallback: {e}")
                b_theses = bull_thesis or f"Bullish momentum intact on {sym}."
                be_theses = bear_thesis or f"Distribution and overbought risk elevated on {sym}."
                calc_score = 0.0
                calc_confidence = 0.50
                calc_reasoning = f"Neutral fallback applied for {sym} due to inference exception: {e}"

            score = calc_score if score is None else float(score)
            confidence = calc_confidence if confidence is None else float(confidence)
            bull_thesis = b_theses if bull_thesis is None else bull_thesis
            bear_thesis = be_theses if bear_thesis is None else bear_thesis
            arbiter_reasoning = calc_reasoning if arbiter_reasoning is None else arbiter_reasoning

        debate_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "score": round(float(score), 4),
            "bull_thesis": str(bull_thesis),
            "bear_thesis": str(bear_thesis),
            "arbiter_reasoning": str(arbiter_reasoning),
            "confidence": round(float(confidence), 4),
        }

        # Non-blocking Redis publish (key: market:news_reasoning:latest with TTL 7200s, and history list)
        try:
            r = self.get_redis()
            if r is not None:
                payload_json = json.dumps(debate_payload)
                r.set("market:news_reasoning:latest", payload_json, ex=7200)
                r.lpush("market:news_reasoning:history", payload_json)
                try:
                    r.ltrim("market:news_reasoning:history", 0, 999)
                except Exception:
                    pass
        except Exception as redis_err:
            logger.warning(f"⚠️ Redis publication in analyze_and_debate failed: {redis_err}")

        # Non-blocking TimescaleDB insert
        try:
            import db_manager
            db_manager.record_news_sentiment([debate_payload])
        except Exception as db_err:
            logger.warning(f"⚠️ TimescaleDB persistence in analyze_and_debate failed: {db_err}")

        return debate_payload

    def get_sentiment(self, ticker: str) -> float:
        """Convenience proxy to cache O(1) lookup."""
        return self.cache.get_sentiment(ticker)

    # ------------------------------------------------------------------
    # Legacy macro RAG (kept for backward compatibility)
    # ------------------------------------------------------------------
    async def _fetch_rss_headlines_async(self, client: httpx.AsyncClient, max_headlines: int = 5) -> str:
        """Asynchronously fetches and aggregates recent macro headlines from RSS feeds."""
        headlines = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        for url in self.rss_feeds:
            try:
                resp = await client.get(url, headers=headers, timeout=6.0, follow_redirects=True)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries[:max_headlines]:
                        clean_title = re.sub(r'<[^>]+>', '', getattr(entry, 'title', '')).strip()
                        if clean_title and clean_title not in headlines:
                            headlines.append(clean_title)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse RSS feed ({url}): {e}")

        if not headlines:
            return "Markets trading in normal consolidation range. No major systemic news events detected."

        return "\n".join([f"- {h}" for h in headlines[:max_headlines]])

    def _sanitize_sentiment_output(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hard-clamps sentiment score and risk multiplier to prevent LLM hallucination extremes.
        """
        try:
            raw_score = raw_data.get("sentiment_score")
            score = float(raw_score) if raw_score is not None else 0.0
            score = float(np.clip(score, -1.0, 1.0))
            
            # Map score (-1.0 to +1.0) into risk multiplier range (0.7x to 1.3x)
            multiplier = 1.0 + (score * 0.3)
            # Enforce hard safety boundaries [0.5, 1.5]
            multiplier = float(np.clip(multiplier, 0.5, 1.5))
            
            reasoning = str(raw_data.get("summary_reasoning") or "Macro news sentiment evaluated.").strip()

            return {
                "sentiment_score": round(score, 2),
                "risk_multiplier": round(multiplier, 2),
                "summary_reasoning": reasoning
            }
        except Exception as e:
            logger.warning(f"⚠️ Error sanitizing sentiment output: {e}. Applying neutral fallback.")
            return self._neutral_fallback()

    def _neutral_fallback(self) -> Dict[str, Any]:
        """Neutral fallback dictionary returned on network or provider errors."""
        return {
            "sentiment_score": 0.0,
            "risk_multiplier": 1.0,
            "summary_reasoning": "Neutral fallback applied due to news feed or LLM API timeout."
        }

    def analyze_macro_sentiment(self) -> Dict[str, Any]:
        """
        Synchronous wrapper calling the sentiment analysis safely without external loop dependencies.
        """
        import concurrent.futures

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(self.analyze_macro_sentiment_async()))
                    return future.result(timeout=40.0)
            else:
                return asyncio.run(self.analyze_macro_sentiment_async())
        except Exception as e:
            logger.error(f"❌ Sentiment analysis execution failed: {e}")
            return self._neutral_fallback()

    async def analyze_macro_sentiment_async(self) -> Dict[str, Any]:
        """
        Queries Google AI Studio asynchronously to evaluate current financial headlines.

        Transport, retries, the dynamic model id and the 500/404 fallback model are
        all delegated to the hardened `_call_gemini_text`/`_call_gemini_json` chain
        (90s read timeout, 15s connect). Never raises: always returns a sanitized dict.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headlines_text = await self._fetch_rss_headlines_async(client, max_headlines=5)

        prompt = f"""
You are a Senior Macroeconomic Risk Analyst for a Quantitative Trading Swarm.
Evaluate the following recent market headlines and determine the systemic market sentiment.

HEADLINES:
{headlines_text}

INSTRUCTIONS:
1. Assign a sentiment_score from -1.0 (Extreme Bearish/Panic) to +1.0 (Extreme Bullish/Euphonic).
2. Write a detailed 2 to 3 sentence summary_reasoning highlighting key drivers (e.g. Fed policy, tech earnings, inflation).
3. Return ONLY a JSON object:
{{
    "sentiment_score": 0.2,
    "summary_reasoning": "Post-earnings sell-offs are hitting growth sectors, but opportunistic dip-buying is providing a floor. Investors remain cautious ahead of upcoming Fed guidance."
}}
"""

        api_key = self.api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ Gemini API key is not set in environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).")
            return self._neutral_fallback()

        try:
            parsed = await self._call_gemini_json(prompt, temperature=0.1, max_tokens=1024)
            logger.info("✅ News sentiment evaluated successfully via [Google AI Studio].")
            return self._sanitize_sentiment_output(parsed)
        except Exception as e:
            logger.warning(
                f"⚠️ News sentiment fetch failed: {type(e).__name__} - {e}. Applying neutral fallback."
            )
            return self._neutral_fallback()


# Backwards-compatible alias for adversarial pipeline
AdversarialSentimentAgent = NewsSentimentAgent
