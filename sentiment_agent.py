import os
import re
import json
import logging
import asyncio
import time
import httpx
import feedparser
import numpy as np
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsSentimentAgent")

# 20-stock universe canonical order for arbiter output validation
CANONICAL_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "QCOM",
    "AVGO", "SPY", "QQQ", "IWM", "GLD",
    "SLV", "TLT", "COIN", "PLTR", "ARM",
]


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
    def __init__(self, api_key: str = None, cache_ttl: float = 1200.0):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.cache = SentimentCache(ttl_seconds=cache_ttl)
        self.rss_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en"
        ]
        self.model = "gemma-4-31b-it"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        self.max_retries = 3
        self.base_delay = 2.0
        self.backoff = 2.0

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
    async def _call_gemini_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Single Gemini text call with retries on 429 / transient errors. Raises on final failure."""
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is not set (GEMINI_API_KEY or GOOGLE_API_KEY).")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient() as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(self.api_url, json=payload, headers=headers, timeout=35.0)
                    if resp.status_code == 429:
                        sleep_time = self.base_delay * (self.backoff ** attempt)
                        logger.warning(
                            f"⚠️ [Rate Limit / 429] debate call. Retrying in {sleep_time:.1f}s "
                            f"(Attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"] or ""
                    return str(content)
                except httpx.HTTPStatusError as hse:
                    code = hse.response.status_code if hse.response is not None else 0
                    logger.warning(f"⚠️ HTTP {code} during debate call: {hse}")
                    if code in (400, 401, 403, 404):
                        raise
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.base_delay * (self.backoff ** attempt))
                except Exception as e:
                    logger.warning(f"⚠️ Debate call attempt failed: {type(e).__name__} - {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.base_delay * (self.backoff ** attempt))
        raise RuntimeError("Gemini debate call failed after retries.")

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
    async def run_adversarial_batch(self, snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
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
        return final

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
        Queries Google AI Studio (Gemma 4-31B) asynchronously to evaluate current financial headlines.
        Includes automated rate-limit exception handling with exponential backoff and robust JSON cleaning.
        """
        async with httpx.AsyncClient() as client:
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

            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if not api_key:
                logger.error("❌ Gemini API key is not set in environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).")
                return self._neutral_fallback()

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gemma-4-31b-it",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"}
            }

            max_retries = 3
            backoff_factor = 2.0

            for attempt in range(max_retries):
                try:
                    resp = await client.post(url, json=payload, headers=headers, timeout=35.0)

                    if resp.status_code == 429:
                        sleep_time = backoff_factor ** (attempt + 1)
                        logger.warning(f"⚠️ [Rate Limit / 429] during news sentiment fetch. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(sleep_time)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    raw_content = data['choices'][0]['message']['content'] or ""

                    # 1. Strip internal thinking tags (<thought>...</thought>)
                    cleaned_content = re.sub(r"<thought>[\s\S]*?</thought>", "", raw_content).strip()

                    # 2. Strip Markdown code fences if present (```json ... ```)
                    cleaned_content = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", cleaned_content).strip()

                    # 3. Extract valid JSON object block {...} using re.DOTALL to strip conversational preambles/postambles
                    json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)

                    if not json_match:
                        raise ValueError(f"Could not locate JSON object pattern in response: '{raw_content[:80]}...'")

                    cleaned_json_str = json_match.group(0).strip()
                    parsed = json.loads(cleaned_json_str)

                    logger.info("✅ News sentiment evaluated successfully via [Google AI Studio - Gemma 4 31B]")
                    return self._sanitize_sentiment_output(parsed)

                except httpx.HTTPStatusError as hse:
                    logger.warning(f"⚠️ HTTP status error {hse.response.status_code} during sentiment fetch: {hse.response.text}")
                    if hse.response.status_code in (400, 401, 403, 404):
                        break
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(backoff_factor ** (attempt + 1))
                except Exception as e:
                    logger.warning(f"⚠️ News sentiment fetch attempt failed: {type(e).__name__} - {e}")
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(backoff_factor ** (attempt + 1))

            return self._neutral_fallback()


# Backwards-compatible alias for adversarial pipeline
AdversarialSentimentAgent = NewsSentimentAgent
