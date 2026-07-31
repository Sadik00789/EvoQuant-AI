import os
import re
import json
import logging
import asyncio
import httpx
import feedparser
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsSentimentAgent")

class NewsSentimentAgent:
    """
    RAG Macro News Sentiment & Regime Scaler Agent.
    Fetches real-time financial headlines from RSS feeds and uses Google AI Studio (Gemma 4-31B)
    to output a strictly bounded risk multiplier [0.5x, 1.5x].
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.rss_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en"
        ]

    async def _fetch_rss_headlines_async(self, client: httpx.AsyncClient, max_headlines: int = 8) -> str:
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
                    return future.result(timeout=30.0)
            else:
                return asyncio.run(self.analyze_macro_sentiment_async())
        except Exception as e:
            logger.error(f"❌ Sentiment analysis execution failed: {e}")
            return self._neutral_fallback()

    async def analyze_macro_sentiment_async(self) -> Dict[str, Any]:
        """
        Queries Google AI Studio (Gemma 4-31B) asynchronously to evaluate current financial headlines.
        Includes automated rate-limit exception handling with exponential backoff.
        """
        async with httpx.AsyncClient() as client:
            headlines_text = await self._fetch_rss_headlines_async(client)

            prompt = f"""
You are a Senior Macroeconomic Risk Analyst for a Quantitative Trading Swarm.
Evaluate the following recent market headlines and determine the systemic market sentiment.

HEADLINES:
{headlines_text}

INSTRUCTIONS:
1. Assign a sentiment_score from -1.0 (Extreme Bearish/Panic) to +1.0 (Extreme Bullish/Euphonic).
2. Write a concise 1-sentence summary_reasoning justifying your score.
3. Return ONLY a JSON object:
{{
    "sentiment_score": 0.2,
    "summary_reasoning": "Tech earnings stability offsetting rate uncertainty."
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
                    resp = await client.post(url, json=payload, headers=headers, timeout=15.0)

                    if resp.status_code == 429:
                        sleep_time = backoff_factor ** (attempt + 1)
                        logger.warning(f"⚠️ [Rate Limit / 429] during news sentiment fetch. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(sleep_time)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    
                    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content).strip()
                    parsed = json.loads(cleaned)

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
                    logger.warning(f"⚠️ News sentiment fetch attempt failed: {e}")
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(backoff_factor ** (attempt + 1))

            return self._neutral_fallback()
