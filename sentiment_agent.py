import os
import re
import json
import logging
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
    Fetches real-time financial headlines from RSS feeds and uses 70B models
    to output a strictly bounded risk multiplier [0.5x, 1.5x].
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.rss_feeds = [
            "https://search.cnbc.com/rs/search/combinedrender?source=cnbcnews&titles=true&displayonly=true&type=news&mincount=10&id=10000664",
            "https://finance.yahoo.com/news/rssindex"
        ]

    def _fetch_rss_headlines(self, max_headlines: int = 8) -> str:
        """Fetches and aggregates recent macro headlines from RSS feeds."""
        headlines = []
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_headlines]:
                    clean_title = re.sub(r'<[^>]+>', '', entry.title).strip()
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
            score = float(raw_data.get("sentiment_score", 0.0))
            score = float(np.clip(score, -1.0, 1.0))
            
            # Map score (-1.0 to +1.0) into risk multiplier range (0.7x to 1.3x)
            multiplier = 1.0 + (score * 0.3)
            # Enforce hard safety boundaries [0.5, 1.5]
            multiplier = float(np.clip(multiplier, 0.5, 1.5))
            
            reasoning = str(raw_data.get("summary_reasoning", "Macro news sentiment evaluated.")).strip()

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
        Synchronous wrapper calling the multi-provider LLM analysis.
        Returns dict with keys: sentiment_score, risk_multiplier, summary_reasoning.
        """
        import asyncio
        try:
            return asyncio.run(self.analyze_macro_sentiment_async())
        except RuntimeError:
            # Handle nested event loop scenarios
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.analyze_macro_sentiment_async())
        except Exception as e:
            logger.error(f"❌ Sentiment analysis execution failed: {e}")
            return self._neutral_fallback()

    async def analyze_macro_sentiment_async(self) -> Dict[str, Any]:
        """
        Queries multi-provider LLMs to evaluate current financial headlines.
        """
        headlines_text = self._fetch_rss_headlines()

        prompt = f"""
You are a Senior Macroeconomic Risk Analyst for an Quantitative Trading Swarm.
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

        providers = [
            {
                "name": "Groq",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "key": self.api_key or os.getenv("GROQ_API_KEY"),
                "model": "llama-3.3-70b-versatile"
            },
            {
                "name": "OpenRouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": os.getenv("OPENROUTER_API_KEY"),
                "model": "meta-llama/llama-3.3-70b-instruct"
            },
            {
                "name": "GitHub Models",
                "url": "https://models.inference.ai.azure.com/chat/completions",
                "key": os.getenv("GITHUB_TOKEN"),
                "model": "Llama-3.3-70B-Instruct"
            }
        ]

        async with httpx.AsyncClient() as client:
            for p in providers:
                if not p["key"]:
                    continue

                headers = {
                    "Authorization": f"Bearer {p['key']}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": p["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1
                }

                try:
                    resp = await client.post(p["url"], json=payload, headers=headers, timeout=10.0)
                    if resp.status_code in (429, 404):
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()
                    parsed = json.loads(cleaned)

                    return self._sanitize_sentiment_output(parsed)

                except Exception as e:
                    logger.warning(f"⚠️ Sentiment fetch failed via [{p['name']}]: {e}")
                    continue

        return self._neutral_fallback()
