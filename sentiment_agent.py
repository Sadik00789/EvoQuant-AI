import os
import re
import json
import logging
import asyncio
import httpx
import feedparser
import numpy as np
import boto3
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsSentimentAgent")

class NewsSentimentAgent:
    """
    RAG Macro News Sentiment & Regime Scaler Agent.
    Fetches real-time financial headlines from RSS feeds and uses multi-provider LLMs
    (Groq -> OpenRouter -> GitHub Models -> SambaNova -> AWS Bedrock)
    to output a strictly bounded risk multiplier [0.5x, 1.5x].
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY2")
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

    def _call_aws_bedrock_llama(self, prompt: str) -> str:
        """
        Synchronous AWS Bedrock Converse API invocation for Llama 3.3 70B Instruct.
        Executed inside an async thread executor to avoid blocking the event loop.
        """
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION", "us-east-1")

        if not aws_key or not aws_secret:
            raise ValueError("AWS credentials not set in environment.")

        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret
        )

        model_id = "us.meta.llama3-3-70b-instruct-v1:0"

        response = bedrock_client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            inferenceConfig={
                "temperature": 0.1,
                "maxTokens": 1024
            }
        )

        output_message = response["output"]["message"]["content"][0]["text"]
        return output_message

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
        Synchronous wrapper calling the multi-provider LLM analysis safely without external loop dependencies.
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
        Queries multi-provider LLMs asynchronously to evaluate current financial headlines.
        Routes through free tiers first, falling back to AWS Bedrock Llama 3.3 70B if rate-limited.
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

            providers = [
                {
                    "name": "Groq",
                    "type": "http",
                    "url": "https://api.groq.com/openai/v1/chat/completions",
                    "key": self.api_key or os.getenv("GROQ_API_KEY2"),
                    "model": "llama-3.3-70b-versatile",
                    "use_json_format": True
                },
                {
                    "name": "OpenRouter",
                    "type": "http",
                    "url": "https://openrouter.ai/api/v1/chat/completions",
                    "key": os.getenv("OPENROUTER_API_KEY2"),
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "headers": {
                        "HTTP-Referer": "https://github.com/EvoQuant-AI",
                        "X-Title": "EvoQuant Trading Swarm"
                    },
                    "use_json_format": True
                },
                {
                    "name": "GitHub Models",
                    "type": "http",
                    "url": "https://models.inference.ai.azure.com/chat/completions",
                    "key": os.getenv("GITHUB_TOKEN2"),
                    "model": "Llama-3.3-70B-Instruct",
                    "use_json_format": True
                },
                {
                    "name": "SambaNova",
                    "type": "http",
                    "url": "https://api.sambanova.ai/v1/chat/completions",
                    "key": os.getenv("SAMBANOVA_API_KEY2"),
                    "model": "Meta-Llama-3.3-70B-Instruct",
                    "use_json_format": False
                },
                {
                    "name": "AWS Bedrock (Llama 3.3 70B)",
                    "type": "bedrock",
                    "key": os.getenv("AWS_ACCESS_KEY_ID")
                }
            ]

            for p in providers:
                if not p["key"]:
                    continue

                if p["type"] == "bedrock":
                    try:
                        logger.info("🛡️ Free tiers exhausted or rate-limited. Routing request to AWS Bedrock (Llama 3.3 70B)...")
                        content = await asyncio.to_thread(self._call_aws_bedrock_llama, prompt)
                        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content).strip()
                        parsed = json.loads(cleaned)
                        return self._sanitize_sentiment_output(parsed)
                    except Exception as e:
                        logger.warning(f"⚠️ AWS Bedrock execution failed: {e}")
                        continue

                headers = {
                    "Authorization": f"Bearer {p['key']}",
                    "Content-Type": "application/json"
                }
                if "headers" in p:
                    headers.update(p["headers"])

                payload = {
                    "model": p["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                if p.get("use_json_format"):
                    payload["response_format"] = {"type": "json_object"}

                try:
                    resp = await client.post(p["url"], json=payload, headers=headers, timeout=12.0)
                    if resp.status_code in (400, 401, 402, 404, 410, 429):
                        logger.warning(f"⚠️ [{p['name']}] News sentiment fetch failed ({resp.status_code}). Trying next provider...")
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    
                    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content).strip()
                    parsed = json.loads(cleaned)

                    return self._sanitize_sentiment_output(parsed)

                except Exception as e:
                    logger.warning(f"⚠️ Sentiment fetch failed via [{p['name']}]: {e}")
                    continue

        return self._neutral_fallback()
