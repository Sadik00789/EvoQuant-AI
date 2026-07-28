import os
import json
import feedparser
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class NewsSentimentAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.rss_urls = [
            "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en",
            "https://finance.yahoo.com/news/rssindex"
        ]

    def fetch_latest_headlines(self, max_headlines: int = 8) -> list:
        headlines = []
        for url in self.rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_headlines]:
                    headlines.append(entry.title)
            except Exception as e:
                print(f"⚠️ RSS Fetch Warning: {e}")
        return headlines[:max_headlines]

    def analyze_macro_sentiment(self) -> dict:
        headlines = self.fetch_latest_headlines()
        if not headlines or not self.client:
            return {"sentiment_score": 0.0, "risk_multiplier": 1.0, "reasoning": "Fallback / No News"}

        prompt = f"""
        Analyze these financial market news headlines and assign an overall macro sentiment score.
        Headlines:
        {json.dumps(headlines, indent=2)}

        Output ONLY valid JSON matching this schema:
        {{
            "sentiment_score": float (-1.0 to +1.0, where -1.0 is severe panic, 0.0 neutral, +1.0 extreme bullishness),
            "risk_multiplier": float (0.5 to 1.2, scaling capital utilization based on headline risk),
            "summary_reasoning": string
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"⚠️ Sentiment Analysis Error: {e}")
            return {"sentiment_score": 0.0, "risk_multiplier": 1.0, "reasoning": "API Error Fallback"}

if __name__ == "__main__":
    agent = NewsSentimentAgent()
    sentiment = agent.analyze_macro_sentiment()
    print("\n📰 LIVE MACRO NEWS SENTIMENT REPORT:")
    print(json.dumps(sentiment, indent=2))