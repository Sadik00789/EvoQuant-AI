"""
News headline enrichment for the adversarial debate (Phase 1).

The producer emits pure 15m technical snapshots; this module injects real
per-ticker headlines (Google News RSS) plus optional macro context before the
bull/bear/arbiter debate, so sentiment is genuinely news-driven rather than a
second technical pass.

No API key required. Results are cached with a TTL to bound request volume.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import feedparser
import httpx

logger = logging.getLogger("NewsFetcher")

_MACRO_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en",
]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


class NewsFetcher:
    def __init__(self, ttl_seconds: float = 1800.0, max_items: int = 2, timeout: float = 6.0):
        self.ttl_seconds = float(ttl_seconds)
        self.max_items = int(max_items)
        self.timeout = float(timeout)
        self._ticker_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._macro_cache: Tuple[float, List[str]] = (0.0, [])

    # ------------------------------------------------------------------
    def _fresh(self, cached: Tuple[float, List[str]]) -> bool:
        return bool(cached and (time.time() - float(cached[0])) <= self.ttl_seconds)

    async def fetch_ticker_headlines(self, client: httpx.AsyncClient, ticker: str) -> List[str]:
        key = str(ticker).upper()
        cached = self._ticker_cache.get(key)
        if cached and self._fresh(cached):
            return cached[1]
        url = (
            f"https://news.google.com/rss/search?q={quote_plus(key + ' stock')}"
            "&hl=en-US&gl=US&ceid=US:en"
        )
        titles: List[str] = []
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=self.timeout, follow_redirects=True)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[: self.max_items]:
                    title = _clean(getattr(entry, "title", ""))
                    if title and title not in titles:
                        titles.append(title[:160])
        except Exception as e:
            logger.debug(f"Headline fetch failed for {key}: {e}")
        self._ticker_cache[key] = (time.time(), titles)
        return titles

    async def fetch_macro_headlines(self, client: httpx.AsyncClient) -> List[str]:
        if self._fresh(self._macro_cache):
            return self._macro_cache[1]
        titles: List[str] = []
        for url in _MACRO_FEEDS:
            try:
                resp = await client.get(url, headers=_HEADERS, timeout=self.timeout, follow_redirects=True)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries[:3]:
                        title = _clean(getattr(entry, "title", ""))
                        if title and title not in titles:
                            titles.append(title[:160])
            except Exception:
                continue
        self._macro_cache = (time.time(), titles[:5])
        return self._macro_cache[1]

    # ------------------------------------------------------------------
    async def enrich(
        self,
        snapshots: Dict[str, Dict],
        client: httpx.AsyncClient = None,
        include_macro: bool = True,
    ) -> Dict[str, Dict]:
        """Return shallow-copied snapshots with a populated `headlines` field."""
        if not snapshots:
            return snapshots
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient()
        try:
            sem = asyncio.Semaphore(6)
            tickers = list(snapshots.keys())

            async def _one(tk: str):
                async with sem:
                    return tk, await self.fetch_ticker_headlines(client, tk)

            pairs = await asyncio.gather(*[_one(tk) for tk in tickers], return_exceptions=True)
            macro: List[str] = []
            if include_macro:
                try:
                    macro = await self.fetch_macro_headlines(client)
                except Exception:
                    macro = []

            enriched: Dict[str, Dict] = {}
            macro_str = " | ".join(macro[:2]) if macro else ""
            for tk, data in snapshots.items():
                entry = dict(data or {})
                titles = []
                for p in pairs:
                    if isinstance(p, tuple) and p[0] == tk:
                        titles = p[1]
                        break
                parts = list(titles)
                if macro_str:
                    parts.append(f"Macro: {macro_str}")
                entry["headlines"] = " | ".join(parts) if parts else str(
                    entry.get("headlines") or "-"
                )
                enriched[tk] = entry
            return enriched
        finally:
            if own_client:
                await client.aclose()
