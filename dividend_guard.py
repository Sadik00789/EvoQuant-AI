"""
Dividend and short-cost guard (Phase 5).

Implements the dividend-aware behavior described in the README but previously
missing from the code:
  * Detect ex-dividend dates and their per-share amount.
  * Compute the dividend liability of holding a short through an ex-date.
  * Compute short borrow cost.
  * Decide when a long should be flattened ahead of an ex-date.

Reads the `dividend_schedule` table through the portfolio manager and caches by
UTC day to avoid repeated queries.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger("DividendGuard")


class DividendGuard:
    def __init__(self, db=None, borrow_annual_bps: float = 50.0):
        self.db = db
        self.borrow_annual_bps = float(borrow_annual_bps)
        self._by_ticker: Dict[str, Dict[str, str]] = {}
        self._cache_day: Optional[date] = None

    def _refresh(self, today: date) -> None:
        if self.db is None or self._cache_day == today:
            return
        try:
            df = self.db.fetch_upcoming_dividends()
            mapping: Dict[str, Dict[str, str]] = {}
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    tk = str(row["ticker"]).upper()
                    mapping[tk] = {
                        "ex_date": str(row["ex_date"])[:10],
                        "amount": float(row.get("amount_per_share", 0.0) or 0.0),
                    }
            self._by_ticker = mapping
            self._cache_day = today
        except Exception as e:
            logger.debug(f"DividendGuard refresh failed: {e}")

    def _record(self, ticker: str, today: Optional[date] = None) -> Optional[Dict[str, str]]:
        today = today or datetime.now(timezone.utc).date()
        self._refresh(today)
        return self._by_ticker.get(str(ticker).upper())

    def ex_dividend_today(self, ticker: str, today: Optional[date] = None) -> float:
        rec = self._record(ticker, today)
        if not rec:
            return 0.0
        ref = (today or datetime.now(timezone.utc).date()).isoformat()
        return float(rec["amount"]) if rec["ex_date"] == ref else 0.0

    def ex_dividend_within(self, ticker: str, days: int = 1, today: Optional[date] = None) -> float:
        rec = self._record(ticker, today)
        if not rec:
            return 0.0
        ref = today or datetime.now(timezone.utc).date()
        try:
            ex = datetime.fromisoformat(rec["ex_date"]).date()
        except Exception:
            return 0.0
        if 0 <= (ex - ref).days <= int(days):
            return float(rec["amount"])
        return 0.0

    def short_dividend_liability(self, ticker: str, shares: float, today: Optional[date] = None) -> float:
        """Cash a short holder owes when the stock goes ex-dividend today."""
        if shares >= 0:
            return 0.0
        amount = self.ex_dividend_today(ticker, today)
        return round(abs(shares) * amount, 2)

    def should_flatten_long(self, ticker: str, today: Optional[date] = None) -> bool:
        """Flatten a long ahead of an ex-dividend date (default: ex-date is tomorrow)."""
        return self.ex_dividend_within(ticker, days=1, today=today) > 0.0

    def borrow_cost(self, shares: float, price: float, days: float = 1.0) -> float:
        """Annualized borrow cost pro-rated for `days` of holding a short."""
        try:
            notional = abs(float(shares)) * float(price)
            return round(notional * (self.borrow_annual_bps / 10000.0) * (float(days) / 365.0), 4)
        except Exception:
            return 0.0

    def short_entry_allowed(
        self,
        ticker: str,
        shares: float = 0.0,
        today: Optional[date] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Block opening a short when the expected dividend liability is likely to
        exceed typical near-term edge; conservative default: never open a new
        short on the day before an ex-dividend date.

        `shares` and `current_time` are accepted for caller ergonomics and are
        fully optional so symbol-eligibility checks that do not yet know the
        exact sizing (e.g. pre-allocation screening in `swarm_consumer`) cannot
        raise a ``TypeError``. When `today` is omitted it is derived from
        `current_time` (or the current UTC date).
        """
        if today is None and current_time is not None:
            try:
                today = current_time.date()
            except Exception:
                today = None
        return self.ex_dividend_within(ticker, days=1, today=today) <= 0.0
