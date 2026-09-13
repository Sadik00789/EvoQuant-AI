"""
Async, idempotent Alpaca execution layer (Phase 3).

Key guarantees:
  * Non-blocking httpx.AsyncClient (no sync `requests` in the event loop).
  * Idempotency via deterministic `client_order_id` per (agent, symbol, window).
  * Explicit order-status polling with a bounded timeout (no fire-and-forget).
  * Per-agent paper sub-account mapping, so each evolutionary agent has an
    isolated book (selected deployment model). Falls back to a shared account.
  * Reconciliation of desired virtual positions against physical broker
    positions, trading only the delta.
  * SHADOW mode: decisions are computed but no orders are submitted.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import httpx

from config import Settings, settings

logger = logging.getLogger("BrokerBridge")


def _quantize(symbol: str, qty: float, allow_fractional: bool) -> float:
    """Whole shares for any short-capable/exit leg unless fractional is allowed."""
    if allow_fractional:
        return round(abs(qty), 4)
    return float(int(abs(qty)))


class AsyncAlpacaBridge:
    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        base_url: str = None,
        fee_bps: float = None,
        shadow_mode: bool = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key if secret_key is not None else os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = (base_url or settings.alpaca_base_url).rstrip("/")
        self.fee_bps = float(fee_bps if fee_bps is not None else settings.fee_bps)
        self.shadow_mode = bool(shadow_mode if shadow_mode is not None else settings.shadow_mode)
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        return bool(self.api_key and self.secret_key)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=10.0)
        return self._client

    async def close(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_active():
            return None
        try:
            client = await self._client_or_create()
            resp = await client.get(f"/v2/positions/{symbol.upper()}", headers=self.headers)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    async def get_all_positions(self) -> Dict[str, float]:
        """Map symbol -> signed qty from the physical account."""
        if not self.is_active():
            return {}
        try:
            client = await self._client_or_create()
            resp = await client.get("/v2/positions", headers=self.headers)
            if resp.status_code != 200:
                return {}
            out: Dict[str, float] = {}
            for pos in resp.json():
                try:
                    out[str(pos["symbol"]).upper()] = float(pos.get("qty", 0.0))
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    @staticmethod
    def _client_order_id(agent_id: str, symbol: str, side: str, window: str) -> str:
        raw = f"{agent_id}|{symbol.upper()}|{side.lower()}|{window}"
        return "eq-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        agent_id: str = "shared",
        window: str = "na",
        allow_fractional: bool = False,
        poll: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a market order idempotently. Returns the broker order dict, or a
        synthetic dict in shadow mode, or None when inactive / skipped.
        """
        side_norm = str(side).lower()
        if side_norm not in ("buy", "sell"):
            logger.warning(f"⚠️ Invalid side '{side}' for {symbol}; skipping.")
            return None

        final_qty = _quantize(symbol, qty, allow_fractional)
        if final_qty <= 0:
            return None

        client_order_id = self._client_order_id(agent_id, symbol, side_norm, window)

        if self.shadow_mode or not self.is_active():
            logger.info(
                f"👻 [SHADOW] {side_norm.upper()} {final_qty} {symbol} "
                f"(agent={agent_id}, id={client_order_id})"
            )
            return {
                "id": client_order_id,
                "status": "shadow",
                "symbol": symbol.upper(),
                "qty": str(final_qty),
                "side": side_norm,
            }

        payload = {
            "symbol": symbol.upper(),
            "qty": str(final_qty),
            "side": side_norm,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        async with self._lock:
            try:
                client = await self._client_or_create()
                resp = await client.post("/v2/orders", json=payload, headers=self.headers)
                if resp.status_code in (200, 201):
                    order = resp.json()
                    logger.info(
                        f"⚡ [BROKER] {side_norm.upper()} {final_qty} {symbol} "
                        f"id={order.get('id')} status={order.get('status')}"
                    )
                    if poll:
                        return await self.wait_for_fill(order.get("id"), order)
                    return order
                if resp.status_code == 422:
                    # Duplicate client_order_id: treat as already submitted (idempotent no-op)
                    body = resp.text
                    if "client_order_id" in body:
                        logger.info(f"♻️ [BROKER] Duplicate order suppressed for {symbol} ({client_order_id}).")
                        return {"id": client_order_id, "status": "duplicate", "symbol": symbol.upper()}
                logger.warning(
                    f"⚠️ [BROKER REJECTED] {side_norm.upper()} {final_qty} {symbol} "
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None
            except Exception as e:
                logger.error(f"❌ [BROKER ERROR] {side_norm.upper()} {final_qty} {symbol}: {e}")
                return None

    async def wait_for_fill(self, order_id: Optional[str], fallback: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Poll order status until filled or timeout."""
        if not order_id:
            return fallback
        deadline = asyncio.get_event_loop().time() + settings.order_poll_timeout_s
        last = fallback or {"id": order_id, "status": "unknown"}
        try:
            client = await self._client_or_create()
            while asyncio.get_event_loop().time() < deadline:
                resp = await client.get(f"/v2/orders/{order_id}", headers=self.headers)
                if resp.status_code == 200:
                    order = resp.json()
                    last = order
                    status = str(order.get("status", "")).lower()
                    if status in ("filled", "canceled", "expired", "rejected"):
                        return order
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"⚠️ Fill poll failed for {order_id}: {e}")
        return last

    async def reconcile_agent(
        self,
        agent_id: str,
        desired: Dict[str, float],
        window: str = "na",
        min_notional: float = 50.0,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Diff desired signed positions against physical positions and trade only
        the delta. Longs may be fractional; shorts/exit legs are whole-share.
        """
        prices = prices or {}
        physical = await self.get_all_positions()
        symbols = set(desired.keys()) | set(set(physical.keys()))
        submitted: Dict[str, Any] = {}
        for sym in symbols:
            want = float(desired.get(sym, 0.0) or 0.0)
            have = float(physical.get(sym, 0.0) or 0.0)
            delta = want - have
            px = float(prices.get(sym, 0.0) or 0.0)
            if px > 0 and abs(delta) * px < min_notional and not (have != 0 and want == 0):
                continue
            if abs(delta) < 1e-9:
                continue
            side = "buy" if delta > 0 else "sell"
            # Opening/increasing a position may be fractional for longs; shorts whole.
            allow_fractional = delta > 0 and want >= 0
            res = await self.submit_order(sym, abs(delta), side, agent_id, window, allow_fractional, poll=True)
            if res is not None:
                submitted[sym] = res
        return submitted


class BrokerRegistry:
    """Maps each agent to its own paper sub-account bridge, with a shared fallback."""

    def __init__(self, cfg: Settings = None):
        self.cfg = cfg or settings
        shared_key = os.getenv("ALPACA_API_KEY", "")
        shared_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._shared = AsyncAlpacaBridge(shared_key, shared_secret, self.cfg.alpaca_base_url)
        self._by_agent: Dict[str, AsyncAlpacaBridge] = {}
        for agent_id, creds in (self.cfg.alpaca_subaccounts or {}).items():
            key = creds.get("key") or shared_key
            secret = creds.get("secret") or shared_secret
            base = creds.get("base_url") or self.cfg.alpaca_base_url
            self._by_agent[agent_id] = AsyncAlpacaBridge(key, secret, base)

    def get(self, agent_id: str) -> AsyncAlpacaBridge:
        return self._by_agent.get(agent_id, self._shared)

    def has_dedicated(self, agent_id: str) -> bool:
        return agent_id in self._by_agent

    async def close_all(self):
        await self._shared.close()
        for b in self._by_agent.values():
            await b.close()
