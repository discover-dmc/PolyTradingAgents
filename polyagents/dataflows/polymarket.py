"""Polymarket API dataflow client.

Provides market data, orderbook depth, and resolution status for Polymarket
prediction markets. Used by the TradeSniper agent and analysts.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

POLYMARKET_CLOB_API = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"


def _get(url: str, params: dict | None = None, timeout: int = 10) -> Any:
    """GET with basic error handling."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Market metadata
# ---------------------------------------------------------------------------

def get_market(condition_id: str) -> dict:
    """Return full market metadata for a given condition_id (Polymarket market ID).

    Returns a dict with keys:
        question, description, end_date_iso, active, closed,
        tokens (list of {outcome, token_id}), volume, liquidity

    Note: the Gamma API does not support GET /markets/{conditionId} as a path
    parameter — it must be passed as a query param: GET /markets?conditionId=...
    """
    data = _get(f"{POLYMARKET_GAMMA_API}/markets", params={"conditionId": condition_id})
    markets = data if isinstance(data, list) else data.get("markets", [])
    if not markets:
        raise ValueError(f"Market not found for condition_id: {condition_id}")
    return markets[0]


def search_markets(query: str, limit: int = 20, active_only: bool = True) -> list[dict]:
    """Search open Polymarket markets by keyword.

    Returns list of market dicts sorted by volume descending.
    """
    params: dict[str, Any] = {"q": query, "limit": limit}
    if active_only:
        params["active"] = "true"
        params["closed"] = "false"
    data = _get(f"{POLYMARKET_GAMMA_API}/markets", params=params)
    markets = data if isinstance(data, list) else data.get("markets", [])
    return sorted(markets, key=lambda m: float(m.get("volume", 0)), reverse=True)


def get_active_markets(
    limit: int = 50,
    min_volume: float = 0.0,
    min_liquidity: float = 0.0,
) -> list[dict]:
    """Return active markets ordered by volume, optionally filtered by thresholds."""
    params: dict[str, Any] = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }
    data = _get(f"{POLYMARKET_GAMMA_API}/markets", params=params)
    markets = data if isinstance(data, list) else data.get("markets", [])

    results = []
    for m in markets:
        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol >= min_volume and liq >= min_liquidity:
            results.append(m)
    return results


# ---------------------------------------------------------------------------
# Orderbook / liquidity
# ---------------------------------------------------------------------------

def get_orderbook(token_id: str) -> dict:
    """Return the current CLOB orderbook for a token.

    Returns dict with 'bids' and 'asks', each a list of {price, size}.
    """
    data = _get(f"{POLYMARKET_CLOB_API}/book", params={"token_id": token_id})
    return data


def get_spread(token_id: str) -> Optional[float]:
    """Return current bid-ask spread as a fraction (0.0 – 1.0).

    Returns None if the orderbook is empty.
    """
    book = get_orderbook(token_id)
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return None
    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return None
    return (best_ask - best_bid) / mid


def get_best_prices(token_id: str) -> dict:
    """Return best bid, best ask, and mid price for a token.

    Returns dict: {best_bid, best_ask, mid, spread}
    """
    book = get_orderbook(token_id)
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    mid = ((best_bid + best_ask) / 2) if (best_bid is not None and best_ask is not None) else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
    }


def get_orderbook_depth(token_id: str, price_range: float = 0.05) -> dict:
    """Return total depth (USD) within price_range of mid price on each side.

    Args:
        token_id: CLOB token ID for the YES or NO side.
        price_range: +/- range from mid to count depth (default 5 cents).

    Returns dict: {bid_depth_usd, ask_depth_usd, total_depth_usd, mid_price}
    """
    book = get_orderbook(token_id)
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return {"bid_depth_usd": 0.0, "ask_depth_usd": 0.0, "total_depth_usd": 0.0, "mid_price": None}

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid = (best_bid + best_ask) / 2

    bid_depth = sum(
        float(b["price"]) * float(b["size"])
        for b in bids
        if float(b["price"]) >= mid - price_range
    )
    ask_depth = sum(
        float(a["price"]) * float(a["size"])
        for a in asks
        if float(a["price"]) <= mid + price_range
    )

    return {
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
        "total_depth_usd": bid_depth + ask_depth,
        "mid_price": mid,
    }


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def get_price_history(
    token_id: str,
    interval: str = "1d",
    limit: int = 30,
) -> list[dict]:
    """Return price history for a token.

    Args:
        token_id: CLOB token ID.
        interval: Candle size — '1m', '5m', '1h', '6h', '1d'.
        limit: Number of candles to return.

    Returns list of {t (timestamp), p (price)} dicts.
    """
    data = _get(
        f"{POLYMARKET_CLOB_API}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": limit},
    )
    return data.get("history", data) if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# Liquidity summary (used by TradeSniper)
# ---------------------------------------------------------------------------

def get_liquidity_summary(condition_id: str) -> dict:
    """Return a combined liquidity summary for a market.

    Fetches market metadata + YES token orderbook depth.
    Returns dict suitable for TradeSniper's filter logic:
        {
            condition_id, question, volume_24h, liquidity,
            yes_mid_price, yes_spread, yes_depth_usd,
            end_date, active, liquid (bool)
        }
    """
    from polyagents.dataflows.config import get_config
    config = get_config()
    min_volume = config.get("polymarket_min_volume", 1000.0)
    min_liquidity = config.get("polymarket_min_liquidity", 500.0)
    max_spread = config.get("polymarket_max_spread", 0.10)
    min_depth = config.get("polymarket_min_depth_usd", 200.0)

    try:
        market = get_market(condition_id)
    except Exception as e:
        logger.warning("Failed to fetch market %s: %s", condition_id, e)
        return {"condition_id": condition_id, "liquid": False, "error": str(e)}

    tokens = market.get("tokens", [])
    yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)

    yes_spread = None
    yes_depth = 0.0
    yes_mid = None

    if yes_token:
        try:
            prices = get_best_prices(yes_token["token_id"])
            yes_mid = prices["mid"]
            yes_spread = prices["spread"]
            depth_info = get_orderbook_depth(yes_token["token_id"])
            yes_depth = depth_info["total_depth_usd"]
        except Exception as e:
            logger.warning("Orderbook fetch failed for %s: %s", condition_id, e)

    volume = float(market.get("volume24hr", 0) or market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)

    # Polymarket's primary liquidity is AMM-based, not CLOB limit orders.
    # The CLOB orderbook is often near-empty even on well-traded markets.
    # Use 5% of AMM liquidity as a conservative depth proxy when CLOB is thin.
    effective_depth = max(yes_depth, liquidity * 0.05)

    liquid = (
        volume >= min_volume
        and liquidity >= min_liquidity
        and (yes_spread is None or yes_spread <= max_spread)
        and effective_depth >= min_depth
    )

    return {
        "condition_id": condition_id,
        "question": market.get("question", ""),
        "description": market.get("description", ""),
        "volume_24h": volume,
        "liquidity": liquidity,
        "yes_mid_price": yes_mid,
        "yes_spread": yes_spread,
        "yes_depth_usd": effective_depth,
        "end_date": market.get("end_date_iso", market.get("endDate", "")),
        "active": market.get("active", True),
        "liquid": liquid,
    }
