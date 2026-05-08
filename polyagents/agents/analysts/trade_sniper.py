"""TradeSniper agent — liquidity gate for PolyAgents.

Runs before the main analyst pipeline. Its only job is to evaluate whether
the target market has enough liquidity to be worth trading. If the market
fails the liquidity filter, the graph short-circuits to SKIP immediately
without invoking any analysts.

Liquidity thresholds are configurable in DEFAULT_CONFIG:
    polymarket_min_volume      (default $1,000 / 24h)
    polymarket_min_liquidity   (default $500 total)
    polymarket_max_spread      (default 10%)
    polymarket_min_depth_usd   (default $200 within ±5 cents of mid)
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from polyagents.dataflows.polymarket import get_liquidity_summary

logger = logging.getLogger(__name__)


def create_trade_sniper(llm):
    """Factory: returns the TradeSniper node function.

    The node writes ``liquidity_summary`` and ``liquidity_ok`` into state.
    Downstream conditional logic checks ``liquidity_ok`` to decide whether
    to fan out to analysts or route directly to the portfolio manager with
    a SKIP decision.
    """

    def trade_sniper_node(state):
        condition_id = state["condition_id"]
        logger.info("TradeSniper: scanning liquidity for %s", condition_id)

        summary = get_liquidity_summary(condition_id)
        liquid = summary.get("liquid", False)

        if not liquid:
            reason_parts = []
            if summary.get("volume_24h", 0) < 1000:
                reason_parts.append(
                    f"low 24h volume (${summary.get('volume_24h', 0):,.0f})"
                )
            if summary.get("liquidity", 0) < 500:
                reason_parts.append(
                    f"low liquidity (${summary.get('liquidity', 0):,.0f})"
                )
            spread = summary.get("yes_spread")
            if spread is not None and spread > 0.10:
                reason_parts.append(f"wide spread ({spread:.1%})")
            if summary.get("yes_depth_usd", 0) < 200:
                reason_parts.append(
                    f"thin orderbook (${summary.get('yes_depth_usd', 0):,.0f} within ±5¢)"
                )
            reason = "; ".join(reason_parts) if reason_parts else "liquidity thresholds not met"
            logger.info("TradeSniper: SKIP — %s", reason)
        else:
            logger.info(
                "TradeSniper: PASS — vol=$%.0f liq=$%.0f spread=%.1f%% depth=$%.0f",
                summary.get("volume_24h", 0),
                summary.get("liquidity", 0),
                (summary.get("yes_spread") or 0) * 100,
                summary.get("yes_depth_usd", 0),
            )

        return {
            "liquidity_summary": summary,
            "liquidity_ok": liquid,
        }

    return trade_sniper_node
