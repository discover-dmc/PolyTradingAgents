"""Canonical node name constants shared between graph setup and conditional routing.

Import these instead of typing the strings inline so that a rename in one
place propagates everywhere automatically, and route mismatches become
import errors rather than silent runtime failures.
"""

from __future__ import annotations


class NodeNames:
    """Static node name constants for the PolyTradingAgents graph."""

    # Liquidity gate (runs before analysts)
    TRADE_SNIPER = "Trade Sniper"

    # Analyst nodes (parallel fan-out — only reached if TradeSniper passes)
    NEWS_ANALYST = "News Analyst"
    BASE_RATE_ANALYST = "Base Rate Analyst"
    CROWD_FORECAST_ANALYST = "Crowd Forecast Analyst"
    DATA_ANALYST = "Data Analyst"

    # Research debate loop (YES vs NO)
    BULL_RESEARCHER = "Yes Researcher"
    BEAR_RESEARCHER = "No Researcher"
    RESEARCH_MANAGER = "Research Manager"

    # Position sizing
    TRADER = "Position Sizer"

    # Risk debate loop
    AGGRESSIVE_ANALYST = "Aggressive Analyst"
    NEUTRAL_ANALYST = "Neutral Analyst"
    CONSERVATIVE_ANALYST = "Conservative Analyst"

    # Final decision
    PORTFOLIO_MANAGER = "Portfolio Manager"


# Analyst key → node name — used for fan-out wiring and validation.
# Keep in sync with the NodeNames analyst constants above.
ANALYST_NODE_NAMES: dict[str, str] = {
    "news": NodeNames.NEWS_ANALYST,
    "base_rate": NodeNames.BASE_RATE_ANALYST,
    "crowd_forecast": NodeNames.CROWD_FORECAST_ANALYST,
    "data": NodeNames.DATA_ANALYST,
}
