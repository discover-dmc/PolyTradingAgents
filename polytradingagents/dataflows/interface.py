from __future__ import annotations

from typing import Any, Annotated

try:
    import pandas as pd
    _DataFrame = pd.DataFrame
except ImportError:  # pragma: no cover
    _DataFrame = Any  # type: ignore[assignment,misc]

from .cache import CacheBackend, InMemoryCache, make_cache, make_key

# ---------------------------------------------------------------------------
# Session-scoped vendor call cache
#
# Identical tool calls within a single propagate() run (e.g. two analysts
# calling get_news() with the same query) are served from this cache instead
# of making a second HTTP round-trip.
#
# The cache is cleared at the start of every propagate() call via
# clear_session_cache().  The default backend is InMemoryCache (thread-safe,
# zero deps).  Switch to Redis by calling configure_session_cache() before
# the first propagate(), or by setting cache_backend="redis" in the config.
# ---------------------------------------------------------------------------

_session_cache: CacheBackend = InMemoryCache()


def configure_session_cache(backend: CacheBackend) -> None:
    """Replace the session cache backend.

    Call this once at startup before any ``propagate()`` run if you want a
    non-default backend (e.g. Redis)::

        from polytradingagents.dataflows.cache import make_cache
        from polytradingagents.dataflows.interface import configure_session_cache
        configure_session_cache(make_cache("redis", namespace="pta:session"))
    """
    global _session_cache
    _session_cache = backend


def clear_session_cache() -> None:
    """Reset the per-run vendor call cache.

    Called at the start of each ``propagate()`` run so stale results from a
    previous run are never served.
    """
    _session_cache.clear()

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Return the TOOLS_CATEGORIES key that owns *method*.

    Raises ``ValueError`` if the method is not registered in any category.
    """
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str | None = None) -> str:
    """Return the configured vendor name for a data category or specific tool.

    Tool-level configuration (``tool_vendors`` key in config) takes precedence
    over the category-level ``data_vendors`` mapping.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")


def route_to_vendor(method: str, *args: Any, **kwargs: Any) -> str | _DataFrame:
    """Route *method* to the appropriate vendor implementation with fallback.

    Tries vendors in priority order (primary first, then remaining available).
    ``AlphaVantageRateLimitError`` triggers an automatic fallback to the next
    vendor; all other exceptions propagate.

    Results are stored in the session cache so that identical calls within a
    single ``propagate()`` run (e.g. two analysts querying the same news for
    the same condition) are served from the cache rather than making a second
    HTTP request.  Call :func:`clear_session_cache` to reset between runs.

    Returns the vendor function's result — typically a formatted string or a
    ``pandas.DataFrame`` depending on the method.
    """
    cache_key = make_key(method, args)
    cached = _session_cache.get(cache_key)
    if cached is not None:
        return cached

    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    result = None
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            result = impl_func(*args, **kwargs)
            break
        except AlphaVantageRateLimitError:
            continue  # Only rate limits trigger fallback
    else:
        raise RuntimeError(f"No available vendor for '{method}'")

    ttl = get_config().get("session_cache_ttl")  # None for memory, int for Redis
    _session_cache.set(cache_key, result, ttl=ttl)
    return result