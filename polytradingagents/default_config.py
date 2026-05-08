import os

_POLYTRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".polytradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("POLYTRADINGAGENTS_RESULTS_DIR", os.path.join(_POLYTRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("POLYTRADINGAGENTS_CACHE_DIR", os.path.join(_POLYTRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("POLYTRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_POLYTRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries.
    "memory_log_max_entries": None,

    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    # Retries on transient LLM errors
    "llm_max_retries": 3,
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,

    # Checkpoint/resume
    "checkpoint_enabled": False,

    # Output language
    "output_language": "English",

    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # ---------------------------------------------------------------------------
    # Polymarket-specific settings
    # ---------------------------------------------------------------------------

    # TradeSniper liquidity thresholds — markets failing any threshold are SKIP'd
    "polymarket_min_volume": 1000.0,        # Minimum 24h volume in USD
    "polymarket_min_liquidity": 500.0,      # Minimum total liquidity in USD
    "polymarket_max_spread": 0.10,          # Maximum bid-ask spread (10%)
    "polymarket_min_depth_usd": 200.0,      # Min orderbook depth within ±5¢ of mid

    # Kelly criterion cap — never risk more than this fraction of bankroll per trade
    "kelly_cap": 0.25,

    # Minimum edge required to take a position (skip if |estimated_prob - market_prob| < this)
    "min_edge": 0.03,

    # Data vendor configuration
    "data_vendors": {
        "news_data": "yfinance",   # Fallback news source
    },
    "tool_vendors": {},
}
