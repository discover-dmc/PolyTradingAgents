"""
PolyAgents — example script.

Run the CLI for an interactive session:

    uv run polyagents

Or use the graph directly (programmatic):
"""

from polyagents.graph.trading_graph import PolyAgentsGraph
from polyagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 1

# ── Graph ─────────────────────────────────────────────────────────────────────
ta = PolyAgentsGraph(debug=True, config=config)

# ── Run analysis on a Polymarket market ───────────────────────────────────────
# condition_id: find it in the market URL, e.g.
#   polymarket.com/event/…?tid=<condition_id>
CONDITION_ID = "0x1234abcd"          # replace with a real condition ID
MARKET_QUESTION = "Will X happen before Y?"
CURRENT_PROBABILITY = 0.45           # current YES mid-price from the order book
TRADE_DATE = "2025-06-01"

_, decision = ta.propagate(
    condition_id=CONDITION_ID,
    trade_date=TRADE_DATE,
    market_question=MARKET_QUESTION,
    current_probability=CURRENT_PROBABILITY,
)
print(decision)
