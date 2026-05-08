<p align="center">
  <img src="assets/logo.png" alt="PolyAgents" style="width: 60%; height: auto;">
</p>

**Multi-agent LLM framework for Polymarket prediction market trading**

Built on [LangGraph](https://github.com/langchain-ai/langgraph), PolyAgents runs a parallel analyst debate and produces a structured YES/NO/SKIP decision with Kelly-sized position sizing — all without wasting tokens on illiquid markets.

---

## Architecture

```
START
  │
  ▼
Trade Sniper ─── illiquid ──────────────────────────────────┐
  │ (liquid)                                                 │
  ├──► News Analyst ──────────────────────────────────────┐  │
  ├──► Base Rate Analyst ─────────────────── (parallel)  │  │
  ├──► Crowd Forecast Analyst ─────────────── fan-out    │  │
  └──► Data Analyst ──────────────────────────────────────┘  │
                                                             │
  Yes Researcher ◄──────────────── (fan-in from analysts)   │
  No Researcher  ◄─────────────────────────────────────────  │
       │                                                     │
       ▼                                                     │
  Research Manager                                           │
       │                                                     │
       ▼                                                     │
  Position Sizer (Trader)                                    │
       │                                                     │
       ▼                                                     │
  Portfolio Manager ◄──────────────── SKIP fast-path ───────┘
       │
       ▼
  PositionDecision (YES / NO / SKIP + Kelly fraction)
```

**TradeSniper** runs first. If the market fails liquidity thresholds (volume, spread, depth) the graph short-circuits directly to the Portfolio Manager which returns `SKIP` with no LLM calls wasted on illiquid markets.

---

## Analysts

| Key | Agent | What it researches |
|-----|-------|--------------------|
| `news` | News Analyst | Recent news, events, catalysts |
| `base_rate` | Base Rate Analyst | Historical base rates, reference classes |
| `crowd_forecast` | Crowd Forecast Analyst | Prediction market consensus, forecaster signals |
| `data` | Data Analyst | Quantitative data, statistics, economic indicators |

---

## PositionDecision schema

Every run produces a structured decision:

```python
class PositionDecision(BaseModel):
    direction: Literal["YES", "NO", "SKIP"]
    estimated_probability: float          # agent's calibrated estimate 0-1
    market_probability: float             # current market mid-price 0-1
    edge: float                           # estimated_probability - market_probability
    kelly_fraction: float                 # suggested Kelly position size (capped)
    confidence: Literal["High", "Medium", "Low"]
    reasoning: str
    catalyst: Optional[str] = None
    resolution_date: Optional[str] = None
```

---

## CLI

The fastest way to run PolyAgents is the interactive terminal UI.

### 1. Install

```bash
git clone https://github.com/discover-dmc/PolyAgents.git
cd PolyAgents
uv sync
```

### 2. Add API keys

```bash
cp .env.example .env   # then fill in your keys
```

At minimum you need one LLM provider key:

```env
OPENAI_API_KEY=sk-...        # OpenAI
ANTHROPIC_API_KEY=sk-ant-... # Anthropic / Claude
GOOGLE_API_KEY=...           # Gemini
```

### 3. Launch

```bash
uv run polyagents
```

The wizard walks you through everything:

1. **Market selection** — three options:
   - **Browse top liquid markets** — fetches the most active markets from Polymarket and shows them in a list, filtered by liquidity thresholds so you only see tradeable markets
   - **Search by keyword** — type a topic (e.g. "Trump", "Fed rate", "Bitcoin") and pick from results
   - **Enter condition ID** — paste a known `0x…` ID directly (power user path)
2. **Confirm details** — question and current YES probability are pre-filled from the Polymarket order book; edit if needed
3. **Analysis date** — defaults to today
4. **Analysts** — pick any combination of News, Base Rate, Crowd Forecast, Data
5. **Research depth** — Shallow (1 round) / Medium (3) / Deep (5)
6. **LLM provider & models** — choose from OpenAI, Anthropic, Google, xAI, DeepSeek, OpenRouter, Ollama, …
7. **Output language** — English, Chinese, Japanese, Spanish, … or custom

A live dashboard streams results as each analyst completes, then shows the final YES / NO / SKIP decision with Kelly fraction.

---

## Quickstart

```python
from polyagents import PolyAgentsGraph

graph = PolyAgentsGraph(
    selected_analysts=["news", "base_rate", "crowd_forecast", "data"],
)

final_state, signal = graph.propagate(
    condition_id="0x...",           # Polymarket condition ID
    trade_date="2026-01-15",
    market_question="Will X happen by Y?",
    current_probability=0.42,
)

print(signal)           # "YES" / "NO" / "SKIP"
print(final_state["final_trade_decision"])
```

### Environment variables

```bash
# Required: at least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: override defaults
POLYAGENTS_RESULTS_DIR=~/results
POLYAGENTS_CACHE_DIR=~/cache
```

---

## Configuration

Key config options (passed to `PolyAgentsGraph` or via `DEFAULT_CONFIG`):

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `"openai"` | LLM backend |
| `deep_think_llm` | `"o4-mini"` | Model for debate/analysis nodes |
| `quick_thinking_llm` | `"gpt-4.1-mini"` | Model for fast utility nodes |
| `polymarket_min_volume` | `1000` | Min 24h volume USD |
| `polymarket_min_liquidity` | `500` | Min total liquidity USD |
| `polymarket_max_spread` | `0.08` | Max bid-ask spread (8%) |
| `polymarket_min_depth_usd` | `200` | Min depth at touch USD |
| `kelly_cap` | `0.25` | Maximum Kelly fraction |
| `min_edge` | `0.03` | Minimum edge to recommend position |
| `memory_log_path` | `~/.polyagents/memory.md` | Path for trade memory log |
| `checkpoint_enabled` | `True` | LangGraph checkpoint resumption |

---

## Analyst selection

Run only the analysts you need:

```python
# Lightweight: just news + base rates
graph = PolyAgentsGraph(selected_analysts=["news", "base_rate"])

# Full ensemble
graph = PolyAgentsGraph(
    selected_analysts=["news", "base_rate", "crowd_forecast", "data"]
)
```

---

## Memory & calibration

After each run, the decision is stored in the memory log as a **pending** entry. When the market resolves, call `graph.propagate(...)` again — it will auto-resolve pending entries, score them (`+1.0` correct / `-1.0` wrong), and generate LLM reflections that are injected as `past_context` in future Portfolio Manager calls.

```python
# Past context is automatically loaded and injected — no manual steps needed
final_state, signal = graph.propagate(condition_id="0x...", ...)
```

---

## Dataflow API

| Function | Module | Description |
|----------|--------|-------------|
| `get_liquidity_summary(condition_id)` | `dataflows.polymarket` | TradeSniper liquidity check |
| `get_market(condition_id)` | `dataflows.polymarket` | Full market data including resolution |
| `get_news(query, days)` | `dataflows.bing_news` | News search via Bing |
| `get_finnhub_news(ticker, days)` | `dataflows.finnhub_utils` | Finnhub news (can point at any topic) |

---

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -q

# Run only smoke tests (fast graph integration)
uv run pytest tests/ -m smoke -q

# Run only unit tests
uv run pytest tests/ -m unit -q
```

---

## Repo structure

```
polyagents/
├── agents/
│   ├── analysts/
│   │   ├── trade_sniper.py       # Liquidity gate
│   │   ├── news_analyst.py
│   │   ├── base_rate_analyst.py
│   │   ├── crowd_forecast_analyst.py
│   │   └── data_analyst.py
│   ├── managers/
│   │   ├── portfolio_manager.py  # Final YES/NO/SKIP decision
│   │   └── research_manager.py
│   ├── prompts/                  # YAML prompt templates
│   └── schemas.py                # PositionDecision, ResearchPlan, …
├── dataflows/
│   ├── polymarket.py             # Polymarket CLOB + Gamma API client
│   └── interface.py              # Cached data fetchers
├── graph/
│   ├── trading_graph.py          # PolyAgentsGraph
│   ├── setup.py                  # LangGraph wiring
│   ├── signal_processing.py      # Deterministic YES/NO/SKIP extraction
│   └── conditional_logic.py      # Liquidity router
└── default_config.py
```

---

## Credits

Adapted from [TradingAgents](https://github.com/TauricResearch/TradingAgents) (TauricResearch). Reoriented for binary prediction markets on Polymarket.
