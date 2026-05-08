"""CLI input helpers for PolyAgents — Polymarket-specific prompts."""
import re
from typing import List, Optional, Tuple  # noqa: F401 – Tuple used in new market-picker helpers

import questionary
from rich.console import Console
from rich.panel import Panel

from cli.models import AnalystType
from polyagents.llm_clients.model_catalog import get_model_options

console = Console()

# ---------------------------------------------------------------------------
# Analyst definitions
# ---------------------------------------------------------------------------

ANALYST_ORDER = [
    ("News Analyst         — recent news, events, catalysts", AnalystType.NEWS),
    ("Base Rate Analyst    — historical base rates, reference classes", AnalystType.BASE_RATE),
    ("Crowd Forecast       — prediction market consensus, forecaster signals", AnalystType.CROWD_FORECAST),
    ("Data Analyst         — quantitative data, statistics, indicators", AnalystType.DATA),
]

_QSTYLE = questionary.Style([
    ("checkbox-selected", "fg:green"),
    ("selected", "fg:green noinherit"),
    ("highlighted", "noinherit"),
    ("pointer", "noinherit"),
])

_SELECT_STYLE = questionary.Style([
    ("selected", "fg:cyan noinherit"),
    ("highlighted", "fg:cyan noinherit"),
    ("pointer", "fg:cyan noinherit"),
])


# ---------------------------------------------------------------------------
# Polymarket-specific inputs
# ---------------------------------------------------------------------------

def pick_market() -> Tuple[str, dict]:
    """Top-level market selection — returns (condition_id, market_info_dict).

    Offers three paths:
      1. Browse top liquid markets fetched from Polymarket
      2. Search by keyword
      3. Enter a condition ID manually
    """
    MODE_OPTIONS = [
        ("Browse top liquid markets  — scan Polymarket for tradeable markets", "browse"),
        ("Search by keyword          — find markets by topic", "search"),
        ("Enter condition ID         — paste a known 0x... ID directly", "manual"),
    ]
    mode = questionary.select(
        "How do you want to find a market?",
        choices=[questionary.Choice(d, value=v) for d, v in MODE_OPTIONS],
        instruction="\n  Arrow keys  |  Enter to select",
        style=_SELECT_STYLE,
    ).ask()

    if mode is None:
        console.print("\n[red]No mode selected. Exiting...[/red]")
        raise SystemExit(1)

    if mode == "browse":
        return _browse_markets()
    if mode == "search":
        return _search_markets()
    return _manual_condition_id()


def _browse_markets() -> Tuple[str, dict]:
    """Fetch and display top liquid markets; user picks one."""
    from polyagents.dataflows.polymarket import get_active_markets

    console.print("\n[cyan]Fetching top liquid markets from Polymarket...[/cyan]")
    try:
        markets = get_active_markets(limit=50, min_volume=500.0, min_liquidity=200.0)
    except Exception as e:
        console.print(f"[yellow]Could not fetch markets: {e}[/yellow]")
        return _manual_condition_id()

    if not markets:
        console.print("[yellow]No liquid markets found. Try entering a condition ID manually.[/yellow]")
        return _manual_condition_id()

    return _pick_from_market_list(markets[:30])


def _search_markets() -> Tuple[str, dict]:
    """Keyword search → pick from results."""
    from polyagents.dataflows.polymarket import search_markets

    query = questionary.text(
        "Search Polymarket markets:",
        validate=lambda x: len(x.strip()) >= 2 or "Enter at least 2 characters.",
        style=_QSTYLE,
    ).ask()

    if not query:
        raise SystemExit(1)

    console.print(f"\n[cyan]Searching for '{query}'...[/cyan]")
    try:
        markets = search_markets(query.strip(), limit=30)
    except Exception as e:
        console.print(f"[yellow]Search failed: {e}[/yellow]")
        return _manual_condition_id()

    if not markets:
        console.print(f"[yellow]No markets found for '{query}'. Try a different term or enter an ID manually.[/yellow]")
        return _manual_condition_id()

    return _pick_from_market_list(markets)


def _pick_from_market_list(markets: list) -> Tuple[str, dict]:
    """Display a market list as a questionary select; return (condition_id, info)."""
    def _fmt(m: dict) -> str:
        q = (m.get("question") or m.get("title") or "Unknown")[:70]
        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        return f"{q:<72}  vol ${vol:>8,.0f}  liq ${liq:>8,.0f}"

    choices = [
        questionary.Choice(_fmt(m), value=m)
        for m in markets
        if m.get("conditionId") or m.get("condition_id")
    ]

    if not choices:
        console.print("[yellow]No markets with valid condition IDs. Falling back to manual entry.[/yellow]")
        return _manual_condition_id()

    choices.append(questionary.Choice("↩  Enter condition ID manually", value=None))

    picked = questionary.select(
        "Select a market:",
        choices=choices,
        instruction="\n  Arrow keys  |  Enter to select",
        style=_SELECT_STYLE,
    ).ask()

    if picked is None:
        return _manual_condition_id()

    cid = (picked.get("conditionId") or picked.get("condition_id", "")).lower()
    info = {
        "question": picked.get("question") or picked.get("title") or "",
        "current_probability": float(picked.get("outcomePrices", [0.5])[0])
            if isinstance(picked.get("outcomePrices"), list)
            else 0.5,
        "end_date": picked.get("end_date_iso") or picked.get("endDate") or "",
        "liquid": True,
    }
    # Try to get a live mid price from the liquidity summary
    live = fetch_market_info(cid)
    if live:
        info.update(live)

    return cid, info


def _manual_condition_id() -> Tuple[str, dict]:
    """Prompt for a raw condition ID, then fetch info."""
    cid = questionary.text(
        "Enter the Polymarket condition ID (0x...):",
        validate=lambda x: (
            bool(re.match(r"^0x[0-9a-fA-F]+$", x.strip()))
            or "Must be a hex string starting with 0x — find it in the market URL."
        ),
        style=_QSTYLE,
    ).ask()

    if not cid:
        console.print("\n[red]No condition ID provided. Exiting...[/red]")
        raise SystemExit(1)

    cid = cid.strip().lower()
    info = fetch_market_info(cid) or {}
    return cid, info


def fetch_market_info(condition_id: str) -> Optional[dict]:
    """Try to fetch market question and current probability from Polymarket.

    Returns a dict with 'question' and 'current_probability', or None on failure.
    """
    try:
        from polyagents.dataflows.polymarket import get_liquidity_summary
        summary = get_liquidity_summary(condition_id)
        return {
            "question": summary.get("question", ""),
            "current_probability": summary.get("yes_mid_price", 0.5),
            "end_date": summary.get("end_date", ""),
            "liquid": summary.get("liquid", False),
        }
    except Exception:
        return None


def get_market_question(prefill: str = "") -> str:
    """Prompt for the market resolution question."""
    question = questionary.text(
        "Market resolution question:",
        default=prefill,
        validate=lambda x: len(x.strip()) > 5 or "Please enter the full resolution question.",
        style=_QSTYLE,
    ).ask()

    if not question:
        console.print("\n[red]No question provided. Exiting...[/red]")
        raise SystemExit(1)

    return question.strip()


def get_current_probability(prefill: float = 0.5) -> float:
    """Prompt for the current YES mid-price (0.01 – 0.99)."""
    def validate(x):
        try:
            v = float(x)
            return 0.01 <= v <= 0.99 or "Enter a probability between 0.01 and 0.99"
        except ValueError:
            return "Enter a number, e.g. 0.45"

    raw = questionary.text(
        "Current YES probability (0–1):",
        default=str(round(prefill, 4)),
        validate=validate,
        style=_QSTYLE,
    ).ask()

    if raw is None:
        raise SystemExit(1)

    return float(raw.strip())


def get_analysis_date() -> str:
    """Prompt for the analysis date (YYYY-MM-DD). Defaults to today."""
    import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Analysis date (YYYY-MM-DD):",
        default=datetime.datetime.now().strftime("%Y-%m-%d"),
        validate=lambda x: validate_date(x.strip()) or "Use YYYY-MM-DD format.",
        style=_QSTYLE,
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        raise SystemExit(1)

    return date.strip()


# ---------------------------------------------------------------------------
# Analyst / depth / LLM selection — mostly unchanged in structure
# ---------------------------------------------------------------------------

def select_analysts() -> List[AnalystType]:
    """Interactive checkbox to select analyst agents."""
    choices = questionary.checkbox(
        "Select analyst agents (Space to toggle, Enter to confirm):",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n  Space = toggle  |  a = all  |  Enter = confirm",
        validate=lambda x: len(x) > 0 or "Select at least one analyst.",
        style=_QSTYLE,
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        raise SystemExit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth (controls debate rounds)."""
    DEPTH_OPTIONS = [
        ("Shallow  — quick, 1 debate round   (cheapest)", 1),
        ("Medium   — balanced, 3 rounds      (recommended)", 3),
        ("Deep     — thorough, 5 rounds      (most thorough)", 5),
    ]
    choice = questionary.select(
        "Research depth:",
        choices=[questionary.Choice(d, value=v) for d, v in DEPTH_OPTIONS],
        instruction="\n  Arrow keys to navigate  |  Enter to select",
        style=_SELECT_STYLE,
    ).ask()

    if choice is None:
        console.print("\n[red]No depth selected. Exiting...[/red]")
        raise SystemExit(1)

    return choice


def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its base URL."""
    PROVIDERS = [
        ("OpenAI",       "openai",    "https://api.openai.com/v1"),
        ("Anthropic",    "anthropic", "https://api.anthropic.com/"),
        ("Google",       "google",    None),
        ("xAI",          "xai",       "https://api.x.ai/v1"),
        ("DeepSeek",     "deepseek",  "https://api.deepseek.com"),
        ("OpenRouter",   "openrouter","https://openrouter.ai/api/v1"),
        ("Ollama",       "ollama",    "http://localhost:11434/v1"),
        ("Azure OpenAI", "azure",     None),
        ("Qwen",         "qwen",      "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("GLM",          "glm",       "https://open.bigmodel.cn/api/paas/v4/"),
    ]
    choice = questionary.select(
        "LLM provider:",
        choices=[questionary.Choice(d, value=(k, u)) for d, k, u in PROVIDERS],
        instruction="\n  Arrow keys  |  Enter to select",
        style=_SELECT_STYLE,
    ).ask()

    if choice is None:
        console.print("\n[red]No provider selected. Exiting...[/red]")
        raise SystemExit(1)

    return choice


def _prompt_custom_model_id() -> str:
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
    ).ask().strip()


def _select_model(provider: str, mode: str) -> str:
    if provider.lower() == "openrouter":
        return _select_openrouter_model()
    if provider.lower() == "azure":
        return questionary.text(
            f"Azure deployment name ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Enter a deployment name.",
        ).ask().strip()

    choice = questionary.select(
        f"{mode.title()}-thinking model:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
        instruction="\n  Arrow keys  |  Enter to select",
        style=_SELECT_STYLE,
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} model selected. Exiting...[/red]")
        raise SystemExit(1)

    return _prompt_custom_model_id() if choice == "custom" else choice


def _select_openrouter_model() -> str:
    try:
        import requests
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        choices = [questionary.Choice(m.get("name") or m["id"], value=m["id"]) for m in models[:8]]
        choices.append(questionary.Choice("Custom model ID", value="custom"))
        choice = questionary.select("OpenRouter model:", choices=choices, style=_SELECT_STYLE).ask()
        if choice == "custom" or choice is None:
            return _prompt_custom_model_id()
        return choice
    except Exception as e:
        console.print(f"[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return _prompt_custom_model_id()


def select_shallow_thinking_agent(provider: str) -> str:
    return _select_model(provider, "quick")


def select_deep_thinking_agent(provider: str) -> str:
    return _select_model(provider, "deep")


def ask_openai_reasoning_effort() -> str:
    return questionary.select(
        "OpenAI reasoning effort:",
        choices=[
            questionary.Choice("Medium (default)", "medium"),
            questionary.Choice("High  (more thorough)", "high"),
            questionary.Choice("Low   (faster/cheaper)", "low"),
        ],
        style=_SELECT_STYLE,
    ).ask()


def ask_anthropic_effort() -> str | None:
    return questionary.select(
        "Claude effort level:",
        choices=[
            questionary.Choice("High   (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low    (faster/cheaper)", "low"),
        ],
        style=_SELECT_STYLE,
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    return questionary.select(
        "Gemini thinking mode:",
        choices=[
            questionary.Choice("Enable thinking (recommended)", "high"),
            questionary.Choice("Minimal / disable thinking", "minimal"),
        ],
        style=_SELECT_STYLE,
    ).ask()


def ask_output_language() -> str:
    choice = questionary.select(
        "Output language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese  (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean   (한국어)", "Korean"),
            questionary.Choice("Spanish  (Español)", "Spanish"),
            questionary.Choice("French   (Français)", "French"),
            questionary.Choice("German   (Deutsch)", "German"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("Arabic   (العربية)", "Arabic"),
            questionary.Choice("Russian  (Русский)", "Russian"),
            questionary.Choice("Hindi    (हिन्दी)", "Hindi"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=_SELECT_STYLE,
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Language name (e.g. Turkish, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice


def format_tool_args(args, max_length: int = 80) -> str:
    result = str(args)
    return result[:max_length - 3] + "..." if len(result) > max_length else result
