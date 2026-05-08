"""
polyagents scan — autonomous market scanner.

Fetches liquid Polymarket markets, runs the full multi-agent pipeline on each,
and produces a portfolio-level summary table of YES / NO / SKIP decisions.

Usage
-----
    uv run polyagents scan                        # defaults: top 10 markets, all analysts
    uv run polyagents scan --limit 25             # scan more markets
    uv run polyagents scan --keyword "Trump"      # restrict to a topic
    uv run polyagents scan --hitl                 # pause before each analysis for approval
    uv run polyagents scan --depth 3              # 3 debate rounds instead of 1
    uv run polyagents scan --min-edge 0.05        # only flag decisions with >5% edge

LLM config is read from environment variables if present, otherwise a minimal
one-time prompt runs at startup (provider → quick model → deep model).

    POLYAGENTS_PROVIDER=openai
    POLYAGENTS_QUICK_MODEL=gpt-4o-mini
    POLYAGENTS_DEEP_MODEL=gpt-4o
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()
load_dotenv(".env.enterprise", override=False)

console = Console()

# ---------------------------------------------------------------------------
# LLM config helpers
# ---------------------------------------------------------------------------

def _llm_config_from_env() -> dict | None:
    """Return LLM config from env vars, or None if any required key is missing."""
    provider = os.environ.get("POLYAGENTS_PROVIDER", "").strip()
    quick    = os.environ.get("POLYAGENTS_QUICK_MODEL", "").strip()
    deep     = os.environ.get("POLYAGENTS_DEEP_MODEL", "").strip()
    if provider and quick and deep:
        return {"provider": provider, "quick": quick, "deep": deep,
                "backend_url": os.environ.get("POLYAGENTS_BACKEND_URL")}
    return None


def _llm_config_interactive() -> dict:
    """Minimal interactive prompt: provider → quick model → deep model."""
    from cli.utils import (
        select_llm_provider,
        select_shallow_thinking_agent,
        select_deep_thinking_agent,
        ask_openai_reasoning_effort,
        ask_anthropic_effort,
        ask_gemini_thinking_config,
    )
    console.print(Panel(
        "[bold]LLM Configuration[/bold]\n"
        "[dim]Set POLYAGENTS_PROVIDER / POLYAGENTS_QUICK_MODEL / POLYAGENTS_DEEP_MODEL "
        "to skip this prompt on future runs.[/dim]",
        border_style="blue", padding=(1, 2),
    ))
    provider, backend_url = select_llm_provider()
    quick = select_shallow_thinking_agent(provider)
    deep  = select_deep_thinking_agent(provider)

    extra: dict = {}
    p = provider.lower()
    if p == "openai":
        extra["openai_reasoning_effort"] = ask_openai_reasoning_effort()
    elif p == "anthropic":
        extra["anthropic_effort"] = ask_anthropic_effort()
    elif p == "google":
        extra["google_thinking_level"] = ask_gemini_thinking_config()

    return {"provider": provider, "quick": quick, "deep": deep,
            "backend_url": backend_url, **extra}


# ---------------------------------------------------------------------------
# Market fetching helpers
# ---------------------------------------------------------------------------

def _prescreen_one(market: dict) -> dict | None:
    """Run get_liquidity_summary() for a single market.

    Returns the market dict enriched with live mid-price and liquidity details,
    or None if the market fails the liquidity gate.
    """
    from polyagents.dataflows.polymarket import get_liquidity_summary
    cid = (market.get("conditionId") or market.get("condition_id") or "").strip().lower()
    if not cid:
        return None
    try:
        summary = get_liquidity_summary(cid)
        if not summary.get("liquid", False):
            return None
        # Enrich the market dict with live data from the summary
        market["_cid"]       = cid
        market["_mid_price"] = summary.get("yes_mid_price") or _market_mid_price(market)
        market["_summary"]   = summary
        return market
    except Exception:
        return None


def _fetch_markets(
    keyword: str | None,
    limit: int,
    min_volume: float,
    min_liquidity: float,
) -> list[dict]:
    """Return a list of pre-screened liquid markets from Polymarket.

    Steps:
    1. Fetch a broad pool (up to 100) from the Gamma API.
    2. Run get_liquidity_summary() on each in parallel — this applies the full
       TradeSniper filter (volume + liquidity + spread + effective depth) so we
       only pass markets to the graph that will actually be analysed.
    3. Return up to `limit` markets, sorted by 24h volume descending.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from polyagents.dataflows.polymarket import get_active_markets, search_markets

    api_fetch = max(100, limit * 3)

    if keyword:
        console.print(f"[cyan]Searching markets for '[bold]{keyword}[/bold]'…[/cyan]")
        raw = search_markets(keyword, limit=api_fetch)
    else:
        console.print(f"[cyan]Fetching markets from Polymarket…[/cyan]")
        raw = get_active_markets(
            limit=api_fetch,
            min_volume=min_volume,
            min_liquidity=min_liquidity,
        )

    # Deduplicate by conditionId before making API calls
    seen: set[str] = set()
    candidates: list[dict] = []
    for m in raw:
        cid = (m.get("conditionId") or m.get("condition_id") or "").strip().lower()
        if cid and cid not in seen:
            seen.add(cid)
            candidates.append(m)

    console.print(f"[dim]Pre-screening {len(candidates)} candidates for liquidity…[/dim]")

    liquid: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_prescreen_one, m): m for m in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                liquid.append(result)

    # Sort by 24h volume descending so highest-activity markets come first
    liquid.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)

    n = len(liquid)
    console.print(
        f"[green]{n} liquid market(s) found[/green] — "
        f"[dim]analysing up to {min(n, limit)}[/dim]"
    )
    return liquid[:limit]


def _market_mid_price(market: dict) -> float:
    """Best-effort YES mid-price from market data."""
    prices = market.get("outcomePrices")
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (ValueError, TypeError):
            pass
    if isinstance(prices, str):
        try:
            parsed = json.loads(prices)
            if isinstance(parsed, list) and parsed:
                return float(parsed[0])
        except Exception:
            pass
    return 0.5


# ---------------------------------------------------------------------------
# Single-market analysis (headless — no Live dashboard)
# ---------------------------------------------------------------------------

def _analyze_market(
    condition_id: str,
    question: str,
    current_probability: float,
    trade_date: str,
    config: dict,
    selected_keys: list[str],
) -> dict:
    """Run the graph on one market and return a result dict."""
    from polyagents.graph.trading_graph import PolyAgentsGraph

    graph = PolyAgentsGraph(
        selected_analysts=selected_keys,
        config=config,
    )

    init_state = graph.propagator.create_initial_state(
        condition_id=condition_id,
        trade_date=trade_date,
        market_question=question,
        current_probability=current_probability,
    )
    args = graph.propagator.get_graph_args()

    trace = []
    for chunk in graph.graph.stream(init_state, **args):
        trace.append(chunk)

    final_state = trace[-1] if trace else {}
    signal = graph.signal_processor.process_signal(
        final_state.get("final_trade_decision", "")
    )

    # Extract decision details from the rendered PositionDecision string.
    # render_position_decision() formats values as percentages, e.g.:
    #   **Edge**: +7.3%
    #   **Kelly Fraction**: 15.0%
    #   **Estimated Probability**: 65.0%
    decision_text = final_state.get("final_trade_decision", "")
    edge = None
    kelly = None
    estimated_prob = None
    market_prob_override = None
    try:
        import re

        def _parse_pct_field(line: str) -> float | None:
            """Parse a value that may be formatted as '7.3%', '+7.3%', or '0.073'."""
            # Prefer explicit percentage notation: captures the sign too
            m = re.search(r"([+-]?\d+\.?\d*)\s*%", line)
            if m:
                val = float(m.group(1)) / 100.0
                return val if -1.0 <= val <= 1.0 else None
            # Fallback: plain decimal already in [0, 1]
            m = re.search(r"([+-]?\d*\.?\d+)", line)
            if m:
                val = float(m.group(1))
                return val if -1.0 <= val <= 1.0 else None
            return None

        for line in (decision_text or "").splitlines():
            ll = line.lower()
            if ("estimated probability" in ll or "estimated_probability" in ll) and estimated_prob is None:
                estimated_prob = _parse_pct_field(line)
            elif "market probability" in ll and market_prob_override is None:
                market_prob_override = _parse_pct_field(line)
            elif "kelly" in ll and kelly is None:
                kelly = _parse_pct_field(line)
            elif "**edge**" in ll and edge is None:
                # Match specifically "**Edge**:" to avoid false hits on other lines
                edge = _parse_pct_field(line)
    except Exception:
        pass

    return {
        "condition_id": condition_id,
        "question":     question,
        "market_prob":  current_probability,
        "estimated_prob": estimated_prob,
        "signal":       signal,
        "edge":         edge,
        "kelly":        kelly,
        "final_state":  final_state,
        "decision_text": decision_text,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SIG_COLOUR = {"YES": "bold green", "NO": "bold red", "SKIP": "dim yellow"}


def _signal_text(signal: str) -> Text:
    t = Text(signal)
    t.stylize(_SIG_COLOUR.get(signal, "white"))
    return t


def _print_market_header(idx: int, total: int, market: dict) -> None:
    q = (market.get("question") or market.get("title") or "Unknown")
    vol = float(market.get("volume", 0) or 0)
    liq = float(market.get("liquidity", 0) or 0)
    console.print(Rule(
        f"[cyan]Market {idx}/{total}[/cyan]  [bold]{q[:70]}[/bold]",
        style="dim"
    ))
    console.print(
        f"  [dim]vol ${vol:,.0f}  liq ${liq:,.0f}  "
        f"cid {(market.get('conditionId') or '')[:20]}…[/dim]"
    )


def _print_result(result: dict) -> None:
    sig = result["signal"]
    colour = _SIG_COLOUR.get(sig, "white")
    parts = [f"  Signal: [{colour}]{sig}[/{colour}]"]
    if result["estimated_prob"] is not None:
        parts.append(f"  Est. prob: {result['estimated_prob']:.1%}")
    parts.append(f"  Market prob: {result['market_prob']:.1%}")
    if result["edge"] is not None:
        parts.append(f"  Edge: {result['edge']:+.1%}")
    if result["kelly"] is not None:
        parts.append(f"  Kelly: {result['kelly']:.2f}")
    console.print("  ".join(parts))


def _summary_table(results: list[dict], min_edge: float) -> Table:
    table = Table(
        title="Scan Results",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#",            width=3,  justify="right")
    table.add_column("Signal",       width=6,  justify="center")
    table.add_column("Market",       min_width=40, no_wrap=False)
    table.add_column("Mkt%",         width=6,  justify="right")
    table.add_column("Est%",         width=6,  justify="right")
    table.add_column("Edge",         width=7,  justify="right")
    table.add_column("Kelly",        width=6,  justify="right")

    actionable = 0
    for i, r in enumerate(results, 1):
        sig     = r["signal"]
        colour  = _SIG_COLOUR.get(sig, "white")
        edge    = r["edge"]
        is_act  = sig != "SKIP" and (edge is None or abs(edge) >= min_edge)
        if is_act:
            actionable += 1

        q = (r["question"] or "")[:60]
        table.add_row(
            str(i),
            Text(sig, style=colour),
            Text(q, style="bold" if is_act else "dim"),
            f"{r['market_prob']:.0%}",
            f"{r['estimated_prob']:.0%}" if r["estimated_prob"] is not None else "—",
            (f"{edge:+.0%}" if edge is not None else "—"),
            (f"{r['kelly']:.2f}" if r["kelly"] is not None else "—"),
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]{actionable}[/bold] actionable decision(s) out of [bold]{len(results)}[/bold] markets scanned."
    )
    return table


def _save_scan_results(results: list[dict], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = save_dir / f"scan_{ts}.json"

    payload = []
    for r in results:
        payload.append({
            "condition_id":    r["condition_id"],
            "question":        r["question"],
            "signal":          r["signal"],
            "market_prob":     r["market_prob"],
            "estimated_prob":  r["estimated_prob"],
            "edge":            r["edge"],
            "kelly":           r["kelly"],
        })

    out_file.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[green]✓ Scan results saved to:[/green] {out_file.resolve()}")


# ---------------------------------------------------------------------------
# Main scan command
# ---------------------------------------------------------------------------

def run_scan(
    limit: int,
    keyword: Optional[str],
    hitl: bool,
    depth: int,
    analysts: Optional[str],
    min_volume: float,
    min_liquidity: float,
    min_edge: float,
    save_dir: Optional[Path],
    no_save: bool,
) -> None:
    """Core scan logic — called by the Typer command in main.py."""

    # 1. LLM config ──────────────────────────────────────────────────────────
    llm_cfg = _llm_config_from_env() or _llm_config_interactive()

    from polyagents.default_config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]            = llm_cfg["provider"]
    config["quick_think_llm"]         = llm_cfg["quick"]
    config["deep_think_llm"]          = llm_cfg["deep"]
    config["backend_url"]             = llm_cfg.get("backend_url")
    config["max_debate_rounds"]       = depth
    config["max_risk_discuss_rounds"] = depth
    config["openai_reasoning_effort"] = llm_cfg.get("openai_reasoning_effort")
    config["anthropic_effort"]        = llm_cfg.get("anthropic_effort")
    config["google_thinking_level"]   = llm_cfg.get("google_thinking_level")

    # 2. Analyst selection ───────────────────────────────────────────────────
    ALL_ANALYSTS = ["news", "base_rate", "crowd_forecast", "data"]
    if analysts:
        selected_keys = [a.strip() for a in analysts.split(",") if a.strip() in ALL_ANALYSTS]
        if not selected_keys:
            console.print(f"[red]No valid analyst keys in '{analysts}'. Valid: {ALL_ANALYSTS}[/red]")
            raise typer.Exit(1)
    else:
        selected_keys = ALL_ANALYSTS

    trade_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 3. Fetch markets ───────────────────────────────────────────────────────
    try:
        markets = _fetch_markets(keyword, limit, min_volume, min_liquidity)
    except Exception as e:
        console.print(f"[red]Failed to fetch markets: {e}[/red]")
        raise typer.Exit(1)

    if not markets:
        console.print("[yellow]No markets matched the filter criteria. Try loosening --min-volume or --min-liquidity.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"\n[green]Found {len(markets)} market(s) to analyse.[/green]  "
        f"[dim]Analysts: {', '.join(selected_keys)}  |  Depth: {depth} round(s)[/dim]\n"
    )

    # 4. Analyse each market ─────────────────────────────────────────────────
    results: list[dict] = []

    for idx, market in enumerate(markets, 1):
        # Use pre-screened CID and live mid-price from _prescreen_one if available
        cid = (market.get("_cid") or market.get("conditionId") or market.get("condition_id") or "").lower()
        question = market.get("question") or market.get("title") or ""
        mid_price = market.get("_mid_price") or _market_mid_price(market)

        _print_market_header(idx, len(markets), market)

        # HITL approval gate
        if hitl:
            choice = typer.prompt(
                f"  Analyse this market? (Y/n/q to quit)",
                default="Y",
            ).strip().upper()
            if choice == "Q":
                console.print("[yellow]Scan stopped by user.[/yellow]")
                break
            if choice not in ("Y", "YES", ""):
                console.print("  [dim]Skipped.[/dim]")
                continue

        # Run analysis with a progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Analysing…", total=None)
            try:
                result = _analyze_market(
                    condition_id=cid,
                    question=question,
                    current_probability=mid_price,
                    trade_date=trade_date,
                    config=config,
                    selected_keys=selected_keys,
                )
            except Exception as e:
                console.print(f"  [red]Analysis failed: {e}[/red]")
                results.append({
                    "condition_id": cid,
                    "question": question,
                    "market_prob": mid_price,
                    "estimated_prob": None,
                    "signal": "ERROR",
                    "edge": None,
                    "kelly": None,
                    "final_state": {},
                    "decision_text": str(e),
                })
                continue
            finally:
                progress.remove_task(task)

        _print_result(result)
        results.append(result)

    if not results:
        console.print("\n[yellow]No markets were analysed.[/yellow]")
        raise typer.Exit(0)

    # 5. Summary table ───────────────────────────────────────────────────────
    console.print(Rule("[bold]Scan Complete[/bold]", style="cyan"))
    _summary_table(results, min_edge)

    # 6. Save ────────────────────────────────────────────────────────────────
    if no_save:
        return

    dest = save_dir or Path.cwd() / "reports" / "scans"
    _save_scan_results(results, dest)
