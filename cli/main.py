"""PolyAgents CLI — interactive terminal UI for Polymarket analysis."""
from __future__ import annotations

import datetime
import time
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any, Dict

import typer
from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

load_dotenv()
load_dotenv(".env.enterprise", override=False)

from polyagents.graph.trading_graph import PolyAgentsGraph
from polyagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import (
    get_condition_id,
    fetch_market_info,
    get_market_question,
    get_current_probability,
    get_analysis_date,
    select_analysts,
    select_research_depth,
    select_llm_provider,
    select_shallow_thinking_agent,
    select_deep_thinking_agent,
    ask_openai_reasoning_effort,
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_output_language,
    format_tool_args,
)
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

app = typer.Typer(
    name="polyagents",
    help="PolyAgents: Multi-Agent LLM framework for Polymarket prediction markets.",
    add_completion=True,
)

# ---------------------------------------------------------------------------
# Analyst ordering (matches ANALYST_NODE_NAMES in node_names.py)
# ---------------------------------------------------------------------------

ANALYST_ORDER = ["news", "base_rate", "crowd_forecast", "data"]

ANALYST_AGENT_NAMES = {
    "news":           "News Analyst",
    "base_rate":      "Base Rate Analyst",
    "crowd_forecast": "Crowd Forecast Analyst",
    "data":           "Data Analyst",
}

ANALYST_REPORT_MAP = {
    "news":           "analyst_report_news",
    "base_rate":      "analyst_report_base_rate",
    "crowd_forecast": "analyst_report_crowd_forecast",
    "data":           "analyst_report_data",
}

# ---------------------------------------------------------------------------
# Message buffer
# ---------------------------------------------------------------------------

class MessageBuffer:
    FIXED_AGENTS = {
        "Research Team":    ["Yes Researcher", "No Researcher", "Research Manager"],
        "Trading Team":     ["Position Sizer"],
        "Risk Management":  ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Mgmt":   ["Portfolio Manager"],
    }

    ANALYST_MAPPING = {
        "news":           "News Analyst",
        "base_rate":      "Base Rate Analyst",
        "crowd_forecast": "Crowd Forecast Analyst",
        "data":           "Data Analyst",
    }

    # (analyst_key_or_None, finalizing_agent)
    REPORT_SECTIONS = {
        "analyst_report_news":           ("news",           "News Analyst"),
        "analyst_report_base_rate":      ("base_rate",      "Base Rate Analyst"),
        "analyst_report_crowd_forecast": ("crowd_forecast", "Crowd Forecast Analyst"),
        "analyst_report_data":           ("data",           "Data Analyst"),
        "investment_plan":               (None,             "Research Manager"),
        "trader_investment_plan":        (None,             "Position Sizer"),
        "final_trade_decision":          (None,             "Portfolio Manager"),
    }

    SECTION_TITLES = {
        "analyst_report_news":           "News Analysis",
        "analyst_report_base_rate":      "Base Rate Analysis",
        "analyst_report_crowd_forecast": "Crowd Forecast Analysis",
        "analyst_report_data":           "Data Analysis",
        "investment_plan":               "Research Team Decision",
        "trader_investment_plan":        "Position Sizer Plan",
        "final_trade_decision":          "Portfolio Manager Decision",
    }

    def __init__(self, max_length: int = 100):
        self.messages: deque = deque(maxlen=max_length)
        self.tool_calls: deque = deque(maxlen=max_length)
        self.current_report: str | None = None
        self.final_report: str | None = None
        self.agent_status: Dict[str, str] = {}
        self.current_agent: str | None = None
        self.report_sections: Dict[str, Any] = {}
        self.selected_analysts: list = []
        self._processed_message_ids: set = set()

    def init_for_analysis(self, selected_analysts: list[str]):
        self.selected_analysts = selected_analysts
        self.agent_status = {}

        for key in selected_analysts:
            name = self.ANALYST_MAPPING.get(key)
            if name:
                self.agent_status[name] = "pending"

        for agents in self.FIXED_AGENTS.values():
            for agent in agents:
                self.agent_status[agent] = "pending"

        self.report_sections = {
            section: None
            for section, (analyst_key, _) in self.REPORT_SECTIONS.items()
            if analyst_key is None or analyst_key in selected_analysts
        }
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self) -> int:
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            if (self.report_sections.get(section) is not None
                    and self.agent_status.get(finalizing_agent) == "completed"):
                count += 1
        return count

    def add_message(self, message_type: str, content: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((ts, message_type, content))

    def add_tool_call(self, tool_name: str, args: dict):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((ts, tool_name, args))

    def update_agent_status(self, agent: str, status: str):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name: str, content: str):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._rebuild_reports()

    def _rebuild_reports(self):
        latest_section = latest_content = None
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section, latest_content = section, content
        if latest_section and latest_content:
            title = self.SECTION_TITLES.get(latest_section, latest_section)
            self.current_report = f"### {title}\n{latest_content}"
        self._build_final_report()

    def _build_final_report(self):
        parts = []
        analyst_sections = [
            ("analyst_report_news", "News Analysis"),
            ("analyst_report_base_rate", "Base Rate Analysis"),
            ("analyst_report_crowd_forecast", "Crowd Forecast Analysis"),
            ("analyst_report_data", "Data Analysis"),
        ]
        analyst_content = [
            (title, self.report_sections[key])
            for key, title in analyst_sections
            if self.report_sections.get(key)
        ]
        if analyst_content:
            parts.append("## Analyst Reports")
            for title, content in analyst_content:
                parts.append(f"### {title}\n{content}")
        for key, heading in [
            ("investment_plan",    "## Research Team Decision"),
            ("trader_investment_plan", "## Position Sizer Plan"),
            ("final_trade_decision",   "## Portfolio Manager Decision"),
        ]:
            if self.report_sections.get(key):
                parts.append(f"{heading}\n{self.report_sections[key]}")
        self.final_report = "\n\n".join(parts) if parts else None


message_buffer = MessageBuffer()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _format_tokens(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def create_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3),
        Layout(name="analysis", ratio=5),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="messages", ratio=3),
    )
    return layout


def update_display(layout: Layout, market_info: dict | None = None,
                   stats_handler=None, start_time: float | None = None):
    # Header
    subtitle = ""
    if market_info:
        q = market_info.get("question", "")
        p = market_info.get("current_probability", 0.5)
        subtitle = f"[dim]{q[:80]}{'…' if len(q) > 80 else ''}[/dim]  [cyan]YES: {p:.0%}[/cyan]"
    layout["header"].update(Panel(
        f"[bold green]PolyAgents[/bold green]   {subtitle}",
        title="Polymarket Multi-Agent Analysis",
        border_style="green",
        padding=(0, 2),
    ))

    # Progress panel
    progress_table = Table(
        show_header=True, header_style="bold magenta",
        box=box.SIMPLE_HEAD, padding=(0, 2), expand=True,
    )
    progress_table.add_column("Team",   style="cyan",  justify="center", width=18)
    progress_table.add_column("Agent",  style="green", justify="center", width=22)
    progress_table.add_column("Status", style="yellow",justify="center", width=12)

    all_teams = {
        "Analyst Team": list(ANALYST_AGENT_NAMES.values()),
        **MessageBuffer.FIXED_AGENTS,
    }
    for team, agents in all_teams.items():
        active = [a for a in agents if a in message_buffer.agent_status]
        if not active:
            continue
        for i, agent in enumerate(active):
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                status_cell = Spinner("dots", text="[blue]running[/blue]", style="bold cyan")
            else:
                colour = {"pending": "yellow", "completed": "green", "error": "red"}.get(status, "white")
                status_cell = f"[{colour}]{status}[/{colour}]"
            progress_table.add_row(team if i == 0 else "", agent, status_cell)
        progress_table.add_row("─" * 18, "─" * 22, "─" * 12, style="dim")

    layout["progress"].update(Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2)))

    # Messages panel
    msg_table = Table(
        show_header=True, header_style="bold magenta",
        box=box.MINIMAL, show_lines=True, expand=True, padding=(0, 1),
    )
    msg_table.add_column("Time",    style="cyan",  width=8,  justify="center")
    msg_table.add_column("Type",    style="green", width=8,  justify="center")
    msg_table.add_column("Content", style="white", ratio=1,  no_wrap=False)

    all_msgs = []
    for ts, tool, args in message_buffer.tool_calls:
        all_msgs.append((ts, "Tool", f"{tool}: {format_tool_args(args)}"))
    for ts, mtype, content in message_buffer.messages:
        s = str(content or "")
        all_msgs.append((ts, mtype, s[:200] + "…" if len(s) > 200 else s))
    for ts, mtype, content in sorted(all_msgs, key=lambda x: x[0], reverse=True)[:12]:
        msg_table.add_row(ts, mtype, Text(content, overflow="fold"))
    layout["messages"].update(Panel(msg_table, title="Messages & Tools", border_style="blue", padding=(1, 2)))

    # Analysis panel
    if message_buffer.current_report:
        layout["analysis"].update(Panel(
            Markdown(message_buffer.current_report),
            title="Current Report", border_style="green", padding=(1, 2),
        ))
    else:
        layout["analysis"].update(Panel(
            "[italic dim]Waiting for first report…[/italic dim]",
            title="Current Report", border_style="green", padding=(1, 2),
        ))

    # Footer
    agents_done  = sum(1 for s in message_buffer.agent_status.values() if s == "completed")
    agents_total = len(message_buffer.agent_status)
    reports_done  = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    parts = [f"Agents: {agents_done}/{agents_total}", f"Reports: {reports_done}/{reports_total}"]
    if stats_handler:
        stats = stats_handler.get_stats()
        parts += [f"LLM: {stats['llm_calls']}", f"Tools: {stats['tool_calls']}"]
        if stats["tokens_in"] or stats["tokens_out"]:
            parts.append(f"Tokens: {_format_tokens(stats['tokens_in'])}↑ {_format_tokens(stats['tokens_out'])}↓")
    if start_time:
        e = time.time() - start_time
        parts.append(f"⏱ {int(e//60):02d}:{int(e%60):02d}")

    footer_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    footer_table.add_column("s", justify="center")
    footer_table.add_row(" | ".join(parts))
    layout["footer"].update(Panel(footer_table, border_style="grey50"))


# ---------------------------------------------------------------------------
# Analyst status updater (parallel fan-out)
# ---------------------------------------------------------------------------

def update_analyst_statuses(chunk: dict):
    analyst_reports = chunk.get("analyst_reports") or {}

    for key in ANALYST_ORDER:
        if key not in message_buffer.selected_analysts:
            continue
        agent_name  = ANALYST_AGENT_NAMES[key]
        section_key = ANALYST_REPORT_MAP[key]

        if key in analyst_reports:
            message_buffer.update_report_section(section_key, analyst_reports[key])

        has_report     = bool(message_buffer.report_sections.get(section_key))
        current_status = message_buffer.agent_status.get(agent_name)

        if has_report:
            if current_status != "completed":
                message_buffer.update_agent_status(agent_name, "completed")
        elif current_status not in ("in_progress", "completed"):
            message_buffer.update_agent_status(agent_name, "in_progress")

    all_done = all(
        message_buffer.agent_status.get(ANALYST_AGENT_NAMES[k]) == "completed"
        for k in message_buffer.selected_analysts
    )
    if all_done and message_buffer.selected_analysts:
        if message_buffer.agent_status.get("Yes Researcher") == "pending":
            message_buffer.update_agent_status("Yes Researcher", "in_progress")
            message_buffer.update_agent_status("No Researcher",  "in_progress")


# ---------------------------------------------------------------------------
# Message content extractor
# ---------------------------------------------------------------------------

def _extract_content(content) -> str | None:
    if not content:
        return None
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, dict):
        t = content.get("text", "")
        return t.strip() or None
    if isinstance(content, list):
        parts = [
            item.get("text", "").strip() if isinstance(item, dict) and item.get("type") == "text"
            else (item.strip() if isinstance(item, str) else "")
            for item in content
        ]
        result = " ".join(p for p in parts if p)
        return result or None
    return str(content).strip() or None


def _classify_message(message) -> tuple[str, str | None]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    content = _extract_content(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("Control" if content and content.strip() == "Continue" else "User", content)
    if isinstance(message, ToolMessage):
        return ("Data", content)
    if isinstance(message, AIMessage):
        return ("Agent", content)
    return ("System", content)


# ---------------------------------------------------------------------------
# Report saving & display
# ---------------------------------------------------------------------------

def save_report_to_disk(final_state: dict, condition_id: str, save_path: Path) -> Path:
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analyst reports
    analysts_dir = save_path / "1_analysts"
    ar = final_state.get("analyst_reports") or {}
    analyst_files = {
        "news":           ("News Analysis",          "news.md"),
        "base_rate":      ("Base Rate Analysis",     "base_rate.md"),
        "crowd_forecast": ("Crowd Forecast Analysis","crowd_forecast.md"),
        "data":           ("Data Analysis",          "data.md"),
    }
    analyst_parts = []
    for key, (title, fname) in analyst_files.items():
        report = ar.get(key)
        if report:
            analysts_dir.mkdir(exist_ok=True)
            (analysts_dir / fname).write_text(report, encoding="utf-8")
            analyst_parts.append((title, report))
    if analyst_parts:
        sections.append("## I. Analyst Reports\n\n" +
                        "\n\n".join(f"### {t}\n{c}" for t, c in analyst_parts))

    # 2. Research debate
    debate = final_state.get("investment_debate_state") or {}
    research_parts = []
    research_dir = save_path / "2_research"
    for field, title, fname in [
        ("bull_history",  "Yes Researcher", "yes_researcher.md"),
        ("bear_history",  "No Researcher",  "no_researcher.md"),
        ("judge_decision","Research Manager","research_manager.md"),
    ]:
        text = (debate.get(field) or "").strip()
        if text:
            research_dir.mkdir(exist_ok=True)
            (research_dir / fname).write_text(text, encoding="utf-8")
            research_parts.append((title, text))
    if research_parts:
        sections.append("## II. Research Team\n\n" +
                        "\n\n".join(f"### {t}\n{c}" for t, c in research_parts))

    # 3. Position sizer
    tp = final_state.get("trader_investment_plan", "").strip()
    if tp:
        trading_dir = save_path / "3_position_sizer"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "plan.md").write_text(tp, encoding="utf-8")
        sections.append(f"## III. Position Sizer\n\n{tp}")

    # 4. Risk management
    risk = final_state.get("risk_debate_state") or {}
    risk_dir = save_path / "4_risk"
    risk_parts = []
    for field, title, fname in [
        ("aggressive_history",  "Aggressive Analyst",  "aggressive.md"),
        ("conservative_history","Conservative Analyst","conservative.md"),
        ("neutral_history",     "Neutral Analyst",     "neutral.md"),
    ]:
        text = (risk.get(field) or "").strip()
        if text:
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / fname).write_text(text, encoding="utf-8")
            risk_parts.append((title, text))
    if risk_parts:
        sections.append("## IV. Risk Management\n\n" +
                        "\n\n".join(f"### {t}\n{c}" for t, c in risk_parts))

    # 5. Portfolio Manager decision (PositionDecision)
    ftd = final_state.get("final_trade_decision", "").strip()
    if ftd:
        pm_dir = save_path / "5_portfolio"
        pm_dir.mkdir(exist_ok=True)
        (pm_dir / "decision.md").write_text(ftd, encoding="utf-8")
        sections.append(f"## V. Portfolio Manager Decision\n\n{ftd}")

    header = (f"# PolyAgents Analysis: {condition_id}\n\n"
              f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_file = save_path / "complete_report.md"
    report_file.write_text(header + "\n\n".join(sections), encoding="utf-8")
    return report_file


def display_complete_report(final_state: dict):
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    ar = final_state.get("analyst_reports") or {}
    analyst_display = {
        "news":           "News Analysis",
        "base_rate":      "Base Rate Analysis",
        "crowd_forecast": "Crowd Forecast Analysis",
        "data":           "Data Analysis",
    }
    analysts = [(name, ar[key]) for key, name in analyst_display.items() if ar.get(key)]
    if analysts:
        console.print(Panel("[bold]I. Analyst Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    debate = final_state.get("investment_debate_state") or {}
    research = [
        ("Yes Researcher",   debate.get("bull_history", "")),
        ("No Researcher",    debate.get("bear_history", "")),
        ("Research Manager", debate.get("judge_decision", "")),
    ]
    research = [(t, c.strip()) for t, c in research if c.strip()]
    if research:
        console.print(Panel("[bold]II. Research Team[/bold]", border_style="magenta"))
        for title, content in research:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    tp = final_state.get("trader_investment_plan", "").strip()
    if tp:
        console.print(Panel("[bold]III. Position Sizer[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(tp), title="Position Sizer", border_style="blue", padding=(1, 2)))

    risk = final_state.get("risk_debate_state") or {}
    risk_reports = [
        ("Aggressive Analyst",  risk.get("aggressive_history",   "")),
        ("Conservative Analyst",risk.get("conservative_history", "")),
        ("Neutral Analyst",     risk.get("neutral_history",      "")),
    ]
    risk_reports = [(t, c.strip()) for t, c in risk_reports if c.strip()]
    if risk_reports:
        console.print(Panel("[bold]IV. Risk Management[/bold]", border_style="red"))
        for title, content in risk_reports:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    ftd = final_state.get("final_trade_decision", "").strip()
    if ftd:
        console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
        console.print(Panel(Markdown(ftd), title="Portfolio Manager", border_style="green", padding=(1, 2)))


# ---------------------------------------------------------------------------
# User input wizard
# ---------------------------------------------------------------------------

def get_user_selections() -> dict:
    # Welcome banner
    welcome_txt = (Path(__file__).parent / "static" / "welcome.txt").read_text(encoding="utf-8")
    console.print(Align.center(Panel(
        f"{welcome_txt}\n"
        "[bold green]PolyAgents — Multi-Agent LLM for Polymarket[/bold green]\n\n"
        "[bold]Pipeline:[/bold]  TradeSniper → Analysts → Research → Risk → Portfolio Manager\n\n"
        "[dim]github.com/discover-dmc/PolyAgents[/dim]",
        border_style="green", padding=(1, 2),
        title="Welcome to PolyAgents",
    )))
    console.print()

    announcements = fetch_announcements()
    display_announcements(console, announcements)

    def box(title: str, prompt: str, hint: str = ""):
        body = f"[bold]{title}[/bold]\n[dim]{prompt}[/dim]"
        if hint:
            body += f"\n[dim italic]{hint}[/dim italic]"
        console.print(Panel(body, border_style="blue", padding=(1, 2)))

    # --- Step 1: Condition ID ---
    box("Step 1: Condition ID",
        "Enter the Polymarket condition ID (0x…)",
        "Find it in the market URL or via the Polymarket API")
    condition_id = get_condition_id()

    # --- Auto-fetch market info ---
    console.print("[dim]Fetching market info from Polymarket…[/dim]")
    info = fetch_market_info(condition_id)
    if info:
        console.print(Panel(
            f"[green]✓ Found market[/green]\n\n"
            f"[bold]Question:[/bold] {info['question']}\n"
            f"[bold]YES price:[/bold] {info['current_probability']:.1%}   "
            f"[bold]Resolves:[/bold] {info.get('end_date', 'unknown')}   "
            f"[bold]Liquid:[/bold] {'✓' if info.get('liquid') else '✗'}",
            border_style="green", padding=(1, 2),
        ))
    else:
        console.print("[yellow]Could not auto-fetch — enter details manually.[/yellow]")

    # --- Step 2: Market question ---
    box("Step 2: Market Question", "The full resolution question for this market")
    market_question = get_market_question(prefill=info["question"] if info else "")

    # --- Step 3: Current probability ---
    box("Step 3: Current YES Probability",
        "Current mid-price for YES shares (0.01–0.99)",
        "Pre-filled from Polymarket if available")
    current_probability = get_current_probability(
        prefill=info["current_probability"] if info else 0.5
    )

    # --- Step 4: Analysis date ---
    box("Step 4: Analysis Date", "Date of this analysis run (YYYY-MM-DD)")
    analysis_date = get_analysis_date()

    # --- Step 5: Analysts ---
    box("Step 5: Analyst Team", "Choose which analysts to run")
    selected_analysts = select_analysts()
    console.print(f"[green]Selected:[/green] {', '.join(a.value for a in selected_analysts)}")

    # --- Step 6: Research depth ---
    box("Step 6: Research Depth", "Controls number of debate rounds")
    research_depth = select_research_depth()

    # --- Step 7: LLM provider ---
    box("Step 7: LLM Provider", "Select your AI provider")
    llm_provider, backend_url = select_llm_provider()

    # --- Step 8: Models ---
    box("Step 8: Quick-Thinking Model", "Fast model for utility nodes")
    shallow_thinker = select_shallow_thinking_agent(llm_provider)
    box("Step 8b: Deep-Thinking Model", "Strong model for analysis and debate")
    deep_thinker = select_deep_thinking_agent(llm_provider)

    # --- Step 9: Provider-specific thinking config ---
    thinking_level = reasoning_effort = anthropic_effort = None
    provider_lower = llm_provider.lower()
    if provider_lower == "google":
        box("Step 9: Thinking Mode", "Configure Gemini thinking")
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        box("Step 9: Reasoning Effort", "Configure OpenAI reasoning effort")
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        box("Step 9: Effort Level", "Configure Claude effort level")
        anthropic_effort = ask_anthropic_effort()

    # --- Step 10: Output language ---
    box("Step 10: Output Language", "Language for all analyst reports")
    output_language = ask_output_language()

    return {
        "condition_id":         condition_id,
        "market_question":      market_question,
        "current_probability":  current_probability,
        "analysis_date":        analysis_date,
        "analysts":             selected_analysts,
        "research_depth":       research_depth,
        "llm_provider":         llm_provider.lower(),
        "backend_url":          backend_url,
        "shallow_thinker":      shallow_thinker,
        "deep_thinker":         deep_thinker,
        "google_thinking_level":thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort":     anthropic_effort,
        "output_language":      output_language,
    }


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_analysis(checkpoint: bool = False):
    selections = get_user_selections()

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"]       = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"]         = selections["shallow_thinker"]
    config["deep_think_llm"]          = selections["deep_thinker"]
    config["backend_url"]             = selections["backend_url"]
    config["llm_provider"]            = selections["llm_provider"]
    config["google_thinking_level"]   = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"]        = selections.get("anthropic_effort")
    config["output_language"]         = selections.get("output_language", "English")
    config["checkpoint_enabled"]      = checkpoint

    stats_handler = StatsCallbackHandler()
    selected_keys = [a.value for a in selections["analysts"]]
    # Preserve canonical analyst ordering
    selected_keys = [k for k in ANALYST_ORDER if k in selected_keys]

    market_info = {
        "question":            selections["market_question"],
        "current_probability": selections["current_probability"],
    }

    graph = PolyAgentsGraph(
        selected_analysts=selected_keys,
        config=config,
        callbacks=[stats_handler],
    )

    message_buffer.init_for_analysis(selected_keys)

    start_time = time.time()
    cid = selections["condition_id"]

    # Result dirs
    results_dir = Path(config["results_dir"]) / cid[:16] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir  = results_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    log_file    = results_dir / "run.log"
    log_file.touch(exist_ok=True)

    # Decorate buffer methods to also write to log
    def _wrap_add_message(obj, fn):
        original = getattr(obj, fn)
        @wraps(original)
        def wrapper(*args, **kwargs):
            original(*args, **kwargs)
            ts, mtype, content = obj.messages[-1]
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{ts} [{mtype}] {str(content).replace(chr(10), ' ')}\n")
        return wrapper

    def _wrap_add_tool_call(obj, fn):
        original = getattr(obj, fn)
        @wraps(original)
        def wrapper(*args, **kwargs):
            original(*args, **kwargs)
            ts, tool, targs = obj.tool_calls[-1]
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{ts} [Tool] {tool}({targs})\n")
        return wrapper

    message_buffer.add_message   = _wrap_add_message(message_buffer,   "add_message")
    message_buffer.add_tool_call = _wrap_add_tool_call(message_buffer, "add_tool_call")

    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:  # noqa: F841
        update_display(layout, market_info, stats_handler, start_time)

        message_buffer.add_message("System", f"Condition ID: {cid}")
        message_buffer.add_message("System", f"Question: {selections['market_question'][:80]}")
        message_buffer.add_message("System", f"YES prob: {selections['current_probability']:.1%}")
        message_buffer.add_message("System", f"Analysts: {', '.join(selected_keys)}")

        # Mark all selected analysts as in_progress (they run in parallel)
        for key in selected_keys:
            message_buffer.update_agent_status(ANALYST_AGENT_NAMES[key], "in_progress")
        update_display(layout, market_info, stats_handler, start_time)

        # Build initial state and stream
        init_state = graph.propagator.create_initial_state(
            condition_id=cid,
            trade_date=selections["analysis_date"],
            market_question=selections["market_question"],
            current_probability=selections["current_probability"],
        )
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        trace = []
        for chunk in graph.graph.stream(init_state, **args):
            # Process messages
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id and msg_id in message_buffer._processed_message_ids:
                    continue
                if msg_id:
                    message_buffer._processed_message_ids.add(msg_id)
                mtype, content = _classify_message(message)
                if content and content.strip():
                    message_buffer.add_message(mtype, content)
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        if isinstance(tc, dict):
                            message_buffer.add_tool_call(tc["name"], tc["args"])
                        else:
                            message_buffer.add_tool_call(tc.name, tc.args)

            # Analyst statuses
            update_analyst_statuses(chunk)

            # Research debate
            if chunk.get("investment_debate_state"):
                d = chunk["investment_debate_state"]
                bull  = (d.get("bull_history") or "").strip()
                bear  = (d.get("bear_history") or "").strip()
                judge = (d.get("judge_decision") or "").strip()

                if bull or bear:
                    for a in ("Yes Researcher", "No Researcher"):
                        if message_buffer.agent_status.get(a) not in ("in_progress", "completed"):
                            message_buffer.update_agent_status(a, "in_progress")
                if bull:
                    message_buffer.update_report_section("investment_plan",
                                                         f"### Yes Researcher\n{bull}")
                if bear:
                    message_buffer.update_report_section("investment_plan",
                                                         f"### No Researcher\n{bear}")
                if judge:
                    message_buffer.update_report_section("investment_plan",
                                                         f"### Research Manager\n{judge}")
                    for a in ("Yes Researcher", "No Researcher", "Research Manager"):
                        message_buffer.update_agent_status(a, "completed")
                    message_buffer.update_agent_status("Position Sizer", "in_progress")

            # Position sizer
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section("trader_investment_plan",
                                                     chunk["trader_investment_plan"])
                message_buffer.update_agent_status("Position Sizer", "completed")
                message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk debate
            if chunk.get("risk_debate_state"):
                r = chunk["risk_debate_state"]
                agg  = (r.get("aggressive_history") or "").strip()
                con  = (r.get("conservative_history") or "").strip()
                neu  = (r.get("neutral_history") or "").strip()
                rjudge = (r.get("judge_decision") or "").strip()
                if agg:
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section("final_trade_decision",
                                                         f"### Aggressive\n{agg}")
                if con:
                    message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section("final_trade_decision",
                                                         f"### Conservative\n{con}")
                if neu:
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section("final_trade_decision",
                                                         f"### Neutral\n{neu}")
                if rjudge:
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section("final_trade_decision",
                                                         f"### Portfolio Manager\n{rjudge}")
                    for a in ("Aggressive Analyst","Conservative Analyst","Neutral Analyst","Portfolio Manager"):
                        message_buffer.update_agent_status(a, "completed")

            update_display(layout, market_info, stats_handler, start_time)
            trace.append(chunk)

        # Finalise
        final_state = trace[-1]
        signal = graph.signal_processor.process_signal(
            final_state.get("final_trade_decision", "")
        )

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        # Sync final report sections from conclusive state
        ar_final = final_state.get("analyst_reports") or {}
        for key, section in ANALYST_REPORT_MAP.items():
            if key in ar_final:
                message_buffer.update_report_section(section, ar_final[key])
        for section in ("investment_plan", "trader_investment_plan", "final_trade_decision"):
            if final_state.get(section):
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, market_info, stats_handler, start_time)

    # Post-run (outside Live so prompts render cleanly)
    signal_colour = {"YES": "bold green", "NO": "bold red", "SKIP": "bold yellow"}.get(signal, "white")
    console.print(f"\n[bold]Signal:[/bold] [{signal_colour}]{signal}[/{signal_colour}]\n")

    # Save report
    save_choice = typer.prompt("Save report? (Y/n)", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{cid[:16]}_{ts}"
        save_str = typer.prompt("Save path (Enter for default)", default=str(default_path)).strip()
        try:
            report_file = save_report_to_disk(final_state, cid, Path(save_str))
            console.print(f"[green]✓ Saved to:[/green] {Path(save_str).resolve()}")
            console.print(f"  [dim]Complete:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving: {e}[/red]")

    # Display full report
    display_choice = typer.prompt("\nDisplay full report? (Y/n)", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    checkpoint: bool = typer.Option(
        False, "--checkpoint",
        help="Enable checkpoint/resume — save state after each node so a crashed run can be resumed.",
    ),
    clear_checkpoints: bool = typer.Option(
        False, "--clear-checkpoints",
        help="Delete all saved checkpoints before running.",
    ),
):
    """Run a full multi-agent analysis on a Polymarket condition."""
    if clear_checkpoints:
        from polyagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    run_analysis(checkpoint=checkpoint)


if __name__ == "__main__":
    app()
