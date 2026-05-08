"""Integration smoke test: exercises the full compile+propagate path.

Uses mock LLMs and a patched TradeSniper so no API calls are made.  Verifies:
- Graph compiles (node wiring correct)
- propagate() returns (final_state dict, signal str)
- signal is one of YES / NO / SKIP
- analyst_reports has an entry per selected analyst
- memory log receives a pending entry
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polyagents.agents.schemas import (
    PositionDecision,
    ResearchPlan,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from polyagents.default_config import DEFAULT_CONFIG
from polyagents.graph.trading_graph import PolyAgentsGraph


# ---------------------------------------------------------------------------
# Mock LLM — zero API calls
# ---------------------------------------------------------------------------

class _MockLLM:
    """Handles every calling pattern in the graph without hitting any API."""

    def bind_tools(self, tools):
        from langchain_core.messages import AIMessage
        stub = MagicMock()
        stub.invoke.return_value = AIMessage(content="Mock analyst report.", tool_calls=[])
        return stub

    def invoke(self, prompt):
        from langchain_core.messages import AIMessage
        return AIMessage(content="Yes Analyst: mock debate argument.")

    def with_structured_output(self, schema, **kwargs):
        stub = MagicMock()
        name = getattr(schema, "__name__", "")
        if name == "ResearchPlan":
            stub.invoke.return_value = ResearchPlan(
                recommendation=PortfolioRating.HOLD,
                rationale="Balanced evidence on both sides.",
                strategic_actions="No position; revisit when edge exceeds 5%.",
            )
        elif name == "TraderProposal":
            stub.invoke.return_value = TraderProposal(
                action=TraderAction.HOLD,
                reasoning="Insufficient edge to size a position.",
            )
        elif name == "PositionDecision":
            stub.invoke.return_value = PositionDecision(
                direction="YES",
                estimated_probability=0.65,
                market_probability=0.50,
                edge=0.15,
                kelly_fraction=0.10,
                confidence="Medium",
                reasoning="Mock: market underprices YES given evidence.",
            )
        else:
            stub.invoke.side_effect = ValueError(f"Unexpected schema: {name}")
        return stub


# ---------------------------------------------------------------------------
# Liquidity pass-through patch — makes TradeSniper always approve
# ---------------------------------------------------------------------------

_LIQUID_SUMMARY = {
    "condition_id": "mock-condition-id",
    "question": "Will X happen by Y?",
    "volume_24h": 5000.0,
    "liquidity": 2000.0,
    "yes_mid_price": 0.50,
    "yes_spread": 0.02,
    "yes_depth_usd": 500.0,
    "end_date": "2026-12-31",
    "active": True,
    "liquid": True,
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_graph(tmp_path):
    config = {
        **DEFAULT_CONFIG,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "mem.md"),
        "checkpoint_enabled": False,
    }
    llm = _MockLLM()
    mock_client = MagicMock()
    mock_client.get_llm.return_value = llm

    with patch("polyagents.graph.trading_graph.create_llm_client", return_value=mock_client):
        graph = PolyAgentsGraph(selected_analysts=["news", "data"], config=config)
    return graph


def _propagate(graph, condition_id="mock-cid-001", trade_date="2026-01-10"):
    """Run propagate() with TradeSniper and resolution patched out."""
    graph._fetch_resolution = MagicMock(return_value=None)
    with patch(
        "polyagents.agents.analysts.trade_sniper.get_liquidity_summary",
        return_value=_LIQUID_SUMMARY,
    ):
        return graph.propagate(
            condition_id=condition_id,
            trade_date=trade_date,
            market_question="Will X happen by Y?",
            current_probability=0.50,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_graph_compiles(mock_graph):
    assert mock_graph.graph is not None
    assert mock_graph.workflow is not None


@pytest.mark.smoke
def test_propagate_returns_tuple(mock_graph):
    final_state, signal = _propagate(mock_graph)
    assert isinstance(final_state, dict)
    assert isinstance(signal, str)


@pytest.mark.smoke
def test_propagate_signal_is_valid(mock_graph):
    _, signal = _propagate(mock_graph)
    assert signal in ("YES", "NO", "SKIP"), f"unexpected signal: {signal!r}"


@pytest.mark.smoke
def test_propagate_final_state_has_required_keys(mock_graph):
    final_state, _ = _propagate(mock_graph)
    required = {"final_trade_decision", "investment_plan", "analyst_reports", "condition_id"}
    missing = required - set(final_state.keys())
    assert not missing, f"final_state missing keys: {missing}"


@pytest.mark.smoke
def test_propagate_analyst_reports_populated(mock_graph):
    final_state, _ = _propagate(mock_graph)
    reports = final_state.get("analyst_reports", {})
    assert "news" in reports, "news analyst report missing"
    assert "data" in reports, "data analyst report missing"


@pytest.mark.smoke
def test_propagate_writes_memory_log_entry(mock_graph):
    _propagate(mock_graph)
    entries = mock_graph.memory_log.load_entries()
    assert len(entries) == 1
    assert entries[0]["ticker"] == "mock-cid-001"
    assert entries[0]["pending"] is True


@pytest.mark.smoke
def test_graph_wiring_all_analysts(tmp_path):
    config = {
        **DEFAULT_CONFIG,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "mem.md"),
        "checkpoint_enabled": False,
    }
    llm = _MockLLM()
    mock_client = MagicMock()
    mock_client.get_llm.return_value = llm

    with patch("polyagents.graph.trading_graph.create_llm_client", return_value=mock_client):
        graph = PolyAgentsGraph(
            selected_analysts=["news", "base_rate", "crowd_forecast", "data"],
            config=config,
        )

    graph._fetch_resolution = MagicMock(return_value=None)
    with patch(
        "polyagents.agents.analysts.trade_sniper.get_liquidity_summary",
        return_value=_LIQUID_SUMMARY,
    ):
        final_state, signal = graph.propagate(
            condition_id="mock-cid-all",
            trade_date="2026-02-01",
            market_question="Will Y happen?",
            current_probability=0.45,
        )

    assert signal in ("YES", "NO", "SKIP")
    reports = final_state.get("analyst_reports", {})
    for key in ("news", "base_rate", "crowd_forecast", "data"):
        assert key in reports, f"{key} analyst report missing"


@pytest.mark.smoke
def test_trade_sniper_skip_path(tmp_path):
    """When TradeSniper returns liquid=False the graph short-circuits to SKIP."""
    config = {
        **DEFAULT_CONFIG,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "mem.md"),
        "checkpoint_enabled": False,
    }
    llm = _MockLLM()
    mock_client = MagicMock()
    mock_client.get_llm.return_value = llm

    with patch("polyagents.graph.trading_graph.create_llm_client", return_value=mock_client):
        graph = PolyAgentsGraph(selected_analysts=["news"], config=config)

    graph._fetch_resolution = MagicMock(return_value=None)
    illiquid = {**_LIQUID_SUMMARY, "liquid": False, "volume_24h": 10.0}
    with patch(
        "polyagents.agents.analysts.trade_sniper.get_liquidity_summary",
        return_value=illiquid,
    ):
        final_state, signal = graph.propagate(
            condition_id="thin-market",
            trade_date="2026-02-01",
            market_question="Will Z happen?",
            current_probability=0.50,
        )

    assert signal == "SKIP", f"expected SKIP for illiquid market, got {signal!r}"


@pytest.mark.smoke
def test_invalid_analyst_key_raises():
    config = {**DEFAULT_CONFIG}
    llm = _MockLLM()
    mock_client = MagicMock()
    mock_client.get_llm.return_value = llm

    with patch("polyagents.graph.trading_graph.create_llm_client", return_value=mock_client):
        with pytest.raises(ValueError, match="Unknown analyst"):
            PolyAgentsGraph(selected_analysts=["markt"], config=config)
