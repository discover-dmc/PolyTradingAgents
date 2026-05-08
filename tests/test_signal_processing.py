"""Tests for the direction heuristic and SignalProcessor adapter.

The Portfolio Manager produces a PositionDecision rendered to markdown
with a ``**Direction**: YES/NO/SKIP`` line. Extraction is deterministic.
"""
import pytest
from unittest.mock import MagicMock

from polyagents.graph.signal_processing import SignalProcessor, parse_direction


@pytest.mark.unit
class TestParseDirection:
    def test_direction_yes(self):
        assert parse_direction("**Direction**: YES\n**Edge**: +0.15") == "YES"

    def test_direction_no(self):
        assert parse_direction("**Direction**: NO\n**Edge**: -0.12") == "NO"

    def test_direction_skip(self):
        assert parse_direction("**Direction**: SKIP\n**Edge**: +0.01") == "SKIP"

    def test_final_position_line(self):
        text = "**Direction**: YES\n\nFINAL POSITION: **YES** @ Kelly 10.0%"
        assert parse_direction(text) == "YES"

    def test_rendered_position_decision_shape(self):
        from polyagents.agents.schemas import PositionDecision, render_position_decision
        dec = PositionDecision(
            direction="NO",
            estimated_probability=0.35,
            market_probability=0.50,
            edge=-0.15,
            kelly_fraction=0.12,
            confidence="High",
            reasoning="Market overprices YES; strong NO case.",
        )
        text = render_position_decision(dec)
        assert parse_direction(text) == "NO"

    def test_no_direction_returns_default(self):
        assert parse_direction("Plain prose with no direction.") == "SKIP"

    def test_all_three_directions_recognised(self):
        for d in ("YES", "NO", "SKIP"):
            assert parse_direction(f"Direction: {d}") == d

    def test_case_insensitive(self):
        assert parse_direction("direction: yes") == "YES"


@pytest.mark.unit
class TestSignalProcessor:
    def test_returns_yes(self):
        sp = SignalProcessor()
        assert sp.process_signal("**Direction**: YES\n**Kelly Fraction**: 10.0%") == "YES"

    def test_returns_skip_for_illiquid(self):
        sp = SignalProcessor()
        text = "**Direction**: SKIP\n**Reasoning**: Market failed liquidity check."
        assert sp.process_signal(text) == "SKIP"

    def test_makes_no_llm_calls(self):
        llm = MagicMock()
        sp = SignalProcessor(llm)
        sp.process_signal("Direction: YES\nDetails.")
        llm.invoke.assert_not_called()
        llm.with_structured_output.assert_not_called()

    def test_default_skip_when_unparseable(self):
        sp = SignalProcessor()
        assert sp.process_signal("Plain prose without a recommendation.") == "SKIP"
