"""Extract the YES/NO/SKIP direction from the Portfolio Manager's PositionDecision.

The PM renders its decision to markdown always containing a
``**Direction**: YES/NO/SKIP`` line — extracted deterministically, no LLM call.
"""
from __future__ import annotations

import re
from typing import Any

_DIRECTION_RE = re.compile(r"direction.*?[:\-][\s*]*(\w+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"FINAL POSITION.*?\*\*(\w+)\*\*", re.IGNORECASE)
_VALID = {"YES", "NO", "SKIP"}


def parse_direction(text: str, default: str = "SKIP") -> str:
    """Extract YES / NO / SKIP from a rendered PositionDecision string."""
    for line in text.splitlines():
        m = _DIRECTION_RE.search(line)
        if m and m.group(1).upper() in _VALID:
            return m.group(1).upper()
    m = _FINAL_RE.search(text)
    if m and m.group(1).upper() in _VALID:
        return m.group(1).upper()
    # Last-resort scan
    for word in text.upper().split():
        clean = word.strip("*:.,")
        if clean in _VALID:
            return clean
    return default


class SignalProcessor:
    """Extract the direction signal from a Portfolio Manager decision."""

    def __init__(self, quick_thinking_llm: Any = None):
        # LLM arg kept for API compat but unused — parsing is deterministic.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return one of YES / NO / SKIP."""
        return parse_direction(full_signal)
