"""
Condition-ID and market-input validation tests (replaces old ticker-symbol tests).
"""
import re
import unittest

import pytest


@pytest.mark.unit
class ConditionIdValidationTests(unittest.TestCase):
    """Validate the hex condition-ID regex used in cli/utils.py."""

    _PATTERN = re.compile(r"^0x[0-9a-fA-F]+$")

    def _valid(self, s: str) -> bool:
        return bool(self._PATTERN.match(s.strip()))

    def test_valid_lowercase_hex(self):
        self.assertTrue(self._valid("0xdeadbeef"))

    def test_valid_uppercase_hex(self):
        self.assertTrue(self._valid("0xDEADBEEF"))

    def test_valid_mixed_case_hex(self):
        self.assertTrue(self._valid("0xAbCdEf1234567890"))

    def test_valid_long_condition_id(self):
        # Real Polymarket condition IDs are 64 hex chars after 0x
        long_id = "0x" + "a1b2c3d4" * 8
        self.assertTrue(self._valid(long_id))

    def test_invalid_no_0x_prefix(self):
        self.assertFalse(self._valid("deadbeef"))

    def test_invalid_non_hex_chars(self):
        self.assertFalse(self._valid("0xGHIJKL"))

    def test_invalid_empty(self):
        self.assertFalse(self._valid(""))

    def test_invalid_only_prefix(self):
        self.assertFalse(self._valid("0x"))

    def test_strips_whitespace_before_validation(self):
        self.assertTrue(self._valid("  0xabcdef  "))


@pytest.mark.unit
class CurrentProbabilityValidationTests(unittest.TestCase):
    """Validate the probability input bounds used in cli/utils.py."""

    def _valid(self, raw: str) -> bool:
        try:
            v = float(raw)
            return 0.01 <= v <= 0.99
        except ValueError:
            return False

    def test_valid_mid_value(self):
        self.assertTrue(self._valid("0.45"))

    def test_valid_lower_bound(self):
        self.assertTrue(self._valid("0.01"))

    def test_valid_upper_bound(self):
        self.assertTrue(self._valid("0.99"))

    def test_invalid_zero(self):
        self.assertFalse(self._valid("0.0"))

    def test_invalid_one(self):
        self.assertFalse(self._valid("1.0"))

    def test_invalid_negative(self):
        self.assertFalse(self._valid("-0.1"))

    def test_invalid_non_numeric(self):
        self.assertFalse(self._valid("abc"))


if __name__ == "__main__":
    unittest.main()
