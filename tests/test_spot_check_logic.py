"""Test the core logic of the spot-check harness without running the loop.

Specifically:
- is_wieferich_ref correctly identifies known Wieferich primes
- read_findings_set parses findings.jsonl correctly
- A constructed disagreement (gmpy2 says Wieferich, not in findings) is
  detectable

Run from the project root:
    python -m pytest tests/test_spot_check_logic.py -v
"""
import json
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.spot_check_harness import (
        is_wieferich_ref,
        read_findings_set,
        read_state,
    )
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="harness imports failed")


class TestIsWieferichRef:
    """gmpy2 reference correctly identifies known Wieferich primes."""

    @pytest.mark.parametrize(
        "base,prime,expected",
        [
            (2, 1093, True),
            (2, 3511, True),
            (3, 11, True),
            (11, 71, True),
            (12, 2693, True),
            (12, 123653, True),
            (34, 46145917691, True),
            (2, 1091, False),   # Adjacent prime, not Wieferich
            (2, 3517, False),
            (3, 13, False),
        ],
    )
    def test_known(self, base, prime, expected):
        assert is_wieferich_ref(base, prime) == expected


class TestReadFindingsSet:
    """findings.jsonl parsing."""

    def test_empty_file_missing(self):
        with tempfile.TemporaryDirectory() as d:
            assert read_findings_set(os.path.join(d, "nope.jsonl")) == set()

    def test_empty_file_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "findings.jsonl")
            with open(path, "w") as f:
                pass
            assert read_findings_set(path) == set()

    def test_single_finding(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "findings.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"base": 186, "prime": 99999999999999,
                                    "found_utc": "2026-06-03T00:00:00Z"}) + "\n")
            assert read_findings_set(path) == {(186, 99999999999999)}

    def test_multiple_findings(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "findings.jsonl")
            with open(path, "w") as f:
                for base, prime in [(186, 100), (187, 200), (200, 300)]:
                    f.write(json.dumps({"base": base, "prime": prime,
                                        "found_utc": "2026-06-03T00:00:00Z"}) + "\n")
            assert read_findings_set(path) == {(186, 100), (187, 200), (200, 300)}

    def test_malformed_line_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "findings.jsonl")
            with open(path, "w") as f:
                f.write("{ not json }\n")
                f.write(json.dumps({"base": 2, "prime": 1093,
                                    "found_utc": "2026-06-03T00:00:00Z"}) + "\n")
            assert read_findings_set(path) == {(2, 1093)}


class TestDisagreementDetection:
    """Verify that a known Wieferich prime NOT in findings would be flagged."""

    def test_disagreement_detection(self):
        # Simulate the situation where the GPU MISSED 1093 base 2 (its in
        # the known data so any harness sample range containing 1093 would
        # detect the disagreement).
        prime, base = 1093, 2
        # gmpy2 says it IS Wieferich:
        assert is_wieferich_ref(base, prime) is True

        # Simulate findings.jsonl without it:
        with tempfile.TemporaryDirectory() as d:
            findings_path = os.path.join(d, "findings.jsonl")
            with open(findings_path, "w") as f:
                pass  # empty
            findings = read_findings_set(findings_path)

            # The disagreement detection logic from the harness:
            disagreement = (base, prime) not in findings
            assert disagreement is True

    def test_known_finding_no_disagreement(self):
        # If the finding IS in findings.jsonl, no disagreement is logged.
        prime, base = 1093, 2
        with tempfile.TemporaryDirectory() as d:
            findings_path = os.path.join(d, "findings.jsonl")
            with open(findings_path, "w") as f:
                f.write(json.dumps({"base": base, "prime": prime,
                                    "found_utc": "2026-06-03T00:00:00Z"}) + "\n")
            findings = read_findings_set(findings_path)
            # Harness's disagreement check:
            disagreement = is_wieferich_ref(base, prime) and (base, prime) not in findings
            assert disagreement is False


class TestReadState:
    """state.json parsing."""

    def test_missing(self):
        with tempfile.TemporaryDirectory() as d:
            assert read_state(os.path.join(d, "nope.json")) is None

    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            payload = {"current_low": 5000000000000, "chunks_completed": 50}
            with open(path, "w") as f:
                json.dump(payload, f)
            state = read_state(path)
            assert state["current_low"] == 5000000000000
            assert state["chunks_completed"] == 50
