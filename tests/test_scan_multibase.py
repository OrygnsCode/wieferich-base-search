"""scan_multibase (optimized decode) produces the same findings as the
single-base kernel run against each base independently.

This is the "post-optimization safety check": after the Python decode
fast path was introduced in Step 1b, we re-verify that the production
scanner still emits the same set of (base, prime) findings as the
trusted reference path.

Run from the project root:
    python -m pytest tests/test_scan_multibase.py -v
"""
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sympy import primerange
    from scripts.wieferich_gpu import gpu_wieferich_batch
    from scripts.scan_multibase_gpu import scan_multibase
    GPU_AVAILABLE = True
except (ImportError, OSError):
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="cupy/CUDA not available"
)


def reference_findings(bases, low, high):
    """Run each base independently with the single-base kernel and aggregate."""
    primes = np.array(list(primerange(max(low, 2), high + 1)), dtype=np.uint64)
    findings = set()
    for base in bases:
        mask = gpu_wieferich_batch(int(base), primes)
        for p in primes[mask]:
            findings.add((int(base), int(p)))
    return findings


class TestScanMultibaseMatchesReference:
    """scan_multibase findings = reference (single-base per base) findings."""

    @pytest.mark.parametrize(
        "bases,low,high",
        [
            ([2, 3, 11, 12], 2, 200000),     # known Wieferich primes hit
            ([47, 72, 186, 187], 2, 100000), # target bases, no Wieferich expected
            ([2, 3, 5, 7, 11], 2, 10000),
            ([186, 187, 200, 203, 222, 231, 304, 311, 335, 355], 2, 50000),
        ],
    )
    def test_findings_match(self, bases, low, high):
        opt_set = set(scan_multibase(bases, low, high))
        ref_set = reference_findings(bases, low, high)
        assert opt_set == ref_set, (
            f"Findings differ. Only in optimized: {opt_set - ref_set}. "
            f"Only in reference: {ref_set - opt_set}."
        )


class TestScanMultibaseAllZeroFastPath:
    """When no Wieferich primes exist in the range, the optimized scan
    short-circuits and yields nothing without doing per-base numpy work."""

    def test_target_bases_no_findings_under_10000(self):
        bases = [186, 187, 200, 304, 311]
        findings = list(scan_multibase(bases, 2, 10000))
        assert findings == []

    def test_empty_high_below_low(self):
        bases = [2, 3]
        findings = list(scan_multibase(bases, 1000, 500))
        assert findings == []


class TestScanMultibaseEmitsExpectedKnown:
    """The known Wieferich primes do get emitted in the optimized path."""

    def test_emits_1093_3511_for_base_2(self):
        bases = [2, 186, 187]
        findings = set(scan_multibase(bases, 2, 5000))
        assert (2, 1093) in findings
        assert (2, 3511) in findings

    def test_emits_11_for_base_3(self):
        bases = [3, 5, 7]
        findings = set(scan_multibase(bases, 2, 100))
        assert (3, 11) in findings

    def test_emits_71_for_base_11(self):
        bases = [11, 13, 17]
        findings = set(scan_multibase(bases, 2, 100))
        assert (11, 71) in findings

    def test_emits_2693_123653_for_base_12(self):
        bases = [12, 13]
        findings = set(scan_multibase(bases, 2, 200000))
        assert (12, 2693) in findings
        assert (12, 123653) in findings
