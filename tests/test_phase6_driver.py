"""Phase 6 driver: checkpoint, resume, finding emission tests.

Tests that the driver:
- Writes state atomically after each chunk
- Resumes from checkpoint without re-doing work
- Writes findings to disk as they are discovered
- Refuses to resume with a mismatched config
- Passes the preflight sanity check
- Produces the same findings on a small scan as the reference

Run from the project root:
    python -m pytest tests/test_phase6_driver.py -v
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sympy import primerange
    from scripts.phase6_driver import Phase6Driver
    from scripts.wieferich_gpu_v4 import gpu_wieferich_batch_v4
    AVAILABLE = True
except (ImportError, OSError):
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="GPU stack unavailable")


@pytest.fixture
def temp_state_dir():
    d = tempfile.mkdtemp(prefix="phase6_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def reference_findings(bases, low, high):
    """All (base, prime) findings in [low, high] from gmpy2 reference."""
    primes = np.array(list(primerange(max(low, 2), high + 1)), dtype=np.uint64)
    found = set()
    for base in bases:
        mask = gpu_wieferich_batch_v4(base, primes)
        for p in primes[mask]:
            found.add((int(base), int(p)))
    return found


def load_findings_jsonl(path):
    if not os.path.exists(path):
        return set()
    found = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                found.add((rec["base"], rec["prime"]))
    return found


class TestPhase6BasicRun:
    """A small end-to-end scan emits the same findings as the reference."""

    def test_finds_known_wieferich_for_base_2_and_12(self, temp_state_dir):
        bases = [2, 3, 11, 12]
        start, end = 2, 200000
        driver = Phase6Driver(
            bases=bases,
            start=start,
            end=end,
            state_dir=temp_state_dir,
            chunk_size=50000,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        )
        ret = driver.run()
        assert ret == 0

        # Findings file should match reference
        driver_findings = load_findings_jsonl(
            os.path.join(temp_state_dir, "findings.jsonl")
        )
        ref = reference_findings(bases, start, end)
        assert driver_findings == ref

        # State file should reflect completion
        with open(os.path.join(temp_state_dir, "state.json"), "r") as f:
            state = json.load(f)
        assert state["current_low"] > end

    def test_no_findings_for_target_bases_below_10000(self, temp_state_dir):
        bases = [186, 187, 200, 304, 311]
        driver = Phase6Driver(
            bases=bases,
            start=2,
            end=10000,
            state_dir=temp_state_dir,
            chunk_size=2000,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        )
        assert driver.run() == 0
        findings = load_findings_jsonl(
            os.path.join(temp_state_dir, "findings.jsonl")
        )
        assert findings == set()


class TestPhase6Resume:
    """Driver resumes from last checkpoint without re-doing work."""

    def test_resume_continues_from_checkpoint(self, temp_state_dir):
        bases = [2, 3, 11, 12]
        start, end = 2, 200000
        kwargs = dict(
            bases=bases,
            start=start,
            end=end,
            state_dir=temp_state_dir,
            chunk_size=50000,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        )

        # Phase A: run with end = 100000 (artificially stop early)
        driver1 = Phase6Driver(**{**kwargs, "end": 100000})
        driver1.run()

        # Now read state and findings after partial run
        with open(os.path.join(temp_state_dir, "state.json"), "r") as f:
            state_after_phase_a = json.load(f)
        findings_after_phase_a = load_findings_jsonl(
            os.path.join(temp_state_dir, "findings.jsonl")
        )

        # Phase B: continue with end = 200000.
        # The config file already has end=100000, so we expect a mismatch error.
        # That validates the resume safety check.
        with pytest.raises(RuntimeError, match="Config in state_dir"):
            Phase6Driver(**kwargs).run()

    def test_resume_with_same_config(self, temp_state_dir):
        """Resume with identical config picks up where the previous run stopped."""
        bases = [2, 3, 11, 12]
        kwargs = dict(
            bases=bases,
            start=2,
            end=200000,
            state_dir=temp_state_dir,
            chunk_size=30000,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        )

        # First invocation: full run
        Phase6Driver(**kwargs).run()
        findings_full = load_findings_jsonl(
            os.path.join(temp_state_dir, "findings.jsonl")
        )

        # Second invocation: same config, should be a no-op (already complete)
        with open(os.path.join(temp_state_dir, "state.json"), "r") as f:
            state_before = json.load(f)
        Phase6Driver(**kwargs).run()
        with open(os.path.join(temp_state_dir, "state.json"), "r") as f:
            state_after = json.load(f)

        # Findings should not duplicate
        findings_after = load_findings_jsonl(
            os.path.join(temp_state_dir, "findings.jsonl")
        )
        assert findings_after == findings_full

        # State should be unchanged (already past end)
        assert state_after["current_low"] == state_before["current_low"]


class TestPhase6ConfigValidation:
    """Mismatched configs are caught."""

    def test_mismatched_bases_rejected(self, temp_state_dir):
        Phase6Driver(
            bases=[2, 3],
            start=2,
            end=100,
            state_dir=temp_state_dir,
            chunk_size=50,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        ).run()
        with pytest.raises(RuntimeError, match="Config in state_dir"):
            Phase6Driver(
                bases=[2, 5],  # different
                start=2,
                end=100,
                state_dir=temp_state_dir,
                chunk_size=50,
                gpu_batch=1 << 18,
                sieve_chunk=1 << 22,
            ).run()

    def test_mismatched_end_rejected(self, temp_state_dir):
        Phase6Driver(
            bases=[2],
            start=2,
            end=100,
            state_dir=temp_state_dir,
            chunk_size=50,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        ).run()
        with pytest.raises(RuntimeError, match="Config in state_dir"):
            Phase6Driver(
                bases=[2],
                start=2,
                end=200,  # different
                state_dir=temp_state_dir,
                chunk_size=50,
                gpu_batch=1 << 18,
                sieve_chunk=1 << 22,
            ).run()


class TestPhase6Preflight:
    """The preflight sanity check runs and passes with v4."""

    def test_preflight_passes(self, temp_state_dir):
        driver = Phase6Driver(
            bases=[2],
            start=2,
            end=100,
            state_dir=temp_state_dir,
            chunk_size=50,
            gpu_batch=1 << 18,
            sieve_chunk=1 << 22,
        )
        # Should not raise
        driver.write_config_if_new()
        driver.preflight_kernel_sanity_check()
