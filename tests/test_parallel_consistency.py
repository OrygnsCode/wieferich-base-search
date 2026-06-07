"""Layer 7 (light): parallel implementation agrees with serial reference.

For every test range, parallel_all_wieferich_in_range and the serial
all_wieferich_in_range must return exactly the same list. Any disagreement
is a hard stop.

This is also a soft cost gate: if the parallel implementation is not
significantly faster than the serial one on a moderately sized range,
something is wrong with the chunking or the pool setup.

Run from the project root:
    python -m pytest tests/test_parallel_consistency.py -v
"""
import os
import sys
import time
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.wieferich_search import all_wieferich_in_range, find_first_wieferich
from scripts.wieferich_search_parallel import (
    parallel_all_wieferich_in_range,
    parallel_find_first_wieferich,
)


class TestParallelAgreesWithSerial:
    """Identical output between serial and parallel on multiple ranges."""

    @pytest.mark.parametrize(
        "base,low,high",
        [
            (2, 2, 10000),
            (3, 2, 1000),
            (5, 2, 1000),
            (7, 2, 1000),
            (11, 2, 1000),
            (12, 2, 200000),
            (47, 2, 100000),  # one of our actual target bases
            (186, 2, 100000),  # another target base
        ],
    )
    def test_parallel_agrees_with_serial(self, base, low, high):
        serial = all_wieferich_in_range(base, low, high)
        parallel = parallel_all_wieferich_in_range(base, low, high)
        assert parallel == serial, (
            f"base={base}, range=[{low}, {high}]: serial={serial}, parallel={parallel}"
        )

    @pytest.mark.parametrize(
        "base,upper",
        [(2, 5000), (3, 100), (11, 100), (12, 5000)],
    )
    def test_parallel_find_first_agrees(self, base, upper):
        serial = find_first_wieferich(base, upper)
        parallel = parallel_find_first_wieferich(base, upper)
        assert parallel == serial


# Speedup is a Phase 5 (capacity estimate) concern, not a Layer 7 correctness
# concern. The multiprocessing.Pool startup overhead dominates at small
# ranges, so a "parallel-must-be-faster" check at small N is uninformative.
# Real timing benchmarks live in scripts/capacity_estimate.py (Phase 5).


class TestParallelEdgeCases:
    """Edge cases on the parallel API."""

    def test_empty_range(self):
        assert parallel_all_wieferich_in_range(2, 100, 50) == []

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            parallel_all_wieferich_in_range(0, 2, 100)

    def test_single_chunk_path(self):
        """Tiny range should still produce correct output."""
        result = parallel_all_wieferich_in_range(2, 2, 5000)
        assert result == [1093, 3511]

    def test_one_worker(self):
        """Forcing n_workers=1 still produces correct output."""
        result = parallel_all_wieferich_in_range(2, 2, 5000, n_workers=1)
        assert result == [1093, 3511]
