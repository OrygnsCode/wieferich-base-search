"""Numba sieve correctness: bit-identical agreement with the pure-Python sieve.

Run from the project root:
    python -m pytest tests/test_numba_sieve.py -v
"""
import os
import sys
import time

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sympy import primerange

from scripts.sieve import segmented_primes as segmented_primes_python
from scripts.sieve_numba import (
    segmented_primes_numba,
    segmented_prime_chunks_numba,
)


class TestNumbaSieveAgreesWithSympy:
    """The numba sieve produces the same primes as sympy.primerange."""

    @pytest.mark.parametrize(
        "low,high",
        [
            (2, 100),
            (2, 1000),
            (2, 10000),
            (10**4, 10**4 + 10000),
            (10**6, 10**6 + 10000),
            (10**9, 10**9 + 10000),
            (10**10, 10**10 + 5000),
            (0, 1000),
            (1, 100),
        ],
    )
    def test_matches_sympy(self, low, high):
        ours = list(segmented_primes_numba(low, high))
        theirs = list(primerange(max(low, 2), high + 1))
        assert ours == theirs


class TestNumbaSieveAgreesWithPython:
    """The numba sieve matches the existing pure-Python sieve exactly."""

    @pytest.mark.parametrize(
        "low,high",
        [
            (2, 1000),
            (2, 100000),
            (10**6, 10**6 + 100000),
            (10**9, 10**9 + 100000),
        ],
    )
    def test_matches_python_sieve(self, low, high):
        ours = list(segmented_primes_numba(low, high))
        theirs = list(segmented_primes_python(low, high))
        assert ours == theirs


class TestNumbaChunkedOutput:
    """The chunked variant yields numpy arrays that concatenate to the right list."""

    def test_chunks_concatenate_correctly(self):
        chunks = list(segmented_prime_chunks_numba(2, 10**5, segment_size=1024))
        flat = np.concatenate(chunks).tolist()
        expected = list(primerange(2, 10**5 + 1))
        assert flat == expected

    def test_chunks_are_uint64(self):
        chunks = list(segmented_prime_chunks_numba(2, 1000))
        assert all(c.dtype == np.uint64 for c in chunks)

    def test_empty_range_yields_nothing(self):
        chunks = list(segmented_prime_chunks_numba(100, 50))
        assert chunks == []


class TestNumbaSieveSpeed:
    """Soft check: numba sieve should be substantially faster than the Python sieve.

    Not a hard correctness gate, but if numba is not faster on a non-trivial
    range, the optimization didn't take effect and something is wrong.
    """

    def test_numba_beats_python_on_large_range(self):
        """Sieve to 10^7 should be faster with numba than pure Python."""
        # Warm up numba JIT first (excluded from timing)
        _ = list(segmented_primes_numba(2, 100))

        t0 = time.perf_counter()
        n_numba = sum(1 for _ in segmented_primes_numba(2, 10**7))
        numba_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_python = sum(1 for _ in segmented_primes_python(2, 10**7))
        python_time = time.perf_counter() - t0

        print(
            f"\nSieve to 10^7: numba={numba_time:.2f}s ({n_numba:,} primes), "
            f"python={python_time:.2f}s ({n_python:,} primes), "
            f"speedup={python_time/numba_time:.1f}x"
        )
        assert n_numba == n_python
        assert numba_time < python_time, (
            f"Numba ({numba_time:.2f}s) should beat Python ({python_time:.2f}s)"
        )
