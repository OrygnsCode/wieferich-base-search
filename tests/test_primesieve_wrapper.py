"""Verify the primesieve ctypes wrapper produces identical output to
sympy.primerange and the existing Python segmented sieve.

Run from the project root:
    python -m pytest tests/test_primesieve_wrapper.py -v -s
"""
import os
import sys
import time

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.sieve_primesieve import (
        primesieve_generate_primes,
        primesieve_count,
        primesieve_chunks,
    )
    PRIMESIEVE_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError) as e:
    PRIMESIEVE_AVAILABLE = False
    PRIMESIEVE_ERROR_MSG = str(e)

pytestmark = pytest.mark.skipif(
    not PRIMESIEVE_AVAILABLE, reason="primesieve.dll not available"
)

from sympy import primerange
from scripts.sieve import segmented_primes as segmented_primes_python


class TestPrimesieveAgreesWithSympy:
    """Output matches sympy.primerange on every test range."""

    @pytest.mark.parametrize(
        "low,high",
        [
            (2, 100),
            (2, 1000),
            (2, 10000),
            (1000, 1200),  # contains 1093, first Wieferich base 2
            (3500, 3600),  # contains 3511, second Wieferich base 2
            (10**4, 10**4 + 1000),
            (10**6, 10**6 + 10000),
            (10**9, 10**9 + 10000),
            (10**10, 10**10 + 5000),
            (10**11, 10**11 + 1000),
            (0, 1000),
            (1, 100),
        ],
    )
    def test_matches_sympy(self, low, high):
        ours = primesieve_generate_primes(low, high).tolist()
        theirs = list(primerange(max(low, 2), high + 1))
        assert ours == theirs


class TestPrimesieveAgreesWithPython:
    """Output matches the existing Python segmented sieve."""

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
        ours = primesieve_generate_primes(low, high).tolist()
        theirs = list(segmented_primes_python(low, high))
        assert ours == theirs


class TestPrimesieveSpecificPrimes:
    """Spot checks for primes we care about."""

    def test_1093_present(self):
        arr = primesieve_generate_primes(1000, 1200)
        assert 1093 in arr

    def test_3511_present(self):
        arr = primesieve_generate_primes(3500, 3600)
        assert 3511 in arr

    def test_71_present(self):
        arr = primesieve_generate_primes(2, 100)
        assert 71 in arr

    def test_dtype_is_uint64(self):
        arr = primesieve_generate_primes(2, 100)
        assert arr.dtype == np.uint64


class TestPrimesieveCount:
    """primesieve_count agrees with len(generate_primes)."""

    @pytest.mark.parametrize("low,high", [(2, 1000), (2, 10**6), (10**6, 10**7)])
    def test_count_matches_length(self, low, high):
        assert primesieve_count(low, high) == len(primesieve_generate_primes(low, high))


class TestPrimesieveChunks:
    """Chunked iteration concatenates to the full prime list."""

    def test_chunks_concatenate(self):
        chunks = list(primesieve_chunks(2, 10**5, chunk_size=1000))
        flat = np.concatenate(chunks).tolist()
        expected = list(primerange(2, 10**5 + 1))
        assert flat == expected

    def test_chunks_each_uint64(self):
        chunks = list(primesieve_chunks(2, 10000))
        assert all(c.dtype == np.uint64 for c in chunks)


class TestPrimesieveSpeed:
    """primesieve must be substantially faster than the Python sieve."""

    def test_primesieve_beats_python_to_1e8(self):
        """Sieve to 10^8: primesieve should be many times faster."""
        t0 = time.perf_counter()
        n_ps = len(primesieve_generate_primes(2, 10**8))
        ps_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_py = sum(1 for _ in segmented_primes_python(2, 10**8))
        py_time = time.perf_counter() - t0

        speedup = py_time / ps_time
        print(
            f"\nSieve to 10^8: primesieve={ps_time:.3f}s ({n_ps:,} primes), "
            f"python={py_time:.3f}s ({n_py:,} primes), speedup={speedup:.0f}x"
        )
        assert n_ps == n_py
        assert speedup > 10, (
            f"Expected primesieve to be at least 10x faster, got {speedup:.1f}x"
        )
