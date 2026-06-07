"""Correctness audit for the production GPU kernel (v4).

The v4 kernel does modular arithmetic modulo p^2 by p-adic decomposition,
which keeps every intermediate product inside 128 bits and stays exact even
when p is large. This test confirms it agrees with gmpy2 (GMP, arbitrary
precision) across the full range of prime sizes used in the search, including
the regime where a naive 128-bit kernel would silently overflow.

Run from the project root:
    python -m pytest tests/test_gpu_v4_audit.py -v
"""
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sympy import nextprime, primerange
    import gmpy2
    from scripts.wieferich_gpu_v4 import (
        gpu_wieferich_batch_v4,
        gpu_wieferich_multibase_v4,
        decode_bitmask_v4,
    )
    GPU_AVAILABLE = True
except (ImportError, OSError):
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="cupy/CUDA not available")


def is_wief_ref(base, prime):
    return int(gmpy2.powmod(base, prime - 1, prime * prime)) == 1


def primes_near(target, count):
    primes = []
    p = int(nextprime(target - 1))
    while len(primes) < count:
        primes.append(p)
        p = int(nextprime(p))
    return np.array(primes, dtype=np.uint64)


KNOWN = [
    (2, 1093), (2, 3511), (3, 11), (5, 2), (7, 5), (11, 71),
    (12, 2693), (12, 123653), (13, 2), (14, 29), (15, 29131),
    (17, 2), (20, 281), (34, 46145917691), (39, 8039),
]

SCALES = [33, 35, 40, 45, 47]
BASES = [2, 3, 47, 186, 200, 304]


class TestV4KnownWieferich:
    @pytest.mark.parametrize("base,prime", KNOWN)
    def test_known_value(self, base, prime):
        result = gpu_wieferich_batch_v4(base, np.array([prime], dtype=np.uint64))
        assert result[0], f"v4 failed on known A039951({base}) = {prime}"


class TestV4PositiveControl:
    def test_base_941(self):
        p = 64501672625861
        result = gpu_wieferich_batch_v4(941, np.array([p], dtype=np.uint64))
        assert result[0]
        assert is_wief_ref(941, p)


class TestV4OverflowScales:
    @pytest.mark.parametrize("k", SCALES)
    @pytest.mark.parametrize("base", BASES)
    def test_agrees_with_gmpy2(self, k, base):
        primes = primes_near(2 ** k, 200)
        gpu = gpu_wieferich_batch_v4(base, primes)
        ref = np.array([is_wief_ref(base, int(p)) for p in primes], dtype=bool)
        if not np.array_equal(gpu, ref):
            bad = primes[gpu != ref]
            raise AssertionError(
                f"v4 vs gmpy2 mismatch at 2^{k}, base {base}: {bad[:3].tolist()}"
            )


class TestV4Multibase:
    @pytest.mark.parametrize("k", SCALES)
    def test_multibase_agrees(self, k):
        primes = primes_near(2 ** k, 100)
        bitmask = gpu_wieferich_multibase_v4(BASES, primes)
        decoded = decode_bitmask_v4(bitmask, len(BASES))
        ref = np.zeros((len(primes), len(BASES)), dtype=bool)
        for bi, base in enumerate(BASES):
            for pi, p in enumerate(primes):
                ref[pi, bi] = is_wief_ref(int(base), int(p))
        assert np.array_equal(decoded, ref), f"multibase mismatch at 2^{k}"


class TestV4SmallScale:
    @pytest.mark.parametrize(
        "base,upper,expected",
        [(2, 5000, [1093, 3511]), (3, 100, [11]), (11, 100, [71]),
         (12, 200000, [2693, 123653])],
    )
    def test_small_findings(self, base, upper, expected):
        primes = np.array(list(primerange(2, upper + 1)), dtype=np.uint64)
        mask = gpu_wieferich_batch_v4(base, primes)
        assert sorted(int(p) for p in primes[mask]) == expected
