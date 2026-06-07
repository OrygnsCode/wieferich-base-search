"""Numba JIT-compiled segmented sieve.

Drop-in replacement for the inner loop of scripts.sieve.segmented_primes
that runs the composite-marking pass in numba-compiled native code rather
than Python bytecode. This bypasses the Python sieve bottleneck observed
in Layer 6.

Two entry points:

- segmented_primes_numba(low, high, segment_size): generator yielding
  individual primes p with low <= p <= high. Drop-in for the existing
  scripts.sieve.segmented_primes.

- segmented_prime_chunks_numba(low, high, segment_size): generator
  yielding numpy uint64 arrays of primes per segment. Better for the
  GPU-batching use case because each chunk can go directly to the GPU
  kernel without per-prime Python work.

This implementation is verified against the pure-Python sieve
(scripts.sieve.segmented_primes) in tests/test_numba_sieve.py.
Bit-identical agreement is required on every test range.
"""
import math
from typing import Iterator

import numpy as np
from numba import njit

from scripts.sieve import small_sieve


@njit(cache=True)
def _mark_segment(seg_low: int, seg_high: int, base_primes: np.ndarray) -> np.ndarray:
    """Mark composites in [seg_low, seg_high] using base_primes.

    Returns a bool array of length (seg_high - seg_low + 1) where True
    indicates the corresponding integer is a candidate prime (not marked
    composite). Caller is responsible for further filtering (e.g.,
    excluding values < 2).

    Inputs:
    - seg_low: int, lower bound of segment
    - seg_high: int, upper bound of segment (inclusive)
    - base_primes: np.ndarray of int64, base primes up to >= sqrt(seg_high)
    """
    size = seg_high - seg_low + 1
    sieve = np.ones(size, dtype=np.bool_)
    for idx in range(len(base_primes)):
        p = base_primes[idx]
        if p * p > seg_high:
            break
        # First multiple of p in [seg_low, seg_high], starting from p*p
        first_mult = ((seg_low + p - 1) // p) * p
        start = first_mult if first_mult > p * p else p * p
        if start > seg_high:
            continue
        offset = start - seg_low
        for j in range(offset, size, p):
            sieve[j] = False
    return sieve


def segmented_prime_chunks_numba(
    low: int,
    high: int,
    segment_size: int = 1 << 22,
) -> Iterator[np.ndarray]:
    """Yield numpy uint64 arrays of primes p with low <= p <= high.

    Each yielded array is the set of primes within one segment. Empty
    segments are skipped (no zero-length arrays yielded).
    """
    if high < 2 or high < low:
        return
    low = max(low, 2)

    base_limit = math.isqrt(high)
    base_primes = np.array(small_sieve(base_limit), dtype=np.int64)
    if len(base_primes) == 0:
        # high is 2 or 3
        primes = np.array(
            [n for n in range(max(low, 2), high + 1) if n in (2, 3)],
            dtype=np.uint64,
        )
        if len(primes) > 0:
            yield primes
        return

    seg_low = low
    while seg_low <= high:
        seg_high = min(seg_low + segment_size - 1, high)
        sieve = _mark_segment(seg_low, seg_high, base_primes)

        # Indices where sieve is True correspond to primes in segment
        positions = np.where(sieve)[0]
        if len(positions) > 0:
            primes = (seg_low + positions).astype(np.uint64)
            # Filter out values < 2 (only matters in the first segment)
            if seg_low < 2:
                primes = primes[primes >= 2]
            if len(primes) > 0:
                yield primes
        seg_low = seg_high + 1


def segmented_primes_numba(
    low: int,
    high: int,
    segment_size: int = 1 << 22,
) -> Iterator[int]:
    """Yield individual primes p with low <= p <= high.

    Drop-in replacement for scripts.sieve.segmented_primes.
    """
    for chunk in segmented_prime_chunks_numba(low, high, segment_size=segment_size):
        for p in chunk:
            yield int(p)
