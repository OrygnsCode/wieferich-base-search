"""Segmented sieve of Eratosthenes for prime enumeration.

Two entry points:

- small_sieve(limit): returns the list of all primes p such that 2 <= p <= limit.
  Used for finding the "base primes" up to sqrt(high) used by the segmented sieve.

- segmented_primes(low, high, segment_size=...): yields all primes p with
  low <= p <= high. Memory cost is segment_size bytes per iteration.

This implementation prioritizes correctness over speed. It is verified
against sympy.primerange in tests/test_sieve.py (Layer 2). For the production
run, this may be wrapped or replaced by a faster implementation, but the
faster implementation must agree with this one on every test range before
it is used in anger.
"""
from typing import Iterator
import math


def small_sieve(limit: int) -> list[int]:
    """Standard sieve of Eratosthenes up to and including limit.

    Returns a list of primes in ascending order. Returns [] for limit < 2.
    """
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            # Mark all multiples of i from i*i to limit
            sieve[i * i : limit + 1 : i] = b"\x00" * (
                (limit - i * i) // i + 1
            )
    return [i for i in range(2, limit + 1) if sieve[i]]


def segmented_primes(
    low: int,
    high: int,
    segment_size: int = 1 << 20,
) -> Iterator[int]:
    """Yield primes p with low <= p <= high.

    Uses segmented Eratosthenes with base primes up to sqrt(high). Each
    segment uses segment_size bytes of bool-array storage.

    Edge cases:
    - If high < 2 or high < low, yields nothing.
    - low is clamped up to 2.
    - p == 2 and p == 3 are included where applicable.
    """
    if high < 2 or high < low:
        return
    low = max(low, 2)

    base_limit = math.isqrt(high)
    base_primes = small_sieve(base_limit)

    seg_low = low
    while seg_low <= high:
        seg_high = min(seg_low + segment_size - 1, high)
        size = seg_high - seg_low + 1
        sieve = bytearray(b"\x01") * size

        for p in base_primes:
            if p * p > seg_high:
                break
            # First multiple of p that is >= seg_low and >= p*p
            first_mult = ((seg_low + p - 1) // p) * p
            start = max(p * p, first_mult)
            if start > seg_high:
                continue
            offset = start - seg_low
            # Mark sieve[offset], sieve[offset + p], ... as composite
            sieve[offset:size:p] = b"\x00" * ((size - offset - 1) // p + 1)

        for i in range(size):
            if sieve[i]:
                yield seg_low + i

        seg_low = seg_high + 1


def count_primes_in_range(low: int, high: int) -> int:
    """Count primes p with low <= p <= high using the segmented sieve."""
    count = 0
    for _ in segmented_primes(low, high):
        count += 1
    return count
