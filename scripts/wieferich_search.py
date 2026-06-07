"""Wieferich prime search drivers.

Composes the prime sieve (scripts.sieve) with the modexp primitive
(scripts.modexp.is_wieferich) into the actual search.

For Phase 1 testing this is single-threaded reference Python. For
production-scale runs, the function will be parallelized across CPU
cores or moved to a GPU kernel; any optimized variant must agree with
this reference implementation on every test range before being used
in anger.

Entry points:

- scan_base_for_wieferich(base, low, high): generator. Yields each
  prime p in [low, high] that is Wieferich to `base`.

- find_first_wieferich(base, upper_bound, lower_bound=2): return the
  smallest Wieferich prime to `base` in [lower_bound, upper_bound], or
  None if none exists in that range.

- all_wieferich_in_range(base, low, high): return the list of all
  Wieferich primes to `base` in [low, high].

Base convention:
- We allow base >= 1. Base 1 is degenerate (1^k = 1, so every prime is
  trivially Wieferich) and is included for OEIS consistency with
  A039951(1) = 2.
- Negative or zero bases raise ValueError.
"""
from typing import Iterator

from scripts.modexp import is_wieferich
from scripts.sieve import segmented_primes


def scan_base_for_wieferich(
    base: int,
    low: int,
    high: int,
    segment_size: int = 1 << 20,
) -> Iterator[int]:
    """Yield each prime p with low <= p <= high that is Wieferich to `base`."""
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    for p in segmented_primes(low, high, segment_size=segment_size):
        if is_wieferich(base, p):
            yield p


def find_first_wieferich(
    base: int,
    upper_bound: int,
    lower_bound: int = 2,
    segment_size: int = 1 << 20,
) -> int | None:
    """Return the smallest Wieferich prime to `base` in [lower_bound, upper_bound].

    Returns None if no Wieferich prime exists in the range.
    """
    for p in scan_base_for_wieferich(
        base, lower_bound, upper_bound, segment_size=segment_size
    ):
        return p
    return None


def all_wieferich_in_range(
    base: int,
    low: int,
    high: int,
    segment_size: int = 1 << 20,
) -> list[int]:
    """Return all Wieferich primes to `base` with low <= p <= high."""
    return list(
        scan_base_for_wieferich(base, low, high, segment_size=segment_size)
    )
