"""Parallel Wieferich prime search via multiprocessing.

Wraps the reference search (scripts.wieferich_search) in a chunked
multiprocessing.Pool. Each worker independently scans a sub-range of
primes for Wieferich-ness using the same is_wieferich primitive
verified in Layer 1.

Correctness is checked against the single-threaded reference in
tests/test_parallel_consistency.py (Layer 4 / Layer 7 light). Any
disagreement at any prime is a hard stop.

For ranges where the search would short-circuit early in the serial
implementation but the actual Wieferich prime is near the upper end
(or does not exist), this parallel version is dramatically faster
because all chunks proceed concurrently.

Entry points:

- parallel_all_wieferich_in_range(base, low, high, n_workers=None):
  return sorted list of all Wieferich primes to `base` in [low, high]

- parallel_find_first_wieferich(base, upper_bound, lower_bound=2,
  n_workers=None): return smallest Wieferich prime in the range, or
  None if none exists
"""
import os
import multiprocessing as mp
from typing import Optional

# Import paths assume the script is run from the project root.
# The PROJECT_ROOT sys.path insertion in test harnesses takes care
# of the development case. For worker processes spawned via
# multiprocessing, the project root must already be on sys.path.
from scripts.modexp import is_wieferich
from scripts.sieve import segmented_primes


def _scan_chunk(args: tuple) -> list[int]:
    """Worker: scan one prime-range chunk for Wieferich primes.

    Args is a 4-tuple: (base, chunk_low, chunk_high, segment_size).
    Returns the list of Wieferich primes found in the chunk.
    """
    base, low, high, segment_size = args
    found = []
    for p in segmented_primes(low, high, segment_size=segment_size):
        if is_wieferich(base, p):
            found.append(p)
    return found


def parallel_all_wieferich_in_range(
    base: int,
    low: int,
    high: int,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    segment_size: int = 1 << 20,
) -> list[int]:
    """Find all Wieferich primes to `base` with low <= p <= high, in parallel.

    Returns a sorted list. n_workers defaults to os.cpu_count(). chunk_size
    defaults to roughly (range size) / (4 * n_workers) so each worker gets
    multiple chunks for load balancing, but never smaller than segment_size.
    """
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if high < low:
        return []
    low = max(low, 2)
    if high < low:
        return []

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    total = high - low + 1
    if chunk_size is None:
        chunk_size = max(total // (n_workers * 4), segment_size)
    chunk_size = max(chunk_size, 1)

    chunks = []
    cur = low
    while cur <= high:
        chunk_high = min(cur + chunk_size - 1, high)
        chunks.append((base, cur, chunk_high, segment_size))
        cur = chunk_high + 1

    if len(chunks) == 1 or n_workers == 1:
        results = [_scan_chunk(chunks[0])] if chunks else []
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_scan_chunk, chunks)

    return sorted(p for sublist in results for p in sublist)


def parallel_find_first_wieferich(
    base: int,
    upper_bound: int,
    lower_bound: int = 2,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    segment_size: int = 1 << 20,
) -> Optional[int]:
    """Return the smallest Wieferich prime to `base` in [lower_bound, upper_bound].

    Returns None if none exists in the range. Unlike the serial
    find_first_wieferich, this does not short-circuit on the first
    discovery: all chunks complete before the minimum can be determined.
    """
    finds = parallel_all_wieferich_in_range(
        base,
        lower_bound,
        upper_bound,
        n_workers=n_workers,
        chunk_size=chunk_size,
        segment_size=segment_size,
    )
    return finds[0] if finds else None
