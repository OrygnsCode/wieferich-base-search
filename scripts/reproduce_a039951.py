"""Layer 4 of the verification framework: reproduce all 46 known A039951 terms.

For each n in {2, 3, ..., 46}, this script independently re-derives
the smallest Wieferich prime to base n by scanning all primes up to
A039951(n) + buffer, and verifies the result equals the published value.

n = 1 is omitted because A039951(1) = 2 is a convention (base 1 is
degenerate, since 1^(p-1) - 1 = 0 for every prime p, so every prime
trivially satisfies the condition).

Most terms verify in well under a second. The slow term is a(34) =
46,145,917,691 (about 4.6e10), which requires scanning ~1.7 billion
primes. Single-threaded with gmpy2 this takes several minutes; on
24 cores it should drop to under 30 seconds.

Run from the project root:
    python scripts/reproduce_a039951.py
    python scripts/reproduce_a039951.py --skip-slow

Pass condition: every reproduced value equals the OEIS-published value.
Exit code 0 on success, 1 on any failure.
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.wieferich_search import find_first_wieferich
from scripts.wieferich_search_parallel import parallel_find_first_wieferich


# Above this expected-value threshold, use the multiprocessing
# implementation. Below it, Pool startup overhead is not worth it.
PARALLEL_THRESHOLD = 10**6


# Verbatim from OEIS A039951 DATA section (n = 1 .. 46).
# 1-indexed; skipping n = 1 because base 1 is degenerate.
A039951_VALUES = {
    2: 1093, 3: 11, 4: 1093, 5: 2, 6: 66161, 7: 5, 8: 3, 9: 2, 10: 3,
    11: 71, 12: 2693, 13: 2, 14: 29, 15: 29131, 16: 1093, 17: 2, 18: 5,
    19: 3, 20: 281, 21: 2, 22: 13, 23: 13, 24: 5, 25: 2, 26: 3, 27: 11,
    28: 3, 29: 2, 30: 7, 31: 7, 32: 5, 33: 2, 34: 46145917691, 35: 3,
    36: 66161, 37: 2, 38: 17, 39: 8039, 40: 11, 41: 2, 42: 23, 43: 5,
    44: 3, 45: 2, 46: 3,
}


def reproduce_one(n: int, expected: int) -> dict:
    """Verify A039951(n) by independently finding the smallest base-n Wieferich.

    Scans primes in [2, expected + buffer]. Buffer is the larger of 100 or
    0.1% of expected. For expected > PARALLEL_THRESHOLD, uses the
    multiprocessing implementation; otherwise uses the serial reference.
    """
    buffer = max(int(expected * 0.001), 100)
    upper = expected + buffer
    use_parallel = expected > PARALLEL_THRESHOLD
    t_start = time.perf_counter()
    if use_parallel:
        found = parallel_find_first_wieferich(n, upper)
    else:
        found = find_first_wieferich(n, upper)
    elapsed = time.perf_counter() - t_start
    return {
        "n": n,
        "expected": expected,
        "found": found,
        "pass": (found == expected),
        "elapsed_seconds": elapsed,
        "scan_upper_bound": upper,
        "implementation": "parallel" if use_parallel else "serial",
    }


def main(skip_slow: bool = False, slow_threshold: int = 10**6) -> int:
    print(f"Layer 4 verification: reproduce A039951 known terms")
    print(f"Start (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Skip slow (a(n) > {slow_threshold:,}): {skip_slow}")
    print(f"Number of terms: {len(A039951_VALUES)}")
    print()
    print(f"  {'n':>3}  {'a(n)':>20}  {'found':>20}  verdict  impl     time")
    print(f"  {'-' * 3}  {'-' * 20}  {'-' * 20}  -------  -------  ----")

    results = []
    failures = 0
    total_time = 0.0
    items_skipped = 0

    for n, expected in sorted(A039951_VALUES.items()):
        if skip_slow and expected > slow_threshold:
            print(
                f"  {n:>3}  {expected:>20,}  {'':>20}  SKIPPED  (a(n) above threshold)"
            )
            items_skipped += 1
            continue
        # Flush stdout so user sees per-line progress for slow terms
        sys.stdout.flush()
        rec = reproduce_one(n, expected)
        results.append(rec)
        total_time += rec["elapsed_seconds"]
        verdict = "PASS" if rec["pass"] else "FAIL"
        print(
            f"  {n:>3}  {expected:>20,}  {rec['found']!s:>20}  {verdict}     "
            f"{rec['implementation']:>7}  {rec['elapsed_seconds']:.2f}s"
        )
        if not rec["pass"]:
            failures += 1

    print()
    print(f"Total: {len(results)} reproduced, {items_skipped} skipped")
    print(f"Failures: {failures}")
    print(f"Total wall time: {total_time:.2f}s")

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"layer4_reproduce_a039951_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "phase": "Layer 4: reproduce A039951",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "skip_slow": skip_slow,
                "slow_threshold": slow_threshold,
                "results": results,
                "skipped": items_skipped,
                "failures": failures,
                "total_elapsed_seconds": total_time,
                "verdict": "PASS" if failures == 0 else "FAIL",
            },
            f,
            indent=2,
        )
    print(f"Log: {log_path}")
    print(f"Verdict: {'PASS' if failures == 0 else 'FAIL'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 4 verification: reproduce all known A039951 terms"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip a(n) where a(n) exceeds the slow threshold (default 10^6)",
    )
    parser.add_argument(
        "--slow-threshold",
        type=int,
        default=10**6,
        help="Threshold above which a(n) is considered 'slow' (default 10^6)",
    )
    args = parser.parse_args()
    sys.exit(main(skip_slow=args.skip_slow, slow_threshold=args.slow_threshold))
