"""Production-scale multi-base Wieferich scan.

Combines:
- primesieve via ctypes (scripts.sieve_primesieve.primesieve_chunks)
- Multi-base GPU kernel (scripts.wieferich_gpu_multibase.gpu_wieferich_multibase)

For each prime in [low, high], tests against every base in `bases` and
yields (base, prime) tuples for each Wieferich hit. The kernel computes
p^2 once per prime and tests all bases in a single launch, sharing the
sieve, data transfer, and modulus computation across bases.

This is the Phase 6 production scanner.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, Sequence

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.sieve_primesieve import primesieve_chunks
# Production now uses the v4 kernel (p-adic decomposition). The earlier
# v1 wieferich_gpu_multibase kernel was found to silently truncate
# unsigned __int128 multiplications for p > 2^32, returning False for
# real Wieferich primes at production scale (see test_gpu_overflow_audit.py).
# v4 exploits the m = p^2 structure and stays in __int128 throughout.
from scripts.wieferich_gpu_v4 import (
    gpu_wieferich_multibase_v4 as gpu_wieferich_multibase,
    MAX_BASES,
)


DEFAULT_CHUNK_SIZE = 10**10
DEFAULT_GPU_BATCH = 1 << 22


# The 27 open bases on the target list (the headline target).
# Bases 47 and 72 are excluded because Fischer has individually pushed
# those past 1.4e14; we do not pick a fight on his focus targets.
TARGET_BASES = [
    186, 187, 200, 203, 222, 231, 304, 311, 335, 355, 435, 454, 546, 554,
    610, 639, 662, 760, 772, 798, 808, 812, 858, 860, 871, 983, 986,
]


def scan_multibase(
    bases: Sequence[int],
    low: int,
    high: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    gpu_batch: int = DEFAULT_GPU_BATCH,
) -> Iterator[tuple]:
    """Yield (base, prime) tuples for every Wieferich hit.

    Streams primes from primesieve in chunks, batches them to the GPU
    multi-base kernel, decodes bitmasks per chunk, yields findings.

    Production fast path: if the bitmask for a batch is all-zero (which
    is the typical case since Wieferich primes are extremely rare), the
    per-base decode loop is skipped entirely. This avoids ~n_bases
    numpy operations on a 4M-element array per all-zero batch, which
    was the dominant Python-side cost identified in Step 1.
    """
    bases_list = list(bases)
    if not bases_list:
        raise ValueError("bases must be non-empty")
    if len(bases_list) > MAX_BASES:
        raise ValueError(
            f"At most {MAX_BASES} bases per scan, got {len(bases_list)}"
        )
    if high < low:
        return

    bases_arr = np.asarray(bases_list, dtype=np.uint64)

    for chunk in primesieve_chunks(low, high, chunk_size=chunk_size):
        for i in range(0, len(chunk), gpu_batch):
            sub = chunk[i : i + gpu_batch]
            bitmask = gpu_wieferich_multibase(bases_arr, sub)

            # Fast path: find indices with any bit set
            nonzero_idx = np.flatnonzero(bitmask)
            if len(nonzero_idx) == 0:
                continue

            # Slow path: decode only the primes that had at least one hit
            for ni in nonzero_idx:
                m = int(bitmask[ni])
                prime_val = int(sub[ni])
                bi = 0
                while m:
                    if m & 1:
                        yield (int(bases_arr[bi]), prime_val)
                    m >>= 1
                    bi += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-base GPU Wieferich scan over a range."
    )
    parser.add_argument(
        "--bases",
        type=str,
        default=",".join(str(b) for b in TARGET_BASES),
        help="Comma-separated bases (default: all 27 target bases)",
    )
    parser.add_argument("--low", type=int, required=True, help="lower bound inclusive")
    parser.add_argument("--high", type=int, required=True, help="upper bound inclusive")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--gpu-batch", type=int, default=DEFAULT_GPU_BATCH)
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Write per-run JSON record to this path (default: auto-generated)",
    )
    args = parser.parse_args()

    bases = [int(b) for b in args.bases.split(",") if b.strip()]
    print(f"Multi-base scan: {len(bases)} bases on [{args.low:,}, {args.high:,}]")
    print(f"Bases: {bases}")
    print(f"chunk_size={args.chunk_size:,} gpu_batch={args.gpu_batch:,}")
    sys.stdout.flush()

    t_start = time.perf_counter()
    findings = []
    primes_tested = 0
    bases_arr = np.asarray(bases, dtype=np.uint64)

    # We re-implement the scan loop here (instead of consuming the
    # generator) so we can track per-chunk timing and primes-tested
    # accurately.
    for chunk in primesieve_chunks(args.low, args.high, chunk_size=args.chunk_size):
        chunk_t0 = time.perf_counter()
        chunk_findings = 0
        for i in range(0, len(chunk), args.gpu_batch):
            sub = chunk[i : i + args.gpu_batch]
            bitmask = gpu_wieferich_multibase(bases_arr, sub)

            # Fast path: skip decode when the entire batch has zero hits.
            nonzero_idx = np.flatnonzero(bitmask)
            if len(nonzero_idx) == 0:
                continue

            for ni in nonzero_idx:
                m = int(bitmask[ni])
                prime_int = int(sub[ni])
                bi = 0
                while m:
                    if m & 1:
                        base_int = int(bases_arr[bi])
                        findings.append({"base": base_int, "prime": prime_int})
                        chunk_findings += 1
                        print(f"FOUND: base={base_int}  prime={prime_int}")
                    m >>= 1
                    bi += 1
        chunk_t1 = time.perf_counter()
        primes_tested += len(chunk)
        chunk_time = chunk_t1 - chunk_t0
        total_elapsed = chunk_t1 - t_start
        rate = len(chunk) / chunk_time if chunk_time > 0 else 0
        overall = primes_tested / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  [chunk] primes_in_chunk={len(chunk):,}  "
            f"chunk_finds={chunk_findings}  "
            f"chunk_rate={rate:.2e}/s  overall_rate={overall:.2e}/s  "
            f"total_tested={primes_tested:,}  elapsed={total_elapsed:.1f}s"
        )
        sys.stdout.flush()

    total_elapsed = time.perf_counter() - t_start
    print()
    print(f"Done. Primes tested: {primes_tested:,}")
    print(f"Total Wieferich findings: {len(findings)}")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    if total_elapsed > 0:
        print(f"End-to-end throughput: {primes_tested / total_elapsed:.2e} primes/sec")
        if len(bases) > 0:
            print(
                f"Effective per-base throughput: "
                f"{primes_tested * len(bases) / total_elapsed:.2e} (base,prime)-pairs/sec"
            )

    log_path = args.log
    if log_path is None:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = os.path.join(log_dir, f"scan_multibase_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "phase": "Phase 6: multi-base scan",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "bases": bases,
                "low": args.low,
                "high": args.high,
                "primes_tested": primes_tested,
                "findings": findings,
                "total_elapsed_seconds": total_elapsed,
                "end_to_end_primes_per_second": (
                    primes_tested / total_elapsed if total_elapsed > 0 else 0
                ),
                "effective_pairs_per_second": (
                    primes_tested * len(bases) / total_elapsed
                    if total_elapsed > 0
                    else 0
                ),
            },
            f,
            indent=2,
        )
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
