"""Continuous spot-check harness for Phase 6.

Runs on the CPU, hunting for false negatives in the Phase 6 GPU pipeline.

How it works:
- Periodically reads state.json to learn how far Phase 6 has scanned.
- Picks random short ranges in the already-scanned territory.
- Sieves primes in each range via primesieve.
- For each prime, runs the Wieferich test against all 27 target bases
  using gmpy2 (the trusted reference).
- Compares the result against findings.jsonl: if gmpy2 reports a prime
  is Wieferich to some base and that (base, prime) is NOT in findings,
  the GPU silently missed it. That's a critical false-negative event;
  the harness logs an alert and writes to a critical file the GPU
  monitor checks.

Logs to:
- gpu_verifier.jsonl  : every checked range and its summary
- gpu_verifier_alerts.log : any disagreements (should stay empty)

Runs in its own PowerShell window in parallel with Phase 6. The work
is CPU-bound and uses gmpy2 (lightweight). Phase 6 uses the GPU and
primesieve (also primarily CPU but small fraction). Should coexist
without competing for the bottleneck.

Run:
    python scripts\\spot_check_harness.py

Or with a different sample size / range size:
    python scripts\\spot_check_harness.py --range-size 100000 --sleep 10
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import gmpy2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.sieve_primesieve import primesieve_generate_primes
from scripts.scan_multibase_gpu import TARGET_BASES


DEFAULT_STATE_DIR = os.path.join("runs", "phase6")
DEFAULT_RANGE_SIZE = 50000          # how many integers wide each sample range is
DEFAULT_SAFETY_MARGIN = 5_000_000_000   # stay this far behind current_low (5 billion)
DEFAULT_SLEEP_SEC = 5               # pause between checks


def is_wieferich_ref(base: int, prime: int) -> bool:
    """gmpy2-backed reference implementation."""
    return int(gmpy2.powmod(base, prime - 1, prime * prime)) == 1


def read_state(state_file: str) -> dict | None:
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_findings_set(findings_file: str) -> set:
    """Return set of (base, prime) tuples from findings.jsonl."""
    if not os.path.exists(findings_file):
        return set()
    result = set()
    with open(findings_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                result.add((int(rec["base"]), int(rec["prime"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return result


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 spot-check harness.")
    parser.add_argument("--state-dir", type=str, default=DEFAULT_STATE_DIR)
    parser.add_argument("--range-size", type=int, default=DEFAULT_RANGE_SIZE,
                        help="Width of each random sample range (default 50000)")
    parser.add_argument("--safety-margin", type=int, default=DEFAULT_SAFETY_MARGIN,
                        help="Stay this many integers behind the GPU position (default 5e9)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SEC,
                        help="Seconds between samples (default 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: use system entropy)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    state_file = os.path.join(args.state_dir, "state.json")
    findings_file = os.path.join(args.state_dir, "findings.jsonl")
    verifier_log = os.path.join(args.state_dir, "gpu_verifier.jsonl")
    alert_log = os.path.join(args.state_dir, "gpu_verifier_alerts.log")
    os.makedirs(args.state_dir, exist_ok=True)

    print(f"Spot-check harness starting.")
    print(f"  State dir: {args.state_dir}")
    print(f"  Bases: {len(TARGET_BASES)}")
    print(f"  Range size per sample: {args.range_size:,}")
    print(f"  Safety margin behind GPU: {args.safety_margin:,}")
    print(f"  Sleep between samples: {args.sleep}s")
    print(f"  Verifier log: {verifier_log}")
    print(f"  Alert log:   {alert_log}")
    print()
    sys.stdout.flush()

    # Wait for Phase 6 state file to exist
    while True:
        state = read_state(state_file)
        if state is not None and "current_low" in state:
            break
        print("Waiting for Phase 6 state.json to appear...")
        sys.stdout.flush()
        time.sleep(5)

    # Get the original start point from config.json so we know our lower bound
    config_file = os.path.join(args.state_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"ERROR: config.json not found at {config_file}. "
              "Is Phase 6 driver running?")
        return 1
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    scan_start = int(config["start"])

    total_ranges_checked = 0
    total_primes_checked = 0
    total_findings_verified = 0
    total_disagreements = 0
    run_start = time.perf_counter()

    try:
        while True:
            state = read_state(state_file)
            if state is None:
                time.sleep(args.sleep)
                continue
            current_low = int(state["current_low"])
            safe_upper = current_low - args.safety_margin
            if safe_upper <= scan_start + args.range_size:
                # Not enough already-tested range yet; sleep and wait
                time.sleep(args.sleep)
                continue

            # Pick a random range within already-tested + safety-margin territory
            sample_low = random.randint(scan_start, safe_upper - args.range_size)
            sample_high = sample_low + args.range_size - 1

            # Sieve primes in the range
            primes = primesieve_generate_primes(sample_low, sample_high)
            if len(primes) == 0:
                continue

            # Load current findings set (changes as Phase 6 finds things)
            findings_set = read_findings_set(findings_file)

            # Test each prime against each base via gmpy2
            disagreements_this_range = []
            findings_verified_this_range = []
            for p in primes:
                p_int = int(p)
                for base in TARGET_BASES:
                    ref = is_wieferich_ref(base, p_int)
                    if ref:
                        if (base, p_int) in findings_set:
                            findings_verified_this_range.append((base, p_int))
                            total_findings_verified += 1
                        else:
                            # gmpy2 says Wieferich, Phase 6 didn't record it.
                            disagreements_this_range.append((base, p_int))
                            total_disagreements += 1
                total_primes_checked += 1

            total_ranges_checked += 1

            # Log the range record
            record = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "range_low": sample_low,
                "range_high": sample_high,
                "n_primes": len(primes),
                "n_disagreements": len(disagreements_this_range),
                "n_findings_verified": len(findings_verified_this_range),
                "gpu_position_at_check": current_low,
            }
            write_jsonl(verifier_log, record)

            # Alert on disagreements
            if disagreements_this_range:
                ts_now = datetime.now(timezone.utc).isoformat()
                alert_line = (
                    f"[{ts_now}] DISAGREEMENT: gmpy2 says Wieferich for "
                    f"{len(disagreements_this_range)} (base,prime) pairs not in "
                    f"findings.jsonl. range=[{sample_low:,}, {sample_high:,}]. "
                    f"Pairs: {disagreements_this_range[:5]}"
                )
                print()
                print("!" * 72)
                print(alert_line)
                print("!" * 72)
                print()
                with open(alert_log, "a", encoding="utf-8") as f:
                    f.write(alert_line + "\n")

            # Occasional progress line
            if total_ranges_checked % 10 == 0:
                run_elapsed = time.perf_counter() - run_start
                rate = total_primes_checked / run_elapsed if run_elapsed > 0 else 0
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"ranges_checked={total_ranges_checked} "
                    f"primes_checked={total_primes_checked:,} "
                    f"rate={rate:.0f}/s "
                    f"findings_verified={total_findings_verified} "
                    f"disagreements={total_disagreements} "
                    f"gpu_pos={current_low:,}"
                )
                sys.stdout.flush()

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print()
        print("Spot-check harness stopped by user.")
        run_elapsed = time.perf_counter() - run_start
        print(f"Total: {total_ranges_checked} ranges, "
              f"{total_primes_checked:,} primes, "
              f"{total_findings_verified} findings verified, "
              f"{total_disagreements} disagreements, "
              f"elapsed {run_elapsed:.0f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
