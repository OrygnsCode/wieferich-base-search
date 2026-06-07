"""Phase 6 production driver: full Wieferich scan with checkpointing.

Runs the complete Phase 6 scan: 27 target bases pushed past Fischer's
4.40e12 broad-sweep bound up to a configurable upper bound (default 10^14).

Features:
- Chunked execution. After each chunk, state is checkpointed atomically
  to disk. If the process dies for any reason (crash, Ctrl+C, power blip,
  Windows reboot), the next invocation resumes from the last completed
  chunk. At most one chunk worth of work is lost.
- Findings are written to disk the moment they are discovered. The
  driver does not hold findings in memory across chunks.
- Verification: at startup the driver runs a v4 sanity check against
  gmpy2 on known Wieferich primes (1093, 3511, 11, a(34) = 46145917691)
  and refuses to proceed if any disagree. This catches regressions
  introduced by kernel changes, driver updates, etc.
- Resume validation: when resuming, the driver checks that the config
  (bases, start, end, chunk_size) matches the state file. Mismatched
  configs trigger a clear error rather than silent corruption.
- Heartbeat: every chunk emits a progress line with primes tested,
  findings count, throughput, and ETA.

Storage layout (under --state-dir, default ./runs/phase6/):
- state.json     : checkpoint state (atomically updated)
- findings.jsonl : one finding per line, append-only
- progress.log   : per-chunk progress, append-only
- config.json    : original config (immutable after first run)

Run:
    python scripts/phase6_driver.py --start 4400000000000 --end 100000000000000
    python scripts/phase6_driver.py --resume    # picks up state.json
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.sieve_primesieve import primesieve_chunks
from scripts.wieferich_gpu_v4 import gpu_wieferich_multibase_v4, MAX_BASES
from scripts.scan_multibase_gpu import TARGET_BASES


DEFAULT_STATE_DIR = os.path.join("runs", "phase6")
DEFAULT_CHUNK_SIZE = 10**10        # ~30 seconds per chunk at production rate
DEFAULT_GPU_BATCH = 1 << 22        # 4M primes per GPU launch
DEFAULT_SIEVE_CHUNK = 1 << 30      # sieve sub-chunk for memory bounding


class Phase6Driver:
    def __init__(
        self,
        bases,
        start: int,
        end: int,
        state_dir: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        gpu_batch: int = DEFAULT_GPU_BATCH,
        sieve_chunk: int = DEFAULT_SIEVE_CHUNK,
    ):
        if len(bases) == 0 or len(bases) > MAX_BASES:
            raise ValueError(f"bases must have 1..{MAX_BASES} entries, got {len(bases)}")
        if end <= start:
            raise ValueError(f"end ({end}) must be greater than start ({start})")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.bases = list(bases)
        self.start = int(start)
        self.end = int(end)
        self.chunk_size = int(chunk_size)
        self.gpu_batch = int(gpu_batch)
        self.sieve_chunk = int(sieve_chunk)
        self.state_dir = state_dir

        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, "state.json")
        self.findings_path = os.path.join(self.state_dir, "findings.jsonl")
        self.progress_path = os.path.join(self.state_dir, "progress.log")
        self.config_path = os.path.join(self.state_dir, "config.json")

        self._interrupted = False

    def install_signal_handler(self):
        def handler(signum, frame):
            self._interrupted = True
            self.log("Interrupt received; finishing current chunk and exiting")
        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):
            pass

    def log(self, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {message}"
        print(line, flush=True)
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_config_if_new(self) -> dict:
        config = {
            "bases": self.bases,
            "start": self.start,
            "end": self.end,
            "chunk_size": self.chunk_size,
            "gpu_batch": self.gpu_batch,
            "sieve_chunk": self.sieve_chunk,
            "v4_kernel": "wieferich_gpu_v4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            mismatches = []
            for key in ("bases", "start", "end", "chunk_size"):
                if existing.get(key) != config.get(key):
                    mismatches.append(
                        f"{key}: existing={existing.get(key)} new={config.get(key)}"
                    )
            if mismatches:
                raise RuntimeError(
                    "Config in state_dir does not match current run config:\n"
                    + "\n".join(mismatches)
                    + "\nEither resume with the original config or use a different "
                    "--state-dir."
                )
            return existing
        else:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return config

    def load_state(self) -> dict:
        if not os.path.exists(self.state_path):
            return {
                "version": 1,
                "current_low": self.start,
                "chunks_completed": 0,
                "primes_tested": 0,
                "findings_count": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state

    def save_state_atomic(self, state: dict) -> None:
        state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        temp_path = self.state_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_path, self.state_path)

    def record_finding(self, base: int, prime: int) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with open(self.findings_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"base": base, "prime": prime, "found_utc": ts}) + "\n")

    def preflight_kernel_sanity_check(self) -> None:
        """Refuse to start if v4 doesn't agree with known A039951 values."""
        import gmpy2
        cases = [
            (2, 1093), (2, 3511), (3, 11), (11, 71),
            (12, 2693), (12, 123653), (34, 46145917691),
        ]
        for base, prime in cases:
            primes_arr = np.array([prime], dtype=np.uint64)
            mask = gpu_wieferich_multibase_v4([base], primes_arr)
            gpu_says_wieferich = bool(mask[0] & 1)
            ref = int(gmpy2.powmod(base, prime - 1, prime * prime)) == 1
            if gpu_says_wieferich != ref:
                raise RuntimeError(
                    f"Preflight sanity check FAILED: "
                    f"base={base} prime={prime} GPU={gpu_says_wieferich} ref={ref}. "
                    "Do not run Phase 6 with this kernel."
                )
        self.log("Preflight kernel sanity check PASS (7 known A039951 values)")

    def process_one_chunk(self, low: int, high: int) -> tuple:
        """Process a single chunk. Returns (n_primes_tested, n_findings)."""
        bases_arr = np.asarray(self.bases, dtype=np.uint64)
        n_bases = len(bases_arr)
        n_primes = 0
        n_findings = 0
        for sub_chunk in primesieve_chunks(low, high, chunk_size=self.sieve_chunk):
            for i in range(0, len(sub_chunk), self.gpu_batch):
                sub = sub_chunk[i : i + self.gpu_batch]
                bitmask = gpu_wieferich_multibase_v4(bases_arr, sub)
                nonzero = np.flatnonzero(bitmask)
                if len(nonzero) > 0:
                    for ni in nonzero:
                        m = int(bitmask[ni])
                        prime_val = int(sub[ni])
                        bi = 0
                        while m:
                            if m & 1:
                                base_val = int(bases_arr[bi])
                                self.record_finding(base_val, prime_val)
                                self.log(f"FINDING: base={base_val} prime={prime_val}")
                                n_findings += 1
                            m >>= 1
                            bi += 1
                n_primes += len(sub)
        return n_primes, n_findings

    def run(self) -> int:
        self.install_signal_handler()
        config = self.write_config_if_new()
        state = self.load_state()

        if state["current_low"] >= self.end:
            self.log(f"State indicates run already complete at "
                     f"current_low={state['current_low']:,} >= end={self.end:,}")
            return 0

        self.log(
            f"Phase 6 driver starting. "
            f"bases={len(self.bases)} start={self.start:,} end={self.end:,} "
            f"chunk_size={self.chunk_size:,} gpu_batch={self.gpu_batch:,}"
        )
        self.log(f"Resume position: current_low={state['current_low']:,} "
                 f"chunks_completed={state['chunks_completed']} "
                 f"primes_tested={state['primes_tested']:,} "
                 f"findings_count={state['findings_count']}")

        self.preflight_kernel_sanity_check()

        run_t_start = time.perf_counter()
        chunks_this_run = 0
        primes_this_run = 0
        findings_this_run = 0

        try:
            while state["current_low"] < self.end:
                if self._interrupted:
                    self.log("Interrupt acknowledged; stopping before next chunk")
                    break

                chunk_low = state["current_low"]
                chunk_high = min(chunk_low + self.chunk_size - 1, self.end)

                t0 = time.perf_counter()
                n_primes, n_findings = self.process_one_chunk(chunk_low, chunk_high)
                elapsed = time.perf_counter() - t0

                state["current_low"] = chunk_high + 1
                state["chunks_completed"] += 1
                state["primes_tested"] += n_primes
                state["findings_count"] += n_findings
                self.save_state_atomic(state)

                chunks_this_run += 1
                primes_this_run += n_primes
                findings_this_run += n_findings

                run_elapsed = time.perf_counter() - run_t_start
                run_rate = primes_this_run / run_elapsed if run_elapsed > 0 else 0
                remaining_range = self.end - state["current_low"]
                eta_seconds = (
                    (remaining_range / (chunk_high - chunk_low + 1)) * elapsed
                    if elapsed > 0
                    else 0
                )
                eta_hours = eta_seconds / 3600

                self.log(
                    f"chunk #{state['chunks_completed']} done: "
                    f"low={chunk_low:,} high={chunk_high:,} "
                    f"primes={n_primes:,} findings={n_findings} "
                    f"time={elapsed:.1f}s rate={n_primes/elapsed:.2e}/s "
                    f"run_rate={run_rate:.2e}/s ETA={eta_hours:.1f}h"
                )

        except Exception as exc:
            self.log(f"Exception in driver: {type(exc).__name__}: {exc}")
            self.save_state_atomic(state)
            raise

        # Final state save
        self.save_state_atomic(state)

        self.log(
            f"Driver stopping. chunks_this_run={chunks_this_run} "
            f"primes_this_run={primes_this_run:,} "
            f"findings_this_run={findings_this_run} "
            f"current_low={state['current_low']:,}"
        )

        if state["current_low"] >= self.end:
            self.log("PHASE 6 SCAN COMPLETE")
            return 0
        else:
            self.log("Run paused (interrupted or exception). Resume by re-invoking.")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 production Wieferich scan driver")
    parser.add_argument(
        "--bases",
        type=str,
        default=",".join(str(b) for b in TARGET_BASES),
        help="comma-separated target bases (default: 27 OEIS-open bases)",
    )
    parser.add_argument("--start", type=int, default=4_400_000_000_000,
                        help="start of scan range, inclusive (default 4.4e12)")
    parser.add_argument("--end", type=int, default=100_000_000_000_000,
                        help="end of scan range, inclusive (default 1e14)")
    parser.add_argument("--state-dir", type=str, default=DEFAULT_STATE_DIR,
                        help=f"directory for checkpoint state (default {DEFAULT_STATE_DIR})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"primes-range chunk size (default {DEFAULT_CHUNK_SIZE:,})")
    parser.add_argument("--gpu-batch", type=int, default=DEFAULT_GPU_BATCH,
                        help=f"GPU batch size in primes (default {DEFAULT_GPU_BATCH:,})")
    parser.add_argument("--sieve-chunk", type=int, default=DEFAULT_SIEVE_CHUNK,
                        help=f"primesieve sub-chunk size (default {DEFAULT_SIEVE_CHUNK:,})")
    args = parser.parse_args()

    bases = [int(b) for b in args.bases.split(",") if b.strip()]

    driver = Phase6Driver(
        bases=bases,
        start=args.start,
        end=args.end,
        state_dir=args.state_dir,
        chunk_size=args.chunk_size,
        gpu_batch=args.gpu_batch,
        sieve_chunk=args.sieve_chunk,
    )
    return driver.run()


if __name__ == "__main__":
    sys.exit(main())
