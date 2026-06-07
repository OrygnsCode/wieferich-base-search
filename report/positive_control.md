# Positive control (false-negative guard)

Date: 2026-06-07

## Purpose

The Phase 6 result is a null result: no Wieferich prime found for 27 bases in
(4.4e12, 1e14]. The risk that would invalidate this is a *silent false
negative*, the pipeline failing to detect a Wieferich prime that is actually
present. This positive control proves the production pipeline detects a KNOWN
Wieferich prime at the same scale.

## Test

A known Wieferich prime exists in our scan range: base 941 has the Fermat
quotient solution **64501672625861** (~6.45e13), found by Richard Fischer,
2025-08-27. This value lies inside (4.4e12, 1e14], the exact interval we
scanned.

We ran the exact production scanner (scripts/scan_multibase_gpu.py, v4 kernel)
over a window bracketing it.

### Run 1: base 941 alone
```
python scripts/scan_multibase_gpu.py --bases 941 --low 64501000000000 --high 64502000000000
-> FOUND: base=941 prime=64501672625861
-> Total Wieferich findings: 1
```

### Run 2: base 941 embedded among real target bases (multibase decode test)
```
python scripts/scan_multibase_gpu.py --bases 186,187,941,200,304,772,986 --low 64501000000000 --high 64502000000000
-> FOUND: base=941 prime=64501672625861
-> Total Wieferich findings: 1   (only 941; the 6 target bases correctly report nothing)
```

## Result: PASS

- The pipeline detects a real Wieferich prime at production scale.
- The multibase bitmask decode attributes the find to the correct base (941),
  and produces no false positives for the other bases in the same window.
- Therefore the 27-base production scan would have surfaced any Wieferich prime
  in (4.4e12, 1e14]. The 0-findings result is a true negative.

## Verification chain summary (for the null result)

1. 250+ unit tests (modexp, sieve, single-base, parallel, GPU consistency).
2. Overflow audit: v4 kernel bit-identical to gmpy2 across 2^33..2^47 (the bug
   that this caught in v1 is documented in tests/test_gpu_overflow_audit.py).
3. Reproduced all 45 known A039951 terms (incl. a(34) = 46145917691).
4. Live spot-check during the run: 54,700 random ranges independently
   re-tested with gmpy2, 0 disagreements.
5. This positive control: known base-941 prime detected at production scale.
