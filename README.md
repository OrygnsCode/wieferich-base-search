# Wieferich base search

A search for Wieferich primes in 27 bases whose smallest example was unknown, run
on a single workstation GPU. No Wieferich prime exists below 10^14 for any of
these 27 bases.

Six of them (311, 454, 554, 662, 772, 983) extend the best previously published
bound: Richard Fischer's detailed table had these at 6.46 x 10^13. The other 21
bases were already carried further by Fischer (to 2.00 x 10^14 in that table), so
for those this run independently confirms his result rather than extending it.

This relates to OEIS [A039951](https://oeis.org/A039951), "smallest prime p such
that p^2 divides n^(p-1) - 1."

## Result

For each of the bases

```
186 187 200 203 222 231 304 311 335 355 435 454 546 554 610
639 662 760 772 798 808 812 858 860 871 983 986
```

we tested every prime p in (4.4 x 10^12, 10^14] for the Wieferich condition
b^(p-1) congruent to 1 (mod p^2). 3,048,213,614,544 primes were tested. None
satisfied the condition for any of the 27 bases, so the smallest Wieferich prime
for each (if one exists) is greater than 10^14.

The lower end of this interval, 4.4 x 10^12, is Fischer's broad-sweep figure. For
the six bases above his detailed table reached 6.46 x 10^13, so the new ground
gained here is from 6.46 x 10^13 to 10^14; for the other 21 bases his table was
already at 2.00 x 10^14, beyond this search.

## How it works

Primes come from a segmented sieve (primesieve). The modular test runs on the GPU.
Because p^2 reaches about 2^94 in this range, a naive 128-bit modular kernel
overflows; the production kernel (`scripts/wieferich_gpu_v4.py`) decomposes each
operand modulo p^2 as a0 + a1*p with a0, a1 < p, so every intermediate product
stays inside 128 bits and the result is exact. One GPU launch tests a prime
against all 27 bases at once and returns a bitmask.

## Reproduce

A single base and range can be checked in PARI/GP:

```
is_wief(b,p) = (Mod(b,p^2)^(p-1) == 1);
forprime(p = lo, hi, if(is_wief(b,p), print(p)));
```

`scripts/wieferich_crosscheck.gp` runs this against known values and a sample of
the search window. The GPU pipeline does the same far faster:

```
python scripts/scan_multibase_gpu.py --bases 186,187,200 --low 4400000000000 --high 4400001000000
```

The full run with checkpointing is `scripts/phase6_driver.py`.

Requirements: Python 3.x, numpy, cupy (CUDA), gmpy2, sympy. primesieve.dll goes in
`scripts/libs/` (build from kimwalisch/primesieve); without it the pure-Python
sieve in `scripts/sieve.py` is the fallback.

## Verification

This is a null result, so the risk is missing a prime that is present. Guards
against that, all in `report/`:

- The production kernel is bit-identical to gmpy2 (GMP) across primes from 2^33
  to 2^47 (`tests/test_gpu_v4_audit.py`).
- All 45 known terms of A039951 are reproduced from scratch
  (`scripts/reproduce_a039951.py`).
- 54,700 random sub-ranges were re-tested with gmpy2 during the run, with no
  disagreements.
- A positive control: the pipeline detects the known base-941 Wieferich prime at
  6.45 x 10^13, which sits inside the search range (`report/positive_control.md`).
- PARI/GP, an independent engine, agrees exactly on a head-to-head window
  (`report/pari_crosscheck.md`).

Run the tests with `python -m pytest tests/ -v`.

## Layout

```
scripts/   search code, GPU kernel, PARI cross-check, full-run driver
tests/     unit and audit tests
report/    findings summary and verification records
```

## Author

Daniel Okwor, Orygn LLC, Houston TX. Contact: daniel@orygn.tech.

Implementation tools include primesieve, CUDA, gmpy2, PARI/GP, Python, and
AI-assisted code development. The author is responsible for all results.

## License

MIT. See LICENSE.
