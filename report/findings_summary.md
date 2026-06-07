# Extending the Wieferich-prime search frontier for 27 bases to 10^14

Daniel Okwor, Orygn LLC (Houston, TX). Contact: daniel@orygn.tech.
Computation completed 2026-06-07.

## Result in one line

For 27 integer bases whose smallest Wieferich prime is unknown, we verified
that no Wieferich prime exists below 10^14, extending the previously published
search bound of 4.4 x 10^12 by a factor of about 23.

## Background

A prime p is a Wieferich prime to base b when p^2 divides b^(p-1) - 1,
equivalently b^(p-1) is congruent to 1 modulo p^2. This is a rare strengthening
of Fermat's little theorem (which guarantees only that p divides b^(p-1) - 1).

OEIS sequence A039951 records a(n) = the smallest prime p such that p^2 divides
n^(p-1) - 1. The value a(n) is known for many n, but is unknown for a set of
bases listed in the entry's comments (source: Richard Fischer). For two of those
bases (47 and 72) Fischer individually searched past 1.4 x 10^14. The remaining
bases sat at his broad-sweep bound of 4.4 x 10^12.

This work extends the search to 10^14 for 27 of those open bases:

186, 187, 200, 203, 222, 231, 304, 311, 335, 355, 435, 454, 546, 554, 610,
639, 662, 760, 772, 798, 808, 812, 858, 860, 871, 983, 986.

## Method

For each prime p in the interval (4.4 x 10^12, 10^14] and each of the 27 bases
b, we tested whether b^(p-1) is congruent to 1 modulo p^2.

- Primes were generated with primesieve 7.5 (segmented sieve of Eratosthenes).
- The modular test ran on an NVIDIA RTX 5080 Laptop GPU via a custom CUDA
  kernel. Because the modulus p^2 reaches about 2^94, a naive 128-bit kernel
  overflows; the production kernel ("v4") exploits the structure m = p^2 by
  decomposing each operand p-adically (a = a0 + a1*p with a0, a1 < p), so every
  intermediate product fits in 128 bits and the result is exact.
- A multi-base kernel tests each prime against all 27 bases in a single GPU
  launch, returning a per-prime bitmask.
- 3,048,213,614,544 primes were tested. Zero Wieferich primes were found.

## Verification

The result is a null result, so the failure mode that matters is a silent false
negative (missing a Wieferich prime that is present). The following independent
checks guard against it:

1. Unit tests (250+) covering the modular-exponentiation primitive, the sieve,
   the single-base and parallel searches, and GPU/CPU consistency.
2. Overflow audit: the production GPU kernel is bit-identical to gmpy2 (GMP)
   across primes from 2^33 to 2^47, the regime where the earlier naive kernel
   silently failed. (That earlier failure was caught by this audit before any
   production run.)
3. Reproduction of all 45 known terms of A039951, including a(34) =
   46145917691, from scratch.
4. Live spot-checking during the production run: 54,700 randomly chosen
   sub-ranges were independently re-tested with gmpy2 as the scan proceeded.
   Zero disagreements.
5. Positive control: the production pipeline was run over a window containing a
   known Wieferich prime (base 941, p = 64501672625861, found by Fischer in
   2025, which lies inside our search interval). The pipeline detected it, and
   the multi-base decode attributed it to the correct base with no false
   positives for other bases. This proves the pipeline detects Wieferich primes
   when they exist at this scale.
6. Independent engine cross-check: PARI/GP 2.17.3, a separate mature math
   system, reproduces the known Wieferich primes, confirms the base-941 control,
   and agrees exactly with the GPU pipeline on a head-to-head window
   ([4.4 x 10^12, 4.4 x 10^12 + 10^6], bases 186/187/200): both enumerate the
   same 34147 primes and both find zero.

## Reproducibility

All code, tests, and the verification records are released publicly (see the
project repository). The search for any single base and range can be reproduced
directly in PARI/GP with:

    is_wief(b,p) = (Mod(b,p^2)^(p-1) == 1);
    forprime(p = lo, hi, if(is_wief(b,p), print(p)));

The GPU pipeline reproduces the same results far faster; PARI is the slow but
independent reference (about 700x slower per test, which is why the full search
required the GPU).

## Honest limitations

- This is a search bound, not a discovery. No new Wieferich prime was found.
- The lower part of the combined bound (below 4.4 x 10^12) rests on Richard
  Fischer's prior published search; our contribution is the extension from
  4.4 x 10^12 to 10^14.
- The result says nothing about whether a Wieferich prime exists for these bases
  above 10^14; the heuristic density (about 1/p per base) suggests examples
  should exist eventually, but none appear in the range searched.

## Tools

primesieve 7.5 (prime generation), a custom CUDA kernel (modular arithmetic),
gmpy2 and PARI/GP (independent verification), Python, and AI-assisted code
development. The author is responsible for all results; every claim is backed by
the verification records in the repository.
