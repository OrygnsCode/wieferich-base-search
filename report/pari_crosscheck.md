# Independent PARI/GP cross-check

Date: 2026-06-07

## Purpose

Our entire pipeline (sieve, GPU Miller-Rabin-free modexp kernel v4, multibase
decode) is one toolchain. PARI/GP is a completely separate, mature math engine,
and is the language OEIS editors use to verify submissions. Agreement between
PARI and our GPU pipeline is independent confirmation from a different codebase.

Script: `scripts/wieferich_crosscheck.gp` (run with PARI/GP 2.17.3,
`gp -q wieferich_crosscheck.gp`).

Wieferich condition in PARI: `is_wief(b,p) = (Mod(b,p^2)^(p-1) == 1)`.

## Results

### Part 1: PARI reproduces known smallest Wieferich primes (A039951)
All 8 reproduced exactly:
base 2 -> 1093, base 3 -> 11, base 5 -> 2, base 7 -> 5, base 11 -> 71,
base 12 -> 2693, base 13 -> 2, base 20 -> 281. **ALL OK.** Confirms the PARI
Wieferich test is itself correct.

### Part 2: positive control
`is_wief(941, 64501672625861) = 1`. **OK.** PARI confirms the known base-941
Wieferich prime (the same one our GPU positive control detected).

### Part 3: negative cross-check, head-to-head on an identical window
Window: [4.4e12, 4.4e12 + 1e6], bases 186, 187, 200.

| Engine | Primes in window | Wieferich primes found |
|---|---|---|
| PARI/GP 2.17.3 (CPU, independent) | 34,147 | 0 |
| Our GPU pipeline (v4 kernel) | 34,147 | 0 |

**Exact agreement**, including the prime count. Two independent engines, same
input, same (empty) result.

## Note on performance (why the GPU was necessary)

PARI single-threaded runs the Wieferich test at roughly 1.5e4 modexp/sec at this
scale; our GPU pipeline runs ~1e7/sec effective (about 700x). A full PARI scan of
the 3.05-trillion-prime range would take years; PARI's role here is independent
spot-confirmation on samples, not bulk scanning. This is exactly why the
production search used the GPU.

## Submission artifact

The PARI function `is_wief(b,p) = (Mod(b,p^2)^(p-1) == 1)` and a `forprime` loop
over a range is the natural PROG-field code for the OEIS submission; editors can
run it directly to verify any specific base/range.
