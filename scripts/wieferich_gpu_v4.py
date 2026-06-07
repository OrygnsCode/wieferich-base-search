"""p-adic decomposition GPU Wieferich kernel (v4).

The v2 kernel was correct but slow because it reduced 192-bit products
mod m via 192 iterations of shift-subtract. The v3 Barrett attempt was
buggy because mu didn't fit cleanly in 128 bits in our prime range.

The v4 kernel exploits the SPECIFIC STRUCTURE of our modulus: m = p^2
for prime p. Any value a < m can be written as a = a0 + a1*p where
0 <= a0 < p and 0 <= a1 < p (this is essentially a p-adic decomposition).

For two such values a = a0 + a1*p and b = b0 + b1*p:
    a * b = a0*b0 + (a0*b1 + a1*b0)*p + a1*b1*p^2
mod p^2 (= m), the a1*b1*p^2 term vanishes:
    a*b mod p^2 = (a0*b0) + ((a0*b1 + a1*b0) mod p) * p
(The mod p on the middle term works because the *p multiplies it,
and (S*p) mod p^2 = (S mod p) * p.)

Each intermediate operand fits in 96 bits or fewer:
- a0, a1, b0, b1 are all < p < 2^48
- Products of two such values are < p^2 < 2^96, fit in __int128
- Sums of such products are < 2*p^2, fit in __int128
- (S mod p) * p < p^2 = m, fits in __int128

So we never need wider than __int128 arithmetic, and we get exact
modular multiplication. No shift-subtract loop, no Barrett mu.

Cost per modmul: 2 decompositions (uint128 / uint64), 3 small
multiplications, 1 (uint128 mod uint64) reduction, 1 small multiplication,
some additions and one final correction. Estimated 100 to 200 GPU cycles
per modmul, vs ~400+ for v2.

Verified bit-identical against v2 (which is verified against gmpy2) in
tests/test_gpu_v4_audit.py.

Constraints (same as v2):
- 1 <= base < 2^64
- 2 <= prime < 2^48
- 1 <= n_bases <= 64
"""
import numpy as np

import cupy as cp


_WIEFERICH_V4_KERNEL_SRC = r"""
// Modular multiplication exploiting m = p^2 structure.
// Returns (a * b) mod m exactly, where a, b in [0, m), m = p*p, p < 2^48.
__device__ __forceinline__ unsigned __int128 mulmod_p_squared(
    unsigned __int128 a,
    unsigned __int128 b,
    unsigned __int128 m,
    unsigned long long p
) {
    // Decompose a = a0 + a1*p, b = b0 + b1*p (with 0 <= a0,a1,b0,b1 < p)
    unsigned long long a1 = (unsigned long long)(a / (unsigned __int128)p);
    unsigned long long a0 = (unsigned long long)(a - (unsigned __int128)a1 * p);
    unsigned long long b1 = (unsigned long long)(b / (unsigned __int128)p);
    unsigned long long b0 = (unsigned long long)(b - (unsigned __int128)b1 * p);

    // Cross products (each < p^2 = m, fits in __int128)
    unsigned __int128 a0_b0 = (unsigned __int128)a0 * b0;
    unsigned __int128 a0_b1 = (unsigned __int128)a0 * b1;
    unsigned __int128 a1_b0 = (unsigned __int128)a1 * b0;

    // Middle coefficient (mod p, because *p makes it wrap mod p^2)
    unsigned __int128 sum = a0_b1 + a1_b0;       // < 2 * p^2
    unsigned long long sum_mod_p = (unsigned long long)(sum % (unsigned __int128)p);
    unsigned __int128 mid_term = (unsigned __int128)sum_mod_p * (unsigned __int128)p;

    // Result = a0*b0 + mid_term  (at most 2*m before correction)
    unsigned __int128 result = a0_b0 + mid_term;
    if (result >= m) result -= m;
    return result;
}

extern "C" __global__ void wieferich_test_v4(
    const unsigned long long* __restrict__ primes,
    const unsigned long long base,
    unsigned char* __restrict__ results,
    const int n_primes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned long long p = primes[idx];
    if (p < 2ULL) {
        results[idx] = 0;
        return;
    }

    unsigned __int128 mod = (unsigned __int128)p * (unsigned __int128)p;
    unsigned long long e = p - 1ULL;

    unsigned __int128 result = (unsigned __int128)1;
    unsigned __int128 b = ((unsigned __int128)base) % mod;

    while (e > 0ULL) {
        if (e & 1ULL) {
            result = mulmod_p_squared(result, b, mod, p);
        }
        b = mulmod_p_squared(b, b, mod, p);
        e >>= 1;
    }

    results[idx] = (result == (unsigned __int128)1) ? 1 : 0;
}

extern "C" __global__ void wieferich_multibase_v4(
    const unsigned long long* __restrict__ primes,
    const unsigned long long* __restrict__ bases,
    const int n_bases,
    unsigned long long* __restrict__ result_bitmask,
    const int n_primes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned long long p = primes[idx];
    if (p < 2ULL) {
        result_bitmask[idx] = 0ULL;
        return;
    }

    unsigned __int128 mod = (unsigned __int128)p * (unsigned __int128)p;
    unsigned long long exp_full = p - 1ULL;

    unsigned long long bitmask = 0ULL;

    for (int bi = 0; bi < n_bases; bi++) {
        unsigned __int128 base_val = (unsigned __int128)bases[bi];

        unsigned __int128 result = (unsigned __int128)1;
        unsigned __int128 b = base_val % mod;
        unsigned long long e = exp_full;

        while (e > 0ULL) {
            if (e & 1ULL) {
                result = mulmod_p_squared(result, b, mod, p);
            }
            b = mulmod_p_squared(b, b, mod, p);
            e >>= 1;
        }

        if (result == (unsigned __int128)1) {
            bitmask |= (1ULL << bi);
        }
    }

    result_bitmask[idx] = bitmask;
}
"""


_wieferich_v4_module = cp.RawModule(
    code=_WIEFERICH_V4_KERNEL_SRC,
    options=("--device-int128",),
)
_wieferich_test_v4 = _wieferich_v4_module.get_function("wieferich_test_v4")
_wieferich_multibase_v4 = _wieferich_v4_module.get_function("wieferich_multibase_v4")


MAX_BASES = 64
MAX_PRIME_V4 = (1 << 48) - 1


def gpu_wieferich_batch_v4(
    base: int,
    primes: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
    """p-adic decomposition single-base Wieferich test."""
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if base >= (1 << 64):
        raise ValueError("base must fit in uint64")
    if primes.dtype != np.uint64:
        primes = primes.astype(np.uint64, copy=False)
    if len(primes) == 0:
        return np.zeros(0, dtype=bool)

    d_primes = cp.asarray(primes)
    d_results = cp.zeros(len(primes), dtype=cp.uint8)
    blocks = (len(primes) + threads_per_block - 1) // threads_per_block
    _wieferich_test_v4(
        (blocks,),
        (threads_per_block,),
        (d_primes, cp.uint64(base), d_results, np.int32(len(primes))),
    )
    return d_results.get().astype(bool)


def gpu_wieferich_multibase_v4(
    bases,
    primes: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
    """p-adic decomposition multi-base Wieferich test, returns bitmask."""
    bases_arr = np.asarray(bases, dtype=np.uint64)
    if bases_arr.ndim != 1:
        raise ValueError(f"bases must be 1D, got shape {bases_arr.shape}")
    n_bases = len(bases_arr)
    if n_bases == 0:
        raise ValueError("bases must be non-empty")
    if n_bases > MAX_BASES:
        raise ValueError(f"At most {MAX_BASES} bases, got {n_bases}")
    if (bases_arr < 1).any():
        raise ValueError("all bases must be >= 1")
    if primes.dtype != np.uint64:
        primes = primes.astype(np.uint64, copy=False)
    if len(primes) == 0:
        return np.zeros(0, dtype=np.uint64)

    d_primes = cp.asarray(primes)
    d_bases = cp.asarray(bases_arr)
    d_results = cp.zeros(len(primes), dtype=cp.uint64)
    blocks = (len(primes) + threads_per_block - 1) // threads_per_block
    _wieferich_multibase_v4(
        (blocks,),
        (threads_per_block,),
        (d_primes, d_bases, np.int32(n_bases), d_results, np.int32(len(primes))),
    )
    return d_results.get()


def decode_bitmask_v4(bitmask_array: np.ndarray, n_bases: int) -> np.ndarray:
    result = np.zeros((len(bitmask_array), n_bases), dtype=bool)
    for i in range(n_bases):
        result[:, i] = (bitmask_array & (np.uint64(1) << np.uint64(i))) != 0
    return result
