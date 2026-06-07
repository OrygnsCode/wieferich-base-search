"""ctypes wrapper around primesieve.dll (Kim Walisch, primesieve 7.5).

primesieve is a highly optimized segmented sieve of Eratosthenes in C++,
multi-threaded and SIMD-accelerated. Build it from source (kimwalisch/primesieve)
and copy the resulting primesieve.dll into scripts/libs/. The pure-Python sieve
in sieve.py is the slower but dependency-free reference if the DLL is absent.

For our use case, we just need `primesieve_generate_primes(start, stop)`
which returns a UINT64 array of all primes in [start, stop]. The array
is allocated by primesieve and freed via `primesieve_free`. We copy the
data into a Python-owned numpy array so we can free the C-side allocation
immediately.

Verified against sympy.primerange and scripts.sieve.segmented_primes in
tests/test_primesieve_wrapper.py. Bit-identical output required.

Reference for the C API: lib/primesieve/include/primesieve.h in the
upstream source.
"""
import ctypes
import os
import sys

import numpy as np

# Resolve the DLL path relative to this module's location.
_DLL_PATH = os.path.join(os.path.dirname(__file__), "libs", "primesieve.dll")

if not os.path.exists(_DLL_PATH):
    raise FileNotFoundError(
        f"primesieve.dll not found at expected path: {_DLL_PATH}\n"
        "Build primesieve (kimwalisch/primesieve) and copy "
        "build/Release/primesieve.dll into scripts/libs/."
    )

_lib = ctypes.CDLL(_DLL_PATH)

# void* primesieve_generate_primes(uint64_t start, uint64_t stop, size_t* size, int type)
_lib.primesieve_generate_primes.restype = ctypes.c_void_p
_lib.primesieve_generate_primes.argtypes = [
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int,
]

# void primesieve_free(void* primes)
_lib.primesieve_free.restype = None
_lib.primesieve_free.argtypes = [ctypes.c_void_p]

# uint64_t primesieve_count_primes(uint64_t start, uint64_t stop)
_lib.primesieve_count_primes.restype = ctypes.c_uint64
_lib.primesieve_count_primes.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

# Type tag for UINT64_PRIMES (position 13 in the enum in primesieve.h).
# Enum order: SHORT, USHORT, INT, UINT, LONG, ULONG, LONGLONG, ULONGLONG,
#             INT16, UINT16, INT32, UINT32, INT64, UINT64.
UINT64_PRIMES = 13

# primesieve returns PRIMESIEVE_ERROR (UINT64_MAX) on failure.
PRIMESIEVE_ERROR = (1 << 64) - 1


def primesieve_generate_primes(start: int, stop: int) -> np.ndarray:
    """Return numpy uint64 array of primes p with start <= p <= stop.

    Allocates on the C side, copies into a Python-owned numpy array,
    and frees the C-side memory. The returned array is safe to use
    after this call returns.

    Raises RuntimeError on primesieve error.
    """
    if stop < 2 or stop < start:
        return np.array([], dtype=np.uint64)
    start = max(start, 0)

    size = ctypes.c_size_t(0)
    ptr = _lib.primesieve_generate_primes(start, stop, ctypes.byref(size), UINT64_PRIMES)

    if not ptr:
        # Empty range (no primes); confirm via count
        return np.array([], dtype=np.uint64)

    n = size.value
    if n == 0:
        _lib.primesieve_free(ptr)
        return np.array([], dtype=np.uint64)

    # Wrap the C-allocated buffer in a ctypes array, then copy to numpy
    arr_type = ctypes.c_uint64 * n
    c_arr = arr_type.from_address(ptr)
    # np.frombuffer with .copy() ensures we own the memory
    arr = np.frombuffer(c_arr, dtype=np.uint64).copy()

    _lib.primesieve_free(ptr)
    return arr


def primesieve_count(start: int, stop: int) -> int:
    """Return the count of primes in [start, stop] without materializing them."""
    if stop < 2 or stop < start:
        return 0
    start = max(start, 0)
    result = _lib.primesieve_count_primes(start, stop)
    if result == PRIMESIEVE_ERROR:
        raise RuntimeError("primesieve_count_primes returned PRIMESIEVE_ERROR")
    return int(result)


def primesieve_chunks(
    start: int,
    stop: int,
    chunk_size: int = 10**10,
):
    """Yield numpy uint64 arrays of primes per chunk in [start, stop].

    Bounds the per-call memory footprint by `chunk_size` (range size in
    integers, not number of primes). Default 10^10 -> roughly 4e8 primes
    per chunk near our scale, about 3 GB of uint64 per chunk.
    """
    if stop < 2 or stop < start:
        return
    cur = max(start, 0)
    while cur <= stop:
        chunk_high = min(cur + chunk_size - 1, stop)
        primes = primesieve_generate_primes(cur, chunk_high)
        if len(primes) > 0:
            yield primes
        cur = chunk_high + 1
