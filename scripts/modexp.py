"""Modular exponentiation primitives used by the Wieferich search.

Two implementations are provided:

- modexp_python(base, exp, mod): uses Python's built-in pow(). Arbitrary
  precision, correct by construction. Slow for large inputs. Used as the
  trust anchor in tests.

- modexp_gmpy2(base, exp, mod): uses gmpy2.powmod, the C-backed GMP
  implementation. Used in the hot path of the actual search. Must
  produce results identical to modexp_python on every input.

The Wieferich condition test is_wieferich(base, prime) returns True
when prime^2 divides base^(prime - 1) - 1, equivalently when
base^(prime - 1) is congruent to 1 modulo prime^2.

All callers in the search code should go through is_wieferich rather than
calling modexp directly, so that the precision (mod = prime^2) and the
exponent (prime - 1) are handled consistently.
"""

try:
    import gmpy2
    _HAVE_GMPY2 = True
except ImportError:
    gmpy2 = None
    _HAVE_GMPY2 = False


def modexp_python(base: int, exp: int, mod: int) -> int:
    """Reference modular exponentiation using Python's built-in pow.

    Arbitrary precision. Slow for large inputs. The trust anchor: any
    other implementation must agree with this one bit for bit.
    """
    return pow(base, exp, mod)


def modexp_gmpy2(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation via gmpy2.

    Raises RuntimeError if gmpy2 is not installed. Result must equal
    modexp_python(base, exp, mod) for every input.
    """
    if not _HAVE_GMPY2:
        raise RuntimeError(
            "gmpy2 is required for modexp_gmpy2; install with pip install gmpy2"
        )
    return int(gmpy2.powmod(base, exp, mod))


def is_wieferich(base: int, prime: int) -> bool:
    """Return True if `prime` is a Wieferich prime to base `base`.

    A prime p is Wieferich to base b when p^2 divides b^(p-1) - 1.

    No primality check is performed on `prime`. Callers must pass a prime.
    Returns False for prime < 2.
    """
    if prime < 2:
        return False
    return modexp_gmpy2(base, prime - 1, prime * prime) == 1


def is_wieferich_reference(base: int, prime: int) -> bool:
    """Slow reference variant using Python's pow. For test correctness only."""
    if prime < 2:
        return False
    return modexp_python(base, prime - 1, prime * prime) == 1
