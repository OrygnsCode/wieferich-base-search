"""Layer 1 of the verification framework: modular exponentiation primitives.

Tests that the gmpy2-backed modular exponentiation primitive produces
results bit-identical to Python's built-in pow() on a broad input space.

Any disagreement is a hard stop: the entire Wieferich search rests on
modexp correctness.

Also tests the is_wieferich function on the small-scale set of known
Wieferich primes across multiple bases (Layer 3 small-scale subset).

Run from the project root:
    python -m pytest tests/test_modexp.py -v
"""
import os
import sys
import random

# Make scripts.modexp importable when pytest runs us from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.modexp import (
    modexp_python,
    modexp_gmpy2,
    is_wieferich,
    is_wieferich_reference,
)


# Reproducible test set: same seed every time so failures are deterministic
SEED = 20260602


class TestModexpPrimitive:
    """Bit-for-bit agreement between gmpy2 and Python pow."""

    def test_small_random(self):
        """10000 small random triples, base/exp/mod < 1000."""
        rng = random.Random(SEED)
        for _ in range(10000):
            base = rng.randint(0, 999)
            exp = rng.randint(0, 999)
            mod = rng.randint(2, 999)
            a = modexp_gmpy2(base, exp, mod)
            b = modexp_python(base, exp, mod)
            assert a == b, f"Mismatch at ({base}, {exp}, {mod}): gmpy2={a}, py={b}"

    def test_medium_random_64bit(self):
        """50000 triples spanning the 64-bit range (typical of small primes p^2)."""
        rng = random.Random(SEED + 1)
        for _ in range(50000):
            base = rng.randint(2, 10**9)
            exp = rng.randint(2, 10**9)
            mod = rng.randint(2, 10**18)
            a = modexp_gmpy2(base, exp, mod)
            b = modexp_python(base, exp, mod)
            assert a == b, f"Mismatch at ({base}, {exp}, {mod}): gmpy2={a}, py={b}"

    def test_large_random_96bit(self):
        """50000 triples in the 96-bit precision regime (p^2 for p ~10^14)."""
        rng = random.Random(SEED + 2)
        for _ in range(50000):
            base = rng.randint(2, 10**14)
            exp = rng.randint(2, 10**14)
            mod = rng.randint(2, 10**28)
            a = modexp_gmpy2(base, exp, mod)
            b = modexp_python(base, exp, mod)
            assert a == b, f"Mismatch at ({base}, {exp}, {mod}): gmpy2={a}, py={b}"

    def test_edge_zero_base(self):
        """0^k mod m is 0 for k >= 1. 0^0 follows Python convention (= 1)."""
        for exp in [1, 5, 100, 999]:
            for mod in [2, 100, 10**10]:
                assert modexp_gmpy2(0, exp, mod) == 0
        # Convention: 0^0 = 1 in Python's pow; check gmpy2 agrees
        for mod in [2, 100, 10**10]:
            assert modexp_gmpy2(0, 0, mod) == modexp_python(0, 0, mod)

    def test_edge_zero_exp(self):
        """x^0 mod m is 1 for any x, m > 1."""
        for base in [2, 3, 7, 11, 47, 100, 999999, 10**15]:
            for mod in [2, 100, 10**10, 10**20]:
                assert modexp_gmpy2(base, 0, mod) == 1

    def test_edge_one_base(self):
        """1^k mod m is 1 for any k, m > 1."""
        for exp in [0, 1, 7, 999, 10**12]:
            for mod in [2, 100, 10**10, 10**20]:
                assert modexp_gmpy2(1, exp, mod) == 1

    def test_edge_mod_minus_one_base(self):
        """(m-1)^k mod m is 1 if k even, m-1 if k odd."""
        for mod in [3, 5, 100, 10**10]:
            assert modexp_gmpy2(mod - 1, 2, mod) == 1
            assert modexp_gmpy2(mod - 1, 3, mod) == mod - 1
            assert modexp_gmpy2(mod - 1, 100, mod) == 1
            assert modexp_gmpy2(mod - 1, 101, mod) == mod - 1

    def test_edge_base_equals_mod(self):
        """When base mod m == 0, result is 0 for exp >= 1."""
        for mod in [3, 5, 100, 10**10]:
            assert modexp_gmpy2(mod, 1, mod) == 0
            assert modexp_gmpy2(mod, 5, mod) == 0
            assert modexp_gmpy2(2 * mod, 5, mod) == 0


class TestIsWieferichBase2:
    """Base 2: only Wieferich primes below 10^4 are 1093 and 3511."""

    def test_1093_is_wieferich(self):
        """The original Wieferich prime."""
        assert is_wieferich(2, 1093)
        # Cross-check against reference
        assert is_wieferich_reference(2, 1093)

    def test_3511_is_wieferich(self):
        """The second known base-2 Wieferich prime."""
        assert is_wieferich(2, 3511)
        assert is_wieferich_reference(2, 3511)

    def test_nearby_primes_not_wieferich(self):
        """Primes near the known Wieferich primes are not themselves Wieferich."""
        # Near 1093
        for p in [1087, 1091, 1097, 1103, 1109]:
            assert not is_wieferich(2, p), f"{p} should not be Wieferich base 2"
        # Near 3511
        for p in [3499, 3511 - 12, 3517, 3527]:
            if p != 3511:
                assert not is_wieferich(2, p), f"{p} should not be Wieferich base 2"

    def test_small_primes_not_wieferich_base_2(self):
        """The first several primes are not Wieferich base 2."""
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            assert not is_wieferich(2, p), f"{p} should not be Wieferich base 2"


class TestIsWieferichOtherBases:
    """The first Wieferich prime for several non-standard bases."""

    def test_base_3_first_wieferich_is_11(self):
        """A039951(3) = 11 (Mirimanoff prime)."""
        assert is_wieferich(3, 11)
        for p in [2, 3, 5, 7]:
            assert not is_wieferich(3, p)

    def test_base_5_first_wieferich_is_2(self):
        """A039951(5) = 2."""
        assert is_wieferich(5, 2)
        # 3 is not Wieferich base 5
        assert not is_wieferich(5, 3)

    def test_base_7_first_wieferich_is_5(self):
        """A039951(7) = 5."""
        assert is_wieferich(7, 5)
        for p in [2, 3]:
            assert not is_wieferich(7, p)

    def test_base_11_first_wieferich_is_71(self):
        """A039951(11) = 71."""
        assert is_wieferich(11, 71)
        # No prime below 71 is Wieferich base 11
        for p in [2, 3, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]:
            assert not is_wieferich(11, p), (
                f"{p} should not be Wieferich base 11 (would contradict A039951(11)=71)"
            )

    def test_base_12_both_known_wieferich(self):
        """A111027 lists 2693 and 123653 as the only known base-12 Wieferich."""
        assert is_wieferich(12, 2693)
        assert is_wieferich(12, 123653)


class TestIsWieferichCrossCheck:
    """is_wieferich (fast) and is_wieferich_reference (slow) must agree."""

    def test_agreement_on_known_set(self):
        """Both implementations agree on every known Wieferich prime tested above."""
        cases = [
            (2, 1093), (2, 3511),
            (3, 11),
            (5, 2),
            (7, 5),
            (11, 71),
            (12, 2693), (12, 123653),
        ]
        for base, prime in cases:
            assert is_wieferich(base, prime) == is_wieferich_reference(base, prime)
            assert is_wieferich(base, prime) is True
