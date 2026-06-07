"""Layer 3 of the verification framework: end-to-end Wieferich search
on small-scale known data.

Composes the sieve (Layer 2) and the modexp primitive (Layer 1) into the
actual search via scripts.wieferich_search, then verifies the output
against the published small-scale data on A039951.

This covers all A039951 entries a(n) where a(n) <= 5000 (a fast subset
of the 46 known terms). The remaining entries (a(6) = 66161, a(15) =
29131, a(34) ~= 4.6 * 10^10, a(36) = 66161, a(39) = 8039) are covered
by scripts/reproduce_a039951.py as part of Layer 4.

Run from the project root:
    python -m pytest tests/test_known_wieferich.py -v
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.wieferich_search import (
    scan_base_for_wieferich,
    find_first_wieferich,
    all_wieferich_in_range,
)


# A039951 values verbatim from the OEIS DATA section:
# 2, 1093, 11, 1093, 2, 66161, 5, 3, 2, 3, 71, 2693, 2, 29, 29131, 1093,
# 2, 5, 3, 281, 2, 13, 13, 5, 2, 3, 11, 3, 2, 7, 7, 5, 2, 46145917691, 3,
# 66161, 2, 17, 8039, 11, 2, 23, 5, 3, 2, 3
A039951_VALUES = {
    1: 2,           # base 1 is degenerate (1^k = 1)
    2: 1093,
    3: 11,
    4: 1093,        # 4 = 2^2 inherits from base 2
    5: 2,
    6: 66161,       # > 5000, deferred to Layer 4
    7: 5,
    8: 3,
    9: 2,
    10: 3,
    11: 71,
    12: 2693,
    13: 2,
    14: 29,
    15: 29131,      # > 5000, deferred to Layer 4
    16: 1093,
    17: 2,
    18: 5,
    19: 3,
    20: 281,
    21: 2,
    22: 13,
    23: 13,
    24: 5,
    25: 2,
    26: 3,
    27: 11,
    28: 3,
    29: 2,
    30: 7,
    31: 7,
    32: 5,
    33: 2,
    34: 46145917691,  # ~ 4.6e10, deferred to Layer 4
    35: 3,
    36: 66161,        # > 5000, deferred to Layer 4
    37: 2,
    38: 17,
    39: 8039,         # > 5000, deferred to Layer 4
    40: 11,
    41: 2,
    42: 23,
    43: 5,
    44: 3,
    45: 2,
    46: 3,
}

# Subset of A039951 with a(n) <= 5000, n >= 2 (skip the degenerate n=1)
SMALL_SUBSET = {
    n: a for n, a in A039951_VALUES.items() if a <= 5000 and n >= 2
}


class TestFindFirstSmallSubset:
    """For each n in SMALL_SUBSET, find_first_wieferich(n, ...) == A039951(n)."""

    @pytest.mark.parametrize("n,expected", sorted(SMALL_SUBSET.items()))
    def test_reproduce_known_term(self, n, expected):
        """find_first_wieferich(n, expected+100) should return expected."""
        upper = max(expected + 100, 200)
        result = find_first_wieferich(n, upper)
        assert result == expected, (
            f"A039951({n}): expected {expected}, got {result}"
        )


class TestBase2KnownSet:
    """Base 2: the two known Wieferich primes (1093, 3511) are the only ones below 10^4."""

    def test_exactly_two_below_10000(self):
        found = all_wieferich_in_range(2, 2, 10000)
        assert found == [1093, 3511]

    def test_first_is_1093(self):
        assert find_first_wieferich(2, 10000) == 1093

    def test_no_wieferich_between_3511_and_5000(self):
        """No other base-2 Wieferich primes exist in (3511, 5000]."""
        found = all_wieferich_in_range(2, 3512, 5000)
        assert found == []


class TestBase11KnownSet:
    """Base 11: the only known Wieferich is 71 (no others below 10000)."""

    def test_only_71_below_10000(self):
        """A039951 lists a(11) = 71. No other base-11 Wieferich below 10^4."""
        found = all_wieferich_in_range(11, 2, 10000)
        # The OEIS lists the smallest. Whether more exist depends on the
        # specific base. For base 11 the known Wieferich primes are scarce;
        # this test asserts what we know: 71 is in the list and nothing
        # below 10000 is wrong.
        assert 71 in found
        assert all(p == 71 for p in found if p < 10000)


class TestBase12KnownSet:
    """Base 12: A111027 says only 2693 and 123653 known."""

    def test_both_known_below_200000(self):
        found = all_wieferich_in_range(12, 2, 200000)
        assert found == [2693, 123653]

    def test_first_is_2693(self):
        assert find_first_wieferich(12, 5000) == 2693


class TestSearchAPI:
    """Boundary conditions on the search functions."""

    def test_invalid_base_zero(self):
        with pytest.raises(ValueError):
            list(scan_base_for_wieferich(0, 2, 100))

    def test_invalid_base_negative(self):
        with pytest.raises(ValueError):
            list(scan_base_for_wieferich(-5, 2, 100))

    def test_base_1_degenerate(self):
        """Base 1 is allowed (degenerate: every prime trivially Wieferich)."""
        # 1^(p-1) = 1, so p^2 | (1 - 1) = 0. Every prime is "Wieferich" to base 1.
        result = list(scan_base_for_wieferich(1, 2, 30))
        # All primes in [2, 30] should be returned
        from sympy import primerange
        expected = list(primerange(2, 31))
        assert result == expected

    def test_empty_range(self):
        assert all_wieferich_in_range(2, 100, 50) == []
        assert all_wieferich_in_range(2, 4, 4) == []

    def test_find_first_returns_none_when_none_exists(self):
        """No base-2 Wieferich exists in (10, 1000)."""
        assert find_first_wieferich(2, upper_bound=1000, lower_bound=10) is None
