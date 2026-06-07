"""Layer 2 of the verification framework: prime sieve correctness.

Our segmented sieve must agree with sympy.primerange exactly on every
range we test. Disagreement at any prime is a hard stop.

Run from the project root:
    python -m pytest tests/test_sieve.py -v
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sympy import primerange

from scripts.sieve import small_sieve, segmented_primes, count_primes_in_range


class TestSmallSieve:
    """Standard sieve up to small limits."""

    def test_limit_below_2(self):
        assert small_sieve(0) == []
        assert small_sieve(1) == []
        assert small_sieve(-5) == []

    def test_limit_2(self):
        assert small_sieve(2) == [2]

    def test_limit_3(self):
        assert small_sieve(3) == [2, 3]

    def test_limit_10(self):
        assert small_sieve(10) == [2, 3, 5, 7]

    def test_limit_100(self):
        assert small_sieve(100) == list(primerange(2, 101))

    def test_limit_10000(self):
        assert small_sieve(10000) == list(primerange(2, 10001))

    def test_limit_100000(self):
        assert small_sieve(100000) == list(primerange(2, 100001))


class TestSegmentedSieve:
    """Segmented sieve agrees with sympy.primerange on various ranges."""

    def test_first_segment(self):
        """Primes in [2, 100]."""
        result = list(segmented_primes(2, 100))
        expected = list(primerange(2, 101))
        assert result == expected

    def test_starting_at_zero(self):
        """Range starting at 0 should clamp to 2."""
        result = list(segmented_primes(0, 100))
        expected = list(primerange(2, 101))
        assert result == expected

    def test_starting_at_one(self):
        result = list(segmented_primes(1, 100))
        expected = list(primerange(2, 101))
        assert result == expected

    def test_first_thousand(self):
        result = list(segmented_primes(2, 1000))
        expected = list(primerange(2, 1001))
        assert result == expected

    def test_mid_range_thousand(self):
        """Primes in [10^4, 10^4 + 1000]."""
        low, high = 10**4, 10**4 + 1000
        result = list(segmented_primes(low, high))
        expected = list(primerange(low, high + 1))
        assert result == expected

    def test_mid_range_million(self):
        """Primes in [10^6, 10^6 + 10000]."""
        low, high = 10**6, 10**6 + 10000
        result = list(segmented_primes(low, high))
        expected = list(primerange(low, high + 1))
        assert result == expected

    def test_large_range(self):
        """Primes in [10^9, 10^9 + 10000]."""
        low, high = 10**9, 10**9 + 10000
        result = list(segmented_primes(low, high))
        expected = list(primerange(low, high + 1))
        assert result == expected

    def test_full_range_to_million(self):
        """All primes up to 10^6, full sieve."""
        result = list(segmented_primes(2, 10**6))
        expected = list(primerange(2, 10**6 + 1))
        assert result == expected

    def test_segment_boundary_size_1000(self):
        """Range that crosses many small segment boundaries."""
        result = list(segmented_primes(0, 100000, segment_size=1000))
        expected = list(primerange(2, 100001))
        assert result == expected

    def test_segment_boundary_size_127(self):
        """Awkward small segment size."""
        result = list(segmented_primes(0, 10000, segment_size=127))
        expected = list(primerange(2, 10001))
        assert result == expected

    def test_segment_boundary_size_3(self):
        """Tiny segment size catches off-by-one bugs."""
        result = list(segmented_primes(0, 1000, segment_size=3))
        expected = list(primerange(2, 1001))
        assert result == expected

    def test_empty_range_high_less_than_low(self):
        assert list(segmented_primes(100, 50)) == []

    def test_range_below_2(self):
        assert list(segmented_primes(0, 1)) == []
        assert list(segmented_primes(-100, -1)) == []

    def test_range_exactly_at_2(self):
        assert list(segmented_primes(2, 2)) == [2]

    def test_range_exactly_at_3(self):
        assert list(segmented_primes(3, 3)) == [3]

    def test_range_at_high_bound_1e10(self):
        """High range close to what production search uses."""
        low, high = 10**10, 10**10 + 5000
        result = list(segmented_primes(low, high))
        expected = list(primerange(low, high + 1))
        assert result == expected

    def test_range_at_high_bound_1e11(self):
        """One order of magnitude higher."""
        low, high = 10**11, 10**11 + 5000
        result = list(segmented_primes(low, high))
        expected = list(primerange(low, high + 1))
        assert result == expected

    def test_specific_known_primes_present(self):
        """1093 (first Wieferich base 2) appears."""
        result = list(segmented_primes(1000, 1200))
        assert 1093 in result
        # And nothing that isn't prime
        non_primes = [p for p in result if not _is_prime_python_fallback(p)]
        assert non_primes == []

    def test_3511_appears(self):
        """3511 (second Wieferich base 2) appears."""
        result = list(segmented_primes(3500, 3600))
        assert 3511 in result

    def test_count_primes_in_range(self):
        """count_primes_in_range agrees with explicit list count."""
        for low, high in [(2, 100), (2, 1000), (10**6, 10**6 + 1000)]:
            counted = count_primes_in_range(low, high)
            listed = sum(1 for _ in segmented_primes(low, high))
            assert counted == listed


def _is_prime_python_fallback(n: int) -> bool:
    """Trivial primality check for cross-verification of sieve output."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    import math
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True
