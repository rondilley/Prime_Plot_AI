"""Tests for prime sieve functionality."""

import numpy as np
import pytest

from prime_plot.core.sieve import (
    generate_primes,
    is_prime,
    is_prime_array,
    prime_sieve_mask,
    nth_prime,
    count_primes,
)


class TestIsPrime:
    """Tests for is_prime function."""

    def test_small_primes(self):
        """Test known small primes."""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in small_primes:
            assert is_prime(p), f"{p} should be prime"

    def test_small_composites(self):
        """Test known small composites."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        for c in composites:
            assert not is_prime(c), f"{c} should not be prime"

    def test_edge_cases(self):
        """Test edge cases."""
        assert not is_prime(0)
        assert not is_prime(1)
        assert is_prime(2)
        assert not is_prime(-5)

    def test_larger_primes(self):
        """Test some larger known primes."""
        large_primes = [97, 101, 103, 107, 109, 113, 127, 131]
        for p in large_primes:
            assert is_prime(p), f"{p} should be prime"


class TestGeneratePrimes:
    """Tests for generate_primes function."""

    def test_primes_up_to_10(self):
        """Test primes up to 10."""
        primes = generate_primes(10)
        expected = np.array([2, 3, 5, 7])
        np.testing.assert_array_equal(primes, expected)

    def test_primes_up_to_100(self):
        """Test primes up to 100."""
        primes = generate_primes(100)
        assert len(primes) == 25  # There are 25 primes <= 100
        assert primes[0] == 2
        assert primes[-1] == 97

    def test_returns_numpy_array(self):
        """Test that result is numpy array."""
        primes = generate_primes(50)
        assert isinstance(primes, np.ndarray)

    def test_invalid_limit(self):
        """Test that invalid limit raises error."""
        with pytest.raises(ValueError):
            generate_primes(1)

        with pytest.raises(ValueError):
            generate_primes(-5)


class TestIsPrimeArray:
    """Tests for is_prime_array function."""

    def test_mixed_array(self):
        """Test array with mix of primes and composites."""
        numbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected = np.array([True, True, False, True, False, True, False, False, False])

        result = is_prime_array(numbers)
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test empty array."""
        result = is_prime_array(np.array([]))
        assert len(result) == 0


class TestPrimeSieveMask:
    """Tests for prime_sieve_mask function."""

    def test_mask_up_to_20(self):
        """Test mask up to 20."""
        mask = prime_sieve_mask(20)
        assert mask[2] == True
        assert mask[3] == True
        assert mask[4] == False
        assert mask[5] == True
        assert mask[6] == False
        assert mask[7] == True

    def test_mask_indexing(self):
        """Test that mask can be used for indexing."""
        mask = prime_sieve_mask(100)
        grid = np.arange(100)
        primes = grid[mask]
        assert len(primes) == 25


class TestNthPrime:
    """Tests for nth_prime function."""

    def test_first_primes(self):
        """Test first few primes."""
        assert nth_prime(1) == 2
        assert nth_prime(2) == 3
        assert nth_prime(3) == 5
        assert nth_prime(4) == 7
        assert nth_prime(5) == 11

    def test_100th_prime(self):
        """Test 100th prime."""
        assert nth_prime(100) == 541


class TestCountPrimes:
    """Tests for count_primes function."""

    def test_count_up_to_100(self):
        """Test counting primes up to 100."""
        assert count_primes(100) == 25

    def test_count_up_to_1000(self):
        """Test counting primes up to 1000."""
        assert count_primes(1000) == 168
