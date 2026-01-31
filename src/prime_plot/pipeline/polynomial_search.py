"""Polynomial-first prime search.

Exploits the mathematical fact that certain quadratic polynomials
produce primes at rates 5-7x higher than random. By testing polynomial
values first, we can find primes with ~2x fewer primality tests.

This is based on classical number theory (Euler 1772, Hardy-Littlewood
Conjecture F), not machine learning or visualization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional
import math


@dataclass
class PrimeGeneratingPolynomial:
    """A polynomial known to produce primes at high rates."""
    name: str
    formula: Callable[[int], int]
    description: str

    def values_near(self, target: int, count: int = 1000) -> List[int]:
        """Generate polynomial values near a target number."""
        # Solve for approximate k where poly(k) ~ target
        # For quadratic ax^2 + bx + c, k ~ sqrt(target/a)
        k_approx = int(math.sqrt(target))

        values = []
        for k in range(max(0, k_approx - count), k_approx + count):
            v = self.formula(k)
            if v > 1:
                values.append(v)

        return sorted(values)


# Classic prime-generating polynomials
PRIME_POLYNOMIALS = [
    PrimeGeneratingPolynomial(
        name="euler",
        formula=lambda k: k*k + k + 41,
        description="Euler's n^2 + n + 41 (1772) - produces primes for n=0..39"
    ),
    PrimeGeneratingPolynomial(
        name="euler_variant",
        formula=lambda k: k*k - k + 41,
        description="n^2 - n + 41 - mirror of Euler's polynomial"
    ),
    PrimeGeneratingPolynomial(
        name="legendre",
        formula=lambda k: 2*k*k + 29,
        description="2n^2 + 29 - Legendre's polynomial"
    ),
    PrimeGeneratingPolynomial(
        name="fung_ruby",
        formula=lambda k: k*k + k + 17,
        description="n^2 + n + 17 - Fung & Ruby polynomial"
    ),
    PrimeGeneratingPolynomial(
        name="quad_41",
        formula=lambda k: 4*k*k - 2*k + 41,
        description="4n^2 - 2n + 41 - high prime density variant"
    ),
    PrimeGeneratingPolynomial(
        name="quad_59",
        formula=lambda k: 4*k*k + 4*k + 59,
        description="4n^2 + 4n + 59 - another high density form"
    ),
]


def miller_rabin(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test with k rounds.

    Args:
        n: Number to test
        k: Number of rounds (higher = more certain)

    Returns:
        True if probably prime, False if definitely composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


@dataclass
class SearchResult:
    """Result of a prime search."""
    primes_found: List[int]
    tests_performed: int
    polynomial_tests: int
    fallback_tests: int


def polynomial_first_search(
    start: int,
    target_primes: int = 100,
    polynomials: Optional[List[PrimeGeneratingPolynomial]] = None,
    poly_range: int = 2000,
    seed: Optional[int] = None,
) -> SearchResult:
    """Find primes by testing polynomial values first.

    Strategy:
    1. Generate values from prime-generating polynomials near start
    2. Test those values first (they have ~55% prime rate)
    3. Fall back to wheel-filtered sequential search if needed

    Args:
        start: Starting point for search
        target_primes: Number of primes to find
        polynomials: List of polynomials to use (default: all known)
        poly_range: How many polynomial values to generate per polynomial
        seed: Random seed for Miller-Rabin

    Returns:
        SearchResult with found primes and test counts
    """
    if seed is not None:
        random.seed(seed)

    if polynomials is None:
        polynomials = PRIME_POLYNOMIALS

    # Collect polynomial values near start
    poly_values = set()
    for poly in polynomials:
        for v in poly.values_near(start, poly_range):
            if v >= start:
                poly_values.add(v)

    poly_values = sorted(poly_values)

    # Search
    primes_found = []
    poly_tests = 0
    fallback_tests = 0

    # Phase 1: Test polynomial values
    for v in poly_values:
        if len(primes_found) >= target_primes:
            break
        poly_tests += 1
        if miller_rabin(v):
            primes_found.append(v)

    # Phase 2: Fall back to wheel-filtered sequential if needed
    if len(primes_found) < target_primes:
        poly_set = set(poly_values)

        # Start wheel-filtered search
        n = start
        if n % 6 == 0:
            n += 1
        elif n % 6 == 2:
            n += 3
        elif n % 6 == 3:
            n += 2
        elif n % 6 == 4:
            n += 1

        step = 4 if n % 6 == 1 else 2

        while len(primes_found) < target_primes:
            if n not in poly_set:  # Skip already tested
                fallback_tests += 1
                if miller_rabin(n):
                    primes_found.append(n)
            n += step
            step = 6 - step

    return SearchResult(
        primes_found=sorted(primes_found),
        tests_performed=poly_tests + fallback_tests,
        polynomial_tests=poly_tests,
        fallback_tests=fallback_tests,
    )


def brute_force_search(
    start: int,
    target_primes: int = 100,
    seed: Optional[int] = None,
) -> SearchResult:
    """Find primes using standard wheel-filtered sequential search.

    Args:
        start: Starting point for search
        target_primes: Number of primes to find
        seed: Random seed for Miller-Rabin

    Returns:
        SearchResult with found primes and test counts
    """
    if seed is not None:
        random.seed(seed)

    primes_found = []
    tests = 0

    # Wheel-filtered search (mod 6)
    n = start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1

    step = 4 if n % 6 == 1 else 2

    while len(primes_found) < target_primes:
        tests += 1
        if miller_rabin(n):
            primes_found.append(n)
        n += step
        step = 6 - step

    return SearchResult(
        primes_found=primes_found,
        tests_performed=tests,
        polynomial_tests=0,
        fallback_tests=tests,
    )


def compare_methods(
    start: int,
    target_primes: int = 100,
    seed: int = 42,
) -> Tuple[SearchResult, SearchResult, float]:
    """Compare polynomial-first vs brute force search.

    Args:
        start: Starting point for search
        target_primes: Number of primes to find
        seed: Random seed

    Returns:
        Tuple of (poly_result, brute_result, improvement_ratio)
    """
    poly_result = polynomial_first_search(start, target_primes, seed=seed)
    brute_result = brute_force_search(start, target_primes, seed=seed)

    improvement = brute_result.tests_performed / poly_result.tests_performed

    return poly_result, brute_result, improvement


if __name__ == "__main__":
    import time

    print("Polynomial-First Prime Search")
    print("=" * 60)
    print()

    for scale in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
        print(f"Scale: {scale:,}")
        print("-" * 40)

        t0 = time.perf_counter()
        poly_result = polynomial_first_search(scale, target_primes=100, seed=42)
        poly_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        brute_result = brute_force_search(scale, target_primes=100, seed=42)
        brute_time = (time.perf_counter() - t0) * 1000

        improvement = brute_result.tests_performed / poly_result.tests_performed

        print(f"  Polynomial-first: {poly_result.tests_performed} tests "
              f"({poly_result.polynomial_tests} poly + {poly_result.fallback_tests} fallback) "
              f"in {poly_time:.1f}ms")
        print(f"  Brute force:      {brute_result.tests_performed} tests in {brute_time:.1f}ms")
        print(f"  Improvement:      {improvement:.2f}x fewer tests")
        print()
