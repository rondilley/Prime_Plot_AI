"""Core prime generation and polynomial utilities."""

from prime_plot.core.sieve import generate_primes, is_prime, is_prime_array
from prime_plot.core.polynomials import PrimePolynomial, FAMOUS_POLYNOMIALS

__all__ = [
    "generate_primes",
    "is_prime",
    "is_prime_array",
    "PrimePolynomial",
    "FAMOUS_POLYNOMIALS",
]
