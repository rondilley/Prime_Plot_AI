"""Prime number generation using high-performance sieve implementations.

Uses external C sieve (primes_new.exe) when available for best performance,
primesieve library as secondary option, with NumPy fallback.
Optional CuPy acceleration for GPU.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import primesieve
    import primesieve.numpy as primesieve_numpy
    HAS_PRIMESIEVE = True
except ImportError:
    HAS_PRIMESIEVE = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _find_c_sieve_executable() -> Path | None:
    """Locate the C sieve executable (primes_new.exe or primes_new)."""
    module_dir = Path(__file__).parent.parent.parent.parent
    candidates = [
        module_dir / "primes_new.exe",
        module_dir / "primes_new",
        Path("primes_new.exe"),
        Path("primes_new"),
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


C_SIEVE_PATH = _find_c_sieve_executable()
HAS_C_SIEVE = C_SIEVE_PATH is not None


def _decode_c_sieve_output(filepath: Path, max_primes: int | None = None) -> np.ndarray:
    """Decode binary output from the C sieve into an array of prime numbers.

    The C sieve outputs:
    - First 8 bytes: max_num as uint64 (little-endian)
    - Remaining bytes: bit array where bit i represents number (2*i + 1)
      A set bit indicates the number is prime.

    Args:
        filepath: Path to the primes.out binary file.
        max_primes: If specified, return only the first N primes.

    Returns:
        Array of prime numbers.
    """
    with open(filepath, "rb") as f:
        max_num = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        bit_array = np.frombuffer(f.read(), dtype=np.uint8)

    bits = np.unpackbits(bit_array, bitorder="little")
    bit_indices = np.nonzero(bits)[0]
    odd_primes = 2 * bit_indices + 1
    odd_primes = odd_primes[(odd_primes > 1) & (odd_primes <= max_num)]

    primes = np.empty(len(odd_primes) + 1, dtype=np.int64)
    primes[0] = 2
    primes[1:] = odd_primes

    if max_primes is not None and len(primes) > max_primes:
        return primes[:max_primes]

    return primes


def _c_sieve(limit: int) -> np.ndarray:
    """Generate primes up to limit using the external C sieve.

    Args:
        limit: Upper bound for prime generation.

    Returns:
        Array of prime numbers up to limit.

    Raises:
        RuntimeError: If the C sieve fails to execute.
    """
    if C_SIEVE_PATH is None:
        raise RuntimeError("C sieve executable not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "primes.out"
        original_cwd = os.getcwd()

        try:
            os.chdir(tmpdir)
            result = subprocess.run(
                [str(C_SIEVE_PATH), str(limit), "-p"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(f"C sieve failed: {result.stderr}")

            if not output_file.exists():
                raise RuntimeError("C sieve did not produce output file")

            return _decode_c_sieve_output(output_file)
        finally:
            os.chdir(original_cwd)


def generate_n_primes(n: int, use_gpu: bool = False) -> np.ndarray:
    """Generate the first n prime numbers.

    Uses the prime number theorem to estimate upper bound, then generates
    primes up to that bound and returns the first n.

    Args:
        n: Number of primes to generate.
        use_gpu: If True and CuPy is available, return CuPy array.

    Returns:
        Array of the first n prime numbers.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        result = np.array([2], dtype=np.int64)
        if use_gpu and HAS_CUPY:
            return cp.asarray(result)
        return result

    upper_bound = max(100, int(n * (np.log(n) + np.log(np.log(n + 1)) + 2)))

    primes = generate_primes(upper_bound, use_gpu=False)
    while len(primes) < n:
        upper_bound = int(upper_bound * 1.5)
        primes = generate_primes(upper_bound, use_gpu=False)

    result = primes[:n]

    if use_gpu and HAS_CUPY:
        return cp.asarray(result)

    return result


def _numpy_sieve(limit: int) -> np.ndarray:
    """NumPy-based Sieve of Eratosthenes (fallback implementation).

    Args:
        limit: Upper bound for prime generation.

    Returns:
        Array of prime numbers up to limit.
    """
    if limit < 2:
        return np.array([], dtype=np.int64)

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    return np.nonzero(is_prime)[0].astype(np.int64)


def generate_primes(limit: int, use_gpu: bool = False) -> np.ndarray:
    """Generate all prime numbers up to and including limit.

    Uses the fastest available method:
    1. External C sieve (primes_new) - fastest for large limits
    2. primesieve library - fast C++ implementation
    3. NumPy sieve - pure Python fallback

    Args:
        limit: Upper bound for prime generation (inclusive).
        use_gpu: If True and CuPy is available, return CuPy array.

    Returns:
        Array of prime numbers up to limit.

    Raises:
        ValueError: If limit is less than 2.
    """
    if limit < 2:
        raise ValueError(f"Limit must be >= 2, got {limit}")

    if HAS_C_SIEVE:
        try:
            primes = _c_sieve(limit)
        except RuntimeError:
            if HAS_PRIMESIEVE:
                primes = primesieve_numpy.primes(limit)
            else:
                primes = _numpy_sieve(limit)
    elif HAS_PRIMESIEVE:
        primes = primesieve_numpy.primes(limit)
    else:
        primes = _numpy_sieve(limit)

    if use_gpu and HAS_CUPY:
        return cp.asarray(primes)

    return primes


def generate_primes_range(start: int, stop: int) -> np.ndarray:
    """Generate prime numbers in range [start, stop].

    Args:
        start: Lower bound (inclusive).
        stop: Upper bound (inclusive).

    Returns:
        Array of primes in the specified range.
    """
    if start > stop:
        raise ValueError(f"start ({start}) must be <= stop ({stop})")

    if HAS_PRIMESIEVE:
        return primesieve_numpy.primes(start, stop)

    all_primes = _numpy_sieve(stop)
    return all_primes[all_primes >= start]


def nth_prime(n: int) -> int:
    """Return the nth prime number (1-indexed).

    Args:
        n: Which prime to return (1 = first prime = 2).

    Returns:
        The nth prime number.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if HAS_PRIMESIEVE:
        return primesieve.nth_prime(n)

    upper_bound = max(100, int(n * (np.log(n) + np.log(np.log(n + 1)) + 2)))
    primes = _numpy_sieve(upper_bound)

    while len(primes) < n:
        upper_bound *= 2
        primes = _numpy_sieve(upper_bound)

    return int(primes[n - 1])


def count_primes(limit: int) -> int:
    """Count prime numbers up to limit.

    Args:
        limit: Upper bound for counting.

    Returns:
        Number of primes <= limit.
    """
    if HAS_PRIMESIEVE:
        return primesieve.count_primes(limit)

    if limit < 2:
        return 0

    return len(_numpy_sieve(limit))


def is_prime(n: int) -> bool:
    """Check if a single number is prime.

    Uses 6k +/- 1 optimization for efficiency.

    Args:
        n: Number to check.

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True


def is_prime_array(numbers: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Check primality for an array of numbers.

    For large arrays, generates primes up to max(numbers) and uses
    set membership for O(1) lookups.

    Args:
        numbers: Array of integers to check.
        use_gpu: If True and CuPy available, use GPU acceleration.

    Returns:
        Boolean array where True indicates prime.
    """
    if len(numbers) == 0:
        return np.array([], dtype=bool)

    numbers = np.asarray(numbers)
    max_val = int(numbers.max())

    if max_val < 2:
        return np.zeros(len(numbers), dtype=bool)

    primes = generate_primes(max_val, use_gpu=False)
    prime_set = set(primes)

    if use_gpu and HAS_CUPY:
        numbers_cpu = cp.asnumpy(numbers) if isinstance(numbers, cp.ndarray) else numbers
        result = np.array([n in prime_set for n in numbers_cpu], dtype=bool)
        return cp.asarray(result)

    return np.array([n in prime_set for n in numbers], dtype=bool)


def prime_sieve_mask(limit: int, use_gpu: bool = False) -> np.ndarray:
    """Generate a boolean mask where mask[i] is True if i is prime.

    Args:
        limit: Size of the mask (0 to limit-1).
        use_gpu: If True and CuPy available, return CuPy array.

    Returns:
        Boolean array of length limit.
    """
    xp = cp if (use_gpu and HAS_CUPY) else np

    mask = xp.zeros(limit, dtype=bool)

    if limit < 2:
        return mask

    primes = generate_primes(limit - 1, use_gpu=False)

    if use_gpu and HAS_CUPY:
        primes = cp.asarray(primes)

    mask[primes] = True
    return mask
