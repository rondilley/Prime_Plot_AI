"""Autonomous Prime Visualization Discovery Engine

This script runs continuously, generating novel visualizations and searching
for reproducible patterns that are NOT explained by known prime mathematics.

Key features:
1. Generates random/evolved N-dimensional coordinate mappings
2. Detects visual patterns using directional analysis
3. Removes known mathematical structure (diagonals, mod-N, polynomials)
4. Tests if residual patterns extend to new number ranges
5. Saves promising discoveries with full metadata
6. Runs for hours/days with periodic checkpointing

Usage:
    python run_autonomous_discovery.py --hours 8 --dimensions 2,3,4
    python run_autonomous_discovery.py --cycles 1000 --min-residual-pattern 0.1
"""

import sys
import json
import time
import random
import signal
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SHUTDOWN] Graceful shutdown requested. Saving progress...")


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from prime_plot.core.sieve import generate_primes, prime_sieve_mask
from prime_plot.utils.run_manager import create_run

# Import enhanced detection (with fallback if not available)
try:
    from prime_plot.analysis.enhanced_detection import (
        comprehensive_pattern_detection,
        evaluate_residual,
        PatternDetectionResult,
        ResidualEvaluationResult
    )
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("Warning: Enhanced detection not available, using basic detection")


def setup_file_logger(log_path: Path) -> logging.Logger:
    """Set up logger that writes to both file and console."""
    logger = logging.getLogger("autonomous_discovery")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # psutil not available, return -1
        return -1.0


def estimate_grid_memory_mb(grid_size: int, dimensions: int) -> float:
    """Estimate memory for a grid of given size and dimensions."""
    # float32 = 4 bytes per element
    elements = grid_size ** dimensions
    return elements * 4 / (1024 * 1024)


# Global prime cache for fast lookup
_prime_cache_limit = 0
_prime_set = set()


def get_prime_set(limit: int) -> set:
    """Get set of primes up to limit, using cached sieve."""
    global _prime_cache_limit, _prime_set

    if limit <= _prime_cache_limit:
        return _prime_set

    # Generate new sieve
    print(f"  Generating prime sieve up to {limit}...", end=" ", flush=True)
    primes = generate_primes(limit)
    _prime_set = set(primes.tolist())
    _prime_cache_limit = limit
    print(f"done ({len(_prime_set)} primes)")

    return _prime_set


def fast_is_prime_mask(numbers: np.ndarray, prime_set: set) -> np.ndarray:
    """Fast primality check using precomputed prime set."""
    return np.array([int(n) in prime_set for n in numbers], dtype=bool)


# =============================================================================
# IMAGE-BASED DEDUPLICATION (Perceptual Hashing)
# =============================================================================

def compute_image_hash(image: np.ndarray, hash_size: int = 16) -> int:
    """Compute perceptual hash of an image using average hash algorithm.

    This creates a hash that is similar for visually similar images,
    allowing us to detect duplicate visualizations even when parameters differ.

    Args:
        image: 2D or 3D numpy array (for 3D, uses max projection)
        hash_size: Size of the hash (default 16 = 256 bits)

    Returns:
        Integer hash value
    """
    # For 3D images, use max projection to 2D
    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    # Resize to hash_size x hash_size using simple block averaging
    h, w = image.shape
    block_h = max(1, h // hash_size)
    block_w = max(1, w // hash_size)

    # Compute block averages
    resized = np.zeros((hash_size, hash_size), dtype=np.float32)
    for i in range(hash_size):
        for j in range(hash_size):
            y_start = i * block_h
            y_end = min((i + 1) * block_h, h)
            x_start = j * block_w
            x_end = min((j + 1) * block_w, w)
            block = image[y_start:y_end, x_start:x_end]
            if block.size > 0:
                resized[i, j] = block.mean()

    # Compute mean and create binary hash
    mean_val = resized.mean()
    binary = (resized > mean_val).flatten()

    # Convert to integer hash
    hash_val = 0
    for i, bit in enumerate(binary):
        if bit:
            hash_val |= (1 << i)

    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two hashes."""
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def image_similarity(hash1: int, hash2: int, hash_size: int = 16) -> float:
    """Compute similarity between two image hashes (0.0 to 1.0)."""
    total_bits = hash_size * hash_size
    distance = hamming_distance(hash1, hash2)
    return 1.0 - (distance / total_bits)


# =============================================================================
# N-DIMENSIONAL COORDINATE GENERATION
# =============================================================================

@dataclass
class NDimensionalGenome:
    """Genome encoding for N-dimensional visualization coordinates.

    Uses CURVE FAMILIES to determine fundamental coordinate mapping,
    then applies modifiers from number-theoretic functions.

    This creates truly diverse visualizations by selecting from
    different space-filling curves, spirals, and traversal patterns.
    """
    dimensions: int = 2

    # PRIMARY: Curve type (determines fundamental coordinate mapping)
    curve_type: int = 0  # See CURVE_* constants

    # Curve-specific parameters
    curve_params: Dict = field(default_factory=dict)

    # Per-dimension parameters (lists of length `dimensions`)
    # Linear component: coeff * n
    linear_coeffs: List[float] = field(default_factory=list)

    # Modular component: coeff * (n mod base)
    mod_coeffs: List[float] = field(default_factory=list)
    mod_bases: List[int] = field(default_factory=list)

    # Sqrt component: coeff * sqrt(n)
    sqrt_coeffs: List[float] = field(default_factory=list)

    # Log component: coeff * log(n)
    log_coeffs: List[float] = field(default_factory=list)

    # Sin component: coeff * sin(freq * n)
    sin_coeffs: List[float] = field(default_factory=list)
    sin_freqs: List[float] = field(default_factory=list)

    # Division component: coeff * (n // divisor)
    div_coeffs: List[float] = field(default_factory=list)
    div_bases: List[int] = field(default_factory=list)

    # Digit sum component
    digit_sum_coeffs: List[float] = field(default_factory=list)

    # Prime counting component: coeff * pi(n) approximation
    prime_count_coeffs: List[float] = field(default_factory=list)

    # NEW: Number-theoretic function coefficients
    totient_coeffs: List[float] = field(default_factory=list)      # Euler's phi
    divisor_count_coeffs: List[float] = field(default_factory=list) # tau(n)
    divisor_sum_coeffs: List[float] = field(default_factory=list)   # sigma(n)
    omega_coeffs: List[float] = field(default_factory=list)         # distinct prime factors
    collatz_coeffs: List[float] = field(default_factory=list)       # Collatz steps
    largest_pf_coeffs: List[float] = field(default_factory=list)    # largest prime factor
    smallest_pf_coeffs: List[float] = field(default_factory=list)   # smallest prime factor
    mobius_coeffs: List[float] = field(default_factory=list)        # Mobius function

    # Cross-dimensional interactions (for 3D+)
    # interaction[i][j] = coefficient for dim_i * dim_j interaction
    interactions: List[List[float]] = field(default_factory=list)

    # Metadata
    fitness: float = 0.0
    generation: int = 0

    def __post_init__(self):
        """Initialize parameter lists if empty."""
        if not self.linear_coeffs:
            self._randomize()

    def _randomize(self):
        """Initialize with random parameters."""
        d = self.dimensions

        # PRIMARY: Choose curve type (fundamental coordinate mapping)
        if d == 2:
            self.curve_type = random.choice(ALL_2D_CURVES)
        elif d >= 3:
            # 50% chance of using a 3D-specific curve, 50% chance of 2D curve with z-extension
            if random.random() < 0.5:
                self.curve_type = random.choice(ALL_3D_CURVES)
            else:
                self.curve_type = random.choice(ALL_2D_CURVES)

        # Curve-specific parameters
        self.curve_params = {
            'hilbert_order': random.randint(4, 10),
            'peano_order': random.randint(3, 5),
            'fermat_a': random.uniform(0.5, 3.0),
            'arch_a': random.uniform(0, 5),
            'arch_b': random.uniform(0.5, 3.0),
            'log_a': random.uniform(0.5, 2.0),
            'log_b': random.uniform(0.05, 0.3),
            'mod_x': random.choice([10, 20, 30, 50, 100, 200]),
            'mod_y': random.choice([10, 20, 30, 50, 100, 200]),
            'helix_pitch': random.uniform(1.0, 20.0),
            'torus_p': random.randint(2, 5),
            'torus_q': random.randint(3, 7),
        }

        # Modifier coefficients - START AT ZERO so curve shape dominates
        # Mutations will gradually introduce small modifiers
        # The curve_type is the PRIMARY coordinate generator
        self.linear_coeffs = [0.0 for _ in range(d)]
        self.mod_coeffs = [0.0 for _ in range(d)]
        self.mod_bases = [random.choice([2, 3, 5, 6, 7, 11, 13, 17, 19, 23, 29, 30, 37, 41]) for _ in range(d)]
        self.sqrt_coeffs = [0.0 for _ in range(d)]
        self.log_coeffs = [0.0 for _ in range(d)]
        self.sin_coeffs = [0.0 for _ in range(d)]
        self.sin_freqs = [random.uniform(0.001, 0.2) for _ in range(d)]
        self.div_coeffs = [0.0 for _ in range(d)]
        self.div_bases = [random.randint(10, 500) for _ in range(d)]
        self.digit_sum_coeffs = [0.0 for _ in range(d)]
        self.prime_count_coeffs = [0.0 for _ in range(d)]

        # Number-theoretic function coefficients - START AT ZERO
        self.totient_coeffs = [0.0 for _ in range(d)]
        self.divisor_count_coeffs = [0.0 for _ in range(d)]
        self.divisor_sum_coeffs = [0.0 for _ in range(d)]
        self.omega_coeffs = [0.0 for _ in range(d)]
        self.collatz_coeffs = [0.0 for _ in range(d)]
        self.largest_pf_coeffs = [0.0 for _ in range(d)]
        self.smallest_pf_coeffs = [0.0 for _ in range(d)]
        self.mobius_coeffs = [0.0 for _ in range(d)]

        # Cross-dimensional interactions
        self.interactions = [[random.uniform(-0.1, 0.1) for _ in range(d)] for _ in range(d)]

    def mutate(self, mutation_rate: float = 0.4, aggressive: bool = False):
        """Apply random mutations to parameters.

        Args:
            mutation_rate: Probability of mutating each parameter (default 0.4)
            aggressive: If True, use larger mutation steps for more exploration
        """
        scale_multiplier = 3.0 if aggressive else 1.0

        def mutate_float(val, scale=1.0):
            if random.random() < mutation_rate:
                if random.random() < 0.1:
                    return random.gauss(0, scale * 2.0 * scale_multiplier)
                return val + random.gauss(0, scale * 0.5 * scale_multiplier)
            return val

        def mutate_int(val, choices):
            if random.random() < mutation_rate:
                return random.choice(choices)
            return val

        d = self.dimensions

        # PRIMARY: Occasionally change curve type (more likely if aggressive)
        curve_change_prob = 0.25 if aggressive else 0.1
        if random.random() < curve_change_prob:
            if d == 2:
                self.curve_type = random.choice(ALL_2D_CURVES)
            elif d >= 3:
                if random.random() < 0.5:
                    self.curve_type = random.choice(ALL_3D_CURVES)
                else:
                    self.curve_type = random.choice(ALL_2D_CURVES)

        # Mutate curve parameters
        if self.curve_params:
            if 'hilbert_order' in self.curve_params:
                self.curve_params['hilbert_order'] = mutate_int(
                    self.curve_params['hilbert_order'], list(range(4, 11)))
            if 'fermat_a' in self.curve_params:
                self.curve_params['fermat_a'] = max(0.1, mutate_float(
                    self.curve_params['fermat_a'], 0.5))
            if 'arch_b' in self.curve_params:
                self.curve_params['arch_b'] = max(0.1, mutate_float(
                    self.curve_params['arch_b'], 0.5))
            if 'mod_x' in self.curve_params:
                self.curve_params['mod_x'] = mutate_int(
                    self.curve_params['mod_x'], [10, 20, 30, 50, 100, 200, 500])
            if 'mod_y' in self.curve_params:
                self.curve_params['mod_y'] = mutate_int(
                    self.curve_params['mod_y'], [10, 20, 30, 50, 100, 200, 500])
            if 'helix_pitch' in self.curve_params:
                self.curve_params['helix_pitch'] = max(0.5, mutate_float(
                    self.curve_params['helix_pitch'], 3.0))
            if 'torus_p' in self.curve_params:
                self.curve_params['torus_p'] = mutate_int(
                    self.curve_params['torus_p'], list(range(2, 8)))
            if 'torus_q' in self.curve_params:
                self.curve_params['torus_q'] = mutate_int(
                    self.curve_params['torus_q'], list(range(3, 10)))

        # Modifier coefficients - KEEP TINY so curve shape dominates
        # Typical curve coords are ~100, modifiers should add max ~10 units
        # Scale = max_contribution / typical_input_value
        # linear: n~15000, want ~5 contribution → scale 0.0003
        # mod: mod result ~40, want ~5 contribution → scale 0.1
        # sqrt: sqrt(15000)~122, want ~5 contribution → scale 0.04
        # log: log(15000)~9.6, want ~3 contribution → scale 0.3
        # sin: output ~1, want ~3 contribution → scale 3
        # div: n/100~150, want ~5 contribution → scale 0.03
        # digit_sum: ~40 max, want ~3 contribution → scale 0.08
        # prime_count: ~1700, want ~3 contribution → scale 0.002
        # totient: ~14000, want ~3 contribution → scale 0.0002
        # divisor_count: ~200, want ~3 contribution → scale 0.015
        # etc.
        self.linear_coeffs = [mutate_float(v, 0.0003) for v in self.linear_coeffs]
        self.mod_coeffs = [mutate_float(v, 0.1) for v in self.mod_coeffs]
        self.mod_bases = [mutate_int(v, [2,3,5,6,7,11,13,17,19,23,29,30,37,41,43,47,53,59,61,67]) for v in self.mod_bases]
        self.sqrt_coeffs = [mutate_float(v, 0.04) for v in self.sqrt_coeffs]
        self.log_coeffs = [mutate_float(v, 0.3) for v in self.log_coeffs]
        self.sin_coeffs = [mutate_float(v, 3.0) for v in self.sin_coeffs]
        self.sin_freqs = [mutate_float(v, 0.01) for v in self.sin_freqs]
        self.div_coeffs = [mutate_float(v, 0.03) for v in self.div_coeffs]
        self.div_bases = [mutate_int(v, list(range(10, 501, 10))) for v in self.div_bases]
        self.digit_sum_coeffs = [mutate_float(v, 0.08) for v in self.digit_sum_coeffs]
        self.prime_count_coeffs = [mutate_float(v, 0.002) for v in self.prime_count_coeffs]

        # Number-theoretic function coefficients - VERY SMALL scales
        self.totient_coeffs = [mutate_float(v, 0.0002) for v in self.totient_coeffs]
        self.divisor_count_coeffs = [mutate_float(v, 0.015) for v in self.divisor_count_coeffs]
        self.divisor_sum_coeffs = [mutate_float(v, 0.0001) for v in self.divisor_sum_coeffs]
        self.omega_coeffs = [mutate_float(v, 0.3) for v in self.omega_coeffs]
        self.collatz_coeffs = [mutate_float(v, 0.03) for v in self.collatz_coeffs]
        self.largest_pf_coeffs = [mutate_float(v, 0.001) for v in self.largest_pf_coeffs]
        self.smallest_pf_coeffs = [mutate_float(v, 0.05) for v in self.smallest_pf_coeffs]
        self.mobius_coeffs = [mutate_float(v, 5.0) for v in self.mobius_coeffs]

        for i in range(d):
            for j in range(d):
                self.interactions[i][j] = mutate_float(self.interactions[i][j], 0.1)

    def crossover(self, other: 'NDimensionalGenome') -> 'NDimensionalGenome':
        """Create child genome by crossing with another genome."""
        child = NDimensionalGenome(dimensions=self.dimensions)

        # Inherit curve_type from one parent
        child.curve_type = random.choice([self.curve_type, other.curve_type])

        # Mix curve_params from both parents
        child.curve_params = {}
        all_keys = set(self.curve_params.keys()) | set(other.curve_params.keys())
        for key in all_keys:
            if key in self.curve_params and key in other.curve_params:
                child.curve_params[key] = random.choice([self.curve_params[key], other.curve_params[key]])
            elif key in self.curve_params:
                child.curve_params[key] = self.curve_params[key]
            else:
                child.curve_params[key] = other.curve_params[key]

        # Crossover all coefficient lists
        all_attrs = [
            'linear_coeffs', 'mod_coeffs', 'mod_bases', 'sqrt_coeffs',
            'log_coeffs', 'sin_coeffs', 'sin_freqs', 'div_coeffs',
            'div_bases', 'digit_sum_coeffs', 'prime_count_coeffs',
            'totient_coeffs', 'divisor_count_coeffs', 'divisor_sum_coeffs',
            'omega_coeffs', 'collatz_coeffs', 'largest_pf_coeffs',
            'smallest_pf_coeffs', 'mobius_coeffs'
        ]

        for attr in all_attrs:
            parent_vals = [getattr(self, attr), getattr(other, attr)]
            child_vals = [random.choice([p[i] for p in parent_vals])
                         for i in range(self.dimensions)]
            setattr(child, attr, child_vals)

        # Crossover interactions
        child.interactions = []
        for i in range(self.dimensions):
            row = []
            for j in range(self.dimensions):
                row.append(random.choice([self.interactions[i][j],
                                          other.interactions[i][j]]))
            child.interactions.append(row)

        return child

    def fingerprint(self, precision: int = 1) -> str:
        """Generate a fingerprint string for this genome strategy.

        Used to detect duplicate/similar strategies and skip re-evaluation.
        Rounds coefficients to specified precision to catch near-duplicates.

        IMPORTANT: precision=1 means coefficients differing by <0.05 are
        considered the same strategy (e.g., 0.12 and 0.14 both round to 0.1).
        This catches visually similar images.

        Args:
            precision: Decimal places for rounding coefficients (default 1 for coarse matching)

        Returns:
            String fingerprint that identifies this strategy
        """
        def round_list(lst):
            return tuple(round(x, precision) if isinstance(x, float) else x for x in lst)

        # Round bases to nearest 5 to catch similar modular patterns
        def round_bases(lst):
            return tuple((x // 5) * 5 for x in lst)

        parts = [
            f"d{self.dimensions}",
            f"c{self.curve_type}",
            f"lin{round_list(self.linear_coeffs)}",
            f"mod{round_list(self.mod_coeffs)}",
            f"modb{round_bases(self.mod_bases)}",
            f"sqrt{round_list(self.sqrt_coeffs)}",
            f"log{round_list(self.log_coeffs)}",
            f"sin{round_list(self.sin_coeffs)}",
            f"sinf{round_list(self.sin_freqs)}",
            f"div{round_list(self.div_coeffs)}",
            f"divb{round_bases(self.div_bases)}",
            f"dig{round_list(self.digit_sum_coeffs)}",
        ]
        return "|".join(parts)


# Strategy tracking for deduplication
@dataclass
class StrategyRecord:
    """Record of an evaluated strategy."""
    fingerprint: str
    fitness: float
    search_improvement: float
    is_interesting: bool
    eval_count: int  # How many times this strategy was encountered


def digit_sum(n: int) -> int:
    """Compute digit sum of integer."""
    return sum(int(d) for d in str(abs(n)))


def approx_prime_count(n: int) -> float:
    """Approximate pi(n) using n/ln(n)."""
    if n < 2:
        return 0
    return n / np.log(n)


# =============================================================================
# NUMBER-THEORETIC FUNCTIONS FOR DIVERSE VISUALIZATIONS
# =============================================================================

def euler_totient(n: int) -> int:
    """Euler's totient function phi(n) - count of coprime integers."""
    if n <= 1:
        return max(1, n)
    result = n
    p = 2
    temp_n = n
    while p * p <= temp_n:
        if temp_n % p == 0:
            while temp_n % p == 0:
                temp_n //= p
            result -= result // p
        p += 1
    if temp_n > 1:
        result -= result // temp_n
    return result


def count_divisors(n: int) -> int:
    """Count number of divisors of n (tau function)."""
    if n <= 1:
        return 1
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count


def sum_divisors(n: int) -> int:
    """Sum of divisors of n (sigma function)."""
    if n <= 1:
        return n
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
        i += 1
    return total


def omega_distinct(n: int) -> int:
    """Count distinct prime factors (little omega)."""
    if n <= 1:
        return 0
    count = 0
    p = 2
    while p * p <= n:
        if n % p == 0:
            count += 1
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        count += 1
    return count


def omega_total(n: int) -> int:
    """Count prime factors with multiplicity (big Omega)."""
    if n <= 1:
        return 0
    count = 0
    p = 2
    while p * p <= n:
        while n % p == 0:
            count += 1
            n //= p
        p += 1
    if n > 1:
        count += 1
    return count


def largest_prime_factor(n: int) -> int:
    """Return largest prime factor of n."""
    if n <= 1:
        return 1
    largest = 1
    p = 2
    while p * p <= n:
        while n % p == 0:
            largest = p
            n //= p
        p += 1
    if n > 1:
        largest = n
    return largest


def smallest_prime_factor(n: int) -> int:
    """Return smallest prime factor of n."""
    if n <= 1:
        return 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            return p
        p += 1
    return n


def collatz_steps(n: int, max_steps: int = 500) -> int:
    """Count steps in Collatz sequence to reach 1."""
    if n <= 1:
        return 0
    steps = 0
    while n != 1 and steps < max_steps:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps


def mobius(n: int) -> int:
    """Mobius function: -1, 0, or 1 based on prime factorization."""
    if n <= 1:
        return 1
    p = 2
    count = 0
    while p * p <= n:
        if n % p == 0:
            n //= p
            count += 1
            if n % p == 0:  # Square factor
                return 0
        p += 1
    if n > 1:
        count += 1
    return -1 if count % 2 == 1 else 1


# =============================================================================
# CURVE FAMILIES - Fundamentally different ways to map integers to coordinates
# =============================================================================

# Curve type constants
CURVE_ULAM = 0          # Ulam spiral (square spiral from center)
CURVE_HILBERT = 1       # Hilbert space-filling curve
CURVE_ZORDER = 2        # Z-order/Morton curve (bit interleaving)
CURVE_SACKS = 3         # Sacks spiral (Archimedean, sqrt-based)
CURVE_VOGEL = 4         # Vogel spiral (golden angle)
CURVE_FERMAT = 5        # Fermat spiral (r^2 proportional to theta)
CURVE_ARCHIMEDEAN = 6   # Archimedean spiral (constant arm spacing)
CURVE_LOGARITHMIC = 7   # Logarithmic spiral (constant angle)
CURVE_THEODORUS = 8     # Spiral of Theodorus (square root spiral)
CURVE_FIBONACCI = 9     # Fibonacci spiral (golden rectangle based)
CURVE_MODULAR_GRID = 10 # Simple modular grid
CURVE_DIAGONAL = 11     # Diagonal traversal (Cantor pairing)
CURVE_PEANO = 12        # Peano curve variant

# 3D curve types
CURVE_3D_HILBERT = 20   # 3D Hilbert curve
CURVE_3D_HELIX = 21     # Helical spiral
CURVE_3D_SPHERICAL = 22 # Spherical spiral
CURVE_3D_TORUS = 23     # Torus knot
CURVE_3D_ZORDER = 24    # 3D Z-order

ALL_2D_CURVES = [CURVE_ULAM, CURVE_HILBERT, CURVE_ZORDER, CURVE_SACKS,
                 CURVE_VOGEL, CURVE_FERMAT, CURVE_ARCHIMEDEAN, CURVE_LOGARITHMIC,
                 CURVE_THEODORUS, CURVE_FIBONACCI, CURVE_MODULAR_GRID,
                 CURVE_DIAGONAL, CURVE_PEANO]

ALL_3D_CURVES = [CURVE_3D_HILBERT, CURVE_3D_HELIX, CURVE_3D_SPHERICAL,
                 CURVE_3D_TORUS, CURVE_3D_ZORDER]


def ulam_coords(n: int) -> Tuple[int, int]:
    """Ulam spiral: square spiral from center outward."""
    if n <= 0:
        return (0, 0)
    if n == 1:
        return (0, 0)

    k = int((np.sqrt(n - 1) + 1) / 2)
    start = (2 * k - 1) ** 2 + 1
    side_len = 2 * k
    pos = n - start
    side = pos // side_len
    offset = pos % side_len

    if side == 0:
        return (k, -k + 1 + offset)
    elif side == 1:
        return (k - 1 - offset, k)
    elif side == 2:
        return (-k, k - 1 - offset)
    else:
        return (-k + 1 + offset, -k)


def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert index d to (x,y) on Hilbert curve of order n."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return (x, y)


def hilbert_coords(n: int, order: int = 8) -> Tuple[int, int]:
    """Get Hilbert curve coordinates for integer n."""
    if n <= 0:
        return (0, 0)
    size = 2 ** order
    n_mod = n % (size * size)
    return hilbert_d2xy(size, n_mod)


def zorder_coords(n: int) -> Tuple[int, int]:
    """Z-order/Morton curve: interleave bits of x and y."""
    if n <= 0:
        return (0, 0)
    x = y = 0
    bit = 0
    temp_n = n
    while temp_n > 0:
        x |= (temp_n & 1) << bit
        temp_n >>= 1
        y |= (temp_n & 1) << bit
        temp_n >>= 1
        bit += 1
    return (x, y)


def sacks_coords(n: int) -> Tuple[float, float]:
    """Sacks spiral: r = sqrt(n), theta = 2*pi*sqrt(n)."""
    if n <= 0:
        return (0.0, 0.0)
    r = np.sqrt(n)
    theta = 2 * np.pi * np.sqrt(n)
    return (r * np.cos(theta), r * np.sin(theta))


def vogel_coords(n: int) -> Tuple[float, float]:
    """Vogel spiral: golden angle spacing."""
    if n <= 0:
        return (0.0, 0.0)
    golden_angle = np.pi * (3 - np.sqrt(5))
    r = np.sqrt(n)
    theta = n * golden_angle
    return (r * np.cos(theta), r * np.sin(theta))


def fermat_coords(n: int, a: float = 1.0) -> Tuple[float, float]:
    """Fermat spiral: r^2 = a^2 * theta."""
    if n <= 0:
        return (0.0, 0.0)
    theta = n * 0.1
    r = a * np.sqrt(theta)
    return (r * np.cos(theta), r * np.sin(theta))


def archimedean_coords(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[float, float]:
    """Archimedean spiral: r = a + b*theta."""
    if n <= 0:
        return (0.0, 0.0)
    theta = n * 0.1
    r = a + b * theta
    return (r * np.cos(theta), r * np.sin(theta))


def logarithmic_coords(n: int, a: float = 1.0, b: float = 0.1) -> Tuple[float, float]:
    """Logarithmic spiral: r = a * e^(b*theta)."""
    if n <= 0:
        return (0.0, 0.0)
    theta = n * 0.05
    # Clamp exponent to avoid overflow (exp(700) ~ 1e304, max float)
    exponent = min(b * theta, 700.0)
    r = a * np.exp(exponent)
    # Clamp r to avoid explosion in coordinate calculation
    r = min(r, 1000)
    return (r * np.cos(theta), r * np.sin(theta))


def theodorus_coords(n: int) -> Tuple[float, float]:
    """Spiral of Theodorus: approximation using closed-form formula.

    The angle grows approximately as 2*sqrt(n), and r = sqrt(n).
    This avoids O(n) iteration per number.
    """
    if n <= 0:
        return (0.0, 0.0)
    # Approximation: angle ~ 2*sqrt(n) radians
    r = np.sqrt(n)
    theta = 2.0 * np.sqrt(n)
    return (r * np.cos(theta), r * np.sin(theta))


def fibonacci_coords(n: int) -> Tuple[float, float]:
    """Fibonacci spiral using golden rectangle construction."""
    if n <= 0:
        return (0.0, 0.0)
    phi = (1 + np.sqrt(5)) / 2
    theta = n * 2 * np.pi / phi
    r = np.power(phi, n * 0.01)
    r = min(r, 500)
    return (r * np.cos(theta), r * np.sin(theta))


def diagonal_coords(n: int) -> Tuple[int, int]:
    """Cantor diagonal pairing - traverse grid diagonally."""
    if n <= 0:
        return (0, 0)
    # Inverse of Cantor pairing function
    w = int((np.sqrt(8 * n + 1) - 1) / 2)
    t = (w * w + w) // 2
    y = n - t
    x = w - y
    return (x, y)


def peano_coords(n: int, order: int = 4) -> Tuple[int, int]:
    """Simplified Peano curve variant."""
    if n <= 0:
        return (0, 0)
    size = 3 ** order
    n_mod = n % (size * size)
    # Use ternary decomposition
    x = y = 0
    s = 1
    temp = n_mod
    for _ in range(order):
        rx = temp % 3
        temp //= 3
        ry = temp % 3
        temp //= 3
        x += s * rx
        y += s * ry
        s *= 3
    return (x, y)


# 3D curve implementations
def hilbert_3d_coords(n: int, order: int = 4) -> Tuple[int, int, int]:
    """3D Hilbert curve coordinates."""
    if n <= 0:
        return (0, 0, 0)
    size = 2 ** order
    n_mod = n % (size ** 3)

    x = y = z = 0
    s = 1
    temp = n_mod

    while s < size:
        rx = 1 & (temp // 4)
        ry = 1 & (temp // 2)
        rz = 1 & temp

        # Rotation based on octant
        if rz == 0:
            if ry == 0:
                x, y = y, x
            else:
                x, z = size - 1 - z, size - 1 - x

        x += s * rx
        y += s * ry
        z += s * rz
        temp //= 8
        s *= 2

    return (x, y, z)


def helix_coords(n: int, pitch: float = 5.0) -> Tuple[float, float, float]:
    """Helical spiral in 3D."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    theta = n * 0.1
    r = np.sqrt(n) * 0.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = theta * pitch / (2 * np.pi)
    return (x, y, z)


def spherical_spiral_coords(n: int) -> Tuple[float, float, float]:
    """Spherical spiral - covers surface of sphere."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    # Fibonacci lattice on sphere
    phi = (1 + np.sqrt(5)) / 2
    i = n
    theta = 2 * np.pi * i / phi
    cos_phi = 1 - (2 * i + 1) / (2 * max(n, 1))
    cos_phi = max(-1, min(1, cos_phi))
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    r = np.sqrt(n) * 0.1

    x = r * sin_phi * np.cos(theta)
    y = r * sin_phi * np.sin(theta)
    z = r * cos_phi
    return (x, y, z)


def torus_knot_coords(n: int, p: int = 2, q: int = 3) -> Tuple[float, float, float]:
    """Torus knot parametric curve."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    t = n * 0.01
    R = 10  # Major radius
    r_minor = 3  # Minor radius

    x = (R + r_minor * np.cos(q * t)) * np.cos(p * t)
    y = (R + r_minor * np.cos(q * t)) * np.sin(p * t)
    z = r_minor * np.sin(q * t)
    return (x, y, z)


def zorder_3d_coords(n: int) -> Tuple[int, int, int]:
    """3D Z-order/Morton curve."""
    if n <= 0:
        return (0, 0, 0)
    x = y = z = 0
    bit = 0
    temp_n = n
    while temp_n > 0:
        x |= (temp_n & 1) << bit
        temp_n >>= 1
        y |= (temp_n & 1) << bit
        temp_n >>= 1
        z |= (temp_n & 1) << bit
        temp_n >>= 1
        bit += 1
    return (x, y, z)


def get_curve_coords_2d(n: int, curve_type: int, params: dict = None) -> Tuple[float, float]:
    """Get 2D coordinates for integer n using specified curve type."""
    params = params or {}

    if curve_type == CURVE_ULAM:
        return ulam_coords(n)
    elif curve_type == CURVE_HILBERT:
        order = params.get('hilbert_order', 8)
        return hilbert_coords(n, order)
    elif curve_type == CURVE_ZORDER:
        return zorder_coords(n)
    elif curve_type == CURVE_SACKS:
        return sacks_coords(n)
    elif curve_type == CURVE_VOGEL:
        return vogel_coords(n)
    elif curve_type == CURVE_FERMAT:
        a = params.get('fermat_a', 1.0)
        return fermat_coords(n, a)
    elif curve_type == CURVE_ARCHIMEDEAN:
        a = params.get('arch_a', 0.0)
        b = params.get('arch_b', 1.0)
        return archimedean_coords(n, a, b)
    elif curve_type == CURVE_LOGARITHMIC:
        a = params.get('log_a', 1.0)
        b = params.get('log_b', 0.1)
        return logarithmic_coords(n, a, b)
    elif curve_type == CURVE_THEODORUS:
        return theodorus_coords(n)
    elif curve_type == CURVE_FIBONACCI:
        return fibonacci_coords(n)
    elif curve_type == CURVE_MODULAR_GRID:
        mod_x = params.get('mod_x', 100)
        mod_y = params.get('mod_y', 100)
        return (n % mod_x, (n // mod_x) % mod_y)
    elif curve_type == CURVE_DIAGONAL:
        return diagonal_coords(n)
    elif curve_type == CURVE_PEANO:
        order = params.get('peano_order', 4)
        return peano_coords(n, order)
    else:
        return (float(n), 0.0)


def get_curve_coords_3d(n: int, curve_type: int, params: dict = None) -> Tuple[float, float, float]:
    """Get 3D coordinates for integer n using specified curve type."""
    params = params or {}

    if curve_type == CURVE_3D_HILBERT:
        order = params.get('hilbert_order', 4)
        return hilbert_3d_coords(n, order)
    elif curve_type == CURVE_3D_HELIX:
        pitch = params.get('helix_pitch', 5.0)
        return helix_coords(n, pitch)
    elif curve_type == CURVE_3D_SPHERICAL:
        return spherical_spiral_coords(n)
    elif curve_type == CURVE_3D_TORUS:
        p = params.get('torus_p', 2)
        q = params.get('torus_q', 3)
        return torus_knot_coords(n, p, q)
    elif curve_type == CURVE_3D_ZORDER:
        return zorder_3d_coords(n)
    else:
        # Fall back to 2D curve + layer
        x, y = get_curve_coords_2d(n, curve_type % len(ALL_2D_CURVES), params)
        z = n // 10000
        return (x, y, float(z))


def compute_nd_coordinates(
    numbers: np.ndarray,
    genome: NDimensionalGenome
) -> np.ndarray:
    """Compute N-dimensional coordinates for array of integers.

    PRIMARY: Uses curve_type to determine fundamental coordinate mapping
    (Hilbert, Ulam, Z-order, Sacks, Vogel, Fermat, etc.)

    SECONDARY: Applies small modifiers from number-theoretic functions

    Args:
        numbers: Array of integers to map
        genome: Coordinate mapping parameters

    Returns:
        Array of shape (len(numbers), genome.dimensions)
    """
    n = len(numbers)
    d = genome.dimensions
    coords = np.zeros((n, d), dtype=np.float64)

    # Get curve parameters
    params = genome.curve_params if hasattr(genome, 'curve_params') and genome.curve_params else {}

    for i, num in enumerate(numbers):
        if num <= 0:
            continue

        int_num = int(num)

        # PRIMARY: Get base coordinates from curve type
        if d >= 3 and genome.curve_type in ALL_3D_CURVES:
            # Use 3D curve
            bx, by, bz = get_curve_coords_3d(int_num, genome.curve_type, params)
            base_curve = [float(bx), float(by), float(bz)]
            for dim in range(3, d):
                base_curve.append(0.0)
        elif d >= 2:
            # Use 2D curve
            bx, by = get_curve_coords_2d(int_num, genome.curve_type, params)
            base_curve = [float(bx), float(by)]
            for dim in range(2, d):
                # For higher dimensions, add layer based on number
                base_curve.append(float(int_num // (10 ** (dim - 1))))
        else:
            base_curve = [float(int_num)]

        # SECONDARY: Compute small modifiers from number-theoretic functions
        modifiers = np.zeros(d)

        for dim in range(d):
            val = 0.0

            # Keep modifier coefficients small so curve shape dominates
            val += genome.linear_coeffs[dim] * num
            val += genome.mod_coeffs[dim] * (num % genome.mod_bases[dim])
            val += genome.sqrt_coeffs[dim] * np.sqrt(num)
            val += genome.log_coeffs[dim] * np.log(max(1, num))
            val += genome.sin_coeffs[dim] * np.sin(genome.sin_freqs[dim] * num)
            val += genome.div_coeffs[dim] * (num // genome.div_bases[dim])
            val += genome.digit_sum_coeffs[dim] * digit_sum(int_num)
            val += genome.prime_count_coeffs[dim] * approx_prime_count(int_num)

            # Number-theoretic functions (expensive, so only compute if coeff is significant)
            # Thresholds set to ~10% of typical mutated values
            if abs(genome.totient_coeffs[dim]) > 0.00002:
                val += genome.totient_coeffs[dim] * euler_totient(int_num)
            if abs(genome.divisor_count_coeffs[dim]) > 0.0015:
                val += genome.divisor_count_coeffs[dim] * count_divisors(int_num)
            if abs(genome.divisor_sum_coeffs[dim]) > 0.00001:
                val += genome.divisor_sum_coeffs[dim] * sum_divisors(int_num)
            if abs(genome.omega_coeffs[dim]) > 0.03:
                val += genome.omega_coeffs[dim] * omega_distinct(int_num)
            if abs(genome.collatz_coeffs[dim]) > 0.003:
                val += genome.collatz_coeffs[dim] * collatz_steps(int_num)
            if abs(genome.largest_pf_coeffs[dim]) > 0.0001:
                val += genome.largest_pf_coeffs[dim] * largest_prime_factor(int_num)
            if abs(genome.smallest_pf_coeffs[dim]) > 0.005:
                val += genome.smallest_pf_coeffs[dim] * smallest_prime_factor(int_num)
            if abs(genome.mobius_coeffs[dim]) > 0.5:
                val += genome.mobius_coeffs[dim] * mobius(int_num)

            modifiers[dim] = val

        # SAFETY: Clip modifiers to max ±30 units so curve shape dominates
        # This prevents runaway modifier accumulation from overwhelming the curve
        max_modifier = 30.0
        modifiers = np.clip(modifiers, -max_modifier, max_modifier)

        # Combine: curve base + modifiers
        for dim in range(d):
            if dim < len(base_curve):
                coords[i, dim] = base_curve[dim] + modifiers[dim]
            else:
                coords[i, dim] = modifiers[dim]

        # Apply cross-dimensional interactions (keep small)
        if genome.interactions:
            final_coords = coords[i].copy()
            for dim in range(d):
                for other_dim in range(d):
                    if dim != other_dim and dim < len(genome.interactions) and other_dim < len(genome.interactions[dim]):
                        final_coords[dim] += genome.interactions[dim][other_dim] * coords[i, other_dim] * 0.01
            coords[i] = final_coords

    return coords


# =============================================================================
# VISUALIZATION RENDERING
# =============================================================================

def render_nd_visualization(
    coords: np.ndarray,
    is_prime: np.ndarray,
    grid_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Render N-dimensional coordinates to grid(s).

    For 2D: Returns single 2D image
    For 3D: Returns 3D volume
    For 4D+: Returns projection to 3D volume

    Args:
        coords: (N, D) array of coordinates
        is_prime: (N,) boolean array
        grid_size: Size of output grid per dimension

    Returns:
        (input_grid, target_grid) - input shows all numbers, target shows only primes
    """
    n_points, dims = coords.shape

    # Normalize coordinates to [0, grid_size-1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero

    normalized = (coords - mins) / ranges * (grid_size - 1)
    normalized = np.clip(normalized, 0, grid_size - 1).astype(np.int32)

    if dims == 2:
        input_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        target_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for i in range(n_points):
            x, y = normalized[i]
            input_grid[y, x] = 1.0
            if is_prime[i]:
                target_grid[y, x] = 1.0

    elif dims == 3:
        input_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        target_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        for i in range(n_points):
            x, y, z = normalized[i]
            input_grid[z, y, x] = 1.0
            if is_prime[i]:
                target_grid[z, y, x] = 1.0
    else:
        # For 4D+, project to 3D by summing over extra dimensions
        input_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        target_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        for i in range(n_points):
            x, y, z = normalized[i, :3]  # Use first 3 dimensions
            input_grid[z, y, x] += 1.0
            if is_prime[i]:
                target_grid[z, y, x] += 1.0

    return input_grid, target_grid


def render_2d_projection(grid_3d: np.ndarray) -> np.ndarray:
    """Project 3D grid to 2D by max projection along z-axis."""
    return np.max(grid_3d, axis=0)


# =============================================================================
# PATTERN DETECTION AND REMOVAL
# =============================================================================

def detect_patterns_nd(
    prime_grid: np.ndarray,
    dimensions: int
) -> Tuple[np.ndarray, float]:
    """Detect patterns in N-dimensional prime grid.

    For 3D+, we use BOTH approaches to avoid missing patterns:
    1. 3D volumetric detection (planes and lines in 3D space)
    2. Multi-axis 2D projections (patterns visible from different angles)

    Returns the maximum pattern fraction from all methods to ensure we
    don't miss patterns that are only visible in one representation.

    Returns:
        (pattern_mask, pattern_fraction) - mask of detected patterns and fraction
    """
    if dimensions == 2:
        return detect_patterns_2d(prime_grid)
    elif dimensions >= 3:
        # Collect pattern fractions from multiple detection methods
        fractions = []
        masks = []

        # Method 1: 3D volumetric detection (planes and lines)
        mask_3d, frac_3d = detect_patterns_3d(prime_grid)
        fractions.append(frac_3d)
        masks.append(('3d_volume', mask_3d))

        # Method 2: Multi-axis 2D projections
        # Project along each axis and detect patterns
        for axis in range(min(3, len(prime_grid.shape))):
            projection = np.max(prime_grid, axis=axis)
            mask_proj, frac_proj = detect_patterns_2d(projection)
            fractions.append(frac_proj)
            masks.append((f'proj_axis{axis}', mask_proj))

        # Take maximum pattern fraction (most conservative - catches patterns
        # visible in ANY representation)
        max_fraction = max(fractions)

        # Return the mask from the method that found the most patterns
        best_idx = fractions.index(max_fraction)
        best_mask = masks[best_idx][1]

        return best_mask, max_fraction


def detect_patterns_2d(prime_grid: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect diagonal and linear patterns in 2D grid.

    Uses enhanced multi-method detection when available:
    - Gabor filter bank (oriented textures)
    - FFT (periodic patterns)
    - Directional kernels (16 orientations)
    - Cluster detection
    """
    # Use > 0 threshold to catch all primes, not just high-intensity pixels
    # (grid may be normalized so single primes have low values like 0.33)
    binary = (prime_grid > 0).astype(np.float32)
    total_primes = binary.sum()

    if total_primes == 0:
        return np.zeros_like(binary, dtype=bool), 0.0

    # Use enhanced detection if available
    if ENHANCED_DETECTION_AVAILABLE:
        result = comprehensive_pattern_detection(binary)
        return result.pattern_mask, result.pattern_fraction

    # Fallback to basic detection
    # Diagonal detection kernels
    k45 = np.eye(7, dtype=np.float32)
    k135 = np.fliplr(k45)

    # Horizontal and vertical
    k_h = np.ones((1, 7), dtype=np.float32)
    k_v = np.ones((7, 1), dtype=np.float32)

    resp_45 = ndimage.convolve(binary, k45 / k45.sum())
    resp_135 = ndimage.convolve(binary, k135 / k135.sum())
    resp_h = ndimage.convolve(binary, k_h / k_h.sum())
    resp_v = ndimage.convolve(binary, k_v / k_v.sum())

    # Combine all linear patterns
    max_resp = np.maximum(np.maximum(resp_45, resp_135), np.maximum(resp_h, resp_v))

    pattern_mask = (max_resp > 0.35) & (binary > 0.5)
    pattern_fraction = pattern_mask.sum() / total_primes

    return pattern_mask, float(pattern_fraction)


def detect_patterns_3d(prime_grid: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect planar and linear patterns in 3D grid.

    Uses multiple kernel types:
    1. Axis-aligned lines (x, y, z directions)
    2. Face diagonals (6 directions on cube faces)
    3. Space diagonals (4 main cube diagonals)
    4. Planar detectors (xy, xz, yz planes)
    """
    # Use > 0 threshold to catch all primes, not just high-intensity pixels
    binary = (prime_grid > 0).astype(np.float32)
    total_primes = binary.sum()

    if total_primes == 0:
        return np.zeros_like(binary, dtype=bool), 0.0

    kernels = []
    k_size = 5

    # 1. Axis-aligned lines (3 directions)
    k_x = np.zeros((1, 1, k_size), dtype=np.float32)
    k_x[0, 0, :] = 1.0
    kernels.append(k_x)

    k_y = np.zeros((1, k_size, 1), dtype=np.float32)
    k_y[0, :, 0] = 1.0
    kernels.append(k_y)

    k_z = np.zeros((k_size, 1, 1), dtype=np.float32)
    k_z[:, 0, 0] = 1.0
    kernels.append(k_z)

    # 2. Face diagonals (6 directions - 2 per face)
    # XY plane diagonals
    k_xy1 = np.zeros((1, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_xy1[0, i, i] = 1.0
    kernels.append(k_xy1)

    k_xy2 = np.zeros((1, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_xy2[0, i, k_size-1-i] = 1.0
    kernels.append(k_xy2)

    # XZ plane diagonals
    k_xz1 = np.zeros((k_size, 1, k_size), dtype=np.float32)
    for i in range(k_size):
        k_xz1[i, 0, i] = 1.0
    kernels.append(k_xz1)

    k_xz2 = np.zeros((k_size, 1, k_size), dtype=np.float32)
    for i in range(k_size):
        k_xz2[i, 0, k_size-1-i] = 1.0
    kernels.append(k_xz2)

    # YZ plane diagonals
    k_yz1 = np.zeros((k_size, k_size, 1), dtype=np.float32)
    for i in range(k_size):
        k_yz1[i, i, 0] = 1.0
    kernels.append(k_yz1)

    k_yz2 = np.zeros((k_size, k_size, 1), dtype=np.float32)
    for i in range(k_size):
        k_yz2[i, k_size-1-i, 0] = 1.0
    kernels.append(k_yz2)

    # 3. Space diagonals (4 main diagonals of cube)
    k_diag1 = np.zeros((k_size, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_diag1[i, i, i] = 1.0
    kernels.append(k_diag1)

    k_diag2 = np.zeros((k_size, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_diag2[i, i, k_size-1-i] = 1.0
    kernels.append(k_diag2)

    k_diag3 = np.zeros((k_size, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_diag3[i, k_size-1-i, i] = 1.0
    kernels.append(k_diag3)

    k_diag4 = np.zeros((k_size, k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        k_diag4[k_size-1-i, i, i] = 1.0
    kernels.append(k_diag4)

    # 4. Planar detectors (detect points on planes)
    # XY plane (constant Z)
    k_plane_xy = np.zeros((1, 3, 3), dtype=np.float32)
    k_plane_xy[0, :, :] = 1.0
    kernels.append(k_plane_xy)

    # XZ plane (constant Y)
    k_plane_xz = np.zeros((3, 1, 3), dtype=np.float32)
    k_plane_xz[:, 0, :] = 1.0
    kernels.append(k_plane_xz)

    # YZ plane (constant X)
    k_plane_yz = np.zeros((3, 3, 1), dtype=np.float32)
    k_plane_yz[:, :, 0] = 1.0
    kernels.append(k_plane_yz)

    # Compute responses for all kernels
    responses = []
    for k in kernels:
        resp = ndimage.convolve(binary, k / k.sum())
        responses.append(resp)

    max_resp = np.maximum.reduce(responses)

    pattern_mask = (max_resp > 0.35) & (binary > 0.5)
    pattern_fraction = pattern_mask.sum() / total_primes if total_primes > 0 else 0.0

    return pattern_mask, float(pattern_fraction)


def compute_residual_pattern_score(
    prime_grid: np.ndarray,
    pattern_mask: np.ndarray,
    genome: Optional['NDimensionalGenome'] = None,
    prime_set: Optional[set] = None,
    test_range: Optional[Tuple[int, int]] = None,
    grid_size: int = 128
) -> Tuple[float, bool, str]:
    """Evaluate residual for exploitable density patterns.

    FIXED: Instead of just running pattern detection again (which was broken),
    we now measure:
    1. Density variance - are there concentration/void regions?
    2. Transfer test - do density patterns predict in new number range?
    3. Exploitability - can this help prime search?

    Returns:
        (exploitability_score, is_interesting, reason)
    """
    # Use > 0 to catch all primes (grid may be normalized with low values)
    residual = (prime_grid > 0) & ~pattern_mask

    if residual.sum() < 20:
        return 0.0, False, "Too few residual primes"

    residual_float = residual.astype(np.float32)

    # Get 2D representation for analysis
    if len(residual_float.shape) == 3:
        residual_2d = np.max(residual_float, axis=0)
    else:
        residual_2d = residual_float

    # Use enhanced evaluation if available
    if ENHANCED_DETECTION_AVAILABLE:
        result = evaluate_residual(
            residual_2d,
            genome=genome,
            prime_set=prime_set,
            test_range=test_range,
            grid_size=grid_size
        )
        return result.exploitability_score, result.is_interesting, result.reason

    # Fallback to basic density variance check
    from scipy import ndimage as ndi

    # Compute local density
    kernel = np.ones((15, 15), dtype=np.float32) / 225
    local_density = ndi.convolve(residual_2d, kernel)

    # Measure variance
    total_primes = residual_2d.sum()
    expected_density = total_primes / residual_2d.size
    expected_variance = expected_density * (1 - expected_density)
    actual_variance = local_density.var()

    variance_ratio = actual_variance / expected_variance if expected_variance > 0 else 1.0

    # Simple scoring
    if variance_ratio > 2.0:
        return min(variance_ratio / 5.0, 1.0), True, f"High variance ratio: {variance_ratio:.2f}"
    else:
        return variance_ratio / 5.0, False, f"Low variance ratio: {variance_ratio:.2f}"


# =============================================================================
# PATTERN EXTENSION TESTING
# =============================================================================

def test_pattern_extends(
    genome: NDimensionalGenome,
    known_range: Tuple[int, int],
    test_range: Tuple[int, int],
    grid_size: int = 128,
    prime_set: Optional[set] = None
) -> Tuple[bool, float]:
    """Test if patterns found in known_range extend to test_range.

    Returns:
        (extends, correlation) - whether pattern extends and correlation score
    """
    # Get prime set if not provided
    if prime_set is None:
        prime_set = get_prime_set(test_range[1])

    # Generate visualization for known range
    known_numbers = np.arange(known_range[0], known_range[1], dtype=np.int64)
    known_primes = fast_is_prime_mask(known_numbers, prime_set)
    known_coords = compute_nd_coordinates(known_numbers, genome)
    _, known_target = render_nd_visualization(known_coords, known_primes, grid_size)

    # Detect patterns in known range
    known_pattern, known_frac = detect_patterns_nd(known_target, genome.dimensions)

    if known_frac < 0.1:
        return False, 0.0

    # Generate visualization for test range
    test_numbers = np.arange(test_range[0], test_range[1], dtype=np.int64)
    test_primes = fast_is_prime_mask(test_numbers, prime_set)
    test_coords = compute_nd_coordinates(test_numbers, genome)
    _, test_target = render_nd_visualization(test_coords, test_primes, grid_size)

    # Detect patterns in test range
    test_pattern, test_frac = detect_patterns_nd(test_target, genome.dimensions)

    # Compare pattern fractions
    if known_frac > 0:
        correlation = min(test_frac / known_frac, known_frac / test_frac) if test_frac > 0 else 0
    else:
        correlation = 0.0

    extends = correlation > 0.7  # Patterns are similar if within 30%

    return extends, correlation


# =============================================================================
# PRIME SEARCH VALIDATION (Critical test: does pattern reduce primality tests?)
# =============================================================================

def miller_rabin_test(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

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
class PrimeSearchValidation:
    """Result of validating a discovery for actual prime search improvement."""
    genome_id: str
    brute_force_tests: int
    pattern_guided_tests: int
    primes_found: int
    improvement_factor: float  # brute_force_tests / pattern_guided_tests
    is_useful: bool  # improvement_factor > 1.05 (5% improvement minimum)
    search_range: Tuple[int, int]
    high_density_hits: int
    low_density_hits: int


@dataclass
class MultiScaleValidation:
    """Result of validating a discovery across multiple scales."""
    genome_id: str
    scale_results: Dict[str, PrimeSearchValidation]  # scale_name -> validation
    min_improvement: float  # Minimum improvement across all scales
    avg_improvement: float  # Average improvement across all scales
    scales_passed: int  # How many scales showed >5% improvement
    total_scales: int
    is_robust: bool  # True if pattern works at ALL scales (>5% at each)


# Scale configurations for multi-scale validation
VALIDATION_SCALES = [
    # (name, train_start, train_end, test_start, test_end, target_primes)
    ("15K", 1, 15_000, 30_000, 60_000, 30),
    ("1M", 100_000, 1_000_000, 2_000_000, 4_000_000, 50),
    ("100M", 10_000_000, 100_000_000, 200_000_000, 400_000_000, 50),
    ("2B", 100_000_000, 2_000_000_000, 2_500_000_000, 3_000_000_000, 30),
]


def validate_at_scale(
    genome: NDimensionalGenome,
    scale_name: str,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    target_primes: int,
    seed: int = 42
) -> PrimeSearchValidation:
    """Validate a genome at a specific scale using Miller-Rabin for large numbers."""
    random.seed(seed)

    genome_id = f"{genome.curve_type}_{genome.dimensions}D_{scale_name}"

    # For large scales, we can't use a sieve - use Miller-Rabin directly
    # Build density map from training range by sampling
    sample_size = min(100_000, train_end - train_start)
    sample_step = max(1, (train_end - train_start) // sample_size)

    # Sample training numbers
    train_samples = []
    n = train_start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1
    step = 4 if n % 6 == 1 else 2

    while n < train_end and len(train_samples) < sample_size:
        train_samples.append(n)
        n += step * sample_step
        step = 6 - step

    # Compute coordinates for training samples
    train_numbers = np.array(train_samples, dtype=np.int64)
    train_coords = compute_nd_coordinates(train_numbers, genome)
    coords_2d = train_coords[:, :2]

    # Normalize
    mins = coords_2d.min(axis=0)
    maxs = coords_2d.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    # Build density map using Miller-Rabin to test primality
    grid_size = 32  # Smaller grid for speed at large scales
    bin_prime_count = np.zeros((grid_size, grid_size), dtype=np.float32)
    bin_total_count = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, num in enumerate(train_numbers):
        x = int((coords_2d[i, 0] - mins[0]) / ranges[0] * (grid_size - 1))
        y = int((coords_2d[i, 1] - mins[1]) / ranges[1] * (grid_size - 1))
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        bin_total_count[y, x] += 1
        if miller_rabin_test(int(num)):
            bin_prime_count[y, x] += 1

    # Compute density
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    valid_mask = bin_total_count >= 3
    density_map[valid_mask] = bin_prime_count[valid_mask] / bin_total_count[valid_mask]

    mean_density = density_map[valid_mask].mean() if valid_mask.any() else 0.1
    high_threshold = mean_density * 1.2

    # Brute force search in test range
    brute_primes = []
    brute_tests = 0
    n = test_start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1
    step = 4 if n % 6 == 1 else 2

    random.seed(seed)
    while len(brute_primes) < target_primes and n < test_end:
        brute_tests += 1
        if miller_rabin_test(n):
            brute_primes.append(n)
        n += step
        step = 6 - step

    # Pattern-guided search
    candidates = []
    n = test_start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1
    step = 4 if n % 6 == 1 else 2

    while n < test_end and len(candidates) < target_primes * 20:
        coord = compute_nd_coordinates(np.array([n], dtype=np.int64), genome)[0, :2]
        x = int((coord[0] - mins[0]) / ranges[0] * (grid_size - 1))
        y = int((coord[1] - mins[1]) / ranges[1] * (grid_size - 1))
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        score = density_map[y, x]
        candidates.append((score, n, x, y))
        n += step
        step = 6 - step

    candidates.sort(key=lambda c: -c[0])

    guided_primes = []
    guided_tests = 0
    high_hits = 0

    random.seed(seed)
    for score, num, x, y in candidates:
        if len(guided_primes) >= target_primes:
            break
        guided_tests += 1
        if miller_rabin_test(num):
            guided_primes.append(num)
            if density_map[y, x] > high_threshold:
                high_hits += 1

    if guided_tests > 0 and len(guided_primes) >= target_primes:
        improvement = brute_tests / guided_tests
    else:
        improvement = 0.0

    return PrimeSearchValidation(
        genome_id=genome_id,
        brute_force_tests=brute_tests,
        pattern_guided_tests=guided_tests,
        primes_found=len(guided_primes),
        improvement_factor=improvement,
        is_useful=improvement > 1.05,
        search_range=(test_start, test_end),
        high_density_hits=high_hits,
        low_density_hits=0
    )


def validate_multi_scale(
    genome: NDimensionalGenome,
    logger=None
) -> MultiScaleValidation:
    """Validate a discovery across multiple scales from 15K to 2B.

    This is the DEFINITIVE test - a pattern must work at ALL scales to be useful.
    """
    genome_id = f"{genome.curve_type}_{genome.dimensions}D_{id(genome) % 10000:04d}"
    scale_results = {}
    improvements = []

    for scale_name, train_start, train_end, test_start, test_end, target_primes in VALIDATION_SCALES:
        try:
            if logger:
                logger.info(f"  Validating at {scale_name} scale...")

            result = validate_at_scale(
                genome=genome,
                scale_name=scale_name,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                target_primes=target_primes
            )
            scale_results[scale_name] = result
            improvements.append(result.improvement_factor)

            if logger:
                status = "PASS" if result.is_useful else "FAIL"
                logger.info(f"    {scale_name}: {result.improvement_factor:.2f}x [{status}]")

        except Exception as e:
            if logger:
                logger.warning(f"    {scale_name}: ERROR - {e}")
            # Record failure
            scale_results[scale_name] = PrimeSearchValidation(
                genome_id=genome_id,
                brute_force_tests=0,
                pattern_guided_tests=0,
                primes_found=0,
                improvement_factor=0.0,
                is_useful=False,
                search_range=(test_start, test_end),
                high_density_hits=0,
                low_density_hits=0
            )
            improvements.append(0.0)

    scales_passed = sum(1 for r in scale_results.values() if r.is_useful)
    min_improvement = min(improvements) if improvements else 0.0
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    is_robust = scales_passed == len(VALIDATION_SCALES) and min_improvement > 1.05

    return MultiScaleValidation(
        genome_id=genome_id,
        scale_results=scale_results,
        min_improvement=min_improvement,
        avg_improvement=avg_improvement,
        scales_passed=scales_passed,
        total_scales=len(VALIDATION_SCALES),
        is_robust=is_robust
    )


def validate_discovery_for_prime_search(
    genome: NDimensionalGenome,
    train_range: Tuple[int, int],
    test_range: Tuple[int, int],
    target_primes: int = 50,
    grid_size: int = 64,
    prime_set: Optional[set] = None,
    seed: int = 42
) -> PrimeSearchValidation:
    """CRITICAL TEST: Validate if a discovered pattern actually reduces primality tests.

    This is the definitive test of whether a discovery is useful. We:
    1. Build a density map from train_range using the genome's coordinates
    2. Use that density map to prioritize candidates in test_range
    3. Count primality tests needed to find target_primes
    4. Compare against brute force baseline
    5. Return improvement factor

    Args:
        genome: The discovered visualization genome
        train_range: (start, end) for building density map
        test_range: (start, end) for searching primes (must be OUTSIDE train_range)
        target_primes: How many primes to find
        grid_size: Resolution of density map
        prime_set: Pre-computed prime set (optional)
        seed: Random seed for reproducibility

    Returns:
        PrimeSearchValidation with improvement factor and statistics
    """
    random.seed(seed)

    if prime_set is None:
        prime_set = get_prime_set(max(train_range[1], test_range[1]))

    # Step 1: Build density map from training range
    train_numbers = np.arange(train_range[0], train_range[1], dtype=np.int64)
    train_primes_mask = fast_is_prime_mask(train_numbers, prime_set)
    train_coords = compute_nd_coordinates(train_numbers, genome)

    # Only use first 2 dimensions for density map
    coords_2d = train_coords[:, :2]

    # Normalize to grid
    mins = coords_2d.min(axis=0)
    maxs = coords_2d.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    # Build density map: count primes per bin
    bin_prime_count = np.zeros((grid_size, grid_size), dtype=np.float32)
    bin_total_count = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, (num, is_p) in enumerate(zip(train_numbers, train_primes_mask)):
        if num % 6 not in [1, 5]:  # Only mod-6 valid
            continue
        x = int((coords_2d[i, 0] - mins[0]) / ranges[0] * (grid_size - 1))
        y = int((coords_2d[i, 1] - mins[1]) / ranges[1] * (grid_size - 1))
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        bin_total_count[y, x] += 1
        if is_p:
            bin_prime_count[y, x] += 1

    # Compute density scores
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    valid_mask = bin_total_count >= 5
    density_map[valid_mask] = bin_prime_count[valid_mask] / bin_total_count[valid_mask]

    mean_density = density_map[valid_mask].mean() if valid_mask.any() else 0.1
    high_threshold = mean_density * 1.2
    low_threshold = mean_density * 0.8

    # Step 2: Search for primes in test range using density guidance
    test_start, test_end = test_range

    # Brute force baseline
    brute_primes = []
    brute_tests = 0
    n = test_start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1
    step = 4 if n % 6 == 1 else 2

    random.seed(seed)  # Same seed for fair comparison
    while len(brute_primes) < target_primes and n < test_end:
        brute_tests += 1
        if miller_rabin_test(n):
            brute_primes.append(n)
        n += step
        step = 6 - step

    # Pattern-guided search
    # First, score all candidates by their density bin
    candidates = []
    n = test_start
    if n % 6 == 0:
        n += 1
    elif n % 6 == 2:
        n += 3
    elif n % 6 == 3:
        n += 2
    elif n % 6 == 4:
        n += 1
    step = 4 if n % 6 == 1 else 2

    while n < test_end and len(candidates) < target_primes * 20:  # Pool of candidates
        # Get coordinates for this number
        coord = compute_nd_coordinates(np.array([n], dtype=np.int64), genome)[0, :2]
        x = int((coord[0] - mins[0]) / ranges[0] * (grid_size - 1))
        y = int((coord[1] - mins[1]) / ranges[1] * (grid_size - 1))
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        score = density_map[y, x]
        candidates.append((score, n, x, y))
        n += step
        step = 6 - step

    # Sort by density score (highest first)
    candidates.sort(key=lambda c: -c[0])

    # Search in density-prioritized order
    guided_primes = []
    guided_tests = 0
    high_hits = 0
    low_hits = 0

    random.seed(seed)  # Same seed for fair comparison
    for score, num, x, y in candidates:
        if len(guided_primes) >= target_primes:
            break
        guided_tests += 1
        if miller_rabin_test(num):
            guided_primes.append(num)
            if density_map[y, x] > high_threshold:
                high_hits += 1
            elif density_map[y, x] < low_threshold:
                low_hits += 1

    # Calculate improvement
    if guided_tests > 0 and len(guided_primes) >= target_primes:
        improvement = brute_tests / guided_tests
    else:
        improvement = 0.0  # Pattern didn't help or couldn't find enough primes

    genome_id = f"{genome.curve_type}_{genome.dimensions}D_{id(genome) % 10000:04d}"

    return PrimeSearchValidation(
        genome_id=genome_id,
        brute_force_tests=brute_tests,
        pattern_guided_tests=guided_tests,
        primes_found=len(guided_primes),
        improvement_factor=improvement,
        is_useful=improvement > 1.05,  # At least 5% improvement required
        search_range=test_range,
        high_density_hits=high_hits,
        low_density_hits=low_hits
    )


def validate_all_discoveries(
    discoveries_dir: Path,
    target_primes: int = 50,
    max_discoveries: int = 100
) -> List[PrimeSearchValidation]:
    """Validate all interesting discoveries in a run directory.

    Args:
        discoveries_dir: Path to run directory with checkpoints/
        target_primes: Primes to find per validation
        max_discoveries: Maximum discoveries to validate

    Returns:
        List of validation results sorted by improvement factor
    """
    checkpoint_dir = discoveries_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoints directory found at {checkpoint_dir}")
        return []

    validations = []
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_discovery_*.json"))[:max_discoveries]

    print(f"Validating {len(checkpoint_files)} discoveries...")

    for i, cp_file in enumerate(checkpoint_files):
        try:
            with open(cp_file) as f:
                data = json.load(f)

            genome_data = data.get('genome', {})
            genome = NDimensionalGenome(dimensions=genome_data.get('dimensions', 2))
            genome.curve_type = genome_data.get('curve_type', 0)
            genome.curve_params = genome_data.get('curve_params', {})

            # Load coefficient lists
            for attr in ['linear_coeffs', 'mod_coeffs', 'mod_bases', 'sqrt_coeffs',
                        'log_coeffs', 'sin_coeffs', 'sin_freqs', 'div_coeffs',
                        'div_bases', 'digit_sum_coeffs']:
                if attr in genome_data:
                    setattr(genome, attr, genome_data[attr])

            # Use training range 1-15000, test range 50000-100000
            # This ensures test range is completely outside training
            validation = validate_discovery_for_prime_search(
                genome=genome,
                train_range=(1, 15000),
                test_range=(50000, 100000),
                target_primes=target_primes
            )

            validations.append(validation)

            status = "USEFUL" if validation.is_useful else "not useful"
            print(f"  [{i+1}/{len(checkpoint_files)}] {cp_file.name}: "
                  f"{validation.improvement_factor:.2f}x ({status})")

        except Exception as e:
            print(f"  [{i+1}/{len(checkpoint_files)}] {cp_file.name}: ERROR - {e}")

    # Sort by improvement factor
    validations.sort(key=lambda v: -v.improvement_factor)

    return validations


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

@dataclass
class DiscoveryResult:
    """Result of evaluating a single visualization genome."""
    genome: NDimensionalGenome
    total_primes: int
    pattern_fraction: float  # Fraction of primes on known patterns
    residual_fraction: float  # Fraction remaining after pattern removal
    residual_exploitability: float  # Density-based exploitability score (FIXED)
    density_variance_ratio: float  # How much more variance than random
    extends_to_new_range: bool
    extension_correlation: float
    fitness: float
    is_interesting: bool  # True if residual shows exploitable density patterns
    interest_reason: str = ""  # Why it is/isn't interesting
    timestamp: str = ""
    eval_id: int = 0  # Unique ID for tracking/auditing

    # CRITICAL: Prime search validation results
    search_validation: Optional[PrimeSearchValidation] = None
    search_improvement: float = 0.0  # Improvement factor (>1 means useful)
    search_validated: bool = False  # Has prime search test been run?

    # Cached arrays for efficient image saving (avoid recomputation)
    _cached_grid_2d: Optional[np.ndarray] = None
    _cached_pattern_mask: Optional[np.ndarray] = None

    # Legacy alias for compatibility
    @property
    def residual_has_pattern(self) -> float:
        return self.residual_exploitability

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def evaluate_genome(
    genome: NDimensionalGenome,
    start_n: int = 1,
    end_n: int = 50000,
    grid_size: int = 200,
    prime_set: Optional[set] = None
) -> DiscoveryResult:
    """Evaluate a genome for exploitable residual patterns.

    FIXED evaluation approach:
    1. Detect and remove known mathematical patterns (polynomials, mod-N, etc.)
    2. Evaluate RESIDUAL for density concentrations/voids (not pattern detection!)
    3. Test if density patterns transfer to new number range
    4. Score based on exploitability for prime search

    A genome is "interesting" if:
    1. After removing known patterns, residual shows density variance > random
    2. High-density regions predict high-density in new ranges
    3. The density ratio (high/low bins) is exploitable for search
    """
    # Get prime set (will be cached for efficiency)
    if prime_set is None:
        # Need primes up to 2*end_n for extension testing
        prime_set = get_prime_set(end_n * 2)

    # Generate numbers and primes
    numbers = np.arange(start_n, end_n, dtype=np.int64)
    primes_mask = fast_is_prime_mask(numbers, prime_set)

    # Compute coordinates
    coords = compute_nd_coordinates(numbers, genome)

    # Render visualization
    _, prime_grid = render_nd_visualization(coords, primes_mask, grid_size)

    total_primes = primes_mask.sum()

    # Step 1: Detect and remove known patterns (enhanced detection)
    pattern_mask, pattern_fraction = detect_patterns_nd(prime_grid, genome.dimensions)

    residual_fraction = 1.0 - pattern_fraction

    # Step 2: Evaluate residual for DENSITY patterns (not line detection!)
    # NOTE: Skip transfer test during fitness computation for performance.
    # Only pass genome/prime_set/test_range when flagging interesting discoveries.
    exploitability, is_interesting, reason = compute_residual_pattern_score(
        prime_grid,
        pattern_mask,
        genome=None,  # Skip transfer test during GA
        prime_set=None,
        test_range=None,
        grid_size=grid_size // 2
    )

    # Step 3: Test if density patterns extend to new range
    extends = False
    correlation = 0.0
    test_range = (end_n, end_n + (end_n - start_n))

    if exploitability > 0.3:  # Only test if residual shows significant structure
        extends, correlation = test_pattern_extends(
            genome,
            known_range=(start_n, end_n),
            test_range=test_range,
            grid_size=grid_size // 2,
            prime_set=prime_set
        )

    # Compute density variance for reporting
    # Use > 0 to catch all primes (grid may be normalized with low values)
    residual = (prime_grid > 0) & ~pattern_mask
    if len(residual.shape) == 3:
        residual_2d = np.max(residual.astype(np.float32), axis=0)
    else:
        residual_2d = residual.astype(np.float32)

    kernel = np.ones((15, 15), dtype=np.float32) / 225
    local_density = ndimage.convolve(residual_2d, kernel)
    total_residual = residual_2d.sum()
    if total_residual > 0:
        expected_density = total_residual / residual_2d.size
        expected_var = expected_density * (1 - expected_density)
        density_variance_ratio = local_density.var() / expected_var if expected_var > 0 else 1.0
    else:
        density_variance_ratio = 1.0

    # Step 4: CRITICAL - Quick search validation to check actual utility
    # The old pattern-based fitness doesn't predict prime search improvement
    # Only validate if initial metrics look promising (to save time)
    search_improvement = 0.0
    search_validation = None

    if exploitability > 0.2 and extends:
        try:
            # Quick validation with 20 primes in a test range outside training
            search_validation = validate_discovery_for_prime_search(
                genome=genome,
                train_range=(start_n, end_n),
                test_range=(end_n * 2, end_n * 4),
                target_primes=20,
                grid_size=48,  # Small grid for speed
                prime_set=prime_set
            )
            search_improvement = search_validation.improvement_factor
        except Exception:
            pass  # Validation failed, assume no improvement

    # Step 5: Compute fitness based on ACTUAL PRIME SEARCH IMPROVEMENT
    # Not pattern detection metrics which don't correlate with utility
    if search_improvement > 1.0:
        # Discovery actually helps - weight improvement heavily
        fitness = (
            0.1 * residual_fraction +
            0.1 * exploitability +
            0.8 * min(search_improvement - 1.0, 0.5) * 2  # Scale improvement to [0, 0.8]
        )
        is_interesting = search_improvement > 1.05  # Only interesting if >5% improvement
        reason = f"Search improvement: {search_improvement:.2f}x"
    else:
        # No search improvement - use old metrics but with low weight
        fitness = (
            0.2 * residual_fraction +
            0.3 * exploitability +
            0.2 * (1.0 if extends else 0.0) +
            0.1 * max(0, correlation)
        ) * 0.5  # Cap at 0.5 since no proven utility
        is_interesting = False
        reason = f"No search improvement ({search_improvement:.2f}x)"

    genome.fitness = fitness

    # Cache 2D grid and pattern mask for efficient image saving
    if len(prime_grid.shape) == 3:
        cached_grid_2d = np.max(prime_grid, axis=0)
    else:
        cached_grid_2d = prime_grid.copy()

    # Normalize cached grid
    if cached_grid_2d.max() > 0:
        cached_grid_2d = cached_grid_2d / cached_grid_2d.max()

    # Get 2D pattern mask for caching
    cached_pattern_mask, _ = detect_patterns_2d(cached_grid_2d)

    return DiscoveryResult(
        genome=genome,
        total_primes=int(total_primes),
        pattern_fraction=pattern_fraction,
        residual_fraction=residual_fraction,
        residual_exploitability=exploitability,
        density_variance_ratio=density_variance_ratio,
        extends_to_new_range=extends,
        extension_correlation=correlation,
        fitness=fitness,
        is_interesting=is_interesting,
        interest_reason=reason,
        search_validation=search_validation,
        search_improvement=search_improvement,
        search_validated=search_validation is not None,
        _cached_grid_2d=cached_grid_2d,
        _cached_pattern_mask=cached_pattern_mask,
    )


# =============================================================================
# AUTONOMOUS DISCOVERY ENGINE
# =============================================================================

class AutonomousDiscoveryEngine:
    """Engine for continuous discovery of novel prime visualizations."""

    def __init__(
        self,
        dimensions: List[int] = [2, 3],
        population_size: int = 20,
        start_n: int = 1,
        end_n: int = 50000,
        grid_size: int = 200,
        output_dir: Optional[Path] = None,
    ):
        self.dimensions = dimensions
        self.population_size = population_size
        self.start_n = start_n
        self.end_n = end_n
        self.grid_size = grid_size

        # Create run directory
        self.run = create_run(
            "discovery",
            f"autonomous_{max(dimensions)}D",
            config={
                "dimensions": dimensions,
                "population_size": population_size,
                "start_n": start_n,
                "end_n": end_n,
                "grid_size": grid_size,
            }
        )

        # Set up file logging
        log_path = self.run.run_dir / "logs" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_file_logger(log_path)

        # Log startup info
        self.logger.info("=" * 60)
        self.logger.info("AUTONOMOUS DISCOVERY ENGINE STARTING")
        self.logger.info("=" * 60)
        self.logger.info(f"Run ID: {self.run.metadata.run_id}")
        self.logger.info(f"Dimensions: {dimensions}")
        self.logger.info(f"Population size: {population_size}")
        self.logger.info(f"Number range: {start_n} to {end_n}")
        self.logger.info(f"Grid size: {grid_size}")
        self.logger.info(f"Output: {self.run.run_dir}")

        # Check memory requirements
        for d in dimensions:
            mem_estimate = estimate_grid_memory_mb(grid_size, d)
            self.logger.info(f"  {d}D grid memory estimate: {mem_estimate:.1f} MB")

        current_mem = get_memory_usage_mb()
        if current_mem > 0:
            self.logger.info(f"Current process memory: {current_mem:.1f} MB")

        # Initialize populations for each dimension
        self.populations: Dict[int, List[NDimensionalGenome]] = {}
        for d in dimensions:
            self.populations[d] = [NDimensionalGenome(dimensions=d) for _ in range(population_size)]

        # Track ALL evaluations for auditing
        self.all_results: List[DiscoveryResult] = []  # Every evaluation
        self.discoveries: List[DiscoveryResult] = []  # Only "interesting" ones
        self.best_fitness: Dict[int, float] = {d: 0.0 for d in dimensions}
        self.cycles_completed = 0
        self.start_time = time.time()
        self.errors_count = 0
        self.last_checkpoint_cycle = 0
        self.eval_counter = 0  # Global counter for unique image filenames

        # Strategy deduplication - track what we've already tested
        self.strategy_cache: Dict[str, StrategyRecord] = {}
        self.strategies_skipped = 0
        self.strategies_evaluated = 0

        # Image-based deduplication - catch visually similar images
        # Maps image hash -> (eval_id, fitness) for lookup
        self.image_hash_cache: Dict[int, Tuple[int, float]] = {}
        self.image_similarity_threshold = 0.95  # 95% similar = duplicate (was 85%, too aggressive)
        self.image_hash_max_size = 500  # Limit cache size to allow exploration
        self.images_skipped_visual = 0

    def run_cycle(self, dimension: int) -> List[DiscoveryResult]:
        """Run one evolutionary cycle for a specific dimension."""
        global _shutdown_requested

        population = self.populations[dimension]
        results = []

        # Evaluate all genomes
        for genome in population:
            # Check for shutdown between genome evaluations
            if _shutdown_requested:
                self.logger.info(f"Shutdown requested during {dimension}D cycle evaluation")
                break

            # Check for duplicate strategy (parameter-based)
            fp = genome.fingerprint()
            if fp in self.strategy_cache:
                # Already evaluated this strategy - skip and use cached result
                cached = self.strategy_cache[fp]
                cached.eval_count += 1
                self.strategies_skipped += 1

                # Reuse the cached fitness for selection
                genome.fitness = cached.fitness

                if cached.eval_count <= 3:  # Only log first few skips
                    self.logger.debug(f"[SKIP] Strategy already tested (seen {cached.eval_count}x): "
                                    f"fitness={cached.fitness:.3f}, improvement={cached.search_improvement:.2f}x")
                continue

            # Check for visually similar image (image-based deduplication)
            # Do quick render to compute hash before expensive evaluation
            try:
                quick_prime_set = get_prime_set(self.end_n * 2)
                quick_numbers = np.arange(self.start_n, self.end_n, dtype=np.int64)
                quick_primes_mask = fast_is_prime_mask(quick_numbers, quick_prime_set)
                quick_coords = compute_nd_coordinates(quick_numbers, genome)
                _, quick_grid = render_nd_visualization(quick_coords, quick_primes_mask, self.grid_size)

                img_hash = compute_image_hash(quick_grid)

                # Check against existing hashes
                is_visual_duplicate = False
                for existing_hash, (existing_eval_id, existing_fitness) in self.image_hash_cache.items():
                    similarity = image_similarity(img_hash, existing_hash)
                    if similarity >= self.image_similarity_threshold:
                        # Found visually similar image - skip
                        is_visual_duplicate = True
                        genome.fitness = existing_fitness
                        self.images_skipped_visual += 1
                        self.logger.debug(f"[SKIP-VISUAL] Image {similarity*100:.0f}% similar to eval_{existing_eval_id:05d} "
                                        f"(fitness={existing_fitness:.3f})")
                        break

                if is_visual_duplicate:
                    continue

            except Exception as e:
                # If quick render fails, proceed with full evaluation
                self.logger.debug(f"Quick render failed, proceeding: {e}")

            try:
                result = evaluate_genome(
                    genome,
                    start_n=self.start_n,
                    end_n=self.end_n,
                    grid_size=self.grid_size
                )
                results.append(result)
                self.strategies_evaluated += 1

                # Cache this strategy's results
                self.strategy_cache[fp] = StrategyRecord(
                    fingerprint=fp,
                    fitness=result.fitness,
                    search_improvement=result.search_improvement,
                    is_interesting=result.is_interesting,
                    eval_count=1
                )

                # Track ALL results for auditing
                self.eval_counter += 1
                result.eval_id = self.eval_counter  # Add eval ID for tracking
                self.all_results.append(result)

                # Cache image hash for visual deduplication
                # Use the grid from result if available, otherwise recompute
                try:
                    if hasattr(result, 'prime_grid') and result.prime_grid is not None:
                        img_hash = compute_image_hash(result.prime_grid)
                    else:
                        # Recompute grid for hash
                        quick_prime_set = get_prime_set(self.end_n * 2)
                        quick_numbers = np.arange(self.start_n, self.end_n, dtype=np.int64)
                        quick_primes_mask = fast_is_prime_mask(quick_numbers, quick_prime_set)
                        quick_coords = compute_nd_coordinates(quick_numbers, genome)
                        _, quick_grid = render_nd_visualization(quick_coords, quick_primes_mask, self.grid_size)
                        img_hash = compute_image_hash(quick_grid)
                    # Prune cache if too large (keep recent entries, allow exploration)
                    if len(self.image_hash_cache) >= self.image_hash_max_size:
                        # Remove oldest entries (by eval_id)
                        sorted_entries = sorted(self.image_hash_cache.items(), key=lambda x: x[1][0])
                        entries_to_remove = len(self.image_hash_cache) - self.image_hash_max_size + 50
                        for hash_key, _ in sorted_entries[:entries_to_remove]:
                            del self.image_hash_cache[hash_key]
                    self.image_hash_cache[img_hash] = (self.eval_counter, result.fitness)
                except Exception as e:
                    self.logger.debug(f"Failed to cache image hash: {e}")

                # Save image for EVERY evaluation (auditable)
                self._save_evaluation_image(result, self.eval_counter)

                if result.is_interesting:
                    # CRITICAL: Validate that discovery actually reduces primality tests
                    try:
                        validation = validate_discovery_for_prime_search(
                            genome=result.genome,
                            train_range=(self.start_n, self.end_n),
                            test_range=(self.end_n * 3, self.end_n * 6),  # Test outside training
                            target_primes=30,
                            grid_size=64
                        )
                        result.search_validation = validation
                        result.search_improvement = validation.improvement_factor
                        result.search_validated = True

                        if validation.is_useful:
                            # Initial validation passed - now test at larger scales
                            self.logger.info(f"[DISCOVERY] Initial validation passed ({validation.improvement_factor:.2f}x)")
                            self.logger.info(f"  Running multi-scale validation (15K to 2B)...")

                            try:
                                multi_scale = validate_multi_scale(result.genome, logger=self.logger)

                                if multi_scale.is_robust:
                                    # Pattern works at ALL scales - this is a true discovery!
                                    self.discoveries.append(result)
                                    self._save_discovery(result, multi_scale)
                                    self.logger.info(f"[BREAKTHROUGH] MULTI-SCALE VALIDATED {dimension}D pattern!")
                                    self.logger.info(f"  Scales passed: {multi_scale.scales_passed}/{multi_scale.total_scales}")
                                    self.logger.info(f"  Min improvement: {multi_scale.min_improvement:.2f}x")
                                    self.logger.info(f"  Avg improvement: {multi_scale.avg_improvement:.2f}x")
                                else:
                                    # Pattern doesn't scale
                                    self.logger.info(f"[REJECTED] Pattern failed multi-scale validation")
                                    self.logger.info(f"  Scales passed: {multi_scale.scales_passed}/{multi_scale.total_scales}")
                                    self.logger.info(f"  Min improvement: {multi_scale.min_improvement:.2f}x (need >1.05x at all scales)")
                                    for scale_name, scale_result in multi_scale.scale_results.items():
                                        status = "PASS" if scale_result.is_useful else "FAIL"
                                        self.logger.debug(f"    {scale_name}: {scale_result.improvement_factor:.2f}x [{status}]")

                            except Exception as e:
                                self.logger.warning(f"  Multi-scale validation error: {e}")
                                # Save with initial validation only
                                self.discoveries.append(result)
                                self._save_discovery(result)
                                self.logger.info(f"  Saved with initial validation only")
                        else:
                            # Pattern looked interesting but doesn't help in practice
                            self.logger.debug(f"[REJECTED] Pattern failed search validation: "
                                            f"{validation.improvement_factor:.2f}x (need >1.05x)")
                    except Exception as e:
                        self.logger.debug(f"Validation error: {e}")
                        # Still save if interesting, mark as not validated
                        self.discoveries.append(result)
                        self._save_discovery(result)
                        self.logger.info(f"[DISCOVERY] Found interesting {dimension}D pattern! "
                                       f"Fitness: {result.fitness:.4f}, "
                                       f"Exploitability: {result.residual_exploitability:.2f}, "
                                       f"Variance: {result.density_variance_ratio:.2f}x (NOT VALIDATED)")
                        self.logger.info(f"  Reason: {result.interest_reason}")

            except MemoryError:
                self.errors_count += 1
                self.logger.error(f"MemoryError evaluating {dimension}D genome. "
                                f"Current memory: {get_memory_usage_mb():.1f} MB")
                # Skip this genome but continue
                continue
            except Exception as e:
                self.errors_count += 1
                self.logger.error(f"Error evaluating {dimension}D genome: {e}")
                self.logger.debug(traceback.format_exc())
                continue

        if not results:
            self.logger.warning(f"No successful evaluations in {dimension}D cycle")
            return results

        # Sort by fitness
        results.sort(key=lambda r: r.fitness, reverse=True)

        # Update best fitness
        if results and results[0].fitness > self.best_fitness[dimension]:
            self.best_fitness[dimension] = results[0].fitness
            self.logger.debug(f"New best {dimension}D fitness: {results[0].fitness:.4f}")

        # Evolution strategy with better diversity:
        # - Keep only top 25% as elite (unchanged)
        # - Mutate next 25% of survivors
        # - Generate 30% children from crossover
        # - Add 20% completely fresh random individuals

        n_elite = max(1, len(results) // 4)  # Top 25%
        n_mutated_survivors = max(1, len(results) // 4)  # Next 25%
        n_fresh = max(2, self.population_size // 5)  # 20% fresh blood
        n_children = self.population_size - n_elite - n_mutated_survivors - n_fresh

        new_population = []

        # 1. Elite (unchanged) - only top 25%
        elite = [r.genome for r in results[:n_elite]]
        new_population.extend(elite)

        # 2. Mutated survivors (next 25%, with aggressive mutation)
        for r in results[n_elite:n_elite + n_mutated_survivors]:
            mutant = NDimensionalGenome(dimensions=dimension)
            # Copy genome - curve type, params, and coefficients
            mutant.curve_type = r.genome.curve_type
            mutant.curve_params = r.genome.curve_params.copy() if r.genome.curve_params else {}
            mutant.linear_coeffs = r.genome.linear_coeffs.copy()
            mutant.mod_coeffs = r.genome.mod_coeffs.copy()
            mutant.mod_bases = r.genome.mod_bases.copy()
            mutant.sqrt_coeffs = r.genome.sqrt_coeffs.copy()
            mutant.log_coeffs = r.genome.log_coeffs.copy()
            mutant.sin_coeffs = r.genome.sin_coeffs.copy()
            mutant.sin_freqs = r.genome.sin_freqs.copy()
            mutant.div_coeffs = r.genome.div_coeffs.copy()
            mutant.div_bases = r.genome.div_bases.copy()
            mutant.digit_sum_coeffs = r.genome.digit_sum_coeffs.copy()
            mutant.prime_count_coeffs = r.genome.prime_count_coeffs.copy()
            mutant.totient_coeffs = r.genome.totient_coeffs.copy()
            mutant.divisor_count_coeffs = r.genome.divisor_count_coeffs.copy()
            mutant.divisor_sum_coeffs = r.genome.divisor_sum_coeffs.copy()
            mutant.omega_coeffs = r.genome.omega_coeffs.copy()
            mutant.collatz_coeffs = r.genome.collatz_coeffs.copy()
            mutant.largest_pf_coeffs = r.genome.largest_pf_coeffs.copy()
            mutant.smallest_pf_coeffs = r.genome.smallest_pf_coeffs.copy()
            mutant.mobius_coeffs = r.genome.mobius_coeffs.copy()
            mutant.interactions = [row.copy() for row in r.genome.interactions]
            mutant.mutate(mutation_rate=0.5, aggressive=True)  # Aggressive mutation
            mutant.generation = self.cycles_completed + 1
            new_population.append(mutant)

        # 3. Children from crossover
        all_parents = [r.genome for r in results[:len(results)//2]]
        for _ in range(n_children):
            if len(all_parents) >= 2:
                parent1, parent2 = random.sample(all_parents, 2)
                child = parent1.crossover(parent2)
            else:
                child = NDimensionalGenome(dimensions=dimension)
            child.mutate(mutation_rate=0.4)  # Standard mutation
            child.generation = self.cycles_completed + 1
            new_population.append(child)

        # 4. Fresh random individuals (critical for exploration!)
        for _ in range(n_fresh):
            fresh = NDimensionalGenome(dimensions=dimension)
            fresh.generation = self.cycles_completed + 1
            new_population.append(fresh)

        self.populations[dimension] = new_population

        return results

    def _compute_diversity(self, dimension: int) -> float:
        """Compute population diversity metric.

        Returns average pairwise distance between genomes (0-1 scale).
        Higher values = more diverse population.
        """
        population = self.populations[dimension]
        if len(population) < 2:
            return 0.0

        # Compare mod_bases (key differentiator)
        distances = []
        for i, g1 in enumerate(population):
            for g2 in population[i+1:]:
                # Count differences in mod_bases
                diff_bases = sum(1 for a, b in zip(g1.mod_bases, g2.mod_bases) if a != b)
                # Add coefficient differences
                diff_coeffs = sum(abs(a - b) for a, b in zip(g1.mod_coeffs, g2.mod_coeffs))
                diff_coeffs += sum(abs(a - b) for a, b in zip(g1.sqrt_coeffs, g2.sqrt_coeffs))
                # Normalize
                distances.append(diff_bases / len(g1.mod_bases) + min(1.0, diff_coeffs / 20.0))

        return sum(distances) / len(distances) / 2.0 if distances else 0.0

    def _save_discovery(self, result: DiscoveryResult, multi_scale: Optional[MultiScaleValidation] = None):
        """Save an interesting discovery."""
        discovery_id = len(self.discoveries)

        # Save genome
        genome_data = {
            "dimensions": result.genome.dimensions,
            "curve_type": result.genome.curve_type,
            "curve_params": result.genome.curve_params,
            "linear_coeffs": result.genome.linear_coeffs,
            "mod_coeffs": result.genome.mod_coeffs,
            "mod_bases": result.genome.mod_bases,
            "sqrt_coeffs": result.genome.sqrt_coeffs,
            "log_coeffs": result.genome.log_coeffs,
            "sin_coeffs": result.genome.sin_coeffs,
            "sin_freqs": result.genome.sin_freqs,
            "div_coeffs": result.genome.div_coeffs,
            "div_bases": result.genome.div_bases,
            "digit_sum_coeffs": result.genome.digit_sum_coeffs,
            "prime_count_coeffs": result.genome.prime_count_coeffs,
            "interactions": result.genome.interactions,
            "fitness": result.fitness,
        }

        # Include search validation results
        validation_data = None
        if result.search_validated and result.search_validation:
            v = result.search_validation
            validation_data = {
                "brute_force_tests": v.brute_force_tests,
                "pattern_guided_tests": v.pattern_guided_tests,
                "primes_found": v.primes_found,
                "improvement_factor": v.improvement_factor,
                "is_useful": v.is_useful,
                "search_range": list(v.search_range),
                "high_density_hits": v.high_density_hits,
                "low_density_hits": v.low_density_hits,
            }

        # Include multi-scale validation results
        multi_scale_data = None
        if multi_scale:
            multi_scale_data = {
                "is_robust": multi_scale.is_robust,
                "scales_passed": multi_scale.scales_passed,
                "total_scales": multi_scale.total_scales,
                "min_improvement": multi_scale.min_improvement,
                "avg_improvement": multi_scale.avg_improvement,
                "scale_results": {}
            }
            for scale_name, scale_result in multi_scale.scale_results.items():
                multi_scale_data["scale_results"][scale_name] = {
                    "improvement_factor": scale_result.improvement_factor,
                    "is_useful": scale_result.is_useful,
                    "brute_force_tests": scale_result.brute_force_tests,
                    "pattern_guided_tests": scale_result.pattern_guided_tests,
                    "search_range": list(scale_result.search_range),
                }

        result_data = {
            "discovery_id": discovery_id,
            "total_primes": result.total_primes,
            "pattern_fraction": result.pattern_fraction,
            "residual_fraction": result.residual_fraction,
            "residual_exploitability": result.residual_exploitability,
            "density_variance_ratio": result.density_variance_ratio,
            "extends_to_new_range": result.extends_to_new_range,
            "extension_correlation": result.extension_correlation,
            "fitness": result.fitness,
            "interest_reason": result.interest_reason,
            "timestamp": result.timestamp,
            "genome": genome_data,
            # CRITICAL: Search validation
            "search_validated": result.search_validated,
            "search_improvement": result.search_improvement,
            "search_validation": validation_data,
            # Multi-scale validation (15K to 2B)
            "multi_scale_validated": multi_scale is not None,
            "multi_scale_robust": multi_scale.is_robust if multi_scale else False,
            "multi_scale_validation": multi_scale_data,
        }

        # Save to checkpoints
        self.run.save_checkpoint(result_data, f"discovery_{discovery_id:04d}")

        # Generate and save visualization
        self._save_discovery_images(result, discovery_id)

    def _save_discovery_images(self, result: DiscoveryResult, discovery_id: int):
        """Generate and save images for a discovery."""
        genome = result.genome

        # Use cached prime set
        prime_set = get_prime_set(self.end_n)

        numbers = np.arange(self.start_n, self.end_n, dtype=np.int64)
        primes_mask = fast_is_prime_mask(numbers, prime_set)
        coords = compute_nd_coordinates(numbers, genome)
        _, prime_grid = render_nd_visualization(coords, primes_mask, self.grid_size)

        if genome.dimensions == 2:
            # Save 2D image directly
            img = (prime_grid * 255).astype(np.uint8)
            self.run.save_image(img, f"discovery_{discovery_id:04d}_{genome.dimensions}D")
        else:
            # Save max projection for 3D+
            projection = render_2d_projection(prime_grid)
            img = (projection / projection.max() * 255).astype(np.uint8) if projection.max() > 0 else projection.astype(np.uint8)
            self.run.save_image(img, f"discovery_{discovery_id:04d}_{genome.dimensions}D_projection")

    def _save_evaluation_image(self, result: DiscoveryResult, eval_id: int):
        """Save images for EVERY evaluation (auditable).

        Creates a 3-panel comparison image showing:
        1. ORIGINAL - All primes (white on black)
        2. KNOWN PATTERNS - Primes on detected patterns (red), others (white)
        3. RESIDUAL - Only primes NOT on known patterns (white on black)

        This provides visual evidence of what the algorithm detected and removed.
        Uses cached arrays from evaluate_genome to avoid expensive recomputation.
        """
        genome = result.genome

        try:
            from PIL import Image, ImageDraw, ImageFont

            # Use cached arrays if available (much faster)
            if result._cached_grid_2d is not None and result._cached_pattern_mask is not None:
                grid_2d = result._cached_grid_2d
                pattern_mask = result._cached_pattern_mask
            else:
                # Fallback: recompute (slower, for backwards compatibility)
                prime_set = get_prime_set(self.end_n)
                numbers = np.arange(self.start_n, self.end_n, dtype=np.int64)
                primes_mask = fast_is_prime_mask(numbers, prime_set)
                coords = compute_nd_coordinates(numbers, genome)
                _, prime_grid = render_nd_visualization(coords, primes_mask, self.grid_size)

                if genome.dimensions == 2:
                    grid_2d = prime_grid
                else:
                    grid_2d = render_2d_projection(prime_grid)

                if grid_2d.max() > 0:
                    grid_2d = grid_2d / grid_2d.max()

                pattern_mask, _ = detect_patterns_2d(grid_2d)

            # Create the three panels
            h, w = grid_2d.shape

            # Panel 1: Original (all primes white)
            original = (grid_2d * 255).astype(np.uint8)

            # Panel 2: Known patterns highlighted
            # Red channel = primes on patterns, Green/Blue = primes not on patterns
            # Use > 0 to catch all primes (grid may be normalized with low values)
            binary = grid_2d > 0
            known_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            known_rgb[pattern_mask, 0] = 255  # Red for detected patterns
            known_rgb[binary & ~pattern_mask, :] = 255  # White for residual

            # Panel 3: Residual only (primes NOT on patterns)
            residual = np.zeros((h, w), dtype=np.uint8)
            residual[binary & ~pattern_mask] = 255

            # Combine into comparison image
            # Add labels
            label_height = 20
            panel_width = w
            total_width = panel_width * 3 + 4  # 2px gaps
            total_height = h + label_height

            comparison = Image.new('RGB', (total_width, total_height), color=(30, 30, 30))
            draw = ImageDraw.Draw(comparison)

            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

            # Paste panels
            img_original = Image.fromarray(original).convert('RGB')
            img_known = Image.fromarray(known_rgb)
            img_residual = Image.fromarray(residual).convert('RGB')

            comparison.paste(img_original, (0, label_height))
            comparison.paste(img_known, (panel_width + 2, label_height))
            comparison.paste(img_residual, (panel_width * 2 + 4, label_height))

            # Add labels
            draw.text((panel_width // 2 - 30, 2), "ORIGINAL", fill=(200, 200, 200), font=font)
            draw.text((panel_width + panel_width // 2 - 50, 2), "KNOWN PATTERNS", fill=(255, 100, 100), font=font)
            draw.text((panel_width * 2 + panel_width // 2 - 30, 2), "RESIDUAL", fill=(100, 255, 100), font=font)

            # Create filename with embedded metrics
            extends_char = "Y" if result.extends_to_new_range else "N"
            interesting_char = "I" if result.is_interesting else "X"
            filename = (f"eval_{eval_id:05d}_{genome.dimensions}D_"
                       f"f{result.fitness:.3f}_"
                       f"p{result.pattern_fraction:.2f}_"
                       f"x{result.residual_exploitability:.2f}_"  # x=exploitability
                       f"v{result.density_variance_ratio:.1f}_"   # v=variance ratio
                       f"e{extends_char}_{interesting_char}")

            # Save comparison image
            comparison_path = self.run.run_dir / "images" / f"{filename}.png"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.save(comparison_path)

        except Exception as e:
            self.logger.error(f"Failed to save evaluation image {eval_id}: {e}")
            self.logger.debug(traceback.format_exc())

    def _generate_mosaic(self):
        """Generate mosaic of ALL evaluations sorted by fitness (most to least interesting).

        Creates a visual audit trail showing what the algorithm explored and why
        it ranked things the way it did.

        FAST VERSION: Reads saved PNG files instead of regenerating visualizations.
        """
        if not self.all_results:
            self.logger.warning("No results to create mosaic from")
            return

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self.logger.error("PIL not available for mosaic generation")
            return

        self.logger.info(f"Generating mosaic of {len(self.all_results)} evaluations (from saved images)...")

        # Sort by fitness (highest first)
        sorted_results = sorted(self.all_results, key=lambda r: r.fitness, reverse=True)

        # Determine grid layout
        n_images = len(sorted_results)
        cols = min(10, n_images)
        rows = (n_images + cols - 1) // cols

        # Thumbnail size
        thumb_size = 150
        label_height = 40
        cell_height = thumb_size + label_height

        # Create mosaic image
        mosaic_width = cols * thumb_size
        mosaic_height = rows * cell_height
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color=(40, 40, 40))
        draw = ImageDraw.Draw(mosaic)

        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        images_dir = self.run.run_dir / "images"

        for idx, result in enumerate(sorted_results):
            row = idx // cols
            col = idx % cols
            x = col * thumb_size
            y = row * cell_height

            try:
                # Find the saved image file for this evaluation
                extends_char = "Y" if result.extends_to_new_range else "N"
                interesting_char = "I" if result.is_interesting else "X"
                filename_pattern = f"eval_{result.eval_id:05d}_{result.genome.dimensions}D_"

                # Find matching file
                matching_files = list(images_dir.glob(f"{filename_pattern}*.png"))

                if matching_files:
                    # Load saved image (FAST)
                    img = Image.open(matching_files[0]).convert('RGB')
                    img = img.resize((thumb_size, thumb_size), Image.Resampling.NEAREST)
                else:
                    # Fallback: create placeholder if image not found
                    img = Image.new('RGB', (thumb_size, thumb_size), color=(60, 60, 60))
                    draw_temp = ImageDraw.Draw(img)
                    draw_temp.text((10, 70), f"No image\n#{result.eval_id}", fill=(150, 150, 150), font=font)

                # Add colored border based on interestingness
                if result.is_interesting:
                    border_color = (0, 255, 0)  # Green for interesting
                elif result.extends_to_new_range:
                    border_color = (255, 255, 0)  # Yellow for extends but not interesting
                else:
                    border_color = (100, 100, 100)  # Gray for not interesting

                # Draw border (3 pixels thick)
                for i in range(3):
                    for j in range(thumb_size):
                        img.putpixel((i, j), border_color)
                        img.putpixel((thumb_size-1-i, j), border_color)
                        img.putpixel((j, i), border_color)
                        img.putpixel((j, thumb_size-1-i), border_color)

                mosaic.paste(img, (x, y))

                # Add label
                extends_str = "Y" if result.extends_to_new_range else "N"
                label = f"#{idx+1} {result.genome.dimensions}D f={result.fitness:.2f}"
                label2 = f"p={result.pattern_fraction:.0%} x={result.residual_exploitability:.2f} v={result.density_variance_ratio:.1f}x"

                # Color code the text
                text_color = (0, 255, 0) if result.is_interesting else (200, 200, 200)
                draw.text((x + 2, y + thumb_size + 2), label, fill=text_color, font=font)
                draw.text((x + 2, y + thumb_size + 14), label2, fill=(150, 150, 150), font=font)

            except Exception as e:
                # Draw error placeholder
                draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill=(80, 0, 0))
                draw.text((x + 2, y + thumb_size + 2), f"Error: {str(e)[:20]}", fill=(255, 0, 0), font=font)

        # Save mosaic
        mosaic_path = self.run.run_dir / "images" / "mosaic_all_sorted.png"
        mosaic_path.parent.mkdir(parents=True, exist_ok=True)
        mosaic.save(mosaic_path)
        self.logger.info(f"Saved mosaic to {mosaic_path}")

        # Also save a legend/summary
        legend_path = self.run.run_dir / "images" / "mosaic_legend.txt"
        with open(legend_path, 'w') as f:
            f.write("MOSAIC LEGEND - All Evaluations Sorted by Fitness\n")
            f.write("=" * 70 + "\n\n")
            f.write("Border colors:\n")
            f.write("  GREEN  = Interesting (exploitable density patterns that transfer)\n")
            f.write("  YELLOW = Extends to new range but low exploitability\n")
            f.write("  GRAY   = Not interesting\n\n")
            f.write("Metrics (FIXED - now measures density, not pattern detection):\n")
            f.write("  f = fitness (0-1, higher is better)\n")
            f.write("  p = pattern fraction (% of primes on KNOWN mathematical patterns)\n")
            f.write("  x = exploitability (density-based score for prime search)\n")
            f.write("  v = variance ratio (actual/expected - >2 means concentrations exist)\n")
            f.write("  e = extends (Y/N - density patterns predict in new number range)\n\n")
            f.write("What makes a discovery interesting:\n")
            f.write("  - Variance ratio > 1.5 (more structure than random)\n")
            f.write("  - High/low density ratio > 1.3 (exploitable difference)\n")
            f.write("  - Transfer correlation > 0.3 (pattern predicts in new range)\n\n")
            f.write("Top 20 evaluations:\n")
            f.write("-" * 70 + "\n")
            for i, r in enumerate(sorted_results[:20]):
                f.write(f"{i+1:3d}. {r.genome.dimensions}D f={r.fitness:.4f} "
                       f"p={r.pattern_fraction:.0%} x={r.residual_exploitability:.2f} "
                       f"v={r.density_variance_ratio:.1f}x e={r.extends_to_new_range}\n")
                if r.interest_reason:
                    f.write(f"     Reason: {r.interest_reason}\n")

        self.logger.info(f"Saved legend to {legend_path}")

    def run_for_duration(self, hours: float = 1.0, report_interval: int = 10):
        """Run discovery for specified duration."""
        global _shutdown_requested

        end_time = time.time() + hours * 3600

        self.logger.info("-" * 60)
        self.logger.info(f"Running for {hours} hours (until {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')})")
        self.logger.info("-" * 60)

        # Pre-generate prime sieve (this is the slow step, do it once)
        self.logger.info("Initializing prime sieve...")
        try:
            _ = get_prime_set(self.end_n * 2)
            self.logger.info("Prime sieve ready. Starting evolution cycles...")
        except Exception as e:
            self.logger.error(f"Failed to initialize prime sieve: {e}")
            self.logger.error(traceback.format_exc())
            self._save_final_summary(status="failed", error=str(e))
            return

        self.logger.info("-" * 60)

        try:
            while time.time() < end_time and not _shutdown_requested:
                # Cycle through dimensions
                for dim in self.dimensions:
                    if _shutdown_requested:
                        break

                    try:
                        results = self.run_cycle(dim)
                        self.cycles_completed += 1
                    except Exception as e:
                        self.errors_count += 1
                        self.logger.error(f"Error in {dim}D cycle {self.cycles_completed}: {e}")
                        self.logger.error(traceback.format_exc())
                        # Save checkpoint on error
                        self._save_progress_checkpoint()
                        continue

                    # Report progress
                    if self.cycles_completed % report_interval == 0:
                        elapsed = time.time() - self.start_time
                        remaining = end_time - time.time()
                        mem_usage = get_memory_usage_mb()

                        # Compute diversity metrics
                        diversity = self._compute_diversity(dim)

                        self.logger.info(f"Cycle {self.cycles_completed} | "
                              f"Elapsed: {elapsed/3600:.2f}h | "
                              f"Remaining: {remaining/3600:.2f}h | "
                              f"Memory: {mem_usage:.0f}MB")
                        self.logger.info(f"  {dim}D: best={self.best_fitness[dim]:.4f} | "
                              f"diversity={diversity:.3f}")
                        self.logger.info(f"  Discoveries: {len(self.discoveries)} | Errors: {self.errors_count} | "
                              f"Strategies: {self.strategies_evaluated} eval, {self.strategies_skipped} param-skip, {self.images_skipped_visual} visual-skip")

                        if results:
                            best = results[0]
                            self.logger.debug(f"  Current best: pattern={best.pattern_fraction:.1%}, "
                                  f"residual_pattern={best.residual_has_pattern:.1%}, "
                                  f"extends={best.extends_to_new_range}")

                    # Save checkpoint periodically
                    if self.cycles_completed - self.last_checkpoint_cycle >= 50:
                        self._save_progress_checkpoint()
                        self.last_checkpoint_cycle = self.cycles_completed

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            _shutdown_requested = True
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            self.logger.error(traceback.format_exc())
            self._save_final_summary(status="crashed", error=str(e))
            raise

        # Final summary
        if _shutdown_requested:
            self._save_final_summary(status="interrupted")
        else:
            self._save_final_summary(status="completed")

    def run_for_cycles(self, num_cycles: int, report_interval: int = 10):
        """Run discovery for specified number of cycles."""
        global _shutdown_requested

        total_cycles = num_cycles * len(self.dimensions)

        self.logger.info("-" * 60)
        self.logger.info(f"Running for {num_cycles} cycles ({total_cycles} total with {len(self.dimensions)} dimensions)")
        self.logger.info("-" * 60)

        # Pre-generate prime sieve (this is the slow step, do it once)
        self.logger.info("Initializing prime sieve...")
        try:
            _ = get_prime_set(self.end_n * 2)
            self.logger.info("Prime sieve ready. Starting evolution cycles...")
        except Exception as e:
            self.logger.error(f"Failed to initialize prime sieve: {e}")
            self.logger.error(traceback.format_exc())
            self._save_final_summary(status="failed", error=str(e))
            return

        self.logger.info("-" * 60)

        try:
            for cycle in range(num_cycles):
                if _shutdown_requested:
                    break

                for dim in self.dimensions:
                    if _shutdown_requested:
                        break

                    try:
                        results = self.run_cycle(dim)
                        self.cycles_completed += 1
                    except Exception as e:
                        self.errors_count += 1
                        self.logger.error(f"Error in {dim}D cycle {self.cycles_completed}: {e}")
                        self.logger.error(traceback.format_exc())
                        self._save_progress_checkpoint()
                        continue

                    if self.cycles_completed % report_interval == 0:
                        elapsed = time.time() - self.start_time
                        mem_usage = get_memory_usage_mb()

                        # Compute diversity metrics
                        diversity = self._compute_diversity(dim)

                        self.logger.info(f"Cycle {self.cycles_completed}/{total_cycles} | "
                              f"Elapsed: {elapsed/60:.1f}min | Memory: {mem_usage:.0f}MB")
                        self.logger.info(f"  {dim}D: best={self.best_fitness[dim]:.4f} | "
                              f"diversity={diversity:.3f}")
                        self.logger.info(f"  Discoveries: {len(self.discoveries)} | Errors: {self.errors_count} | "
                              f"Strategies: {self.strategies_evaluated} eval, {self.strategies_skipped} param-skip, {self.images_skipped_visual} visual-skip")

                    # Save checkpoint periodically
                    if self.cycles_completed - self.last_checkpoint_cycle >= 50:
                        self._save_progress_checkpoint()
                        self.last_checkpoint_cycle = self.cycles_completed

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            _shutdown_requested = True
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            self.logger.error(traceback.format_exc())
            self._save_final_summary(status="crashed", error=str(e))
            raise

        # Final summary
        if _shutdown_requested:
            self._save_final_summary(status="interrupted")
        else:
            self._save_final_summary(status="completed")

    def _save_progress_checkpoint(self):
        """Save progress checkpoint."""
        checkpoint = {
            "cycles_completed": self.cycles_completed,
            "discoveries_count": len(self.discoveries),
            "best_fitness": self.best_fitness,
            "errors_count": self.errors_count,
            "elapsed_hours": (time.time() - self.start_time) / 3600,
            "memory_mb": get_memory_usage_mb(),
        }
        self.run.save_checkpoint(checkpoint, f"progress_{self.cycles_completed:06d}")
        self.logger.debug(f"Saved progress checkpoint at cycle {self.cycles_completed}")

    def _save_final_summary(self, status: str = "completed", error: Optional[str] = None):
        """Save final summary of discovery run."""
        elapsed = time.time() - self.start_time

        summary = {
            "status": status,
            "total_cycles": self.cycles_completed,
            "total_evaluations": len(self.all_results),  # ALL evaluations for auditing
            "total_discoveries": len(self.discoveries),
            "total_errors": self.errors_count,
            "best_fitness_per_dimension": self.best_fitness,
            "elapsed_hours": elapsed / 3600,
            "cycles_per_hour": self.cycles_completed / (elapsed / 3600) if elapsed > 0 else 0,
            "final_memory_mb": get_memory_usage_mb(),
            # Strategy deduplication stats
            "strategies_evaluated": self.strategies_evaluated,
            "strategies_skipped": self.strategies_skipped,
            "unique_strategies": len(self.strategy_cache),
        }

        if error:
            summary["error"] = error

        # Save strategy cache summary (top strategies by fitness)
        top_strategies = sorted(
            self.strategy_cache.values(),
            key=lambda s: s.fitness,
            reverse=True
        )[:100]  # Top 100

        strategy_summary = []
        for s in top_strategies:
            strategy_summary.append({
                "fingerprint": s.fingerprint[:100] + "..." if len(s.fingerprint) > 100 else s.fingerprint,
                "fitness": s.fitness,
                "search_improvement": s.search_improvement,
                "is_interesting": s.is_interesting,
                "eval_count": s.eval_count,
            })

        self.run.save_checkpoint({"strategies": strategy_summary}, "strategy_cache_summary")

        # List interesting discoveries
        interesting = [d for d in self.discoveries if d.is_interesting]
        summary["interesting_discoveries"] = len(interesting)

        if interesting:
            summary["best_discoveries"] = [
                {
                    "dimensions": d.genome.dimensions,
                    "fitness": d.fitness,
                    "residual_pattern": d.residual_has_pattern,
                    "extends": d.extends_to_new_range,
                }
                for d in sorted(interesting, key=lambda x: x.fitness, reverse=True)[:10]
            ]

        # Generate mosaic of ALL evaluations for visual audit
        self.logger.info("Generating audit mosaic...")
        try:
            self._generate_mosaic()
        except Exception as e:
            self.logger.error(f"Failed to generate mosaic: {e}")
            self.logger.error(traceback.format_exc())

        self.run.save_results(summary, summary=summary)
        self.run.complete(status=status, summary=summary)

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"DISCOVERY RUN {status.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Total cycles: {self.cycles_completed}")
        self.logger.info(f"Total evaluations: {len(self.all_results)}")
        self.logger.info(f"Interesting discoveries: {len(interesting)}")
        self.logger.info(f"Total errors: {self.errors_count}")
        self.logger.info(f"Elapsed time: {elapsed/3600:.2f} hours")
        self.logger.info(f"Output directory: {self.run.run_dir}")
        self.logger.info(f"Mosaic: {self.run.run_dir / 'images' / 'mosaic_all_sorted.png'}")
        if error:
            self.logger.error(f"Error: {error}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Prime Visualization Discovery Engine"
    )

    parser.add_argument(
        "--hours", type=float, default=None,
        help="Run for this many hours"
    )
    parser.add_argument(
        "--cycles", type=int, default=100,
        help="Run for this many cycles (if --hours not specified)"
    )
    parser.add_argument(
        "--dimensions", type=str, default="2,3",
        help="Comma-separated list of dimensions to explore (e.g., '2,3,4')"
    )
    parser.add_argument(
        "--population", type=int, default=20,
        help="Population size for genetic algorithm"
    )
    parser.add_argument(
        "--start-n", type=int, default=1,
        help="Start of number range"
    )
    parser.add_argument(
        "--end-n", type=int, default=50000,
        help="End of number range"
    )
    parser.add_argument(
        "--grid-size", type=int, default=200,
        help="Grid size for visualization rendering"
    )
    parser.add_argument(
        "--report-interval", type=int, default=10,
        help="Report progress every N cycles"
    )

    args = parser.parse_args()

    dimensions = [int(d.strip()) for d in args.dimensions.split(",")]

    # Warn about memory for high-dimension grids
    for d in dimensions:
        mem_estimate = estimate_grid_memory_mb(args.grid_size, d)
        if mem_estimate > 500:
            print(f"WARNING: {d}D grid with size {args.grid_size} will use ~{mem_estimate:.0f}MB per grid")
            print("Consider reducing --grid-size or removing higher dimensions")

    engine = None
    try:
        engine = AutonomousDiscoveryEngine(
            dimensions=dimensions,
            population_size=args.population,
            start_n=args.start_n,
            end_n=args.end_n,
            grid_size=args.grid_size,
        )

        if args.hours is not None:
            engine.run_for_duration(hours=args.hours, report_interval=args.report_interval)
        else:
            engine.run_for_cycles(num_cycles=args.cycles, report_interval=args.report_interval)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print(traceback.format_exc())

        # Try to save what we have
        if engine is not None:
            try:
                engine._save_progress_checkpoint()
                engine._save_final_summary(status="crashed", error=str(e))
            except Exception:
                pass  # Can't save, just exit

        sys.exit(1)


if __name__ == "__main__":
    main()
