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
# N-DIMENSIONAL COORDINATE GENERATION
# =============================================================================

@dataclass
class NDimensionalGenome:
    """Genome encoding for N-dimensional visualization coordinates.

    Each dimension has its own set of parameters controlling how
    integer n maps to that dimension's coordinate.
    """
    dimensions: int = 2

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

        self.linear_coeffs = [random.uniform(-0.001, 0.001) for _ in range(d)]
        self.mod_coeffs = [random.uniform(0, 10) for _ in range(d)]
        self.mod_bases = [random.choice([2, 3, 5, 6, 7, 11, 13, 17, 19, 23, 29, 30, 37, 41]) for _ in range(d)]
        self.sqrt_coeffs = [random.uniform(0, 2) for _ in range(d)]
        self.log_coeffs = [random.uniform(0, 5) for _ in range(d)]
        self.sin_coeffs = [random.uniform(0, 3) for _ in range(d)]
        self.sin_freqs = [random.uniform(0.001, 0.1) for _ in range(d)]
        self.div_coeffs = [random.uniform(0, 5) for _ in range(d)]
        self.div_bases = [random.randint(10, 1000) for _ in range(d)]
        self.digit_sum_coeffs = [random.uniform(0, 2) for _ in range(d)]
        self.prime_count_coeffs = [random.uniform(0, 0.5) for _ in range(d)]

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
                # Occasionally do large jumps for exploration
                if random.random() < 0.1:  # 10% chance of big jump
                    return random.gauss(0, scale * 2.0 * scale_multiplier)
                return val + random.gauss(0, scale * 0.5 * scale_multiplier)
            return val

        def mutate_int(val, choices):
            if random.random() < mutation_rate:
                return random.choice(choices)
            return val

        d = self.dimensions

        self.linear_coeffs = [mutate_float(v, 0.001) for v in self.linear_coeffs]
        self.mod_coeffs = [mutate_float(v, 3) for v in self.mod_coeffs]
        self.mod_bases = [mutate_int(v, [2,3,5,6,7,11,13,17,19,23,29,30,37,41,43,47,53,59,61,67]) for v in self.mod_bases]
        self.sqrt_coeffs = [mutate_float(v, 0.8) for v in self.sqrt_coeffs]
        self.log_coeffs = [mutate_float(v, 2) for v in self.log_coeffs]
        self.sin_coeffs = [mutate_float(v, 1.0) for v in self.sin_coeffs]
        self.sin_freqs = [mutate_float(v, 0.05) for v in self.sin_freqs]
        self.div_coeffs = [mutate_float(v, 2) for v in self.div_coeffs]
        self.div_bases = [mutate_int(v, list(range(10, 1001, 10))) for v in self.div_bases]
        self.digit_sum_coeffs = [mutate_float(v, 1.0) for v in self.digit_sum_coeffs]
        self.prime_count_coeffs = [mutate_float(v, 0.2) for v in self.prime_count_coeffs]

        for i in range(d):
            for j in range(d):
                self.interactions[i][j] = mutate_float(self.interactions[i][j], 0.1)

    def crossover(self, other: 'NDimensionalGenome') -> 'NDimensionalGenome':
        """Create child genome by crossing with another genome."""
        child = NDimensionalGenome(dimensions=self.dimensions)

        for attr in ['linear_coeffs', 'mod_coeffs', 'mod_bases', 'sqrt_coeffs',
                     'log_coeffs', 'sin_coeffs', 'sin_freqs', 'div_coeffs',
                     'div_bases', 'digit_sum_coeffs', 'prime_count_coeffs']:
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


def digit_sum(n: int) -> int:
    """Compute digit sum of integer."""
    return sum(int(d) for d in str(abs(n)))


def approx_prime_count(n: int) -> float:
    """Approximate pi(n) using n/ln(n)."""
    if n < 2:
        return 0
    return n / np.log(n)


def compute_nd_coordinates(
    numbers: np.ndarray,
    genome: NDimensionalGenome
) -> np.ndarray:
    """Compute N-dimensional coordinates for array of integers.

    Args:
        numbers: Array of integers to map
        genome: Coordinate mapping parameters

    Returns:
        Array of shape (len(numbers), genome.dimensions)
    """
    n = len(numbers)
    d = genome.dimensions
    coords = np.zeros((n, d), dtype=np.float64)

    for i, num in enumerate(numbers):
        if num <= 0:
            continue

        # Compute base coordinates for each dimension
        base_coords = np.zeros(d)

        for dim in range(d):
            val = 0.0

            # Linear
            val += genome.linear_coeffs[dim] * num

            # Modular
            val += genome.mod_coeffs[dim] * (num % genome.mod_bases[dim])

            # Sqrt
            val += genome.sqrt_coeffs[dim] * np.sqrt(num)

            # Log
            val += genome.log_coeffs[dim] * np.log(num)

            # Sinusoidal
            val += genome.sin_coeffs[dim] * np.sin(genome.sin_freqs[dim] * num)

            # Division
            val += genome.div_coeffs[dim] * (num // genome.div_bases[dim])

            # Digit sum
            val += genome.digit_sum_coeffs[dim] * digit_sum(num)

            # Prime count approximation
            val += genome.prime_count_coeffs[dim] * approx_prime_count(num)

            base_coords[dim] = val

        # Apply cross-dimensional interactions
        for dim in range(d):
            for other_dim in range(d):
                if dim != other_dim:
                    coords[i, dim] += genome.interactions[dim][other_dim] * base_coords[other_dim]
            coords[i, dim] += base_coords[dim]

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
    binary = (prime_grid > 0.5).astype(np.float32)
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
    binary = (prime_grid > 0.5).astype(np.float32)
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
    residual = (prime_grid > 0.5) & ~pattern_mask

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
    residual = (prime_grid > 0.5) & ~pattern_mask
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

    # Step 4: Compute fitness based on EXPLOITABILITY (not pattern detection)
    # We want:
    # - High exploitability (density variations in residual)
    # - extends = True (patterns transfer to new range)
    # - High correlation (strong prediction)

    fitness = (
        0.2 * residual_fraction +          # Some credit for unexplained primes
        0.4 * exploitability +             # Main score: density exploitability
        0.2 * (1.0 if extends else 0.0) +  # Bonus for transfer
        0.2 * max(0, correlation)          # Bonus for correlation
    )

    # Update interest determination with better criteria
    if not is_interesting and extends and correlation > 0.2:
        is_interesting = True
        reason = f"Extends with r={correlation:.2f}"

    genome.fitness = fitness

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

            try:
                result = evaluate_genome(
                    genome,
                    start_n=self.start_n,
                    end_n=self.end_n,
                    grid_size=self.grid_size
                )
                results.append(result)

                # Track ALL results for auditing
                self.eval_counter += 1
                result.eval_id = self.eval_counter  # Add eval ID for tracking
                self.all_results.append(result)

                # Save image only for top performers or interesting discoveries
                # (Skip most evaluations for performance - mosaic uses saved images)
                save_this_image = (
                    result.is_interesting or
                    result.fitness >= 0.5 or
                    self.eval_counter <= 5 or  # First few for debugging
                    self.eval_counter % 50 == 0  # Sample every 50
                )
                if save_this_image:
                    self._save_evaluation_image(result, self.eval_counter)

                if result.is_interesting:
                    self.discoveries.append(result)
                    self._save_discovery(result)
                    self.logger.info(f"[DISCOVERY] Found interesting {dimension}D pattern! "
                                   f"Fitness: {result.fitness:.4f}, "
                                   f"Exploitability: {result.residual_exploitability:.2f}, "
                                   f"Variance: {result.density_variance_ratio:.2f}x")
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
            # Copy genome
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

    def _save_discovery(self, result: DiscoveryResult):
        """Save an interesting discovery."""
        discovery_id = len(self.discoveries)

        # Save genome
        genome_data = {
            "dimensions": result.genome.dimensions,
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
        """
        genome = result.genome

        try:
            from PIL import Image, ImageDraw, ImageFont

            # Use cached prime set
            prime_set = get_prime_set(self.end_n)

            numbers = np.arange(self.start_n, self.end_n, dtype=np.int64)
            primes_mask = fast_is_prime_mask(numbers, prime_set)
            coords = compute_nd_coordinates(numbers, genome)
            _, prime_grid = render_nd_visualization(coords, primes_mask, self.grid_size)

            # Get 2D representation
            if genome.dimensions == 2:
                grid_2d = prime_grid
            else:
                grid_2d = render_2d_projection(prime_grid)

            # Normalize to 0-1
            if grid_2d.max() > 0:
                grid_2d = grid_2d / grid_2d.max()

            # Detect patterns on the 2D representation
            pattern_mask, _ = detect_patterns_2d(grid_2d)

            # Create the three panels
            h, w = grid_2d.shape

            # Panel 1: Original (all primes white)
            original = (grid_2d * 255).astype(np.uint8)

            # Panel 2: Known patterns highlighted
            # Red channel = primes on patterns, Green/Blue = primes not on patterns
            binary = grid_2d > 0.5
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
                        self.logger.info(f"  Discoveries: {len(self.discoveries)} | Errors: {self.errors_count}")

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
                        self.logger.info(f"  Discoveries: {len(self.discoveries)} | Errors: {self.errors_count}")

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
        }

        if error:
            summary["error"] = error

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
