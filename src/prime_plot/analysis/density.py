"""Prime density analysis tools.

Computes local and global prime density metrics for visualization
and pattern analysis.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, count_primes


def compute_local_density(
    prime_grid: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute local prime density using a sliding window.

    Args:
        prime_grid: Binary grid where 1 indicates prime.
        window_size: Size of the averaging window.

    Returns:
        Float array of local densities.
    """
    from scipy.ndimage import uniform_filter

    density = uniform_filter(
        prime_grid.astype(np.float32),
        size=window_size,
        mode='constant',
    )

    return density


def prime_density_map(
    grid: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Create density heatmap from integer grid.

    Combines prime detection and local density computation.

    Args:
        grid: 2D integer grid (e.g., from UlamSpiral).
        window_size: Size of the density window.

    Returns:
        Float array of local prime densities.
    """
    from prime_plot.core.sieve import prime_sieve_mask

    max_val = int(grid.max())
    if max_val < 2:
        return np.zeros_like(grid, dtype=np.float32)

    mask = prime_sieve_mask(max_val + 1)
    prime_grid = mask[grid].astype(np.float32)

    return compute_local_density(prime_grid, window_size)


def radial_density_profile(
    prime_grid: np.ndarray,
    center: tuple[int, int] | None = None,
    max_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prime density as a function of distance from center.

    Useful for analyzing how prime density decreases as numbers grow
    (following the prime number theorem).

    Args:
        prime_grid: Binary grid where 1 indicates prime.
        center: (x, y) center point. Defaults to grid center.
        max_radius: Maximum radius to consider.

    Returns:
        Tuple of (radii, densities) arrays.
    """
    height, width = prime_grid.shape

    if center is None:
        center = (width // 2, height // 2)

    if max_radius is None:
        max_radius = min(
            center[0], center[1],
            width - center[0] - 1,
            height - center[1] - 1
        )

    y_coords, x_coords = np.ogrid[:height, :width]
    distances = np.sqrt(
        (x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2
    )

    radii = np.arange(1, max_radius + 1)
    densities = np.zeros(len(radii))

    for i, r in enumerate(radii):
        ring_mask = (distances >= r - 0.5) & (distances < r + 0.5)
        ring_primes = prime_grid[ring_mask]

        if len(ring_primes) > 0:
            densities[i] = ring_primes.mean()

    return radii, densities


def theoretical_density(n: int) -> float:
    """Compute theoretical prime density at n using prime number theorem.

    The density of primes near n is approximately 1/ln(n).

    Args:
        n: The number to compute density at.

    Returns:
        Theoretical density (primes per integer near n).
    """
    if n < 2:
        return 0.0
    return 1.0 / np.log(n)


def density_ratio(
    prime_grid: np.ndarray,
    integer_grid: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """Compute ratio of observed to expected prime density.

    Values > 1 indicate higher-than-expected prime concentration,
    which may correspond to prime-generating polynomials.

    Args:
        prime_grid: Binary grid where 1 indicates prime.
        integer_grid: Grid of integer values.
        window_size: Size of the averaging window.

    Returns:
        Float array of density ratios.
    """
    from scipy.ndimage import uniform_filter

    observed = uniform_filter(
        prime_grid.astype(np.float32),
        size=window_size,
        mode='constant',
    )

    log_values = np.log(np.maximum(integer_grid, 2).astype(np.float32))
    expected = 1.0 / log_values

    expected_smoothed = uniform_filter(
        expected,
        size=window_size,
        mode='constant',
    )

    ratio = np.divide(
        observed,
        expected_smoothed,
        out=np.zeros_like(observed),
        where=expected_smoothed > 0.001,
    )

    return ratio


def cumulative_prime_count(limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative prime count function pi(n).

    Args:
        limit: Upper bound for computation.

    Returns:
        Tuple of (n_values, pi_values) arrays.
    """
    primes = generate_primes(limit)

    n_values = np.arange(2, limit + 1)
    pi_values = np.zeros(len(n_values), dtype=np.int64)

    prime_idx = 0
    for i, n in enumerate(n_values):
        while prime_idx < len(primes) and primes[prime_idx] <= n:
            prime_idx += 1
        pi_values[i] = prime_idx

    return n_values, pi_values


def prime_counting_error(limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute error between pi(n) and n/ln(n) approximation.

    Args:
        limit: Upper bound for computation.

    Returns:
        Tuple of (n_values, relative_error) arrays.
    """
    n_values, pi_values = cumulative_prime_count(limit)

    approx = n_values / np.log(n_values)

    error = (pi_values - approx) / pi_values

    return n_values, error


def segment_density_analysis(
    start: int,
    end: int,
    segment_size: int = 1000,
) -> dict[str, np.ndarray]:
    """Analyze prime density across segments of integers.

    Args:
        start: Starting integer.
        end: Ending integer.
        segment_size: Size of each analysis segment.

    Returns:
        Dictionary with segment statistics.
    """
    from prime_plot.core.sieve import generate_primes_range

    segments = list(range(start, end, segment_size))
    densities = []
    counts = []
    theoretical = []

    for seg_start in segments:
        seg_end = min(seg_start + segment_size, end)
        primes = generate_primes_range(seg_start, seg_end)

        count = len(primes)
        density = count / (seg_end - seg_start)

        densities.append(density)
        counts.append(count)

        mid = (seg_start + seg_end) / 2
        theoretical.append(theoretical_density(mid))

    return {
        "segments": np.array(segments),
        "densities": np.array(densities),
        "counts": np.array(counts),
        "theoretical": np.array(theoretical),
        "ratio": np.array(densities) / np.array(theoretical),
    }
