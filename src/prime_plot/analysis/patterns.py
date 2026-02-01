"""Pattern detection in prime visualizations.

Detects diagonal lines, clusters, and other structural patterns
in Ulam spirals and related visualizations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from prime_plot.core.sieve import is_prime


@dataclass
class DiagonalPattern:
    """Represents a detected diagonal pattern.

    Attributes:
        start: (x, y) starting position.
        direction: (dx, dy) direction vector.
        length: Number of points in the pattern.
        prime_count: Number of primes along the diagonal.
        density: Fraction of points that are prime.
        values: Integer values along the diagonal.
    """
    start: tuple[int, int]
    direction: tuple[int, int]
    length: int
    prime_count: int
    density: float
    values: list[int]


def detect_diagonal_patterns(
    grid: np.ndarray,
    min_length: int = 10,
    min_density: float = 0.3,
) -> list[DiagonalPattern]:
    """Detect high-density diagonal patterns in a grid.

    Scans all 4 diagonal directions (NE, NW, SE, SW) for lines
    with above-threshold prime density.

    Args:
        grid: 2D integer grid (e.g., from UlamSpiral.generate_grid()).
        min_length: Minimum diagonal length to consider.
        min_density: Minimum prime density threshold.

    Returns:
        List of DiagonalPattern objects sorted by density.
    """
    height, width = grid.shape
    patterns = []

    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for dy, dx in directions:
        if dx > 0:
            start_cols = range(width)
        else:
            start_cols = range(width - 1, -1, -1)

        for start_row in range(height):
            for start_col in start_cols:
                values = []
                y, x = start_row, start_col

                while 0 <= y < height and 0 <= x < width:
                    val = grid[y, x]
                    if val > 0:
                        values.append(int(val))
                    y += dy
                    x += dx

                if len(values) >= min_length:
                    prime_count = sum(1 for v in values if is_prime(v))
                    density = prime_count / len(values)

                    if density >= min_density:
                        patterns.append(DiagonalPattern(
                            start=(start_col, start_row),
                            direction=(dx, dy),
                            length=len(values),
                            prime_count=prime_count,
                            density=density,
                            values=values,
                        ))

    patterns.sort(key=lambda p: p.density, reverse=True)

    return _deduplicate_patterns(patterns)


def _deduplicate_patterns(patterns: list[DiagonalPattern]) -> list[DiagonalPattern]:
    """Remove overlapping patterns, keeping highest density."""
    if not patterns:
        return []

    seen_values: set[int] = set()
    unique = []

    for p in patterns:
        value_set = frozenset(p.values)

        overlap = len(value_set & seen_values) / len(value_set) if value_set else 1.0

        if overlap < 0.5:
            unique.append(p)
            seen_values.update(value_set)

    return unique


def analyze_line_density(
    grid: np.ndarray,
    direction: str = "diagonal",
) -> dict[str, np.ndarray]:
    """Analyze prime density along lines in the grid.

    Args:
        grid: 2D integer grid.
        direction: "diagonal", "horizontal", or "vertical".

    Returns:
        Dictionary with 'densities' array and 'counts' array.
    """
    height, width = grid.shape

    if direction == "horizontal":
        densities = []
        counts = []
        for row in range(height):
            values = [int(v) for v in grid[row, :] if v > 0]
            if values:
                prime_count = sum(1 for v in values if is_prime(v))
                densities.append(prime_count / len(values))
                counts.append(prime_count)
            else:
                densities.append(0.0)
                counts.append(0)

        return {
            "densities": np.array(densities),
            "counts": np.array(counts),
            "positions": np.arange(height),
        }

    elif direction == "vertical":
        densities = []
        counts = []
        for col in range(width):
            values = [int(v) for v in grid[:, col] if v > 0]
            if values:
                prime_count = sum(1 for v in values if is_prime(v))
                densities.append(prime_count / len(values))
                counts.append(prime_count)
            else:
                densities.append(0.0)
                counts.append(0)

        return {
            "densities": np.array(densities),
            "counts": np.array(counts),
            "positions": np.arange(width),
        }

    else:
        main_densities = []
        main_counts = []

        for offset in range(-height + 1, width):
            diag = np.diagonal(grid, offset=offset)
            values = [int(v) for v in diag if v > 0]
            if values:
                prime_count = sum(1 for v in values if is_prime(v))
                main_densities.append(prime_count / len(values))
                main_counts.append(prime_count)
            else:
                main_densities.append(0.0)
                main_counts.append(0)

        anti_densities = []
        anti_counts = []
        flipped = np.fliplr(grid)

        for offset in range(-height + 1, width):
            diag = np.diagonal(flipped, offset=offset)
            values = [int(v) for v in diag if v > 0]
            if values:
                prime_count = sum(1 for v in values if is_prime(v))
                anti_densities.append(prime_count / len(values))
                anti_counts.append(prime_count)
            else:
                anti_densities.append(0.0)
                anti_counts.append(0)

        return {
            "main_densities": np.array(main_densities),
            "main_counts": np.array(main_counts),
            "anti_densities": np.array(anti_densities),
            "anti_counts": np.array(anti_counts),
            "offsets": np.arange(-height + 1, width),
        }


@dataclass
class HighDensityRegion:
    """A rectangular region with high prime density.

    Attributes:
        x: Left column index.
        y: Top row index.
        width: Region width.
        height: Region height.
        density: Prime density in the region.
        prime_count: Number of primes in the region.
    """
    x: int
    y: int
    width: int
    height: int
    density: float
    prime_count: int


def extract_high_density_regions(
    prime_grid: np.ndarray,
    window_size: int = 32,
    stride: int = 16,
    min_density: float = 0.2,
) -> list[HighDensityRegion]:
    """Extract rectangular regions with high prime density.

    Uses a sliding window approach to find areas with
    above-threshold prime concentration.

    Args:
        prime_grid: Binary grid where 1 indicates prime.
        window_size: Size of the sliding window.
        stride: Step size for the sliding window.
        min_density: Minimum density threshold.

    Returns:
        List of HighDensityRegion objects.
    """
    height, width = prime_grid.shape
    regions = []

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = prime_grid[y:y + window_size, x:x + window_size]
            prime_count = int(window.sum())
            total = window_size * window_size
            density = prime_count / total

            if density >= min_density:
                regions.append(HighDensityRegion(
                    x=x,
                    y=y,
                    width=window_size,
                    height=window_size,
                    density=density,
                    prime_count=prime_count,
                ))

    regions.sort(key=lambda r: r.density, reverse=True)
    return regions


def find_prime_clusters(
    prime_grid: np.ndarray,
    radius: int = 3,
    min_primes: int = 5,
) -> list[tuple[int, int, int]]:
    """Find clusters of primes within a given radius.

    Args:
        prime_grid: Binary grid where 1 indicates prime.
        radius: Search radius around each point.
        min_primes: Minimum number of primes to form a cluster.

    Returns:
        List of (x, y, count) tuples for cluster centers.
    """
    from scipy.ndimage import uniform_filter

    kernel_size = 2 * radius + 1
    counts = uniform_filter(
        prime_grid.astype(float),
        size=kernel_size,
        mode='constant'
    ) * (kernel_size ** 2)

    counts = counts.astype(int)

    clusters = []
    height, width = prime_grid.shape

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            count = counts[y, x]
            if count >= min_primes:
                is_local_max = True
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if (dy != 0 or dx != 0) and counts[y + dy, x + dx] > count:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break

                if is_local_max:
                    clusters.append((x, y, count))

    clusters.sort(key=lambda c: c[2], reverse=True)
    return clusters
