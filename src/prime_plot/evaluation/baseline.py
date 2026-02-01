"""Random baseline generation for pattern comparison.

Provides methods to generate random point distributions that match
the density of prime visualizations but lack the structural patterns.
This allows measuring how much better than random the patterns are.
"""

from __future__ import annotations

import numpy as np


def generate_random_baseline(
    shape: tuple[int, int],
    density: float,
    seed: int | None = None
) -> np.ndarray:
    """Generate uniformly random point distribution.

    Args:
        shape: (height, width) of output image.
        density: Fraction of pixels to turn on (0-1).
        seed: Random seed for reproducibility.

    Returns:
        2D uint8 array with random points.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    num_points = int(h * w * density)

    image = np.zeros(shape, dtype=np.uint8)

    if num_points > 0:
        # Generate random positions
        positions = np.random.choice(h * w, size=num_points, replace=False)
        y_coords = positions // w
        x_coords = positions % w
        image[y_coords, x_coords] = 255

    return image


def generate_density_matched_baseline(
    reference_image: np.ndarray,
    threshold: float = 0.5,
    seed: int | None = None
) -> np.ndarray:
    """Generate random baseline matching density of reference image.

    Args:
        reference_image: Image to match density of.
        threshold: Value above which pixel is considered "on".
        seed: Random seed for reproducibility.

    Returns:
        Random image with same density as reference.
    """
    img = reference_image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    density = (img > threshold).mean()

    h, w = reference_image.shape[:2]
    return generate_random_baseline((h, w), density, seed)


def generate_radial_density_baseline(
    shape: tuple[int, int],
    total_points: int,
    center: tuple[int, int] | None = None,
    density_profile: str = "uniform",
    seed: int | None = None
) -> np.ndarray:
    """Generate random baseline with radial density profile.

    Matches the radial distribution of points in spiral visualizations
    while randomizing angular positions.

    Args:
        shape: (height, width) of output image.
        total_points: Total number of points to place.
        center: Center point (y, x). Default: image center.
        density_profile: 'uniform', 'sqrt', or 'linear'.
        seed: Random seed for reproducibility.

    Returns:
        2D uint8 array with radially-distributed random points.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    if center is None:
        center = (h // 2, w // 2)

    center_y, center_x = center
    max_r = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)

    image = np.zeros(shape, dtype=np.uint8)

    # Generate random radii based on density profile
    if density_profile == "uniform":
        # Uniform in area means r ~ sqrt(uniform)
        r_values = max_r * np.sqrt(np.random.random(total_points))
    elif density_profile == "sqrt":
        # sqrt(n) radial scaling (like Sacks spiral)
        r_values = max_r * np.sqrt(np.random.random(total_points))
    elif density_profile == "linear":
        # Linear radial distribution
        r_values = max_r * np.random.random(total_points)
    else:
        raise ValueError(f"Unknown density_profile: {density_profile}")

    # Random angles
    theta_values = 2 * np.pi * np.random.random(total_points)

    # Convert to coordinates
    x_coords = (center_x + r_values * np.cos(theta_values)).astype(np.int32)
    y_coords = (center_y + r_values * np.sin(theta_values)).astype(np.int32)

    # Filter valid coordinates and place points
    valid = (0 <= x_coords) & (x_coords < w) & (0 <= y_coords) & (y_coords < h)
    image[y_coords[valid], x_coords[valid]] = 255

    return image


def generate_local_density_baseline(
    reference_image: np.ndarray,
    window_size: int = 32,
    threshold: float = 0.5,
    seed: int | None = None
) -> np.ndarray:
    """Generate random baseline matching local density variations.

    Divides image into windows and matches density in each window,
    preserving large-scale density variations but randomizing local structure.

    Args:
        reference_image: Image to match.
        window_size: Size of density matching windows.
        threshold: Value above which pixel is considered "on".
        seed: Random seed for reproducibility.

    Returns:
        Random image matching local densities of reference.
    """
    if seed is not None:
        np.random.seed(seed)

    img = reference_image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.float64)
    h, w = binary.shape

    output = np.zeros_like(binary, dtype=np.uint8)

    # Process each window
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)

            window = binary[y:y_end, x:x_end]
            window_h, window_w = window.shape

            # Count points in reference window
            num_points = int(window.sum())

            if num_points > 0:
                # Randomly place same number of points
                positions = np.random.choice(
                    window_h * window_w,
                    size=min(num_points, window_h * window_w),
                    replace=False
                )
                wy = positions // window_w
                wx = positions % window_w
                output[y + wy, x + wx] = 255

    return output


def compute_baseline_statistics(
    prime_image: np.ndarray,
    num_baselines: int = 10,
    threshold: float = 0.5,
    seed: int = 42
) -> dict:
    """Compute statistics from multiple random baselines.

    Useful for establishing statistical significance of patterns.

    Args:
        prime_image: Original prime visualization.
        num_baselines: Number of random baselines to generate.
        threshold: Pixel threshold.
        seed: Base random seed.

    Returns:
        Dictionary with baseline statistics.
    """
    from prime_plot.evaluation.metrics import (
        calculate_line_density,
        calculate_entropy,
        calculate_cluster_coherence,
    )

    img = prime_image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    density = (img > threshold).mean()

    line_scores = []
    entropies = []
    cluster_scores = []

    h, w = prime_image.shape[:2]
    for i in range(num_baselines):
        baseline = generate_random_baseline(
            (h, w), density, seed=seed + i
        )

        line_score, _ = calculate_line_density(baseline)
        line_scores.append(line_score)

        ent = calculate_entropy(baseline)
        entropies.append(ent)

        cluster_score = calculate_cluster_coherence(baseline)
        cluster_scores.append(cluster_score)

    return {
        'num_baselines': num_baselines,
        'density': density,
        'line_density': {
            'mean': np.mean(line_scores),
            'std': np.std(line_scores),
            'min': np.min(line_scores),
            'max': np.max(line_scores),
        },
        'entropy': {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'min': np.min(entropies),
            'max': np.max(entropies),
        },
        'cluster_coherence': {
            'mean': np.mean(cluster_scores),
            'std': np.std(cluster_scores),
            'min': np.min(cluster_scores),
            'max': np.max(cluster_scores),
        },
    }
