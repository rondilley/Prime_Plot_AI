"""Predictive power evaluation for prime visualizations.

Measures whether a visualization creates regions where prime density
varies significantly - i.e., whether the visualization enables
better-than-random prime prediction.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy import ndimage


@dataclass
class PredictivePowerMetrics:
    """Results from predictive power analysis."""

    # Core metrics
    density_variance: float  # Variance in prime density across regions
    density_range: float     # Max - min density across regions
    information_gain: float  # Bits of information about primality from position

    # Regional analysis
    num_high_density_regions: int   # Regions with >1.5x average density
    num_low_density_regions: int    # Regions with <0.5x average density
    max_regional_density: float     # Highest regional prime density
    min_regional_density: float     # Lowest regional prime density

    # Prediction potential
    separation_score: float  # How well high/low regions separate
    predictive_value: float  # Overall usefulness for prediction (0-1)


def calculate_regional_densities(
    prime_image: np.ndarray,
    grid_size: int = 10
) -> Tuple[np.ndarray, float]:
    """Divide image into grid and calculate prime density per region.

    Args:
        prime_image: Binary image where white=prime, black=composite
        grid_size: Number of divisions per axis

    Returns:
        (density_grid, global_density)
    """
    h, w = prime_image.shape[:2]
    if len(prime_image.shape) == 3:
        prime_image = prime_image[:, :, 0]

    # Normalize to 0-1
    img = prime_image.astype(float)
    if img.max() > 1:
        img = img / 255.0

    cell_h = h // grid_size
    cell_w = w // grid_size

    densities = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            region = img[y1:y2, x1:x2]

            # Density = fraction of pixels that are prime (white)
            densities[i, j] = region.mean()

    global_density = img.mean()

    return densities, global_density


def calculate_information_gain(
    densities: np.ndarray,
    global_density: float
) -> float:
    """Calculate information gain about primality from knowing position.

    Uses mutual information: I(Prime; Position) = H(Prime) - H(Prime|Position)

    Higher values mean position tells you more about primality.
    """
    if global_density <= 0 or global_density >= 1:
        return 0.0

    # Global entropy H(Prime)
    p = global_density
    h_prime = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)

    # Conditional entropy H(Prime|Position)
    # Average entropy within each region, weighted by region size
    h_conditional = 0.0
    num_regions = densities.size

    for d in densities.flatten():
        if d > 0 and d < 1:
            region_entropy = -d * np.log2(d + 1e-10) - (1-d) * np.log2(1-d + 1e-10)
        else:
            region_entropy = 0.0
        h_conditional += region_entropy / num_regions

    # Information gain
    info_gain = h_prime - h_conditional

    return max(0.0, info_gain)


def calculate_separation_score(
    densities: np.ndarray,
    global_density: float
) -> float:
    """Measure how well high and low density regions separate.

    Returns value 0-1 where 1 means perfect separation (some regions
    have all primes, others have none).
    """
    flat = densities.flatten()

    if len(flat) < 2:
        return 0.0

    # Normalize by global density
    if global_density > 0:
        normalized = flat / global_density
    else:
        return 0.0

    # Count regions significantly above/below average
    high = (normalized > 1.5).sum()
    low = (normalized < 0.5).sum()

    # Separation score based on having distinct high/low regions
    total = len(flat)
    separation = (high + low) / total

    # Also factor in the magnitude of difference
    if flat.std() > 0:
        cv = flat.std() / (flat.mean() + 1e-10)  # Coefficient of variation
        magnitude = min(cv, 1.0)
    else:
        magnitude = 0.0

    return separation * 0.5 + magnitude * 0.5


def calculate_predictive_power(
    prime_image: np.ndarray,
    grid_sizes: List[int] = [5, 10, 20],
    all_integers_image: Optional[np.ndarray] = None
) -> PredictivePowerMetrics:
    """Full predictive power analysis of a prime visualization.

    Args:
        prime_image: Image with only primes plotted
        grid_sizes: Grid resolutions to analyze
        all_integers_image: Optional image with all integers plotted
                           (to normalize for plotting density)

    Returns:
        PredictivePowerMetrics with comprehensive analysis
    """
    # Aggregate metrics across multiple grid sizes
    all_densities = []
    all_info_gains = []
    all_separations = []

    for grid_size in grid_sizes:
        densities, global_density = calculate_regional_densities(prime_image, grid_size)
        all_densities.append(densities)

        info_gain = calculate_information_gain(densities, global_density)
        all_info_gains.append(info_gain)

        separation = calculate_separation_score(densities, global_density)
        all_separations.append(separation)

    # Use finest grid for detailed metrics
    finest_densities = all_densities[-1]
    _, global_density = calculate_regional_densities(prime_image, grid_sizes[-1])

    flat_densities = finest_densities.flatten()

    # Handle edge cases
    if global_density == 0:
        return PredictivePowerMetrics(
            density_variance=0.0,
            density_range=0.0,
            information_gain=0.0,
            num_high_density_regions=0,
            num_low_density_regions=0,
            max_regional_density=0.0,
            min_regional_density=0.0,
            separation_score=0.0,
            predictive_value=0.0
        )

    # Normalized densities (relative to average)
    normalized = flat_densities / (global_density + 1e-10)

    # Core metrics
    density_variance = float(flat_densities.var())
    density_range = float(flat_densities.max() - flat_densities.min())
    information_gain = float(np.mean(all_info_gains))

    # Regional counts
    num_high = int((normalized > 1.5).sum())
    num_low = int((normalized < 0.5).sum())

    # Extremes
    max_density = float(flat_densities.max())
    min_density = float(flat_densities.min())

    # Separation
    separation_score = float(np.mean(all_separations))

    # Overall predictive value (0-1 composite score)
    # Weight: info gain most important, then separation, then variance
    predictive_value = (
        0.4 * min(information_gain / 0.5, 1.0) +  # Normalize info gain
        0.3 * separation_score +
        0.2 * min(density_variance / (global_density ** 2 + 1e-10), 1.0) +
        0.1 * (1.0 if (num_high > 0 and num_low > 0) else 0.0)
    )

    return PredictivePowerMetrics(
        density_variance=density_variance,
        density_range=density_range,
        information_gain=information_gain,
        num_high_density_regions=num_high,
        num_low_density_regions=num_low,
        max_regional_density=max_density,
        min_regional_density=min_density,
        separation_score=separation_score,
        predictive_value=min(predictive_value, 1.0)
    )


def compare_to_random_baseline(
    prime_image: np.ndarray,
    num_samples: int = 10
) -> Dict[str, float]:
    """Compare visualization's predictive power to random baseline.

    Generates random images with same density and measures whether
    the actual visualization has significantly higher predictive power.

    Returns:
        Dictionary with actual metrics, baseline metrics, and improvement ratio
    """
    actual_metrics = calculate_predictive_power(prime_image)

    # Generate random baselines with same density
    img = prime_image.astype(float)
    if img.max() > 1:
        img = img / 255.0
    density = img.mean()

    baseline_values = []
    for _ in range(num_samples):
        random_img = (np.random.random(img.shape) < density).astype(float) * 255
        baseline_metrics = calculate_predictive_power(random_img.astype(np.uint8))
        baseline_values.append(baseline_metrics.predictive_value)

    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)

    # How many standard deviations above baseline?
    z_score: float
    if baseline_std > 0:
        z_score = float((actual_metrics.predictive_value - baseline_mean) / baseline_std)
    else:
        z_score = 0.0 if actual_metrics.predictive_value == baseline_mean else float('inf')

    return {
        'actual_predictive_value': float(actual_metrics.predictive_value),
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(baseline_std),
        'z_score': float(z_score),
        'improvement_ratio': float(actual_metrics.predictive_value / (baseline_mean + 1e-10)),
        'significantly_better': bool(z_score > 2.0)  # p < 0.05
    }
