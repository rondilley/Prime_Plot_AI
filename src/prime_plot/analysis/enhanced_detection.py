"""Enhanced Pattern Detection for Prime Visualizations

This module provides comprehensive pattern detection using multiple methods:
1. Gabor filter bank (multi-orientation, multi-scale texture detection)
2. FFT-based periodic pattern detection
3. Connected component / blob detection
4. Morphological void/concentration detection
5. Directional convolution kernels (enhanced)
6. Local density variance analysis

The key insight: we need to detect KNOWN mathematical patterns (diagonals from
polynomials, modular grids, etc.) and REMOVE them. Then evaluate if the RESIDUAL
shows exploitable density variations.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PatternDetectionResult:
    """Result of comprehensive pattern detection."""
    pattern_mask: np.ndarray  # Boolean mask of detected patterns
    pattern_fraction: float   # Fraction of primes on patterns

    # Per-method breakdown
    gabor_fraction: float
    fft_fraction: float
    directional_fraction: float
    cluster_fraction: float

    # Confidence metrics
    detection_confidence: float  # How confident are we in detection?
    method_agreement: float      # Do methods agree on what's a pattern?


def create_gabor_filter_bank(
    ksize: int = 15,
    orientations: int = 12,
    scales: List[float] = None
) -> List[np.ndarray]:
    """Create a bank of Gabor filters at multiple orientations and scales.

    Gabor filters are optimal for detecting oriented texture patterns,
    which is exactly what polynomial diagonals and modular grids are.

    Args:
        ksize: Kernel size (should be odd)
        orientations: Number of orientation steps (12 = every 15 degrees)
        scales: List of wavelengths (frequencies) to detect

    Returns:
        List of Gabor kernels
    """
    if scales is None:
        scales = [4.0, 6.0, 8.0, 12.0]  # Multiple wavelengths

    kernels = []

    for theta_idx in range(orientations):
        theta = theta_idx * np.pi / orientations

        for wavelength in scales:
            # Gabor parameters
            sigma = wavelength * 0.5  # Bandwidth
            gamma = 0.5  # Aspect ratio
            psi = 0  # Phase offset

            # Create kernel
            kernel = _gabor_kernel(ksize, sigma, theta, wavelength, gamma, psi)
            kernels.append(kernel)

    return kernels


def _gabor_kernel(
    ksize: int,
    sigma: float,
    theta: float,
    wavelength: float,
    gamma: float,
    psi: float
) -> np.ndarray:
    """Generate a single Gabor kernel."""
    half = ksize // 2
    y, x = np.meshgrid(
        np.arange(-half, half + 1),
        np.arange(-half, half + 1)
    )

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor function
    gb = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    gb *= np.cos(2 * np.pi * x_theta / wavelength + psi)

    return gb.astype(np.float32)


def detect_with_gabor_bank(
    binary_image: np.ndarray,
    kernels: List[np.ndarray],
    threshold: float = 0.25
) -> Tuple[np.ndarray, float]:
    """Detect patterns using Gabor filter bank.

    Returns mask of pixels that respond strongly to ANY Gabor filter,
    indicating oriented texture (lines, diagonals, grids).
    """
    total_primes = binary_image.sum()
    if total_primes == 0:
        return np.zeros_like(binary_image, dtype=bool), 0.0

    max_response = np.zeros_like(binary_image, dtype=np.float32)

    for kernel in kernels:
        # Convolve and take absolute value (both positive and negative responses)
        response = np.abs(ndimage.convolve(binary_image, kernel))
        max_response = np.maximum(max_response, response)

    # Normalize
    if max_response.max() > 0:
        max_response /= max_response.max()

    # Threshold
    pattern_mask = (max_response > threshold) & (binary_image > 0.5)
    pattern_fraction = pattern_mask.sum() / total_primes

    return pattern_mask, float(pattern_fraction)


def detect_with_fft(
    binary_image: np.ndarray,
    threshold_sigma: float = 3.0
) -> Tuple[np.ndarray, float, List[Tuple[float, float]]]:
    """Detect periodic patterns using FFT.

    Looks for strong peaks in the frequency domain, indicating
    regular periodic structure (modular patterns, grids).

    Returns:
        (pattern_mask, pattern_fraction, dominant_frequencies)
    """
    total_primes = binary_image.sum()
    if total_primes == 0:
        return np.zeros_like(binary_image, dtype=bool), 0.0, []

    # Compute FFT
    fft_result = fftshift(fft2(binary_image))
    magnitude = np.abs(fft_result)

    # Find significant peaks (above threshold_sigma standard deviations)
    mean_mag = magnitude.mean()
    std_mag = magnitude.std()
    threshold = mean_mag + threshold_sigma * std_mag

    # Mask central DC component
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude[cy-2:cy+3, cx-2:cx+3] = 0

    peak_mask = magnitude > threshold

    # Find dominant frequencies
    dominant_freqs = []
    peak_coords = np.where(peak_mask)
    for y, x in zip(peak_coords[0], peak_coords[1]):
        freq_y = (y - cy) / h
        freq_x = (x - cx) / w
        if freq_y != 0 or freq_x != 0:
            dominant_freqs.append((freq_x, freq_y))

    # For each dominant frequency, mark pixels that match that periodicity
    pattern_mask = np.zeros_like(binary_image, dtype=bool)

    for freq_x, freq_y in dominant_freqs[:10]:  # Top 10 frequencies
        # Create mask for this frequency
        period_x = int(abs(1 / freq_x)) if freq_x != 0 else w
        period_y = int(abs(1 / freq_y)) if freq_y != 0 else h

        if period_x < w and period_y < h:
            # Mark pixels that align with this periodicity
            for y in range(h):
                for x in range(w):
                    if binary_image[y, x] > 0.5:
                        if (x % max(1, period_x) < 2) or (y % max(1, period_y) < 2):
                            pattern_mask[y, x] = True

    pattern_fraction = pattern_mask.sum() / total_primes if total_primes > 0 else 0.0

    return pattern_mask, float(pattern_fraction), dominant_freqs[:10]


def detect_clusters(
    binary_image: np.ndarray,
    min_cluster_size: int = 5,
    connectivity_radius: float = 2.0
) -> Tuple[np.ndarray, float, int]:
    """Detect clustered patterns using connected component analysis.

    Primes that cluster together (more than expected by chance) indicate
    structure that might be exploitable.

    Returns:
        (cluster_mask, cluster_fraction, num_clusters)
    """
    total_primes = binary_image.sum()
    if total_primes == 0:
        return np.zeros_like(binary_image, dtype=bool), 0.0, 0

    # Dilate to connect nearby primes
    structure = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(
        binary_image > 0.5,
        structure=structure,
        iterations=int(connectivity_radius)
    )

    # Find connected components
    labeled, num_features = ndimage.label(dilated)

    # Find significant clusters
    cluster_mask = np.zeros_like(binary_image, dtype=bool)
    significant_clusters = 0

    for i in range(1, num_features + 1):
        component_mask = labeled == i
        primes_in_component = (binary_image > 0.5) & component_mask
        count = primes_in_component.sum()

        if count >= min_cluster_size:
            cluster_mask |= primes_in_component
            significant_clusters += 1

    cluster_fraction = cluster_mask.sum() / total_primes if total_primes > 0 else 0.0

    return cluster_mask, float(cluster_fraction), significant_clusters


def detect_voids_and_concentrations(
    binary_image: np.ndarray,
    window_size: int = 15,
    threshold_sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detect regions of unusually high or low prime density.

    This is key for residual evaluation - we want to find regions where
    primes concentrate or avoid, beyond what's expected by chance.

    Returns:
        (concentration_mask, void_mask, density_variance_ratio)
    """
    total_primes = binary_image.sum()
    if total_primes == 0:
        return (np.zeros_like(binary_image, dtype=bool),
                np.zeros_like(binary_image, dtype=bool), 0.0)

    # Compute local density using Gaussian smoothing
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    local_count = ndimage.convolve(binary_image, kernel / kernel.sum())

    # Expected density (uniform)
    expected_density = total_primes / binary_image.size

    # Find regions significantly above/below expected
    mean_local = local_count.mean()
    std_local = local_count.std()

    if std_local > 0:
        high_threshold = mean_local + threshold_sigma * std_local
        low_threshold = mean_local - threshold_sigma * std_local

        concentration_mask = (local_count > high_threshold) & (binary_image > 0.5)
        void_mask = (local_count < low_threshold) & (binary_image > 0.5)
    else:
        concentration_mask = np.zeros_like(binary_image, dtype=bool)
        void_mask = np.zeros_like(binary_image, dtype=bool)

    # Compute variance ratio vs random expectation
    # For random Poisson, variance = mean
    expected_variance = expected_density * (1 - expected_density)
    actual_variance = local_count.var()

    variance_ratio = actual_variance / expected_variance if expected_variance > 0 else 1.0

    return concentration_mask, void_mask, float(variance_ratio)


def enhanced_directional_detection(
    binary_image: np.ndarray,
    num_orientations: int = 16,
    kernel_sizes: List[int] = None,
    threshold: float = 0.30
) -> Tuple[np.ndarray, float]:
    """Enhanced directional detection with more orientations and scales.

    Uses 16+ orientations (22.5 degree steps) and multiple kernel sizes
    to catch patterns at different scales.
    """
    if kernel_sizes is None:
        kernel_sizes = [5, 7, 9, 11]

    total_primes = binary_image.sum()
    if total_primes == 0:
        return np.zeros_like(binary_image, dtype=bool), 0.0

    max_response = np.zeros_like(binary_image, dtype=np.float32)

    for ksize in kernel_sizes:
        for theta_idx in range(num_orientations):
            theta = theta_idx * np.pi / num_orientations

            # Create oriented line kernel
            kernel = _create_oriented_kernel(ksize, theta)

            response = ndimage.convolve(binary_image, kernel / kernel.sum())
            max_response = np.maximum(max_response, response)

    pattern_mask = (max_response > threshold) & (binary_image > 0.5)
    pattern_fraction = pattern_mask.sum() / total_primes

    return pattern_mask, float(pattern_fraction)


def _create_oriented_kernel(ksize: int, theta: float) -> np.ndarray:
    """Create a line kernel at specified orientation."""
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    # Draw line through center at angle theta
    length = ksize // 2
    for i in range(-length, length + 1):
        x = int(center + i * np.cos(theta))
        y = int(center + i * np.sin(theta))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0

    return kernel


def comprehensive_pattern_detection(
    binary_image: np.ndarray,
    use_gabor: bool = True,
    use_fft: bool = True,
    use_directional: bool = True,
    use_clusters: bool = True
) -> PatternDetectionResult:
    """Run all pattern detection methods and combine results.

    This provides the most thorough detection of known mathematical
    patterns (polynomial diagonals, modular grids, periodic structure).
    """
    h, w = binary_image.shape
    total_primes = binary_image.sum()

    if total_primes == 0:
        return PatternDetectionResult(
            pattern_mask=np.zeros_like(binary_image, dtype=bool),
            pattern_fraction=0.0,
            gabor_fraction=0.0,
            fft_fraction=0.0,
            directional_fraction=0.0,
            cluster_fraction=0.0,
            detection_confidence=0.0,
            method_agreement=0.0
        )

    combined_mask = np.zeros_like(binary_image, dtype=bool)
    method_masks = []
    fractions = []

    # 1. Gabor filter bank
    if use_gabor:
        gabor_kernels = create_gabor_filter_bank(ksize=15, orientations=12)
        gabor_mask, gabor_frac = detect_with_gabor_bank(binary_image, gabor_kernels)
        combined_mask |= gabor_mask
        method_masks.append(gabor_mask)
        fractions.append(gabor_frac)
    else:
        gabor_frac = 0.0

    # 2. FFT periodic detection
    if use_fft:
        fft_mask, fft_frac, _ = detect_with_fft(binary_image)
        combined_mask |= fft_mask
        method_masks.append(fft_mask)
        fractions.append(fft_frac)
    else:
        fft_frac = 0.0

    # 3. Enhanced directional detection
    if use_directional:
        dir_mask, dir_frac = enhanced_directional_detection(binary_image)
        combined_mask |= dir_mask
        method_masks.append(dir_mask)
        fractions.append(dir_frac)
    else:
        dir_frac = 0.0

    # 4. Cluster detection
    if use_clusters:
        cluster_mask, cluster_frac, _ = detect_clusters(binary_image)
        combined_mask |= cluster_mask
        method_masks.append(cluster_mask)
        fractions.append(cluster_frac)
    else:
        cluster_frac = 0.0

    # Compute combined fraction
    combined_fraction = combined_mask.sum() / total_primes

    # Compute method agreement (how many methods agree on each pixel)
    if method_masks:
        agreement_count = sum(m.astype(int) for m in method_masks)
        # Points where 2+ methods agree
        high_agreement = (agreement_count >= 2) & (binary_image > 0.5)
        method_agreement = high_agreement.sum() / total_primes
    else:
        method_agreement = 0.0

    # Detection confidence based on method agreement
    detection_confidence = method_agreement / combined_fraction if combined_fraction > 0 else 0.0

    return PatternDetectionResult(
        pattern_mask=combined_mask,
        pattern_fraction=float(combined_fraction),
        gabor_fraction=float(gabor_frac),
        fft_fraction=float(fft_frac),
        directional_fraction=float(dir_frac),
        cluster_fraction=float(cluster_frac),
        detection_confidence=float(detection_confidence),
        method_agreement=float(method_agreement)
    )


# =============================================================================
# RESIDUAL EVALUATION (The key fix)
# =============================================================================

@dataclass
class ResidualEvaluationResult:
    """Result of evaluating the residual for exploitable patterns."""

    # Density analysis
    density_variance_ratio: float  # Actual variance / expected variance
    concentration_fraction: float  # Fraction in high-density regions
    void_fraction: float           # Fraction in low-density regions

    # Transfer test
    transfer_score: float          # Do density patterns predict in new range?
    transfer_correlation: float    # Correlation between ranges

    # Exploitability
    exploitability_score: float    # Overall score for search prioritization
    high_density_ratio: float      # Prime rate in high bins / rate in low bins

    # Is this residual interesting?
    is_interesting: bool
    reason: str


def evaluate_residual_density(
    residual_image: np.ndarray,
    bin_size: int = 20,
    min_primes_per_bin: int = 5
) -> Tuple[np.ndarray, float, float]:
    """Evaluate residual for density variations.

    Bins the image into regions and measures prime density per region.
    High variance = potential for exploitation.

    Returns:
        (density_map, variance_ratio, high_low_ratio)
    """
    h, w = residual_image.shape
    binary = residual_image > 0.5

    # Compute density per bin
    n_bins_y = max(1, h // bin_size)
    n_bins_x = max(1, w // bin_size)

    density_map = np.zeros((n_bins_y, n_bins_x), dtype=np.float32)
    counts = []

    for by in range(n_bins_y):
        for bx in range(n_bins_x):
            y_start = by * bin_size
            y_end = min((by + 1) * bin_size, h)
            x_start = bx * bin_size
            x_end = min((bx + 1) * bin_size, w)

            bin_region = binary[y_start:y_end, x_start:x_end]
            count = bin_region.sum()
            bin_area = bin_region.size

            density = count / bin_area if bin_area > 0 else 0
            density_map[by, bx] = density

            if count >= min_primes_per_bin:
                counts.append(count)

    if len(counts) < 4:
        return density_map, 1.0, 1.0

    # Compute variance ratio
    counts = np.array(counts)
    expected_variance = counts.mean()  # Poisson assumption
    actual_variance = counts.var()
    variance_ratio = actual_variance / expected_variance if expected_variance > 0 else 1.0

    # Compute high/low density ratio
    sorted_densities = np.sort(density_map.flatten())
    n_bins_total = len(sorted_densities)

    if n_bins_total >= 4:
        high_density = sorted_densities[-n_bins_total//4:].mean()
        low_density = sorted_densities[:n_bins_total//4].mean()
        high_low_ratio = high_density / low_density if low_density > 0 else 1.0
    else:
        high_low_ratio = 1.0

    return density_map, float(variance_ratio), float(high_low_ratio)


def test_density_transfer(
    genome,
    range_a: Tuple[int, int],
    range_b: Tuple[int, int],
    prime_set: set,
    grid_size: int = 128,
    bin_size: int = 16
) -> Tuple[float, float]:
    """Test if density patterns in range_a predict density in range_b.

    This is the key test: if high-density bins in range A are also
    high-density in range B, then the pattern is reproducible and
    potentially exploitable for prime search.

    Returns:
        (transfer_score, correlation)
    """
    # Import here to avoid circular dependency
    from run_autonomous_discovery import (
        compute_nd_coordinates, render_nd_visualization,
        fast_is_prime_mask, detect_patterns_nd
    )

    # Generate visualizations for both ranges
    numbers_a = np.arange(range_a[0], range_a[1], dtype=np.int64)
    primes_a = fast_is_prime_mask(numbers_a, prime_set)
    coords_a = compute_nd_coordinates(numbers_a, genome)
    _, target_a = render_nd_visualization(coords_a, primes_a, grid_size)

    numbers_b = np.arange(range_b[0], range_b[1], dtype=np.int64)
    primes_b = fast_is_prime_mask(numbers_b, prime_set)
    coords_b = compute_nd_coordinates(numbers_b, genome)
    _, target_b = render_nd_visualization(coords_b, primes_b, grid_size)

    # Get 2D representations
    if len(target_a.shape) == 3:
        target_a = np.max(target_a, axis=0)
        target_b = np.max(target_b, axis=0)

    # Remove patterns from both
    pattern_mask_a, _ = detect_patterns_nd(target_a, 2)
    pattern_mask_b, _ = detect_patterns_nd(target_b, 2)

    residual_a = (target_a > 0.5) & ~pattern_mask_a
    residual_b = (target_b > 0.5) & ~pattern_mask_b

    # Compute density maps
    density_a, _, _ = evaluate_residual_density(residual_a.astype(np.float32), bin_size)
    density_b, _, _ = evaluate_residual_density(residual_b.astype(np.float32), bin_size)

    # Flatten for correlation
    flat_a = density_a.flatten()
    flat_b = density_b.flatten()

    if len(flat_a) != len(flat_b):
        # Resize to match
        min_len = min(len(flat_a), len(flat_b))
        flat_a = flat_a[:min_len]
        flat_b = flat_b[:min_len]

    # Compute correlation
    if flat_a.std() > 0 and flat_b.std() > 0:
        correlation = np.corrcoef(flat_a, flat_b)[0, 1]
    else:
        correlation = 0.0

    # Compute transfer score
    # High bins in A should predict high bins in B
    n_bins = len(flat_a)
    if n_bins >= 4:
        high_bins_a = flat_a > np.percentile(flat_a, 75)
        density_b_in_high_a = flat_b[high_bins_a].mean() if high_bins_a.sum() > 0 else 0
        density_b_overall = flat_b.mean()

        transfer_score = density_b_in_high_a / density_b_overall if density_b_overall > 0 else 1.0
    else:
        transfer_score = 1.0

    return float(transfer_score), float(correlation)


def evaluate_residual(
    residual_image: np.ndarray,
    genome=None,
    prime_set: set = None,
    test_range: Tuple[int, int] = None,
    grid_size: int = 128
) -> ResidualEvaluationResult:
    """Comprehensive evaluation of residual patterns.

    This replaces the broken "run pattern detection again on residual" approach.
    Instead, we:
    1. Measure density variance (are there concentration/void regions?)
    2. Test if density patterns transfer to new range
    3. Score exploitability for prime search
    """
    binary = residual_image > 0.5
    total_primes = binary.sum()

    if total_primes < 20:
        return ResidualEvaluationResult(
            density_variance_ratio=1.0,
            concentration_fraction=0.0,
            void_fraction=0.0,
            transfer_score=1.0,
            transfer_correlation=0.0,
            exploitability_score=0.0,
            high_density_ratio=1.0,
            is_interesting=False,
            reason="Too few residual primes"
        )

    # 1. Density variance analysis
    density_map, variance_ratio, high_low_ratio = evaluate_residual_density(
        residual_image
    )

    # 2. Concentration/void detection
    conc_mask, void_mask, _ = detect_voids_and_concentrations(residual_image)
    concentration_fraction = conc_mask.sum() / total_primes
    void_fraction = void_mask.sum() / total_primes

    # 3. Transfer test (if genome and range provided)
    if genome is not None and prime_set is not None and test_range is not None:
        try:
            transfer_score, correlation = test_density_transfer(
                genome,
                range_a=(1, test_range[0]),
                range_b=test_range,
                prime_set=prime_set,
                grid_size=grid_size
            )
        except Exception:
            transfer_score = 1.0
            correlation = 0.0
    else:
        transfer_score = 1.0
        correlation = 0.0

    # 4. Compute exploitability score
    # High variance + high transfer + high ratio = exploitable
    exploitability = (
        0.3 * min(variance_ratio / 2.0, 1.0) +  # Variance contribution
        0.3 * min((high_low_ratio - 1.0) / 2.0, 1.0) +  # Ratio contribution
        0.4 * max(0, correlation)  # Transfer contribution
    )

    # 5. Determine if interesting
    is_interesting = (
        variance_ratio > 1.5 and  # More variance than random
        high_low_ratio > 1.3 and  # Meaningful density difference
        (correlation > 0.3 or transfer_score > 1.2)  # Pattern transfers
    )

    if is_interesting:
        reason = f"Variance {variance_ratio:.2f}x, density ratio {high_low_ratio:.2f}x, transfer r={correlation:.2f}"
    else:
        reasons = []
        if variance_ratio <= 1.5:
            reasons.append(f"low variance ({variance_ratio:.2f})")
        if high_low_ratio <= 1.3:
            reasons.append(f"low density ratio ({high_low_ratio:.2f})")
        if correlation <= 0.3 and transfer_score <= 1.2:
            reasons.append(f"no transfer (r={correlation:.2f})")
        reason = "Not interesting: " + ", ".join(reasons)

    return ResidualEvaluationResult(
        density_variance_ratio=variance_ratio,
        concentration_fraction=concentration_fraction,
        void_fraction=void_fraction,
        transfer_score=transfer_score,
        transfer_correlation=correlation,
        exploitability_score=exploitability,
        high_density_ratio=high_low_ratio,
        is_interesting=is_interesting,
        reason=reason
    )
