"""Pattern quality metrics for prime visualizations.

Provides quantitative measures of pattern strength, including:
- Line density: How strongly primes align along diagonals/curves
- Cluster coherence: How well-defined prime clusters are
- Entropy: Information content / randomness measure
- Signal-to-noise ratio: Pattern strength vs random baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy


@dataclass
class PatternMetrics:
    """Container for all pattern quality metrics.

    Attributes:
        name: Name of the visualization method.
        line_density: Score for linear pattern strength (0-1).
        diagonal_density: Score for diagonal patterns specifically.
        cluster_coherence: How well-defined clusters are (0-1).
        entropy: Shannon entropy of the image.
        entropy_ratio: Entropy relative to random baseline.
        snr: Signal-to-noise ratio in dB.
        snr_linear: Linear signal-to-noise ratio.
        fft_peak_strength: Strength of dominant frequency components.
        autocorr_score: Self-similarity measure.
        sparsity: Fraction of non-zero pixels.
        ml_learnability: Optional ML-based learnability score.
        raw_scores: Dictionary of additional raw scores.
    """
    name: str
    line_density: float = 0.0
    diagonal_density: float = 0.0
    cluster_coherence: float = 0.0
    entropy: float = 0.0
    entropy_ratio: float = 1.0
    snr: float = 0.0
    snr_linear: float = 1.0
    fft_peak_strength: float = 0.0
    autocorr_score: float = 0.0
    sparsity: float = 0.0
    ml_learnability: float | None = None
    raw_scores: dict[str, Any] = field(default_factory=dict)

    def overall_score(self, weights: dict[str, float] | None = None) -> float:
        """Calculate weighted overall pattern quality score.

        Args:
            weights: Optional weight dictionary. Keys should match attribute names.
                     Default weights emphasize SNR and line density.

        Returns:
            Weighted score between 0 and 1.
        """
        if weights is None:
            weights = {
                'snr_linear': 0.25,
                'line_density': 0.20,
                'diagonal_density': 0.15,
                'cluster_coherence': 0.15,
                'fft_peak_strength': 0.10,
                'autocorr_score': 0.10,
                'entropy_ratio': 0.05,
            }

        score = 0.0
        total_weight = 0.0

        for attr, weight in weights.items():
            value = getattr(self, attr, None)
            if value is not None:
                # Normalize SNR to 0-1 range (assume max useful SNR ~20dB)
                if attr == 'snr':
                    value = min(max(value / 20, 0), 1)
                elif attr == 'snr_linear':
                    value = min(max((value - 1) / 10, 0), 1)
                elif attr == 'entropy_ratio':
                    # Lower entropy ratio = more structured = better
                    value = max(0, 1 - value)

                score += weight * value
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'line_density': self.line_density,
            'diagonal_density': self.diagonal_density,
            'cluster_coherence': self.cluster_coherence,
            'entropy': self.entropy,
            'entropy_ratio': self.entropy_ratio,
            'snr': self.snr,
            'snr_linear': self.snr_linear,
            'fft_peak_strength': self.fft_peak_strength,
            'autocorr_score': self.autocorr_score,
            'sparsity': self.sparsity,
            'ml_learnability': self.ml_learnability,
            'overall_score': self.overall_score(),
        }


def calculate_line_density(
    image: np.ndarray,
    angles: list[float] | None = None,
    threshold: float = 0.5
) -> tuple[float, dict[str, float]]:
    """Calculate how strongly points align along lines at various angles.

    Uses Radon transform / projection approach to find line patterns.

    Args:
        image: 2D binary or grayscale image.
        angles: List of angles (degrees) to check. Default: 0, 45, 90, 135.
        threshold: Minimum pixel value to consider as "on".

    Returns:
        Tuple of (overall_score, per_angle_scores).
    """
    if angles is None:
        angles = [0, 45, 90, 135]

    # Normalize image
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    # Threshold to binary
    binary = (img > threshold).astype(np.float64)

    if binary.sum() == 0:
        return 0.0, {f"angle_{a}": 0.0 for a in angles}

    scores = {}
    h, w = binary.shape

    for angle in angles:
        # Rotate image and project onto vertical axis
        rotated = ndimage.rotate(binary, angle, reshape=True, order=0)

        # Sum along columns (projection)
        projection = rotated.sum(axis=0)

        # Score based on variance of projection
        # High variance = points concentrated on certain lines
        if projection.sum() > 0:
            projection_norm = projection / projection.sum()
            # Use coefficient of variation
            mean_proj = projection.mean()
            if mean_proj > 0:
                cv = projection.std() / mean_proj
                # Normalize to 0-1 range (empirically, CV > 2 is very strong)
                score = min(cv / 2.0, 1.0)
            else:
                score = 0.0
        else:
            score = 0.0

        scores[f"angle_{angle}"] = score

    # Overall score is max of all angles (best alignment direction)
    overall = max(scores.values()) if scores else 0.0

    return overall, scores


def calculate_diagonal_density(image: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate density of points along main diagonals.

    Diagonals are significant in Ulam spirals where primes cluster.

    Args:
        image: 2D binary or grayscale image.
        threshold: Minimum pixel value to consider.

    Returns:
        Score between 0 and 1.
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.float64)
    if binary.sum() == 0:
        return 0.0

    h, w = binary.shape
    total_on = binary.sum()

    # Extract main diagonals
    diag_sum = 0
    diag_pixels = 0

    # Main diagonal and nearby
    for offset in range(-min(h, w) // 10, min(h, w) // 10 + 1):
        diag = np.diag(binary, k=offset)
        diag_sum += diag.sum()
        diag_pixels += len(diag)

    # Anti-diagonal and nearby
    flipped = np.fliplr(binary)
    for offset in range(-min(h, w) // 10, min(h, w) // 10 + 1):
        diag = np.diag(flipped, k=offset)
        diag_sum += diag.sum()
        diag_pixels += len(diag)

    # Compare diagonal density to overall density
    if diag_pixels == 0 or total_on == 0:
        return 0.0

    diagonal_density = diag_sum / diag_pixels
    overall_density = total_on / (h * w)

    # Ratio of diagonal to overall density
    # Value > 1 means diagonals are denser than average
    if overall_density > 0:
        ratio = diagonal_density / overall_density
        # Normalize: ratio of 2 = score of 1
        return min((ratio - 1) / 1.0, 1.0) if ratio > 1 else 0.0

    return 0.0


def calculate_cluster_coherence(
    image: np.ndarray,
    threshold: float = 0.5,
    min_cluster_size: int = 3
) -> float:
    """Measure how well-defined clusters are in the image.

    Uses connected component analysis and cluster shape metrics.

    Args:
        image: 2D binary or grayscale image.
        threshold: Minimum pixel value for binary conversion.
        min_cluster_size: Minimum pixels to count as a cluster.

    Returns:
        Coherence score between 0 and 1.
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.int32)

    if binary.sum() == 0:
        return 0.0

    # Find connected components
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        return 0.0

    # Analyze cluster properties
    cluster_sizes = []
    cluster_compactness = []

    for i in range(1, num_features + 1):
        mask = labeled == i
        size = mask.sum()

        if size < min_cluster_size:
            continue

        cluster_sizes.append(size)

        # Compactness = 4*pi*area / perimeter^2
        # For a circle, compactness = 1; for irregular shapes, < 1
        # Use erosion to estimate perimeter
        eroded = ndimage.binary_erosion(mask)
        perimeter = mask.sum() - eroded.sum()

        if perimeter > 0:
            compactness = 4 * np.pi * size / (perimeter ** 2)
            compactness = min(compactness, 1.0)  # Cap at 1
            cluster_compactness.append(compactness)

    if not cluster_sizes:
        return 0.0

    # Score based on:
    # 1. Number of significant clusters (more = more structured)
    # 2. Size consistency (similar sizes = more coherent)
    # 3. Compactness (round clusters = more coherent)

    num_clusters = len(cluster_sizes)
    size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
    avg_compactness = np.mean(cluster_compactness) if cluster_compactness else 0

    # Combine metrics
    # Many clusters of consistent size and shape = high coherence
    cluster_score = min(num_clusters / 50, 1.0)  # Cap at 50 clusters
    consistency_score = max(0, 1 - size_cv / 2)  # Lower CV = better
    compactness_score = avg_compactness

    coherence = (cluster_score * 0.3 + consistency_score * 0.3 + compactness_score * 0.4)

    return coherence


def calculate_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Calculate Shannon entropy of the image.

    Lower entropy indicates more structure/predictability.

    Args:
        image: 2D array.
        bins: Number of histogram bins.

    Returns:
        Entropy value (higher = more random).
    """
    img = image.flatten().astype(np.float64)

    if img.max() > 1:
        img = img / 255.0

    # Create histogram
    hist, _ = np.histogram(img, bins=bins, range=(0, 1), density=True)
    hist = hist[hist > 0]  # Remove zero bins

    if len(hist) == 0:
        return 0.0

    # Shannon entropy
    return scipy_entropy(hist, base=2)


def calculate_snr(
    signal_image: np.ndarray,
    noise_image: np.ndarray | None = None,
    method: str = "variance"
) -> tuple[float, float]:
    """Calculate signal-to-noise ratio.

    Args:
        signal_image: Image with patterns (prime visualization).
        noise_image: Random baseline image. If None, uses image statistics.
        method: 'variance' or 'peak' method.

    Returns:
        Tuple of (snr_db, snr_linear).
    """
    signal = signal_image.astype(np.float64)
    if signal.max() > 1:
        signal = signal / 255.0

    if noise_image is not None:
        noise = noise_image.astype(np.float64)
        if noise.max() > 1:
            noise = noise / 255.0
    else:
        noise = None

    if method == "variance":
        # SNR based on variance ratio
        signal_var = signal.var()

        if noise is not None:
            noise_var = noise.var()
        else:
            # Estimate noise as local variance
            # Use Laplacian to estimate noise
            laplacian = ndimage.laplace(signal)
            noise_var = (laplacian.var() / 2)  # Approximate

        if noise_var > 0:
            snr_linear = signal_var / noise_var
        else:
            snr_linear = float('inf') if signal_var > 0 else 1.0

    else:  # peak method
        signal_peak = signal.max()
        if noise is not None:
            noise_std = noise.std()
        else:
            noise_std = signal.std() * 0.1  # Rough estimate

        if noise_std > 0:
            snr_linear = signal_peak / noise_std
        else:
            snr_linear = float('inf') if signal_peak > 0 else 1.0

    # Convert to dB
    if snr_linear > 0 and snr_linear != float('inf'):
        snr_db = 10 * np.log10(snr_linear)
    elif snr_linear == float('inf'):
        snr_db = 100.0  # Cap at 100 dB
    else:
        snr_db = 0.0

    return snr_db, snr_linear


def calculate_sparsity(image: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate image sparsity (fraction of non-zero pixels).

    Args:
        image: 2D array.
        threshold: Value above which pixel is "on".

    Returns:
        Sparsity value between 0 and 1.
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    return (img > threshold).mean()


def compute_all_metrics(
    image: np.ndarray,
    name: str,
    baseline_image: np.ndarray | None = None
) -> PatternMetrics:
    """Compute all pattern metrics for an image.

    Args:
        image: Prime visualization image.
        name: Name of the visualization method.
        baseline_image: Optional random baseline for comparison.

    Returns:
        PatternMetrics with all scores populated.
    """
    # Line density
    line_score, line_details = calculate_line_density(image)

    # Diagonal density
    diag_score = calculate_diagonal_density(image)

    # Cluster coherence
    cluster_score = calculate_cluster_coherence(image)

    # Entropy
    img_entropy = calculate_entropy(image)
    if baseline_image is not None:
        baseline_entropy = calculate_entropy(baseline_image)
        entropy_ratio = img_entropy / baseline_entropy if baseline_entropy > 0 else 1.0
    else:
        # Estimate max entropy for binary image
        sparsity = calculate_sparsity(image)
        if 0 < sparsity < 1:
            max_entropy = -sparsity * np.log2(sparsity) - (1 - sparsity) * np.log2(1 - sparsity)
            entropy_ratio = img_entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            entropy_ratio = 1.0

    # SNR
    snr_db, snr_linear = calculate_snr(image, baseline_image)

    # Sparsity
    sparsity = calculate_sparsity(image)

    return PatternMetrics(
        name=name,
        line_density=line_score,
        diagonal_density=diag_score,
        cluster_coherence=cluster_score,
        entropy=img_entropy,
        entropy_ratio=entropy_ratio,
        snr=snr_db,
        snr_linear=snr_linear,
        sparsity=sparsity,
        raw_scores={'line_angles': line_details},
    )
