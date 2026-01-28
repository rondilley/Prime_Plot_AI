"""Pattern detection algorithms for prime visualizations.

Provides algorithms for detecting:
- Lines and curves (Hough transform)
- Clusters (density-based)
- Frequency patterns (FFT)
- Self-similarity (autocorrelation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import correlate2d


@dataclass
class DetectedLine:
    """Represents a detected line in an image.

    Attributes:
        rho: Distance from origin to line.
        theta: Angle of line normal (radians).
        strength: Accumulator value / confidence.
        num_points: Number of points on/near line.
    """
    rho: float
    theta: float
    strength: float
    num_points: int


@dataclass
class DetectedCluster:
    """Represents a detected cluster.

    Attributes:
        center_x: X coordinate of cluster center.
        center_y: Y coordinate of cluster center.
        size: Number of pixels in cluster.
        density: Points per unit area.
        extent: Bounding box (x_min, y_min, x_max, y_max).
    """
    center_x: float
    center_y: float
    size: int
    density: float
    extent: tuple[int, int, int, int]


def detect_lines(
    image: np.ndarray,
    threshold: float = 0.5,
    num_angles: int = 180,
    min_strength: float = 0.1,
    max_lines: int = 20
) -> tuple[list[DetectedLine], float]:
    """Detect lines using a simplified Hough transform.

    Args:
        image: 2D binary or grayscale image.
        threshold: Pixel threshold for binary conversion.
        num_angles: Number of angles to test.
        min_strength: Minimum relative strength to report.
        max_lines: Maximum number of lines to return.

    Returns:
        Tuple of (list of detected lines, overall line score).
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.float64)
    points = np.argwhere(binary > 0)

    if len(points) < 2:
        return [], 0.0

    h, w = binary.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))

    # Hough accumulator
    thetas = np.linspace(0, np.pi, num_angles, endpoint=False)
    rhos = np.arange(-diag, diag + 1)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.float64)

    # Precompute cos and sin
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Vote in accumulator
    for y, x in points:
        for t_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho = int(x * cos_t + y * sin_t) + diag
            if 0 <= rho < len(rhos):
                accumulator[rho, t_idx] += 1

    # Normalize
    max_acc = accumulator.max()
    if max_acc > 0:
        accumulator /= max_acc

    # Find peaks
    lines = []
    # Use non-maximum suppression
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(accumulator, size=neighborhood_size)
    peaks = (accumulator == data_max) & (accumulator >= min_strength)

    peak_coords = np.argwhere(peaks)

    # Sort by strength
    peak_strengths = [(accumulator[r, t], r, t) for r, t in peak_coords]
    peak_strengths.sort(reverse=True)

    for strength, r_idx, t_idx in peak_strengths[:max_lines]:
        rho_val = rhos[r_idx]
        theta_val = thetas[t_idx]

        # Count points near this line
        num_points = 0
        for y, x in points:
            rho_point = x * cos_thetas[t_idx] + y * sin_thetas[t_idx]
            if abs(rho_point - rho_val) < 2:  # Within 2 pixels
                num_points += 1

        lines.append(DetectedLine(
            rho=float(rho_val),
            theta=float(theta_val),
            strength=float(strength),
            num_points=num_points,
        ))

    # Overall score based on strongest lines
    if lines:
        # Score = sum of top line strengths weighted by point coverage
        total_points = len(points)
        score = sum(
            line.strength * (line.num_points / total_points)
            for line in lines[:5]
        ) / min(5, len(lines))
    else:
        score = 0.0

    return lines, score


def detect_clusters(
    image: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 5,
    max_clusters: int = 50
) -> tuple[list[DetectedCluster], float]:
    """Detect clusters using connected component analysis.

    Args:
        image: 2D binary or grayscale image.
        threshold: Pixel threshold for binary conversion.
        min_size: Minimum cluster size in pixels.
        max_clusters: Maximum clusters to return.

    Returns:
        Tuple of (list of detected clusters, cluster quality score).
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.int32)

    if binary.sum() == 0:
        return [], 0.0

    # Apply slight dilation to connect nearby points
    structure = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(binary, structure=structure, iterations=1)

    # Find connected components
    labeled, num_features = ndimage.label(dilated)

    if num_features == 0:
        return [], 0.0

    clusters = []

    for i in range(1, num_features + 1):
        mask = labeled == i
        # Get original points in this cluster region
        original_points = binary & mask
        size = original_points.sum()

        if size < min_size:
            continue

        # Find cluster properties
        coords = np.argwhere(original_points)
        center_y = coords[:, 0].mean()
        center_x = coords[:, 1].mean()

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Density = points / bounding box area
        bbox_area = max((y_max - y_min + 1) * (x_max - x_min + 1), 1)
        density = size / bbox_area

        clusters.append(DetectedCluster(
            center_x=float(center_x),
            center_y=float(center_y),
            size=int(size),
            density=float(density),
            extent=(int(x_min), int(y_min), int(x_max), int(y_max)),
        ))

    # Sort by size
    clusters.sort(key=lambda c: c.size, reverse=True)
    clusters = clusters[:max_clusters]

    # Score based on cluster quality
    if clusters:
        # Good clustering = many well-defined clusters with consistent density
        avg_density = np.mean([c.density for c in clusters])
        density_consistency = 1 - np.std([c.density for c in clusters]) / (avg_density + 1e-6)
        density_consistency = max(0, density_consistency)

        # Coverage = fraction of points in clusters
        total_points = binary.sum()
        clustered_points = sum(c.size for c in clusters)
        coverage = clustered_points / total_points if total_points > 0 else 0

        # Number of clusters (more = more structured, up to a point)
        num_score = min(len(clusters) / 20, 1.0)

        score = (avg_density * 0.3 + density_consistency * 0.3 +
                coverage * 0.2 + num_score * 0.2)
    else:
        score = 0.0

    return clusters, score


def compute_fft_spectrum(
    image: np.ndarray,
    threshold: float = 0.5
) -> tuple[np.ndarray, float, list[tuple[float, float, float]]]:
    """Compute 2D FFT and analyze frequency components.

    Strong peaks in FFT indicate periodic/regular patterns.

    Args:
        image: 2D image.
        threshold: Pixel threshold for preprocessing.

    Returns:
        Tuple of (magnitude spectrum, peak strength score, list of (freq_x, freq_y, magnitude)).
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    # Apply threshold
    img = (img > threshold).astype(np.float64)

    # Zero-pad to power of 2 for efficiency
    h, w = img.shape
    new_h = 2 ** int(np.ceil(np.log2(h)))
    new_w = 2 ** int(np.ceil(np.log2(w)))
    padded = np.zeros((new_h, new_w))
    padded[:h, :w] = img

    # Compute 2D FFT
    fft = np.fft.fft2(padded)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Normalize (ignore DC component)
    center_y, center_x = new_h // 2, new_w // 2
    magnitude[center_y, center_x] = 0  # Zero out DC

    max_mag = magnitude.max()
    if max_mag > 0:
        magnitude_norm = magnitude / max_mag
    else:
        return magnitude, 0.0, []

    # Find peaks (strong frequency components)
    peak_threshold = 0.1
    peaks = []

    # Use non-maximum suppression
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(magnitude_norm, size=neighborhood_size)
    peak_mask = (magnitude_norm == data_max) & (magnitude_norm >= peak_threshold)
    peak_coords = np.argwhere(peak_mask)

    for y, x in peak_coords:
        # Convert to frequency
        freq_y = (y - center_y) / new_h
        freq_x = (x - center_x) / new_w
        mag = magnitude_norm[y, x]
        peaks.append((freq_x, freq_y, mag))

    # Sort by magnitude
    peaks.sort(key=lambda p: p[2], reverse=True)
    peaks = peaks[:20]  # Top 20 peaks

    # Score based on peak concentration
    # Strong periodic patterns have few dominant peaks
    if peaks:
        top_5_energy = sum(p[2] for p in peaks[:5])
        total_energy = magnitude_norm.sum()
        if total_energy > 0:
            concentration = top_5_energy / (total_energy / (new_h * new_w) * 25 + 1e-6)
            score = min(concentration / 10, 1.0)
        else:
            score = 0.0
    else:
        score = 0.0

    return magnitude, score, peaks


def compute_autocorrelation(
    image: np.ndarray,
    threshold: float = 0.5,
    max_lag: int | None = None
) -> tuple[np.ndarray, float]:
    """Compute 2D autocorrelation to measure self-similarity.

    Repeating patterns show strong off-center peaks in autocorrelation.

    Args:
        image: 2D image.
        threshold: Pixel threshold for preprocessing.
        max_lag: Maximum lag to compute. Default: min(dim) // 4.

    Returns:
        Tuple of (autocorrelation image, self-similarity score).
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    img = (img > threshold).astype(np.float64)

    # Subtract mean
    img = img - img.mean()

    h, w = img.shape
    if max_lag is None:
        max_lag = min(h, w) // 4

    # Compute autocorrelation using correlate2d
    # This is slow for large images, so we may crop
    if h > 200 or w > 200:
        # Use center crop for large images
        crop_h = min(200, h)
        crop_w = min(200, w)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
        h, w = img.shape
        max_lag = min(max_lag, min(h, w) // 4)

    # Compute autocorrelation
    autocorr = correlate2d(img, img, mode='same')

    # Normalize
    center_val = autocorr[h // 2, w // 2]
    if center_val > 0:
        autocorr_norm = autocorr / center_val
    else:
        return autocorr, 0.0

    # Find off-center peaks
    center_y, center_x = h // 2, w // 2

    # Mask out center region
    y_coords, x_coords = np.ogrid[:h, :w]
    center_mask = ((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2) < (max_lag // 4) ** 2
    autocorr_masked = autocorr_norm.copy()
    autocorr_masked[center_mask] = 0

    # Find peaks in masked autocorrelation
    peak_threshold = 0.2
    data_max = ndimage.maximum_filter(autocorr_masked, size=5)
    peaks = (autocorr_masked == data_max) & (autocorr_masked >= peak_threshold)

    peak_coords = np.argwhere(peaks)
    peak_values = [autocorr_norm[y, x] for y, x in peak_coords]

    # Score based on off-center peak strength
    if peak_values:
        # Higher off-center peaks = more self-similar pattern
        max_off_center = max(peak_values)
        avg_off_center = np.mean(peak_values)
        score = (max_off_center * 0.6 + avg_off_center * 0.4)
    else:
        score = 0.0

    return autocorr_norm, score


def detect_radial_patterns(
    image: np.ndarray,
    threshold: float = 0.5,
    num_rays: int = 36
) -> tuple[np.ndarray, float]:
    """Detect radial patterns emanating from center.

    Useful for Vogel and Sacks spirals where primes align on rays.

    Args:
        image: 2D image.
        threshold: Pixel threshold.
        num_rays: Number of angular bins.

    Returns:
        Tuple of (ray density array, radial pattern score).
    """
    img = image.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > threshold).astype(np.float64)
    points = np.argwhere(binary > 0)

    if len(points) < 10:
        return np.zeros(num_rays), 0.0

    h, w = img.shape
    center_y, center_x = h // 2, w // 2

    # Compute angle for each point
    angles = np.arctan2(points[:, 0] - center_y, points[:, 1] - center_x)
    angles = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1)

    # Bin into rays
    ray_counts = np.zeros(num_rays)
    for angle in angles:
        bin_idx = int(angle * num_rays) % num_rays
        ray_counts[bin_idx] += 1

    # Normalize
    if ray_counts.sum() > 0:
        ray_density = ray_counts / ray_counts.sum()
    else:
        ray_density = ray_counts

    # Score based on variance of ray density
    # High variance = points concentrated on certain rays
    if ray_density.std() > 0:
        cv = ray_density.std() / ray_density.mean()
        score = min(cv * 2, 1.0)  # CV of 0.5 = score of 1.0
    else:
        score = 0.0

    return ray_density, score
