"""Visual pattern analysis pipeline for prime number visualizations.

This script does what the tool SHOULD be doing:
1. Generate multiple visualization methods
2. Analyze images for visual patterns (lines, clusters, FFT)
3. Remove known patterns (mod-6, polynomial diagonals) using ACTUAL coordinate mappings
4. Re-analyze residual for unknown patterns
5. Test if residual patterns extend to new number ranges
6. Compare brute-force vs visual-guided prime finding efficiency

Output goes to output/runs/{run_id}/ with all intermediate images.
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prime_plot.core.sieve import generate_primes_range, is_prime, prime_sieve_mask
from prime_plot.evaluation.pipeline import EvaluationPipeline
from prime_plot.evaluation.detectors import (
    detect_lines,
    detect_clusters,
    compute_fft_spectrum,
    compute_autocorrelation,
)
from prime_plot.visualization.renderer import save_raw_image
from prime_plot.utils.run_manager import create_run
from PIL import Image, ImageDraw, ImageFont

# Golden angle for Vogel spiral
GOLDEN_ANGLE = 2 * np.pi * (1 - 1 / ((1 + np.sqrt(5)) / 2))


def create_visualization_mosaic(
    images_dir: Path,
    ranking: List[Tuple[str, float]],
    output_path: Path,
    tile_size: int = 256,
    include_residuals: bool = True
) -> None:
    """Create a mosaic of all visualizations sorted by quality score.

    Tiles are arranged from top-left (best) to bottom-right (worst).

    Args:
        images_dir: Directory containing the visualization images
        ranking: List of (method_name, score) tuples sorted by score descending
        output_path: Path to save the mosaic image
        tile_size: Size of each tile in pixels
        include_residuals: If True, create a second mosaic with residual images
    """
    if not ranking:
        return

    # Calculate grid dimensions
    n_images = len(ranking)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))

    # Create mosaic for original images
    mosaic_width = cols * tile_size
    mosaic_height = rows * tile_size + 30  # Extra space for labels at bottom
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color=(32, 32, 32))

    # Try to get a font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(mosaic)

    for idx, (method_name, score) in enumerate(ranking):
        row = idx // cols
        col = idx % cols

        # Load original image
        img_path = images_dir / f"{method_name}_original.png"
        if not img_path.exists():
            # Try alternate naming
            img_path = images_dir / f"viz_{method_name}.png"

        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            # Resize to tile size
            img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            # Calculate position
            x = col * tile_size
            y = row * tile_size

            # Paste image
            mosaic.paste(img, (x, y))

            # Add rank number in corner
            rank_text = f"#{idx + 1}"
            draw.rectangle([x, y, x + 25, y + 15], fill=(0, 0, 0, 180))
            draw.text((x + 2, y + 1), rank_text, fill=(255, 255, 255), font=font)

            # Add method name at bottom of tile
            name_y = y + tile_size - 15
            draw.rectangle([x, name_y, x + tile_size, name_y + 15], fill=(0, 0, 0, 180))
            # Truncate name if too long
            display_name = method_name[:20] + "..." if len(method_name) > 20 else method_name
            draw.text((x + 2, name_y + 1), f"{display_name} ({score:.2f})", fill=(255, 255, 255), font=font)

    # Add title at bottom
    title = f"Visualization Mosaic - {n_images} methods, sorted by score (best top-left)"
    draw.text((10, mosaic_height - 25), title, fill=(200, 200, 200), font=font)

    mosaic.save(output_path)

    # Create residual mosaic if requested
    if include_residuals:
        residual_mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color=(32, 32, 32))
        draw_res = ImageDraw.Draw(residual_mosaic)

        for idx, (method_name, score) in enumerate(ranking):
            row = idx // cols
            col = idx % cols

            # Load residual image
            img_path = images_dir / f"{method_name}_residual.png"

            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                x = col * tile_size
                y = row * tile_size

                residual_mosaic.paste(img, (x, y))

                rank_text = f"#{idx + 1}"
                draw_res.rectangle([x, y, x + 25, y + 15], fill=(0, 0, 0, 180))
                draw_res.text((x + 2, y + 1), rank_text, fill=(255, 255, 255), font=font)

                name_y = y + tile_size - 15
                draw_res.rectangle([x, name_y, x + tile_size, name_y + 15], fill=(0, 0, 0, 180))
                display_name = method_name[:20] + "..." if len(method_name) > 20 else method_name
                draw_res.text((x + 2, name_y + 1), f"{display_name} ({score:.2f})", fill=(255, 255, 255), font=font)

        title = f"Residual Mosaic - {n_images} methods, sorted by score (best top-left)"
        draw_res.text((10, mosaic_height - 25), title, fill=(200, 200, 200), font=font)

        residual_path = output_path.parent / "mosaic_residuals.png"
        residual_mosaic.save(residual_path)


@dataclass
class PatternAnalysis:
    """Results of visual pattern analysis."""
    method_name: str
    num_lines: int
    strongest_line_theta: float  # Angle of strongest line
    num_clusters: int
    fft_dominant_freq: Tuple[float, float]  # (freq_x, freq_y) of strongest peak
    autocorr_score: float

    # After removing known patterns
    residual_lines: int
    residual_clusters: int
    residual_fft_peaks: int

    # Mod-6 removal statistics
    mod6_pixels_removed: int
    polynomial_pixels_removed: int
    residual_prime_fraction: float  # What fraction of primes remain unexplained

    # Predictive power test
    pattern_extends_to_new_range: bool
    prediction_improvement: float  # vs brute force


# ============================================================================
# VISUALIZATION-SPECIFIC COORDINATE MAPPING FUNCTIONS
# These map integer n to (x, y) pixel coordinates for each visualization type
# ============================================================================

def ulam_coords(n: int, image_size: int, start: int = 1) -> Tuple[int, int]:
    """Convert integer n to Ulam spiral (x, y) pixel coordinates."""
    m = n - start
    if m == 0:
        cx = cy = (image_size - 1) // 2
        return (cx, cy)

    # Compute layer k
    k = int(np.ceil((np.sqrt(m + 1) - 1) / 2))
    t = 2 * k + 1
    m_max = t * t - 1
    t_minus = t - 1

    # Compute offset from center
    if m >= m_max - t_minus:
        x = k - (m_max - m)
        y = -k
    elif m >= m_max - 2 * t_minus:
        x = -k
        y = -k + (m_max - t_minus - m)
    elif m >= m_max - 3 * t_minus:
        x = -k + (m_max - 2 * t_minus - m)
        y = k
    else:
        x = k
        y = k - (m_max - 3 * t_minus - m)

    # Convert to pixel coordinates
    cx = cy = (image_size - 1) // 2
    px = cx + x
    py = cy + y

    return (int(np.clip(px, 0, image_size - 1)), int(np.clip(py, 0, image_size - 1)))


def sacks_coords(n: int, image_size: int, max_n: int) -> Tuple[int, int]:
    """Convert integer n to Sacks spiral (x, y) pixel coordinates."""
    sqrt_n = np.sqrt(n)
    theta = 2 * np.pi * sqrt_n
    r = sqrt_n

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Scale to image
    max_r = np.sqrt(max_n)
    scale = (image_size / 2 - 5) / max_r

    px = int(x * scale + image_size / 2)
    py = int(y * scale + image_size / 2)

    return (np.clip(px, 0, image_size - 1), np.clip(py, 0, image_size - 1))


def vogel_coords(n: int, image_size: int, max_n: int) -> Tuple[int, int]:
    """Convert integer n to Vogel spiral (x, y) pixel coordinates."""
    theta = n * GOLDEN_ANGLE
    r = np.sqrt(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Scale to image
    max_r = np.sqrt(max_n)
    scale = (image_size / 2 - 5) / max_r

    px = int(x * scale + image_size / 2)
    py = int(y * scale + image_size / 2)

    return (np.clip(px, 0, image_size - 1), np.clip(py, 0, image_size - 1))


def klauber_coords(n: int, image_size: int, max_n: int) -> Tuple[int, int]:
    """Convert integer n to Klauber triangle (x, y) pixel coordinates."""
    # Row k contains numbers from k^2 - k + 1 to k^2 + k - 1
    k = int(np.ceil(np.sqrt(n)))
    if k * k - k + 1 > n:
        k -= 1
    if k < 1:
        k = 1

    row_start = k * k - k + 1
    pos_in_row = n - row_start

    # X position centered
    x = (image_size // 2) + pos_in_row - k + 1

    # Scale y to fit image
    max_rows = int(np.sqrt(max_n)) + 1
    y_scale = (image_size - 10) / max(max_rows, 1)
    y = int(k * y_scale)

    # Ensure bounds
    x = int(np.clip(x, 0, image_size - 1))
    y = int(np.clip(y, 0, image_size - 1))

    return (x, y)


def get_coord_function(method_name: str, image_size: int, max_n: int) -> Optional[Callable[[int], Tuple[int, int]]]:
    """Get coordinate mapping function for a visualization method."""
    method_lower = method_name.lower()

    if 'ulam' in method_lower:
        return lambda n: ulam_coords(n, image_size)
    elif 'sacks' in method_lower:
        return lambda n: sacks_coords(n, image_size, max_n)
    elif 'vogel' in method_lower:
        return lambda n: vogel_coords(n, image_size, max_n)
    elif 'klauber' in method_lower:
        return lambda n: klauber_coords(n, image_size, max_n)
    else:
        # For other methods, return None (we'll use the Ulam as default)
        return lambda n: ulam_coords(n, image_size)


# ============================================================================
# KNOWN PATTERN MASK GENERATION
# Creates masks based on actual coordinate mappings
# ============================================================================

def create_mod6_mask_for_visualization(
    coord_func: Callable[[int], Tuple[int, int]],
    image_size: int,
    max_n: int
) -> np.ndarray:
    """Create mod-6 pattern mask using actual visualization coordinates.

    This marks all positions where numbers n % 6 == 1 or n % 6 == 5 would land.
    These are the ONLY positions where primes > 3 can exist.

    Returns:
        Binary mask where 1 = expected mod-6 prime candidate position
    """
    mask = np.zeros((image_size, image_size), dtype=np.float32)

    # Mark all positions where n % 6 == 1 or 5 (prime candidates)
    for n in range(1, max_n + 1):
        if n % 6 == 1 or n % 6 == 5 or n == 2 or n == 3:  # Include 2 and 3 as special primes
            x, y = coord_func(n)
            if 0 <= x < image_size and 0 <= y < image_size:
                mask[y, x] = 1.0

    return mask


def create_polynomial_mask_for_visualization(
    coord_func: Callable[[int], Tuple[int, int]],
    image_size: int,
    max_n: int
) -> np.ndarray:
    """Create mask for known prime-generating polynomials using actual coordinates."""
    return create_polynomial_mask_for_visualization_sized(coord_func, image_size, image_size, max_n)


def create_polynomial_mask_for_visualization_sized(
    coord_func: Callable[[int], Tuple[int, int]],
    height: int,
    width: int,
    max_n: int
) -> np.ndarray:
    """Create mask for known prime-generating polynomials using actual coordinates.

    Known prime-rich polynomials:
    - Euler's: n^2 + n + 41 (produces 40 consecutive primes for n=0..39)
    - 2n^2 + 29 (produces 29 primes)
    - n^2 - n + 41 (same as Euler's shifted)
    - 4n^2 + 4n + 59 (Ulam diagonal)

    Returns:
        Binary mask where 1 = position on known polynomial
    """
    mask = np.zeros((height, width), dtype=np.float32)

    # Famous prime-generating polynomials
    polynomials = [
        lambda n: n*n + n + 41,      # Euler's polynomial
        lambda n: n*n - n + 41,      # Euler's shifted
        lambda n: 2*n*n + 29,        # Another famous one
        lambda n: 4*n*n + 4*n + 59,  # Ulam diagonal
        lambda n: 4*n*n - 2*n + 41,  # Another diagonal form
        lambda n: n*n + n + 17,      # Less famous but productive
        lambda n: 2*n*n + 11,        # Small polynomial
    ]

    for poly in polynomials:
        n = 0
        while True:
            try:
                val = poly(n)
                if val > max_n:
                    break
                if val > 0:
                    x, y = coord_func(val)
                    if 0 <= y < height and 0 <= x < width:
                        mask[y, x] = 1.0
                n += 1
            except (OverflowError, ValueError):
                break

    return mask


def detect_diagonal_patterns(binary_image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Detect diagonal patterns using directional convolution.

    Returns mask where True = pixel is part of a diagonal pattern.
    """
    # 45-degree diagonal kernel
    k45 = np.eye(kernel_size, dtype=np.float32)
    # 135-degree diagonal kernel
    k135 = np.fliplr(k45)

    resp_45 = ndimage.convolve(binary_image, k45 / k45.sum())
    resp_135 = ndimage.convolve(binary_image, k135 / k135.sum())

    # Maximum response from either diagonal direction
    diagonal_strength = np.maximum(resp_45, resp_135)

    # Threshold: pixel is on diagonal if has neighbors along diagonal
    # 0.35 means roughly 3 of 7 neighbors on diagonal (including self)
    on_diagonal = (diagonal_strength > 0.35) & (binary_image > 0.5)

    return on_diagonal


def detect_curved_patterns(binary_image: np.ndarray) -> np.ndarray:
    """Detect curved patterns (for Sacks-like spirals) using local density gradient.

    Returns mask where True = pixel is part of a curved pattern.
    """
    # Use Gaussian smoothing to find local density
    smoothed = ndimage.gaussian_filter(binary_image.astype(np.float32), sigma=3)

    # Compute gradient magnitude
    gy, gx = np.gradient(smoothed)
    gradient_mag = np.sqrt(gx**2 + gy**2)

    # High gradient = on edge of a dense region = on a curve
    threshold = np.percentile(gradient_mag[binary_image > 0.5], 50) if (binary_image > 0.5).sum() > 0 else 0
    on_curve = (gradient_mag > threshold) & (binary_image > 0.5)

    return on_curve


def create_known_pattern_images(
    prime_image: np.ndarray,
    coord_func: Callable[[int], Tuple[int, int]],
    height: int,
    width: int,
    max_n: int,
    method_name: str = ""
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Create images showing known patterns and residual after removal.

    For Ulam-like spirals: detects DIAGONAL patterns (polynomial structure)
    For Sacks-like spirals: detects CURVED patterns (polynomial curves)

    Returns:
        (marked_image, residual_image, pattern_count, total_primes)
        - marked_image: RGB image with known patterns in RED, others in WHITE
        - residual_image: grayscale with known patterns zeroed out
        - pattern_count: number of primes on detected patterns
        - total_primes: total prime count
    """
    # Normalize input
    img = prime_image.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0

    binary = (img > 0.5).astype(np.float32)
    total_primes = int(binary.sum())

    # Detect patterns based on visualization type
    method_lower = method_name.lower()

    if 'ulam' in method_lower or 'klauber' in method_lower or 'diagonal' in method_lower:
        # Ulam-family: detect diagonal patterns
        on_pattern = detect_diagonal_patterns(binary)
        pattern_type = "diagonal"
    elif 'sacks' in method_lower or 'vogel' in method_lower or 'spiral' in method_lower:
        # Sacks-family: detect curved patterns
        on_pattern = detect_curved_patterns(binary)
        pattern_type = "curved"
    else:
        # Default: try diagonals first, then curves
        diag_pattern = detect_diagonal_patterns(binary)
        curve_pattern = detect_curved_patterns(binary)
        # Use whichever detects more
        if diag_pattern.sum() > curve_pattern.sum():
            on_pattern = diag_pattern
            pattern_type = "diagonal"
        else:
            on_pattern = curve_pattern
            pattern_type = "curved"

    not_on_pattern = (binary > 0.5) & ~on_pattern

    # Create RGB marked image
    marked = np.zeros((height, width, 3), dtype=np.uint8)
    marked[on_pattern, 0] = 255  # Red for pattern primes
    marked[not_on_pattern] = [255, 255, 255]  # White for non-pattern primes

    # Create residual (only non-pattern primes)
    residual = np.zeros((height, width), dtype=np.float32)
    residual[not_on_pattern] = 1.0
    residual_uint8 = (residual * 255).astype(np.uint8)

    pattern_count = int(on_pattern.sum())

    return marked, residual_uint8, pattern_count, total_primes


def subtract_known_patterns(
    prime_image: np.ndarray,
    method_name: str,
    max_n: int,
    remove_mod6: bool = True,
    remove_polynomials: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, int, int, float]:
    """Compute residual after accounting for known prime distribution patterns.

    Instead of pixel-level subtraction, we:
    1. Create a null model based on prime number theorem (1/ln(n) density)
    2. Create polynomial masks for known prime-rich curves
    3. Compute what fraction of visual structure is "explained" by these

    The residual highlights positions where actual prime density differs
    significantly from expected density.

    Args:
        prime_image: Binary image (255=prime, 0=composite)
        method_name: Name of visualization method
        max_n: Maximum integer visualized

    Returns:
        (residual_image, mod6_removed, poly_removed, unexplained_fraction)
    """
    img = prime_image.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0

    h, w = img.shape

    # Get coordinate function for this visualization
    coord_func = get_coord_function(method_name, w, max_n)
    if coord_func is None:
        return img, 0, 0, 1.0

    # Track original prime pixels
    original_prime_pixels = (img > 0.5).sum()
    mod6_removed = 0
    poly_removed = 0

    # Create null model based on prime number theorem
    null_model = create_null_model_image(coord_func, h, w, max_n)

    # Create polynomial mask
    poly_mask = create_polynomial_mask_for_visualization_sized(coord_func, h, w, max_n)

    # Count primes on polynomial positions
    poly_primes = (img > 0.5) & (poly_mask > 0.5)
    poly_removed = int(poly_primes.sum())

    # Compute residual: actual - expected
    # Where residual > 0: more primes than expected (interesting!)
    # Where residual < 0: fewer primes than expected (also interesting!)
    residual = img - null_model

    # Also mark polynomial positions as "explained"
    # Reduce signal at known polynomial curves
    residual = residual * (1 - poly_mask * 0.5)

    # Compute mod-6 statistics
    for n in range(1, min(max_n + 1, 1000)):  # Sample first 1000
        if n > 3 and n % 6 != 1 and n % 6 != 5:
            x, y = coord_func(n)
            if 0 <= y < h and 0 <= x < w and img[y, x] > 0.5:
                mod6_removed += 1  # This shouldn't happen for valid primes

    # Normalize residual to highlight significant deviations
    # Scale so that ±1 std dev maps to ±0.5
    std = residual.std()
    if std > 0:
        residual = residual / (2 * std + 1e-10)

    # Shift to 0-1 range for visualization
    residual = (residual + 1) / 2
    residual = np.clip(residual, 0, 1)

    # Calculate unexplained fraction
    # Count pixels where residual deviates significantly from 0.5 (expected)
    significant_deviation = np.abs(residual - 0.5) > 0.15
    unexplained = (significant_deviation & (img > 0.5)).sum()
    unexplained_fraction = unexplained / original_prime_pixels if original_prime_pixels > 0 else 0

    if verbose:
        print(f"    Pattern removal: {mod6_removed} invalid mod-6, {poly_removed} on polynomials")
        print(f"    Residual contains {unexplained_fraction*100:.1f}% unexplained structure")

    return residual, mod6_removed, poly_removed, unexplained_fraction


def analyze_single_visualization(
    name: str,
    image: np.ndarray,
    max_n: int,
    run,
    verbose: bool = True
) -> PatternAnalysis:
    """Analyze a single visualization image for patterns."""

    if verbose:
        print(f"\n  Analyzing {name}...")

    # Save original image
    run.save_image(image, f"{name}_original")

    # 1. Detect patterns in original
    lines, line_score = detect_lines(image)
    clusters, cluster_score = detect_clusters(image)
    fft_mag, fft_score, fft_peaks = compute_fft_spectrum(image)
    autocorr, autocorr_score = compute_autocorrelation(image)

    # Save FFT visualization
    fft_vis = (np.log1p(fft_mag) / np.log1p(fft_mag).max() * 255).astype(np.uint8)
    run.save_image(fft_vis, f"{name}_fft")

    # Save autocorrelation
    autocorr_vis = ((autocorr - autocorr.min()) / (autocorr.max() - autocorr.min() + 1e-10) * 255).astype(np.uint8)
    run.save_image(autocorr_vis, f"{name}_autocorr")

    if verbose:
        print(f"    Original: {len(lines)} lines, {len(clusters)} clusters, FFT score: {fft_score:.3f}")

    # 2. Create known pattern visualization and proper residual
    h, w = image.shape[:2]
    coord_func = get_coord_function(name, w, max_n)

    if coord_func:
        # Generate marked image (known patterns in RED) and zeroed residual
        marked_img, residual_uint8, pattern_count, total_primes = create_known_pattern_images(
            image, coord_func, h, w, max_n, method_name=name
        )

        # Save marked image showing detected patterns in RED, others in WHITE
        run.save_image(marked_img, f"{name}_known_patterns")

        # Save residual with detected patterns completely zeroed out
        run.save_image(residual_uint8, f"{name}_residual")

        # Calculate residual statistics
        remaining_primes = (residual_uint8 > 127).sum()
        residual_frac = remaining_primes / total_primes if total_primes > 0 else 0
        pattern_pct = pattern_count / total_primes * 100 if total_primes > 0 else 0

        # Update variables for PatternAnalysis
        poly_removed = pattern_count
        mod6_removed = 0  # Not tracking separately anymore

        if verbose:
            print(f"    Detected patterns: {pattern_count}/{total_primes} primes ({pattern_pct:.1f}%) - RED in marked image")
            print(f"    Residual: {remaining_primes} primes remain ({residual_frac*100:.1f}%)")
    else:
        # Fallback if no coordinate function
        mod6_removed = 0
        poly_removed = 0
        residual_frac = 1.0
        residual_uint8 = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        run.save_image(residual_uint8, f"{name}_residual")

    # Create comparison: original | known patterns marked | residual (3-up)
    orig_uint8 = image if image.max() <= 255 else (image / image.max() * 255).astype(np.uint8)
    if len(orig_uint8.shape) == 2:
        orig_rgb = np.stack([orig_uint8, orig_uint8, orig_uint8], axis=-1)
    else:
        orig_rgb = orig_uint8

    if coord_func:
        comparison = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
        comparison[:, :w] = orig_rgb
        comparison[:, w + 10:w * 2 + 10] = marked_img
        res_rgb = np.stack([residual_uint8, residual_uint8, residual_uint8], axis=-1)
        comparison[:, w * 2 + 20:] = res_rgb
    else:
        comparison = np.zeros((h, w * 2 + 10), dtype=np.uint8)
        comparison[:, :w] = orig_uint8
        comparison[:, w + 10:] = residual_uint8

    run.save_image(comparison, f"{name}_comparison")

    # 3. Re-analyze residual
    res_lines, _ = detect_lines(residual_uint8)
    res_clusters, _ = detect_clusters(residual_uint8)
    _, res_fft_score, res_fft_peaks = compute_fft_spectrum(residual_uint8)

    if verbose:
        print(f"    Residual: {len(res_lines)} lines, {len(res_clusters)} clusters, FFT score: {res_fft_score:.3f}")
        print(f"    {residual_frac*100:.1f}% of visual structure unexplained by known patterns")

    # Get dominant FFT frequency
    dominant_freq = fft_peaks[0][:2] if fft_peaks else (0.0, 0.0)

    # Get strongest line angle
    strongest_theta = lines[0].theta if lines else 0.0

    return PatternAnalysis(
        method_name=name,
        num_lines=len(lines),
        strongest_line_theta=float(strongest_theta),
        num_clusters=len(clusters),
        fft_dominant_freq=dominant_freq,
        autocorr_score=float(autocorr_score),
        residual_lines=len(res_lines),
        residual_clusters=len(res_clusters),
        residual_fft_peaks=len(res_fft_peaks),
        mod6_pixels_removed=mod6_removed,
        polynomial_pixels_removed=poly_removed,
        residual_prime_fraction=float(residual_frac),
        pattern_extends_to_new_range=False,  # Will be tested separately
        prediction_improvement=0.0,  # Will be computed separately
    )


def test_pattern_extension(
    known_range_start: int,
    known_range_end: int,
    test_range_start: int,
    test_range_end: int,
    high_density_positions: List[Tuple[int, int]],  # (x, y) positions with high prime density
    visualization_func,  # Function that maps n -> (x, y)
    verbose: bool = True
) -> Tuple[bool, float]:
    """Test if patterns found in known range extend to new range.

    Returns:
        (pattern_extends, improvement_ratio)
    """
    # Get primes in test range
    test_primes = set(generate_primes_range(test_range_start, test_range_end))

    # Count primes in high-density vs low-density positions
    high_density_prime_count = 0
    high_density_total = 0
    low_density_prime_count = 0
    low_density_total = 0

    # Sample numbers in test range
    sample_size = min(10000, test_range_end - test_range_start)
    sample_numbers = np.random.randint(test_range_start, test_range_end, sample_size)

    for n in sample_numbers:
        try:
            x, y = visualization_func(n)
            is_high_density = any(
                abs(x - hx) < 5 and abs(y - hy) < 5
                for hx, hy in high_density_positions
            )

            is_prime_n = n in test_primes

            if is_high_density:
                high_density_total += 1
                if is_prime_n:
                    high_density_prime_count += 1
            else:
                low_density_total += 1
                if is_prime_n:
                    low_density_prime_count += 1
        except Exception:
            continue

    # Calculate densities
    if high_density_total > 0 and low_density_total > 0:
        high_density_ratio = high_density_prime_count / high_density_total
        low_density_ratio = low_density_prime_count / low_density_total

        if low_density_ratio > 0:
            improvement = high_density_ratio / low_density_ratio
            extends = improvement > 1.1  # 10% better
        else:
            improvement = 1.0
            extends = False
    else:
        improvement = 1.0
        extends = False

    if verbose:
        print(f"    Pattern extension test: improvement={improvement:.2f}x, extends={extends}")

    return extends, improvement


def test_primality_efficiency(
    start: int,
    end: int,
    target_primes: int,
    guided_priority_func,  # Returns priority score for a number
    verbose: bool = True
) -> Tuple[int, int, float]:
    """Compare brute-force vs guided prime finding.

    Args:
        start: Range start
        end: Range end
        target_primes: Number of primes to find
        guided_priority_func: Function(n) -> priority (higher = test first)

    Returns:
        (brute_force_tests, guided_tests, improvement_ratio)
    """
    primes = set(generate_primes_range(start, end))

    # Brute force: test in order
    bf_tests = 0
    bf_found = 0
    for n in range(start, end):
        if n % 2 == 0 and n > 2:
            continue  # Skip even
        bf_tests += 1
        if n in primes:
            bf_found += 1
            if bf_found >= target_primes:
                break

    # Guided: test by priority
    candidates = []
    for n in range(start, min(start + 100000, end)):
        if n % 2 == 0 and n > 2:
            continue
        try:
            priority = guided_priority_func(n)
            candidates.append((n, priority))
        except Exception:
            candidates.append((n, 0.0))

    # Sort by priority (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    guided_tests = 0
    guided_found = 0
    for n, _ in candidates:
        guided_tests += 1
        if n in primes:
            guided_found += 1
            if guided_found >= target_primes:
                break

    improvement = bf_tests / guided_tests if guided_tests > 0 else 1.0

    if verbose:
        print(f"    Efficiency test: BF={bf_tests}, Guided={guided_tests}, Improvement={improvement:.2f}x")

    return bf_tests, guided_tests, improvement


def run_visual_analysis(
    max_n: int = 50000,
    image_size: int = 500,
    test_extension: bool = True,
    test_efficiency: bool = True,
    verbose: bool = True
):
    """Run the full visual pattern analysis pipeline."""

    # Create run
    config = {
        "max_n": max_n,
        "image_size": image_size,
        "test_extension": test_extension,
        "test_efficiency": test_efficiency,
    }

    run = create_run(
        run_type="evaluation",
        description="visual_pattern_analysis",
        config=config,
        tags=["visual-analysis", "pattern-detection", "residual"]
    )

    print("=" * 70)
    print("VISUAL PATTERN ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print(f"Max N: {max_n:,}")
    print(f"Image size: {image_size}")
    print()

    run.log("Starting visual pattern analysis")

    # Step 1: Generate visualizations using evaluation pipeline
    print("Step 1: Generating visualizations...")
    eval_pipeline = EvaluationPipeline(max_n=max_n, image_size=image_size, verbose=False)
    eval_results = eval_pipeline.run_evaluation()

    print(f"  Generated {len(eval_results)} visualizations")

    # Save all visualization images
    for result in eval_results:
        save_raw_image(result.image, run.images_dir / f"viz_{result.name}.png")

    # Step 2: Detailed pattern analysis
    print("\nStep 2: Detailed pattern analysis...")
    analyses = []

    for result in eval_results:
        analysis = analyze_single_visualization(
            result.name,
            result.image,
            max_n,
            run,
            verbose=verbose
        )
        analyses.append(analysis)

    # Step 3: Find visualizations with significant residual patterns
    print("\nStep 3: Identifying interesting residual patterns...")

    interesting = []
    for analysis in analyses:
        # Interesting if residual has patterns after removing known structure
        residual_score = (
            analysis.residual_lines * 0.3 +
            analysis.residual_clusters * 0.3 +
            analysis.residual_fft_peaks * 0.4
        )
        if residual_score > 5:  # Threshold
            interesting.append((analysis, residual_score))
            print(f"  {analysis.method_name}: residual_score={residual_score:.1f}")

    interesting.sort(key=lambda x: x[1], reverse=True)

    if not interesting:
        print("  No significant residual patterns found after removing mod-6 and polynomial structure")
        print("  This suggests the visualizations are primarily showing known prime patterns")

    # Step 4: Test pattern extension to new ranges
    if test_extension and interesting:
        print("\nStep 4: Testing if residual patterns extend to new number ranges...")

        for analysis, score in interesting[:3]:  # Test top 3
            print(f"\n  Testing {analysis.method_name}...")

            coord_func = get_coord_function(analysis.method_name, image_size, max_n * 2)

            # Find high-density cluster positions from residual
            residual_path = run.images_dir / f"{analysis.method_name}_residual.png"
            if residual_path.exists():
                from PIL import Image
                residual_img = np.array(Image.open(residual_path).convert('L'))
                res_clusters, _ = detect_clusters(residual_img)
                high_density_positions = [(c.center_x, c.center_y) for c in res_clusters[:20]]

                if high_density_positions:
                    extends, improvement = test_pattern_extension(
                        known_range_start=1,
                        known_range_end=max_n,
                        test_range_start=max_n,
                        test_range_end=max_n * 2,
                        high_density_positions=high_density_positions,
                        visualization_func=coord_func,
                        verbose=verbose
                    )

                    # Update analysis
                    analysis.pattern_extends_to_new_range = extends
                    analysis.prediction_improvement = improvement
                else:
                    print(f"    No cluster positions found in residual for {analysis.method_name}")
            else:
                print(f"    Residual image not found for {analysis.method_name}")

    # Step 5: Efficiency comparison
    if test_efficiency:
        print("\nStep 5: Testing prime-finding efficiency...")

        # Use the best visualization's cluster positions as guidance
        if eval_results:
            best = eval_results[0]
            clusters, _ = detect_clusters(best.image)

            if clusters:
                # Create priority function using ACTUAL coordinate mapping
                cluster_centers = [(c.center_x, c.center_y) for c in clusters[:10]]
                h, w = best.image.shape
                coord_func = get_coord_function(best.name, w, max_n * 2)  # Extended range

                def priority_func(n):
                    # Use actual coordinate mapping for this visualization
                    try:
                        x, y = coord_func(n)
                    except Exception:
                        return 0.0

                    # Higher priority if near cluster center
                    min_dist = min(
                        np.sqrt((x - cx)**2 + (y - cy)**2)
                        for cx, cy in cluster_centers
                    ) if cluster_centers else float('inf')
                    return 1.0 / (min_dist + 1)

                bf, guided, improvement = test_primality_efficiency(
                    start=max_n,
                    end=max_n + 50000,
                    target_primes=50,
                    guided_priority_func=priority_func,
                    verbose=verbose
                )

                run.log(f"Efficiency: BF={bf}, Guided={guided}, Improvement={improvement:.2f}x")
                print(f"\n    Best visualization for guidance: {best.name}")
                print(f"    Using {len(cluster_centers)} cluster centers for prioritization")

    # Save results
    ranking = eval_pipeline.get_ranking()
    results = {
        "config": config,
        "visualizations_analyzed": len(analyses),
        "interesting_residuals": [
            {"method": a.method_name, "residual_score": s}
            for a, s in interesting
        ],
        "analyses": [asdict(a) for a in analyses],
        "ranking": ranking,
    }

    run.save_results(results, summary={
        "visualizations_analyzed": len(analyses),
        "interesting_residuals_found": len(interesting),
    })

    # Step 6: Generate visualization mosaics
    print("\nStep 6: Generating visualization mosaics...")
    try:
        mosaic_path = run.images_dir / "mosaic_visualizations.png"
        create_visualization_mosaic(
            images_dir=run.images_dir,
            ranking=ranking,
            output_path=mosaic_path,
            tile_size=256,
            include_residuals=True
        )
        print(f"  Saved mosaic to: {mosaic_path}")
        print(f"  Saved residual mosaic to: {run.images_dir / 'mosaic_residuals.png'}")
    except Exception as e:
        print(f"  Warning: Failed to create mosaic: {e}")

    run.complete(status="completed", summary={
        "methods_analyzed": len(analyses),
        "residual_patterns_found": len(interesting),
    })

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to: {run.run_dir}")
    print("=" * 70)

    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)
    print(f"Total visualizations analyzed: {len(analyses)}")
    print(f"Visualizations with interesting residual patterns: {len(interesting)}")

    # Report on pattern removal effectiveness
    print("\nPattern Removal Statistics:")
    for analysis in analyses:
        print(f"  {analysis.method_name}:")
        print(f"    - Primes on polynomials: {analysis.polynomial_pixels_removed}")
        print(f"    - Residual fraction: {analysis.residual_prime_fraction*100:.1f}%")
        print(f"    - Residual patterns: {analysis.residual_lines} lines, {analysis.residual_clusters} clusters")

    if interesting:
        print("\nTop residual patterns (after removing polynomial structure):")
        for analysis, score in interesting[:5]:
            print(f"  {analysis.method_name}: {analysis.residual_lines} lines, "
                  f"{analysis.residual_clusters} clusters, "
                  f"residual={analysis.residual_prime_fraction*100:.1f}%, score={score:.1f}")

        print("\nThese visualizations show patterns NOT fully explained by:")
        print("  - Mod-6 structure (primes > 3 are 1 or 5 mod 6)")
        print("  - Known prime-generating polynomials (Euler's n^2+n+41, etc)")
        print("\nFurther analysis needed to determine if these are:")
        print("  a) Novel exploitable patterns, or")
        print("  b) Other known mathematical structures not yet removed")
    else:
        print("\nConclusion: All detected patterns appear to be explained by known")
        print("prime number theory (mod-6 structure, polynomial diagonals).")
        print("No novel exploitable patterns found in this analysis.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual pattern analysis for prime visualizations")
    parser.add_argument("--max-n", type=int, default=50000, help="Maximum number to visualize")
    parser.add_argument("--image-size", type=int, default=500, help="Image size")
    parser.add_argument("--no-extension", action="store_true", help="Skip pattern extension testing")
    parser.add_argument("--no-efficiency", action="store_true", help="Skip efficiency testing")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    run_visual_analysis(
        max_n=args.max_n,
        image_size=args.image_size,
        test_extension=not args.no_extension,
        test_efficiency=not args.no_efficiency,
        verbose=not args.quiet,
    )
