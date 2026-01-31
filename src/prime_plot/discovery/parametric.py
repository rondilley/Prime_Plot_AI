"""Parametric visualization renderer for evolved genomes.

Converts a VisualizationGenome into actual prime visualizations
that can be evaluated for predictive power.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from prime_plot.discovery.genome import VisualizationGenome
from prime_plot.core.sieve import prime_sieve_mask


def digit_sum(n: int) -> int:
    """Calculate sum of digits of n."""
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total


def digit_sum_vectorized(arr: np.ndarray) -> np.ndarray:
    """Vectorized digit sum calculation."""
    result = np.zeros_like(arr)
    temp = arr.copy()
    while np.any(temp > 0):
        result += temp % 10
        temp //= 10
    return result


def is_quadratic_residue(n: int, p: int) -> bool:
    """Check if n is a quadratic residue mod p."""
    if n % p == 0:
        return True
    return pow(int(n), (p - 1) // 2, int(p)) == 1


def quadratic_residue_score(arr: np.ndarray, base: int) -> np.ndarray:
    """Calculate quadratic residue score for array of values."""
    base = max(3, int(base))
    if base % 2 == 0:
        base += 1  # Ensure odd for proper QR calculation

    result = np.zeros_like(arr, dtype=np.float64)
    for i, n in enumerate(arr):
        if n > 0:
            result[i] = 1.0 if is_quadratic_residue(int(n), base) else 0.0
    return result


class ParametricVisualization:
    """Renders a visualization from a genome.

    Converts genome parameters into coordinate mappings and renders
    prime positions to create an image for evaluation.
    """

    def __init__(
        self,
        genome: VisualizationGenome,
        max_n: int = 10000,
        image_size: int = 256,
    ):
        self.genome = genome
        self.max_n = max_n
        self.image_size = image_size

    def compute_coordinates(self, n_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (x, y) coordinates for given n values."""
        g = self.genome
        n = n_values.astype(np.float64)

        # Avoid division by zero and invalid values
        n_safe = np.maximum(n, 1)
        sqrt_n = np.sqrt(n_safe)

        # Polar component
        r = (
            g.r_const
            + g.r_sqrt * sqrt_n
            + g.r_lin * n
            + g.r_sin * np.sin(g.r_freq * n)
        )

        # Modular angle component
        t_mod_base = max(2, int(g.t_mod_base))
        theta = (
            g.t_const
            + g.t_sqrt * sqrt_n
            + g.t_lin * n
            + g.t_mod * (n % t_mod_base)
        )

        x_polar = r * np.cos(theta)
        y_polar = r * np.sin(theta)

        # Cartesian/grid component
        x_mod_base = max(2, int(g.x_mod_base))
        x_div_base = max(1, int(g.x_div_base))
        y_mod_base = max(2, int(g.y_mod_base))
        y_div_base = max(1, int(g.y_div_base))

        x_cart = g.x_mod * (n % x_mod_base) + g.x_div * (n // x_div_base)
        y_cart = g.y_mod * (n % y_mod_base) + g.y_div * (n // y_div_base)

        # Blend polar and cartesian
        blend = np.clip(g.blend, 0, 1)
        x = blend * x_polar + (1 - blend) * x_cart
        y = blend * y_polar + (1 - blend) * y_cart

        # Add digit sum influence
        if g.x_digit_sum > 0 or g.y_digit_sum > 0:
            ds = digit_sum_vectorized(n_values.astype(np.int64))
            x += g.x_digit_sum * ds
            y += g.y_digit_sum * ds

        # Add quadratic residue influence
        if g.qr_mod > 0:
            qr = quadratic_residue_score(n_values, g.qr_base)
            # Use QR to create offset pattern
            x += g.qr_mod * qr * np.cos(n * 0.1)
            y += g.qr_mod * qr * np.sin(n * 0.1)

        return x, y

    def render_primes(self) -> np.ndarray:
        """Render prime positions to image."""
        # Get prime mask
        prime_mask = prime_sieve_mask(self.max_n)

        # Get prime indices
        prime_indices = np.where(prime_mask)[0]

        if len(prime_indices) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Compute coordinates for primes
        x, y = self.compute_coordinates(prime_indices)

        # Handle invalid values
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Normalize to image coordinates
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Avoid division by zero
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        # Add small margin
        margin = 0.02
        x_norm = (x - x_min) / x_range * (1 - 2*margin) + margin
        y_norm = (y - y_min) / y_range * (1 - 2*margin) + margin

        # Convert to pixel coordinates
        px = (x_norm * (self.image_size - 1)).astype(np.int32)
        py = (y_norm * (self.image_size - 1)).astype(np.int32)

        # Clip to valid range
        px = np.clip(px, 0, self.image_size - 1)
        py = np.clip(py, 0, self.image_size - 1)

        # Create image
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        image[py, px] = 255

        return image

    def render_all(self) -> np.ndarray:
        """Render all integers (not just primes) to image."""
        n_values = np.arange(2, self.max_n + 1)

        # Compute coordinates
        x, y = self.compute_coordinates(n_values)

        # Handle invalid values
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        n_valid = n_values[valid]

        if len(x) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Normalize
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        margin = 0.02
        x_norm = (x - x_min) / x_range * (1 - 2*margin) + margin
        y_norm = (y - y_min) / y_range * (1 - 2*margin) + margin

        px = (x_norm * (self.image_size - 1)).astype(np.int32)
        py = (y_norm * (self.image_size - 1)).astype(np.int32)

        px = np.clip(px, 0, self.image_size - 1)
        py = np.clip(py, 0, self.image_size - 1)

        # Create image - use prime mask for value
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        prime_mask = prime_sieve_mask(self.max_n)

        for i in range(len(px)):
            n = n_valid[i]
            if prime_mask[n]:
                image[py[i], px[i]] = 255

        return image

    def get_stats(self) -> dict:
        """Get statistics about the visualization."""
        image = self.render_primes()
        prime_count = np.sum(image > 0)
        coverage = prime_count / (self.image_size * self.image_size)

        # Check for degenerate cases
        unique_pixels = len(np.unique(np.where(image > 0)[0] * self.image_size +
                                       np.where(image > 0)[1]))

        return {
            'prime_pixels': int(prime_count),
            'unique_positions': int(unique_pixels),
            'coverage': float(coverage),
            'is_degenerate': bool(unique_pixels < 10 or coverage < 0.001),
        }
