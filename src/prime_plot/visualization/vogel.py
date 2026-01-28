"""Vogel spiral visualization using the golden angle.

The Vogel spiral (Helmut Vogel, 1979) arranges points using the golden angle
(~137.5 degrees), which is related to the golden ratio. This creates a pattern
where points are optimally distributed without overlapping, similar to sunflower
seed arrangements.

Primes plotted on this spiral often align along specific rays, revealing
connections between prime distribution and the golden ratio.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Golden ratio and golden angle constants
PHI = (1 + np.sqrt(5)) / 2  # ~1.618033988749895
GOLDEN_ANGLE = 2 * np.pi * (1 - 1 / PHI)  # ~2.399963229728653 radians (~137.5077 degrees)
GOLDEN_ANGLE_DEGREES = 360 * (1 - 1 / PHI)  # ~137.5077 degrees


class VogelSpiral:
    """Generator for Vogel spiral visualizations.

    In the Vogel spiral:
    - Each point n is placed at angle n * golden_angle
    - Radius increases as sqrt(n) for uniform density
    - Points naturally avoid overlap due to golden angle properties

    This creates spiraling arms similar to sunflower seed patterns.

    Attributes:
        max_n: Maximum integer to include in the spiral.
        image_size: Size of the output image in pixels.
        scaling: Radius scaling mode ('sqrt', 'linear', 'log').
    """

    def __init__(
        self,
        max_n: int,
        image_size: int = 1000,
        scaling: str = "sqrt"
    ):
        """Initialize Vogel spiral generator.

        Args:
            max_n: Maximum integer to include.
            image_size: Width/height of rendered image.
            scaling: How radius scales with n ('sqrt', 'linear', 'log').

        Raises:
            ValueError: If max_n < 1 or image_size < 10.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")
        if scaling not in ("sqrt", "linear", "log"):
            raise ValueError(f"scaling must be 'sqrt', 'linear', or 'log', got {scaling}")

        self.max_n = max_n
        self.image_size = image_size
        self.scaling = scaling
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) coordinates for integers 1 to max_n.

        Returns:
            Tuple of (x_coords, y_coords) arrays in continuous space.
        """
        if self._coords is not None:
            return self._coords

        n = np.arange(1, self.max_n + 1, dtype=np.float64)

        # Angle increases by golden angle for each point
        theta = n * GOLDEN_ANGLE

        # Radius scaling
        if self.scaling == "sqrt":
            r = np.sqrt(n)
        elif self.scaling == "linear":
            r = n / np.sqrt(self.max_n)  # Normalize to similar range as sqrt
        else:  # log
            r = np.log1p(n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        self._coords = (x, y)
        return self._coords

    def render_primes(
        self,
        point_size: int = 1,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render spiral with primes as white points on black background.

        Args:
            point_size: Radius of each point in pixels.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array with primes marked.
        """
        x, y = self.generate_coordinates()

        # Scale to fit image
        max_r = max(np.abs(x).max(), np.abs(y).max())
        scale = (self.image_size / 2 - point_size - 5) / max_r

        px = (x * scale + self.image_size / 2).astype(np.int32)
        py = (y * scale + self.image_size / 2).astype(np.int32)

        primes = generate_primes(self.max_n)
        prime_set = set(primes)

        xp = cp if (use_gpu and HAS_CUPY) else np
        image = xp.zeros((self.image_size, self.image_size), dtype=xp.uint8)

        if use_gpu and HAS_CUPY:
            px = cp.asarray(px)
            py = cp.asarray(py)

        for i in range(1, self.max_n + 1):
            if i in prime_set:
                ix, iy = int(px[i-1]), int(py[i-1])

                if point_size == 1:
                    if 0 <= ix < self.image_size and 0 <= iy < self.image_size:
                        image[iy, ix] = 255
                else:
                    for dx in range(-point_size, point_size + 1):
                        for dy in range(-point_size, point_size + 1):
                            if dx*dx + dy*dy <= point_size*point_size:
                                nx, ny = ix + dx, iy + dy
                                if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                    image[ny, nx] = 255

        if use_gpu and HAS_CUPY:
            return cp.asnumpy(image)
        return image

    def render_all_integers(
        self,
        point_size: int = 1,
        prime_color: int = 255,
        composite_color: int = 64
    ) -> np.ndarray:
        """Render spiral showing both primes and composites.

        Args:
            point_size: Radius of each point in pixels.
            prime_color: Grayscale value for primes (0-255).
            composite_color: Grayscale value for composites (0-255).

        Returns:
            2D uint8 array.
        """
        x, y = self.generate_coordinates()

        max_r = max(np.abs(x).max(), np.abs(y).max())
        scale = (self.image_size / 2 - point_size - 5) / max_r

        px = (x * scale + self.image_size / 2).astype(np.int32)
        py = (y * scale + self.image_size / 2).astype(np.int32)

        mask = prime_sieve_mask(self.max_n + 1)
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i in range(1, self.max_n + 1):
            ix, iy = int(px[i-1]), int(py[i-1])
            color = prime_color if mask[i] else composite_color

            if 0 <= ix < self.image_size and 0 <= iy < self.image_size:
                image[iy, ix] = max(image[iy, ix], color)

        return image

    def count_primes_on_rays(self, num_rays: int = 10) -> dict[int, int]:
        """Count primes falling on specific angular rays.

        Args:
            num_rays: Number of rays to analyze (divides 360 degrees).

        Returns:
            Dictionary mapping ray index to prime count.
        """
        x, y = self.generate_coordinates()
        primes = set(generate_primes(self.max_n))

        ray_counts = {i: 0 for i in range(num_rays)}
        ray_width = 2 * np.pi / num_rays

        for i in range(1, self.max_n + 1):
            if i in primes:
                angle = np.arctan2(y[i-1], x[i-1])
                if angle < 0:
                    angle += 2 * np.pi
                ray_idx = int(angle / ray_width) % num_rays
                ray_counts[ray_idx] += 1

        return ray_counts


def generate_vogel_image(
    max_n: int,
    image_size: int = 1000,
    point_size: int = 1,
    scaling: str = "sqrt",
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate Vogel spiral image.

    Args:
        max_n: Maximum integer to include.
        image_size: Width/height of output image.
        point_size: Radius of each point.
        scaling: Radius scaling mode.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    spiral = VogelSpiral(max_n, image_size, scaling)
    return spiral.render_primes(point_size=point_size, use_gpu=use_gpu)
