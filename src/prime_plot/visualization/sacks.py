"""Sacks spiral visualization.

The Sacks spiral (Robert Sacks, 1994) arranges integers on an Archimedean
spiral where perfect squares lie on the positive x-axis. This creates
continuous curves of primes rather than the broken diagonals of the Ulam spiral.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class SacksSpiral:
    """Generator for Sacks spiral visualizations.

    In the Sacks spiral:
    - Perfect squares (1, 4, 9, 16, ...) lie on the positive x-axis
    - Each rotation contains one more integer than the previous
    - Position of integer n: angle = 2*pi*sqrt(n), radius = sqrt(n)

    Attributes:
        max_n: Maximum integer to include in the spiral.
        image_size: Size of the output image in pixels.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        """Initialize Sacks spiral generator.

        Args:
            max_n: Maximum integer to include.
            image_size: Width/height of rendered image.

        Raises:
            ValueError: If max_n < 1 or image_size < 10.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")

        self.max_n = max_n
        self.image_size = image_size
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) coordinates for integers 1 to max_n.

        Returns:
            Tuple of (x_coords, y_coords) arrays in continuous space.
        """
        if self._coords is not None:
            return self._coords

        n = np.arange(1, self.max_n + 1, dtype=np.float64)

        sqrt_n = np.sqrt(n)
        theta = 2 * np.pi * sqrt_n
        r = sqrt_n

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

        max_r = np.sqrt(self.max_n)
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

        max_r = np.sqrt(self.max_n)
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

    def get_curve_points(self, polynomial_func) -> tuple[np.ndarray, np.ndarray]:
        """Get coordinates of points satisfying a polynomial.

        Args:
            polynomial_func: Function f(n) returning integer values.

        Returns:
            (x, y) coordinates for polynomial values in range.
        """
        x_all, y_all = self.generate_coordinates()

        n_values = []
        for n in range(self.max_n):
            val = polynomial_func(n)
            if 1 <= val <= self.max_n:
                n_values.append(val - 1)

        if not n_values:
            return np.array([]), np.array([])

        indices = np.array(n_values)
        return x_all[indices], y_all[indices]


def generate_sacks_image(
    max_n: int,
    image_size: int = 1000,
    point_size: int = 1,
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate Sacks spiral image.

    Args:
        max_n: Maximum integer to include.
        image_size: Width/height of output image.
        point_size: Radius of each point.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    spiral = SacksSpiral(max_n, image_size)
    return spiral.render_primes(point_size=point_size, use_gpu=use_gpu)
