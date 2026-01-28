"""Ulam spiral generation and visualization.

The Ulam spiral arranges positive integers in a square spiral pattern,
starting from the center and spiraling outward. Prime numbers are marked,
revealing diagonal patterns corresponding to prime-generating polynomials.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class UlamSpiral:
    """Generator for Ulam spiral visualizations.

    The spiral starts at 1 in the center and proceeds:
    1 -> right -> up -> left -> left -> down -> down -> right -> right -> right -> ...

    Attributes:
        size: Width/height of the spiral grid.
        start: Starting number at the center (default 1).
    """

    def __init__(self, size: int, start: int = 1):
        """Initialize Ulam spiral generator.

        Args:
            size: Width and height of the output grid.
            start: Number at the center of the spiral.

        Raises:
            ValueError: If size < 1 or start < 0.
        """
        if size < 1:
            raise ValueError(f"Size must be >= 1, got {size}")
        if start < 0:
            raise ValueError(f"Start must be >= 0, got {start}")

        self.size = size
        self.start = start
        self._grid: np.ndarray | None = None
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def max_value(self) -> int:
        """Maximum integer value in the spiral."""
        return self.start + self.size * self.size - 1

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) coordinates for each integer.

        Returns:
            Tuple of (x_coords, y_coords) arrays where index i corresponds
            to the integer (start + i).
        """
        if self._coords is not None:
            return self._coords

        n = self.size * self.size
        x = np.zeros(n, dtype=np.int32)
        y = np.zeros(n, dtype=np.int32)

        cx = (self.size - 1) // 2
        cy = (self.size - 1) // 2

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        px, py = cx, cy
        direction = 0
        steps_in_direction = 1
        steps_taken = 0
        direction_changes = 0

        for i in range(n):
            if 0 <= px < self.size and 0 <= py < self.size:
                x[i] = px
                y[i] = py
            else:
                x[i] = np.clip(px, 0, self.size - 1)
                y[i] = np.clip(py, 0, self.size - 1)

            steps_taken += 1

            if steps_taken <= steps_in_direction:
                dx, dy = directions[direction]
                px += dx
                py += dy

            if steps_taken == steps_in_direction:
                steps_taken = 0
                direction = (direction + 1) % 4
                direction_changes += 1

                if direction_changes % 2 == 0:
                    steps_in_direction += 1

        self._coords = (x, y)
        return self._coords

    def generate_grid(self) -> np.ndarray:
        """Generate 2D grid with integer values at each position.

        Returns:
            2D array where grid[y, x] contains the integer at that position.
        """
        if self._grid is not None:
            return self._grid

        x_coords, y_coords = self.generate_coordinates()

        grid = np.zeros((self.size, self.size), dtype=np.int64)
        values = np.arange(self.start, self.start + len(x_coords))

        grid[y_coords, x_coords] = values

        self._grid = grid
        return self._grid

    def render_primes(self, use_gpu: bool = False) -> np.ndarray:
        """Render spiral with primes marked as 1, composites as 0.

        Args:
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D binary array (uint8) where 1 indicates prime.
        """
        grid = self.generate_grid()

        if self.max_value < 2:
            return np.zeros_like(grid, dtype=np.uint8)

        mask = prime_sieve_mask(self.max_value + 1, use_gpu=False)

        if use_gpu and HAS_CUPY:
            grid_gpu = cp.asarray(grid)
            mask_gpu = cp.asarray(mask)
            result = mask_gpu[grid_gpu].astype(cp.uint8)
            return cp.asnumpy(result)

        return mask[grid].astype(np.uint8)

    def render_density(
        self,
        window_size: int = 5,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render local prime density map.

        Args:
            window_size: Size of the averaging window.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D float array with local prime densities.
        """
        prime_grid = self.render_primes(use_gpu=use_gpu).astype(np.float32)

        if use_gpu and HAS_CUPY:
            prime_grid = cp.asarray(prime_grid)
            kernel = cp.ones((window_size, window_size), dtype=cp.float32)
            kernel /= kernel.sum()

            from cupyx.scipy.ndimage import convolve
            density = convolve(prime_grid, kernel, mode='constant')
            return cp.asnumpy(density)

        from scipy.ndimage import convolve
        kernel = np.ones((window_size, window_size), dtype=np.float32)
        kernel /= kernel.sum()

        return convolve(prime_grid, kernel, mode='constant')

    def get_diagonal_values(self, diagonal: str = "main") -> np.ndarray:
        """Extract values along a diagonal of the spiral.

        Args:
            diagonal: One of "main", "anti", "horizontal", "vertical".

        Returns:
            Array of integer values along the diagonal.
        """
        grid = self.generate_grid()
        center = self.size // 2

        if diagonal == "main":
            return np.diag(grid)
        elif diagonal == "anti":
            return np.diag(np.fliplr(grid))
        elif diagonal == "horizontal":
            return grid[center, :]
        elif diagonal == "vertical":
            return grid[:, center]
        else:
            raise ValueError(f"Unknown diagonal: {diagonal}")

    @staticmethod
    def integer_to_coords(n: int, start: int = 1) -> tuple[int, int]:
        """Convert an integer to its (x, y) offset from center.

        Args:
            n: The integer value.
            start: The value at the center.

        Returns:
            (x, y) offset from center position.
        """
        if n < start:
            raise ValueError(f"n ({n}) must be >= start ({start})")

        m = n - start

        if m == 0:
            return (0, 0)

        k = int(np.ceil((np.sqrt(m + 1) - 1) / 2))
        t = 2 * k + 1
        m_max = t * t - 1

        t_minus = t - 1

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

        return (x, y)


def generate_ulam_image(
    size: int,
    start: int = 1,
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate Ulam spiral image.

    Args:
        size: Width/height of the output image.
        start: Starting number at center.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    spiral = UlamSpiral(size, start)
    return spiral.render_primes(use_gpu=use_gpu) * 255
