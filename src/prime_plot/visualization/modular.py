"""Modular arithmetic visualizations for prime numbers.

This module implements visualizations based on modular (clock) arithmetic,
revealing patterns in prime distribution related to residue classes.

Visualizations:
1. ModularGrid: 2D grid where position (x, y) = (n mod m1, n mod m2)
2. ModularClock: Circular "clock face" showing primes by residue
3. ModularMatrix: Matrix showing prime counts in each residue class
4. CageMatch: "Boxed in" patterns using Fibonacci modular arithmetic
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class ModularGrid:
    """Visualize primes on a 2D modular grid.

    Each integer n is placed at position (n mod m1, n mod m2), creating
    a grid that reveals patterns in prime residue classes.

    Common choices:
    - m1=6, m2=6: Shows primes clustered at (1,1), (1,5), (5,1), (5,5)
    - m1=30, m2=30: Reveals more detailed wheel factorization patterns

    Attributes:
        max_n: Maximum integer to include.
        mod1: First modulus (x-axis).
        mod2: Second modulus (y-axis).
    """

    def __init__(self, max_n: int, mod1: int = 6, mod2: int = 6):
        """Initialize modular grid visualization.

        Args:
            max_n: Maximum integer to include.
            mod1: Modulus for x-coordinate.
            mod2: Modulus for y-coordinate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if mod1 < 2 or mod2 < 2:
            raise ValueError(f"Moduli must be >= 2, got {mod1}, {mod2}")

        self.max_n = max_n
        self.mod1 = mod1
        self.mod2 = mod2

    def generate_density_grid(self) -> np.ndarray:
        """Generate grid showing prime density at each (mod1, mod2) position.

        Returns:
            2D array of shape (mod2, mod1) with prime counts.
        """
        primes = set(generate_primes(self.max_n))
        grid = np.zeros((self.mod2, self.mod1), dtype=np.int32)

        for p in primes:
            x = p % self.mod1
            y = p % self.mod2
            grid[y, x] += 1

        return grid

    def render(self, scale: int = 20) -> np.ndarray:
        """Render density grid as image.

        Args:
            scale: Pixels per grid cell.

        Returns:
            2D uint8 array.
        """
        density = self.generate_density_grid()

        # Normalize to 0-255
        if density.max() > 0:
            normalized = (density / density.max() * 255).astype(np.uint8)
        else:
            normalized = density.astype(np.uint8)

        # Scale up
        image = np.repeat(np.repeat(normalized, scale, axis=0), scale, axis=1)
        return image

    def analyze_residue_classes(self) -> dict:
        """Analyze which residue classes contain primes.

        Returns:
            Dictionary with residue class statistics.
        """
        density = self.generate_density_grid()
        total_primes = density.sum()

        # Find active residue classes
        active = []
        for y in range(self.mod2):
            for x in range(self.mod1):
                if density[y, x] > 0:
                    active.append({
                        'residue1': x,
                        'residue2': y,
                        'count': int(density[y, x]),
                        'fraction': density[y, x] / total_primes if total_primes > 0 else 0,
                    })

        # Sort by count descending
        active.sort(key=lambda r: r['count'], reverse=True)

        return {
            'total_primes': int(total_primes),
            'active_classes': len(active),
            'total_classes': self.mod1 * self.mod2,
            'residues': active,
        }


class ModularClock:
    """Circular "clock face" visualization of prime residues.

    Plots primes on a circle where the angle is determined by n mod m,
    and radius by the magnitude of n. Creates radial patterns showing
    prime distribution across residue classes.

    Attributes:
        max_n: Maximum integer to include.
        modulus: Clock face divisions (like hours on a clock).
        image_size: Size of output image.
    """

    def __init__(self, max_n: int, modulus: int = 12, image_size: int = 1000):
        """Initialize modular clock visualization.

        Args:
            max_n: Maximum integer to include.
            modulus: Number of divisions (residue classes).
            image_size: Width/height of rendered image.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {modulus}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")

        self.max_n = max_n
        self.modulus = modulus
        self.image_size = image_size
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate polar coordinates for all integers.

        Returns:
            Tuple of (x_coords, y_coords) arrays.
        """
        if self._coords is not None:
            return self._coords

        n = np.arange(1, self.max_n + 1, dtype=np.float64)

        # Angle determined by residue class
        theta = 2 * np.pi * (n % self.modulus) / self.modulus

        # Radius by magnitude (log scale for better distribution)
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
        """Render clock with primes marked.

        Args:
            point_size: Radius of each point in pixels.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array with primes marked.
        """
        x, y = self.generate_coordinates()

        max_r = max(np.abs(x).max(), np.abs(y).max())
        scale = (self.image_size / 2 - point_size - 5) / max_r

        px = (x * scale + self.image_size / 2).astype(np.int32)
        py = (y * scale + self.image_size / 2).astype(np.int32)

        primes = set(generate_primes(self.max_n))

        xp = cp if (use_gpu and HAS_CUPY) else np
        image = xp.zeros((self.image_size, self.image_size), dtype=xp.uint8)

        for i in range(1, self.max_n + 1):
            if i in primes:
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

    def render_with_spokes(
        self,
        point_size: int = 1,
        spoke_intensity: int = 32
    ) -> np.ndarray:
        """Render clock with guide spokes showing residue class divisions.

        Args:
            point_size: Radius of each point in pixels.
            spoke_intensity: Grayscale value for spoke lines.

        Returns:
            2D uint8 array.
        """
        image = self.render_primes(point_size=point_size)
        center = self.image_size // 2
        max_radius = self.image_size // 2 - 5

        # Draw spokes for each residue class
        for i in range(self.modulus):
            theta = 2 * np.pi * i / self.modulus
            for r in range(0, max_radius, 1):
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    if image[y, x] == 0:  # Don't overwrite primes
                        image[y, x] = spoke_intensity

        return image


class ModularMatrix:
    """Matrix visualization showing prime behavior across moduli.

    Creates a 2D matrix where:
    - Row i represents modulus i
    - Column j represents residue class j
    - Cell value is the count/density of primes in that class

    Attributes:
        max_n: Maximum integer to include.
        max_modulus: Maximum modulus to analyze.
    """

    def __init__(self, max_n: int, max_modulus: int = 30):
        """Initialize modular matrix visualization.

        Args:
            max_n: Maximum integer to include.
            max_modulus: Largest modulus to include in matrix.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if max_modulus < 2:
            raise ValueError(f"max_modulus must be >= 2, got {max_modulus}")

        self.max_n = max_n
        self.max_modulus = max_modulus

    def generate_matrix(self) -> np.ndarray:
        """Generate the modular matrix.

        Returns:
            2D array of shape (max_modulus-1, max_modulus) with prime counts.
            Row m contains counts for modulus (m+2).
        """
        primes = list(generate_primes(self.max_n))
        matrix = np.zeros((self.max_modulus - 1, self.max_modulus), dtype=np.int32)

        for m in range(2, self.max_modulus + 1):
            for p in primes:
                residue = p % m
                matrix[m - 2, residue] += 1

        return matrix

    def render(self, scale: int = 10) -> np.ndarray:
        """Render matrix as image.

        Args:
            scale: Pixels per matrix cell.

        Returns:
            2D uint8 array.
        """
        matrix = self.generate_matrix()

        # Normalize each row independently (different moduli have different scales)
        normalized = np.zeros_like(matrix, dtype=np.float64)
        for i in range(matrix.shape[0]):
            row_max = matrix[i].max()
            if row_max > 0:
                normalized[i] = matrix[i] / row_max

        image = (normalized * 255).astype(np.uint8)
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        return image

    def find_forbidden_residues(self) -> dict[int, list[int]]:
        """Find residue classes that contain no primes (except possibly 2, 3, etc.).

        These are residues coprime to the modulus that still have no primes,
        which would be surprising (there shouldn't be any for large enough max_n).

        Returns:
            Dictionary mapping modulus to list of prime-free residue classes.
        """
        matrix = self.generate_matrix()
        forbidden = {}

        for m in range(2, self.max_modulus + 1):
            row = matrix[m - 2]
            empty_residues = []
            for r in range(m):
                # Check if r is coprime to m but has no primes
                if np.gcd(r, m) == 1 and row[r] == 0:
                    empty_residues.append(r)
            if empty_residues:
                forbidden[m] = empty_residues

        return forbidden


class CageMatch:
    """Cage Match visualization using Fibonacci modular arithmetic.

    Plots primes using modular arithmetic on Fibonacci sequence,
    revealing "boxed in" patterns where primes create bounded regions.

    The key insight is that F_n mod m is periodic (Pisano period),
    and primes fall into specific patterns within this periodicity.

    Attributes:
        max_n: Maximum integer to include.
        modulus: Modulus for Fibonacci residues.
        image_size: Size of output image.
    """

    def __init__(self, max_n: int, modulus: int = 10, image_size: int = 1000):
        """Initialize Cage Match visualization.

        Args:
            max_n: Maximum integer to include.
            modulus: Modulus for computing Fibonacci residues.
            image_size: Width/height of rendered image.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {modulus}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")

        self.max_n = max_n
        self.modulus = modulus
        self.image_size = image_size
        self._fib_mod = self._compute_fibonacci_mod()
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def _compute_fibonacci_mod(self) -> list[int]:
        """Compute Fibonacci sequence mod m up to max_n terms."""
        fib_mod = [0, 1]
        for _ in range(2, self.max_n + 1):
            fib_mod.append((fib_mod[-1] + fib_mod[-2]) % self.modulus)
        return fib_mod

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate coordinates using Fibonacci modular mapping.

        Position is based on:
        - x: F_n mod m (Fibonacci residue)
        - y: n mod m (linear residue)

        Returns:
            Tuple of (x_coords, y_coords) arrays.
        """
        if self._coords is not None:
            return self._coords

        x = np.zeros(self.max_n, dtype=np.float64)
        y = np.zeros(self.max_n, dtype=np.float64)

        for n in range(1, self.max_n + 1):
            # x is Fibonacci residue
            fib_idx = n if n < len(self._fib_mod) else n % len(self._fib_mod)
            x[n - 1] = self._fib_mod[fib_idx]

            # y is linear residue
            y[n - 1] = n % self.modulus

        self._coords = (x, y)
        return self._coords

    def render_primes(
        self,
        point_size: int = 3,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render cage match pattern with primes marked.

        Args:
            point_size: Radius of each point in pixels.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array with primes marked.
        """
        x, y = self.generate_coordinates()

        # Scale to fit image with padding
        scale = (self.image_size - 2 * point_size - 20) / self.modulus
        offset = point_size + 10

        px = (x * scale + offset).astype(np.int32)
        py = (y * scale + offset).astype(np.int32)

        primes = set(generate_primes(self.max_n))

        xp = cp if (use_gpu and HAS_CUPY) else np
        image = xp.zeros((self.image_size, self.image_size), dtype=xp.uint8)

        for i in range(1, self.max_n + 1):
            if i in primes:
                ix, iy = int(px[i-1]), int(py[i-1])

                if point_size <= 1:
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

    def render_density(self, scale: int = 50) -> np.ndarray:
        """Render density grid of primes in (Fib_mod, linear_mod) space.

        Args:
            scale: Pixels per grid cell.

        Returns:
            2D uint8 array.
        """
        primes = set(generate_primes(self.max_n))
        grid = np.zeros((self.modulus, self.modulus), dtype=np.int32)

        for n in range(1, self.max_n + 1):
            if n in primes:
                fib_idx = n if n < len(self._fib_mod) else n % len(self._fib_mod)
                x = self._fib_mod[fib_idx]
                y = n % self.modulus
                grid[y, x] += 1

        # Normalize
        if grid.max() > 0:
            normalized = (grid / grid.max() * 255).astype(np.uint8)
        else:
            normalized = grid.astype(np.uint8)

        image = np.repeat(np.repeat(normalized, scale, axis=0), scale, axis=1)
        return image

    def pisano_period(self) -> int:
        """Compute the Pisano period (period of Fibonacci mod m).

        Returns:
            Length of the period.
        """
        # Pisano period always starts with 0, 1
        # Find when we return to 0, 1
        a, b = 0, 1
        for i in range(1, self.modulus * self.modulus + 1):
            a, b = b, (a + b) % self.modulus
            if a == 0 and b == 1:
                return i
        return -1  # Should not happen for valid modulus


def generate_modular_image(
    max_n: int,
    viz_type: str = "grid",
    modulus: int = 6,
    image_size: int = 1000,
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate modular visualization.

    Args:
        max_n: Maximum integer to include.
        viz_type: 'grid', 'clock', 'matrix', or 'cage'.
        modulus: Modulus to use.
        image_size: Width/height of output image.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    if viz_type == "grid":
        grid_viz = ModularGrid(max_n, modulus, modulus)
        return grid_viz.render(scale=image_size // modulus)
    elif viz_type == "clock":
        clock_viz = ModularClock(max_n, modulus, image_size)
        return clock_viz.render_primes(use_gpu=use_gpu)
    elif viz_type == "matrix":
        matrix_viz = ModularMatrix(max_n, modulus)
        return matrix_viz.render(scale=image_size // modulus)
    elif viz_type == "cage":
        cage_viz = CageMatch(max_n, modulus, image_size)
        return cage_viz.render_primes(use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown viz_type: {viz_type}")
