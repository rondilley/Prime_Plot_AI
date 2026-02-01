"""Fibonacci spiral and related visualizations.

This module implements several Fibonacci-based visualizations for prime numbers:

1. FibonacciSpiral: Classic Fibonacci spiral where positions are determined by
   Fibonacci-indexed shells and golden angle rotation.

2. ReverseFibonacciSpiral: Spiral constructed by traversing Fibonacci sequence
   in reverse, converging toward center.

3. FibonacciShellPlot: Plots primes based on which Fibonacci "shell" they fall
   into (between consecutive Fibonacci numbers).

The golden angle (~137.5 degrees) appears naturally in these visualizations due
to its relationship with the Fibonacci sequence and golden ratio.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import generate_primes, prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2  # ~1.618033988749895
PSI = (1 - np.sqrt(5)) / 2  # ~-0.618033988749895 (conjugate golden ratio)
GOLDEN_ANGLE = 2 * np.pi / (PHI * PHI)  # ~137.5077 degrees


def fibonacci_sequence(n: int) -> list[int]:
    """Generate first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate.

    Returns:
        List of Fibonacci numbers [0, 1, 1, 2, 3, 5, 8, ...].
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]

    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci_up_to(limit: int) -> list[int]:
    """Generate Fibonacci numbers up to a limit.

    Args:
        limit: Maximum Fibonacci number to include.

    Returns:
        List of Fibonacci numbers <= limit.
    """
    fib = [0, 1]
    while fib[-1] + fib[-2] <= limit:
        fib.append(fib[-1] + fib[-2])
    return fib


def find_fibonacci_shell(n: int, fib_sequence: list[int]) -> int:
    """Find which Fibonacci shell a number belongs to.

    A number n is in shell k if F_k <= n < F_{k+1}.

    Args:
        n: Number to classify.
        fib_sequence: Pre-computed Fibonacci sequence.

    Returns:
        Shell index (0-indexed).
    """
    for i in range(len(fib_sequence) - 1):
        if fib_sequence[i] <= n < fib_sequence[i + 1]:
            return i
    return len(fib_sequence) - 1


class FibonacciSpiral:
    """Fibonacci spiral visualization for prime numbers.

    Maps integers to positions on a spiral where:
    - Radius is determined by Fibonacci shell (distance from origin grows
      by golden ratio between shells)
    - Angle within each shell is distributed using golden angle

    Attributes:
        max_n: Maximum integer to include.
        image_size: Size of output image.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        """Initialize Fibonacci spiral generator.

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
        self._fib = fibonacci_up_to(max_n + 1)
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) coordinates for integers 1 to max_n.

        Returns:
            Tuple of (x_coords, y_coords) arrays.
        """
        if self._coords is not None:
            return self._coords

        x = np.zeros(self.max_n, dtype=np.float64)
        y = np.zeros(self.max_n, dtype=np.float64)

        for n in range(1, self.max_n + 1):
            shell = find_fibonacci_shell(n, self._fib)

            # Radius grows by golden ratio per shell
            r = PHI ** shell

            # Position within shell determines angle
            shell_start = self._fib[shell] if shell < len(self._fib) else 0
            shell_end = self._fib[shell + 1] if shell + 1 < len(self._fib) else self.max_n
            shell_size = max(shell_end - shell_start, 1)
            position_in_shell = n - shell_start

            # Angle uses golden angle for optimal distribution
            theta = position_in_shell * GOLDEN_ANGLE

            x[n - 1] = r * np.cos(theta)
            y[n - 1] = r * np.sin(theta)

        self._coords = (x, y)
        return self._coords

    def render_primes(
        self,
        point_size: int = 2,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render spiral with primes marked.

        Args:
            point_size: Radius of each point in pixels.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array with primes marked.
        """
        x, y = self.generate_coordinates()

        max_r = max(np.abs(x).max(), np.abs(y).max())
        if max_r == 0:
            max_r = 1
        scale = (self.image_size / 2 - point_size - 10) / max_r

        px = (x * scale + self.image_size / 2).astype(np.int32)
        py = (y * scale + self.image_size / 2).astype(np.int32)

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

    def primes_between_fibonacci(self) -> list[tuple[int, int, int]]:
        """Count primes between consecutive Fibonacci numbers.

        Returns:
            List of (F_n, F_{n+1}, prime_count) tuples.
        """
        primes = set(generate_primes(self.max_n))
        results = []

        for i in range(len(self._fib) - 1):
            f_n = self._fib[i]
            f_n1 = self._fib[i + 1]
            count = sum(1 for p in primes if f_n <= p < f_n1)
            results.append((f_n, f_n1, count))

        return results


class ReverseFibonacciSpiral:
    """Reverse Fibonacci spiral - converges toward center.

    Instead of expanding outward, this spiral starts from the outer edge
    and winds inward, using the "reverse" Fibonacci property where the
    sequence can be extended to negative indices.

    F_{-n} = (-1)^{n+1} * F_n

    This creates patterns that converge toward a different equilibrium
    point related to the conjugate golden ratio (1 - sqrt(5))/2.

    Attributes:
        max_n: Maximum integer to include.
        image_size: Size of output image.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        """Initialize reverse Fibonacci spiral.

        Args:
            max_n: Maximum integer to include.
            image_size: Width/height of rendered image.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")

        self.max_n = max_n
        self.image_size = image_size
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate coordinates for reverse spiral.

        Returns:
            Tuple of (x_coords, y_coords) arrays.
        """
        if self._coords is not None:
            return self._coords

        n = np.arange(1, self.max_n + 1, dtype=np.float64)

        # Reverse spiral: radius decreases, angle winds inward
        # Use inverse relationship: larger n -> smaller r
        r = np.sqrt(self.max_n) / np.sqrt(n)

        # Angle still uses golden angle but in reverse direction
        theta = -n * GOLDEN_ANGLE

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        self._coords = (x, y)
        return self._coords

    def render_primes(
        self,
        point_size: int = 2,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render reverse spiral with primes marked.

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


class FibonacciShellPlot:
    """Visualize primes based on Fibonacci shell membership.

    Creates a polar plot where:
    - Radial position = Fibonacci shell number
    - Angular position = position within shell (golden angle distributed)

    This reveals density patterns of primes across Fibonacci intervals.

    Attributes:
        max_n: Maximum integer to include.
        image_size: Size of output image.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        """Initialize Fibonacci shell plot.

        Args:
            max_n: Maximum integer to include.
            image_size: Width/height of rendered image.
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        if image_size < 10:
            raise ValueError(f"image_size must be >= 10, got {image_size}")

        self.max_n = max_n
        self.image_size = image_size
        self._fib = fibonacci_up_to(max_n + 1)
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    def generate_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate coordinates based on Fibonacci shells.

        Returns:
            Tuple of (x_coords, y_coords) arrays.
        """
        if self._coords is not None:
            return self._coords

        x = np.zeros(self.max_n, dtype=np.float64)
        y = np.zeros(self.max_n, dtype=np.float64)

        num_shells = len(self._fib) - 1

        for n in range(1, self.max_n + 1):
            shell = find_fibonacci_shell(n, self._fib)

            # Radius is simply the shell number (linear)
            r = shell + 1

            # Angle based on position within shell
            shell_start = self._fib[shell] if shell < len(self._fib) else 0
            shell_end = self._fib[shell + 1] if shell + 1 < len(self._fib) else self.max_n
            shell_size = max(shell_end - shell_start, 1)
            position_in_shell = n - shell_start

            # Distribute points around the circle proportionally
            theta = 2 * np.pi * position_in_shell / shell_size

            x[n - 1] = r * np.cos(theta)
            y[n - 1] = r * np.sin(theta)

        self._coords = (x, y)
        return self._coords

    def render_primes(
        self,
        point_size: int = 2,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render shell plot with primes marked.

        Args:
            point_size: Radius of each point in pixels.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array with primes marked.
        """
        x, y = self.generate_coordinates()

        max_r = max(np.abs(x).max(), np.abs(y).max())
        if max_r == 0:
            max_r = 1
        scale = (self.image_size / 2 - point_size - 10) / max_r

        px = (x * scale + self.image_size / 2).astype(np.int32)
        py = (y * scale + self.image_size / 2).astype(np.int32)

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

    def shell_statistics(self) -> list[dict]:
        """Calculate prime statistics for each Fibonacci shell.

        Returns:
            List of dicts with shell info and prime statistics.
        """
        primes = set(generate_primes(self.max_n))
        stats = []

        for i in range(len(self._fib) - 1):
            f_n = self._fib[i]
            f_n1 = self._fib[i + 1]
            shell_size = f_n1 - f_n
            prime_count = sum(1 for p in primes if f_n <= p < f_n1)

            # Expected primes by prime number theorem: ~n/ln(n)
            if f_n > 1:
                expected = shell_size / np.log((f_n + f_n1) / 2)
            else:
                expected = shell_size / 2  # Rough estimate for small numbers

            stats.append({
                'shell': i,
                'start': f_n,
                'end': f_n1,
                'size': shell_size,
                'primes': prime_count,
                'density': prime_count / shell_size if shell_size > 0 else 0,
                'expected': expected,
                'ratio': prime_count / expected if expected > 0 else 0,
            })

        return stats


def generate_fibonacci_image(
    max_n: int,
    image_size: int = 1000,
    point_size: int = 2,
    spiral_type: str = "forward",
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate Fibonacci spiral image.

    Args:
        max_n: Maximum integer to include.
        image_size: Width/height of output image.
        point_size: Radius of each point.
        spiral_type: 'forward', 'reverse', or 'shell'.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    if spiral_type == "forward":
        fwd_spiral = FibonacciSpiral(max_n, image_size)
        return fwd_spiral.render_primes(point_size=point_size, use_gpu=use_gpu)
    elif spiral_type == "reverse":
        rev_spiral = ReverseFibonacciSpiral(max_n, image_size)
        return rev_spiral.render_primes(point_size=point_size, use_gpu=use_gpu)
    elif spiral_type == "shell":
        shell_plot = FibonacciShellPlot(max_n, image_size)
        return shell_plot.render_primes(point_size=point_size, use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown spiral_type: {spiral_type}")
