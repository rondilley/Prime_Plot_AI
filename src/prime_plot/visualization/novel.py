"""Novel prime visualization methods.

Experimental approaches including 3D projections, alternative spirals,
and AI-generated coordinate mappings.
"""

import numpy as np
from typing import Optional, Tuple
from ..core.sieve import generate_primes


class PyramidPlot:
    """Map integers to a triangular pyramid structure.

    Numbers fill layers of a pyramid from bottom to top.
    Each layer n has n^2 positions arranged in a triangle.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))

    def generate_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D pyramid coordinates for each integer."""
        # Calculate number of layers needed
        # Layer k has k^2 positions, total positions = sum(k^2) = n(n+1)(2n+1)/6
        layers = int(np.ceil((6 * self.max_n) ** (1/3)))

        x_coords = []
        y_coords = []
        z_coords = []

        n = 1
        for layer in range(1, layers + 1):
            # Each layer is a triangular grid
            for row in range(layer):
                for col in range(row + 1):
                    if n > self.max_n:
                        break
                    # Center the triangle
                    x = col - row / 2
                    y = row * np.sqrt(3) / 2
                    z = layer
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    n += 1
                if n > self.max_n:
                    break
            if n > self.max_n:
                break

        return np.array(x_coords), np.array(y_coords), np.array(z_coords)

    def render_primes(self, projection: str = 'top') -> np.ndarray:
        """Render primes as 2D projection of pyramid.

        Args:
            projection: 'top', 'side', or 'iso' (isometric)
        """
        x, y, z = self.generate_coordinates()

        # Project to 2D
        if projection == 'top':
            px, py = x, y
        elif projection == 'side':
            px, py = x, z
        elif projection == 'iso':
            # Isometric projection
            angle = np.pi / 6
            px = x * np.cos(angle) - y * np.sin(angle)
            py = z + (x * np.sin(angle) + y * np.cos(angle)) * 0.5
        else:
            px, py = x, y

        # Normalize to image coordinates
        if len(px) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        px_norm = (px - px.min()) / (px.max() - px.min() + 1e-10) * (self.image_size - 20) + 10
        py_norm = (py - py.min()) / (py.max() - py.min() + 1e-10) * (self.image_size - 20) + 10

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, n in enumerate(range(1, min(self.max_n + 1, len(px) + 1))):
            if n in self.primes:
                xi, yi = int(px_norm[i]), int(py_norm[i])
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class ConePlot:
    """Map integers to a cone structure.

    Numbers spiral up a cone, with radius decreasing as height increases.
    """

    def __init__(self, max_n: int, image_size: int = 1000, turns_per_layer: float = 1.0):
        self.max_n = max_n
        self.image_size = image_size
        self.turns_per_layer = turns_per_layer
        self.primes = set(generate_primes(max_n))

    def generate_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D cone coordinates."""
        n = np.arange(1, self.max_n + 1)

        # Height increases with sqrt(n)
        z = np.sqrt(n)

        # Radius decreases as we go up (cone shape)
        max_z = np.sqrt(self.max_n)
        r = (max_z - z) / max_z  # 1 at bottom, 0 at top

        # Angle spirals
        theta = 2 * np.pi * self.turns_per_layer * z

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y, z

    def render_primes(self, projection: str = 'top') -> np.ndarray:
        """Render primes as 2D projection of cone."""
        x, y, z = self.generate_coordinates()

        if projection == 'top':
            px, py = x, y
        elif projection == 'side':
            px, py = x, z
        elif projection == 'iso':
            angle = np.pi / 6
            px = x * np.cos(angle) - y * np.sin(angle)
            py = z * 0.5 + (x * np.sin(angle) + y * np.cos(angle)) * 0.3
        else:
            px, py = x, y

        # Normalize
        center = self.image_size // 2
        scale = (self.image_size - 20) / 2 / max(abs(px).max(), abs(py).max(), 1e-10)

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, n in enumerate(range(1, self.max_n + 1)):
            if n in self.primes:
                xi = int(center + px[i] * scale)
                yi = int(center + py[i] * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class HexagonalSpiral:
    """Hexagonal spiral arrangement.

    Like Ulam but on a hexagonal grid - each cell has 6 neighbors.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))

    def generate_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate hexagonal spiral coordinates."""
        # Directions for hexagonal grid (6 directions)
        # Using axial coordinates
        directions = [
            (1, 0), (0, 1), (-1, 1),
            (-1, 0), (0, -1), (1, -1)
        ]

        x_coords = [0]
        y_coords = [0]

        x, y = 0, 0
        direction = 0
        steps_in_direction = 1
        steps_taken = 0
        direction_changes = 0

        for n in range(2, self.max_n + 1):
            dx, dy = directions[direction]
            x += dx
            y += dy
            x_coords.append(x)
            y_coords.append(y)

            steps_taken += 1
            if steps_taken >= steps_in_direction:
                steps_taken = 0
                direction = (direction + 1) % 6
                direction_changes += 1
                if direction_changes % 2 == 0:
                    steps_in_direction += 1

        return np.array(x_coords), np.array(y_coords)

    def render_primes(self) -> np.ndarray:
        """Render primes on hexagonal spiral."""
        ax_x, ax_y = self.generate_coordinates()

        # Convert axial to pixel coordinates
        # x_pixel = x + y/2, y_pixel = y * sqrt(3)/2
        px = ax_x + ax_y / 2
        py = ax_y * np.sqrt(3) / 2

        # Normalize
        center = self.image_size // 2
        max_coord = max(abs(px).max(), abs(py).max(), 1)
        scale = (self.image_size - 20) / 2 / max_coord

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, n in enumerate(range(1, self.max_n + 1)):
            if n in self.primes:
                xi = int(center + px[i] * scale)
                yi = int(center + py[i] * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class PrimeGapSpiral:
    """Visualize based on gaps between consecutive primes.

    Radius = prime value, angle = cumulative gap sum.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = generate_primes(max_n)

    def render_primes(self, angle_scale: float = 0.1) -> np.ndarray:
        """Render spiral based on prime gaps."""
        if len(self.primes) < 2:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Calculate gaps
        gaps = np.diff(self.primes)
        cumulative_angle = np.cumsum(gaps) * angle_scale
        cumulative_angle = np.insert(cumulative_angle, 0, 0)

        # Radius based on prime value
        r = np.sqrt(self.primes)

        # Convert to Cartesian
        x = r * np.cos(cumulative_angle)
        y = r * np.sin(cumulative_angle)

        # Normalize
        center = self.image_size // 2
        max_r = max(abs(x).max(), abs(y).max(), 1)
        scale = (self.image_size - 20) / 2 / max_r

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i in range(len(self.primes)):
            xi = int(center + x[i] * scale)
            yi = int(center + y[i] * scale)
            if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                image[yi, xi] = 255

        return image


class PolynomialSpiral:
    """Spiral with polynomial radius function.

    r = n^power, theta = n * angle_mult
    """

    def __init__(self, max_n: int, image_size: int = 1000,
                 power: float = 0.5, angle_mult: float = 2.39996):
        self.max_n = max_n
        self.image_size = image_size
        self.power = power
        self.angle_mult = angle_mult  # Default is golden angle
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render primes with polynomial spiral."""
        n = np.arange(1, self.max_n + 1)

        r = n ** self.power
        theta = n * self.angle_mult

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        center = self.image_size // 2
        max_r = max(abs(x).max(), abs(y).max(), 1)
        scale = (self.image_size - 20) / 2 / max_r

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, num in enumerate(n):
            if num in self.primes:
                xi = int(center + x[i] * scale)
                yi = int(center + y[i] * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class DiagonalWave:
    """Wave-like diagonal arrangement.

    x = n mod width, y = n // width + sin(n * freq)
    Creates wavy diagonal patterns.
    """

    def __init__(self, max_n: int, image_size: int = 1000,
                 wave_freq: float = 0.1, wave_amp: float = 5.0):
        self.max_n = max_n
        self.image_size = image_size
        self.wave_freq = wave_freq
        self.wave_amp = wave_amp
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render with diagonal wave pattern."""
        width = int(np.sqrt(self.max_n)) + 1

        n = np.arange(1, self.max_n + 1)
        x = n % width
        y = n // width + self.wave_amp * np.sin(n * self.wave_freq)

        # Normalize
        x_norm = x / width * (self.image_size - 20) + 10
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10) * (self.image_size - 20) + 10

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, num in enumerate(n):
            if num in self.primes:
                xi, yi = int(x_norm[i]), int(y_norm[i])
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class LogarithmicSpiral:
    """True logarithmic spiral.

    r = a * e^(b * theta), position based on n
    """

    def __init__(self, max_n: int, image_size: int = 1000,
                 growth_rate: float = 0.1):
        self.max_n = max_n
        self.image_size = image_size
        self.growth_rate = growth_rate
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render logarithmic spiral."""
        n = np.arange(1, self.max_n + 1)

        theta = np.sqrt(n) * 2 * np.pi / 10
        r = np.exp(self.growth_rate * theta)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        center = self.image_size // 2
        max_r = max(abs(x).max(), abs(y).max(), 1)
        scale = (self.image_size - 20) / 2 / max_r

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, num in enumerate(n):
            if num in self.primes:
                xi = int(center + x[i] * scale)
                yi = int(center + y[i] * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class SquareRootSpiral:
    """Square root based spiral with varying angle functions."""

    def __init__(self, max_n: int, image_size: int = 1000,
                 angle_func: str = 'linear'):
        self.max_n = max_n
        self.image_size = image_size
        self.angle_func = angle_func
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render with different angle functions."""
        n = np.arange(1, self.max_n + 1)

        r = np.sqrt(n)

        if self.angle_func == 'linear':
            theta = n * 0.5
        elif self.angle_func == 'sqrt':
            theta = np.sqrt(n) * 2 * np.pi
        elif self.angle_func == 'log':
            theta = np.log(n + 1) * 2 * np.pi
        elif self.angle_func == 'prime_count':
            # Angle based on prime counting function approximation
            theta = n / np.log(n + 2) * 0.1
        else:
            theta = n * 0.5

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        center = self.image_size // 2
        max_r = max(abs(x).max(), abs(y).max(), 1)
        scale = (self.image_size - 20) / 2 / max_r

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for i, num in enumerate(n):
            if num in self.primes:
                xi = int(center + x[i] * scale)
                yi = int(center + y[i] * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class PrimeFactorSpiral:
    """Position based on prime factorization properties.

    Angle = sum of prime factors, radius = number of factors.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))
        self._precompute_factors()

    def _precompute_factors(self):
        """Precompute smallest prime factor for each number."""
        self.spf = list(range(self.max_n + 1))
        for i in range(2, int(np.sqrt(self.max_n)) + 1):
            if self.spf[i] == i:  # i is prime
                for j in range(i * i, self.max_n + 1, i):
                    if self.spf[j] == j:
                        self.spf[j] = i

    def _factor_sum_and_count(self, n: int) -> Tuple[int, int]:
        """Get sum and count of prime factors."""
        if n <= 1:
            return 0, 0
        total = 0
        count = 0
        while n > 1:
            p = self.spf[n]
            total += p
            count += 1
            n //= p
        return total, count

    def render_primes(self) -> np.ndarray:
        """Render based on factorization."""
        coords = []
        for n in range(1, self.max_n + 1):
            fsum, fcount = self._factor_sum_and_count(n)
            theta = fsum * 0.1
            r = fcount + 1
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coords.append((n, x, y))

        # Normalize
        xs = np.array([c[1] for c in coords])
        ys = np.array([c[2] for c in coords])

        center = self.image_size // 2
        max_r = max(abs(xs).max(), abs(ys).max(), 1)
        scale = (self.image_size - 20) / 2 / max_r

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for n, x, y in coords:
            if n in self.primes:
                xi = int(center + x * scale)
                yi = int(center + y * scale)
                if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                    image[yi, xi] = 255

        return image


class DoubleSpiralInterleave:
    """Two interleaved spirals - odd and even positions.

    Creates interference patterns between two spiral arms.
    """

    def __init__(self, max_n: int, image_size: int = 1000,
                 spiral_type: str = 'archimedean'):
        self.max_n = max_n
        self.image_size = image_size
        self.spiral_type = spiral_type
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render double spiral."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for n in range(1, self.max_n + 1):
            if n not in self.primes:
                continue

            # Split into two spirals
            half_n = n // 2
            offset = np.pi if n % 2 == 0 else 0

            if self.spiral_type == 'archimedean':
                r = np.sqrt(half_n)
                theta = np.sqrt(half_n) * 2 * np.pi + offset
            else:  # fermat
                r = np.sqrt(half_n)
                theta = half_n * 137.508 * np.pi / 180 + offset

            max_r = np.sqrt(self.max_n // 2)
            scale = (self.image_size - 20) / 2 / max_r

            x = r * np.cos(theta) * scale
            y = r * np.sin(theta) * scale

            xi = int(center + x)
            yi = int(center + y)
            if 0 <= xi < self.image_size and 0 <= yi < self.image_size:
                image[yi, xi] = 255

        return image
