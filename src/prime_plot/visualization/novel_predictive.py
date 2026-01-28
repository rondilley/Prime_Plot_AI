"""Novel prime visualizations designed for predictive power.

These visualizations are designed to create regions with varying
prime density - making them useful for ML-based prime prediction.

Each method is based on number-theoretic properties that correlate
with primality.
"""

import numpy as np
from typing import Tuple, Optional, List
from ..core.sieve import generate_primes


class TwinPrimeSpiral:
    """Visualization emphasizing twin prime gaps.

    Position based on distance to nearest twin prime pair.
    Twin primes (p, p+2) are relatively rare - positions near them
    may have different prime density.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))
        self._find_twin_primes()

    def _find_twin_primes(self):
        """Find all twin prime pairs."""
        self.twin_primes = []
        prime_list = sorted(self.primes)
        for i in range(len(prime_list) - 1):
            if prime_list[i + 1] - prime_list[i] == 2:
                self.twin_primes.append((prime_list[i], prime_list[i + 1]))

    def render_primes(self) -> np.ndarray:
        """Render with position based on twin prime proximity."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        if not self.twin_primes:
            return image

        twin_positions = np.array([t[0] for t in self.twin_primes])

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            # Distance to nearest twin prime
            distances = np.abs(twin_positions - n)
            min_dist = distances.min()
            nearest_idx = distances.argmin()

            # Angle based on which twin prime is nearest
            theta = nearest_idx * 2 * np.pi / len(self.twin_primes)

            # Radius based on distance (closer to twin = smaller radius)
            r = np.log1p(min_dist + 1) * 20

            x = int(center + r * np.cos(theta))
            y = int(center + r * np.sin(theta))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class QuadraticResidueGrid:
    """Grid based on quadratic residue properties.

    x = n mod p1, y = n mod p2 where p1, p2 are primes.
    Quadratic residues create specific patterns that primes follow.
    """

    def __init__(self, max_n: int, image_size: int = 1000,
                 mod1: int = 7, mod2: int = 11):
        self.max_n = max_n
        self.image_size = image_size
        self.mod1 = mod1
        self.mod2 = mod2
        self.primes = set(generate_primes(max_n))

    def render_primes(self) -> np.ndarray:
        """Render primes on quadratic residue grid."""
        # Create accumulator grid
        grid = np.zeros((self.mod2, self.mod1), dtype=np.float32)
        counts = np.zeros((self.mod2, self.mod1), dtype=np.float32)

        for n in range(1, self.max_n + 1):
            x = n % self.mod1
            y = n % self.mod2
            counts[y, x] += 1
            if n in self.primes:
                grid[y, x] += 1

        # Normalize to density
        density = np.divide(grid, counts, where=counts > 0, out=np.zeros_like(grid))

        # Scale up to image size
        scale_y = self.image_size // self.mod2
        scale_x = self.image_size // self.mod1

        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        for y in range(self.mod2):
            for x in range(self.mod1):
                y1, y2 = y * scale_y, (y + 1) * scale_y
                x1, x2 = x * scale_x, (x + 1) * scale_x
                image[y1:y2, x1:x2] = int(density[y, x] * 255)

        return image


class SophieGermainHighlight:
    """Highlight Sophie Germain primes.

    Sophie Germain prime p: both p and 2p+1 are prime.
    These are rarer than regular primes and may indicate special regions.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max(max_n, max_n * 2 + 2)))
        self._find_sophie_germain()

    def _find_sophie_germain(self):
        """Find Sophie Germain primes."""
        self.sophie_germain = set()
        for p in self.primes:
            if p <= self.max_n and (2 * p + 1) in self.primes:
                self.sophie_germain.add(p)

    def render_primes(self) -> np.ndarray:
        """Spiral with Sophie Germain primes emphasized."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            # Base spiral coordinates
            r = np.sqrt(n)
            theta = np.sqrt(n) * 2 * np.pi

            # Modify based on Sophie Germain proximity
            is_sg = n in self.sophie_germain

            # Sophie Germain primes get different radius scaling
            if is_sg:
                r = r * 0.8  # Pull inward

            x = int(center + r * np.cos(theta) * (self.image_size / 2 - 10) / np.sqrt(self.max_n))
            y = int(center + r * np.sin(theta) * (self.image_size / 2 - 10) / np.sqrt(self.max_n))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class MersenneProximityPlot:
    """Position primes based on proximity to Mersenne numbers.

    Mersenne numbers 2^n - 1 are prime-rich regions.
    Distance to nearest Mersenne may correlate with primality.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))
        self._compute_mersenne()

    def _compute_mersenne(self):
        """Compute Mersenne numbers up to max_n."""
        self.mersenne = []
        n = 1
        while (2 ** n - 1) <= self.max_n * 2:
            self.mersenne.append(2 ** n - 1)
            n += 1

    def render_primes(self) -> np.ndarray:
        """Render based on Mersenne proximity."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        mersenne_arr = np.array(self.mersenne)
        center = self.image_size // 2
        max_r = self.image_size // 2 - 10

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            # Distance to nearest Mersenne
            distances = np.abs(mersenne_arr - n)
            min_dist = distances.min()
            nearest_idx = distances.argmin()

            # Angle from Mersenne index
            theta = nearest_idx * 2.39996  # Golden angle

            # Radius from log distance
            r = (np.log1p(min_dist) / np.log1p(self.max_n)) * max_r

            x = int(center + r * np.cos(theta))
            y = int(center + r * np.sin(theta))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class PrimeGapHistogramPlot:
    """2D histogram where x=gap before, y=gap after.

    For each prime p, plot at (gap to previous prime, gap to next prime).
    Creates regions showing gap distribution patterns.
    """

    def __init__(self, max_n: int, image_size: int = 1000, max_gap: int = 100):
        self.max_n = max_n
        self.image_size = image_size
        self.max_gap = max_gap
        self.primes = generate_primes(max_n)

    def render_primes(self) -> np.ndarray:
        """Render gap histogram."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        scale = (self.image_size - 1) / self.max_gap

        for i in range(1, len(self.primes) - 1):
            gap_before = self.primes[i] - self.primes[i - 1]
            gap_after = self.primes[i + 1] - self.primes[i]

            if gap_before <= self.max_gap and gap_after <= self.max_gap:
                x = int(gap_before * scale)
                y = int(gap_after * scale)

                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    # Accumulate for density
                    image[y, x] = min(255, image[y, x] + 10)

        return image


class DigitSumModularPlot:
    """Position based on digit sum modular properties.

    Digit sum mod 9 relates to divisibility by 3 and 9.
    Primes (except 3) never have digit sum divisible by 3.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))

    def _digit_sum(self, n: int) -> int:
        """Sum of digits."""
        return sum(int(d) for d in str(n))

    def render_primes(self) -> np.ndarray:
        """Render based on digit sum properties."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            ds = self._digit_sum(n)
            ds_mod9 = ds % 9

            # Angle from digit sum mod 9 (primes avoid 0, 3, 6)
            theta = ds_mod9 * 2 * np.pi / 9

            # Radius from n
            r = np.sqrt(n) / np.sqrt(self.max_n) * (self.image_size // 2 - 10)

            x = int(center + r * np.cos(theta))
            y = int(center + r * np.sin(theta))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class FermatResidueSpiral:
    """Spiral based on Fermat's Little Theorem residues.

    For prime p, a^(p-1) = 1 mod p.
    Position based on 2^n mod small primes gives primality hints.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))
        self.small_primes = [3, 5, 7, 11, 13]

    def render_primes(self) -> np.ndarray:
        """Render based on Fermat residue patterns."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            # Compute 2^n mod each small prime
            residues = [pow(2, n, p) for p in self.small_primes]

            # Use residues to compute angle
            theta = sum(r * (i + 1) for i, r in enumerate(residues)) * 0.1

            # Radius from n
            r = np.sqrt(n) / np.sqrt(self.max_n) * (self.image_size // 2 - 10)

            x = int(center + r * np.cos(theta))
            y = int(center + r * np.sin(theta))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class CollatzStepsPlot:
    """Position based on Collatz sequence length.

    Number of steps to reach 1 in Collatz sequence.
    May reveal structure in how primes behave under Collatz iteration.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))
        self._cache = {}

    def _collatz_steps(self, n: int) -> int:
        """Count steps to reach 1."""
        if n in self._cache:
            return self._cache[n]

        original = n
        steps = 0
        while n != 1 and steps < 1000:
            if n in self._cache:
                steps += self._cache[n]
                break
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1

        self._cache[original] = steps
        return steps

    def render_primes(self) -> np.ndarray:
        """Render primes by Collatz step count."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Find max steps for scaling
        max_steps = 0
        for n in range(2, min(self.max_n + 1, 100000)):
            if n in self.primes:
                steps = self._collatz_steps(n)
                max_steps = max(max_steps, steps)

        if max_steps == 0:
            return image

        center = self.image_size // 2

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            steps = self._collatz_steps(n)

            # x = n (scaled), y = steps
            x = int((n / self.max_n) * (self.image_size - 1))
            y = int((steps / max_steps) * (self.image_size - 1))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


class PrimitiveRootPattern:
    """Pattern based on primitive root existence.

    n has a primitive root iff n is 1, 2, 4, p^k, or 2p^k for odd prime p.
    This creates a specific subset that primes naturally belong to.
    """

    def __init__(self, max_n: int, image_size: int = 1000):
        self.max_n = max_n
        self.image_size = image_size
        self.primes = set(generate_primes(max_n))

    def _has_primitive_root(self, n: int) -> bool:
        """Check if n has a primitive root."""
        if n <= 1:
            return n == 1
        if n in (2, 4):
            return True
        if n % 2 == 0:
            n //= 2
            if n % 2 == 0:
                return False
        # Check if n is prime power
        for p in self.primes:
            if p > n:
                break
            if p * p > n:
                return n in self.primes
            power = p
            while power <= n:
                if power == n:
                    return True
                power *= p
        return False

    def render_primes(self) -> np.ndarray:
        """Render showing primitive root relationship."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for n in range(2, self.max_n + 1):
            if n not in self.primes:
                continue

            has_pr = self._has_primitive_root(n)  # Always true for primes

            # Check neighbors for primitive root
            neighbors_with_pr = sum(
                1 for k in range(max(1, n-5), min(self.max_n, n+6))
                if k != n and self._has_primitive_root(k)
            )

            # Position based on n and neighbor count
            theta = n * 2.39996
            r = np.sqrt(n) * (1 + 0.1 * (neighbors_with_pr - 5))

            scale = (self.image_size // 2 - 10) / np.sqrt(self.max_n)
            x = int(center + r * np.cos(theta) * scale)
            y = int(center + r * np.sin(theta) * scale)

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = 255

        return image


def get_novel_predictive_methods(max_n: int, image_size: int) -> dict:
    """Get all novel predictive visualization methods.

    Returns:
        Dictionary mapping method names to render functions.
    """
    return {
        'twin_prime_spiral': lambda: TwinPrimeSpiral(max_n, image_size).render_primes(),
        'quadratic_residue_7x11': lambda: QuadraticResidueGrid(max_n, image_size, 7, 11).render_primes(),
        'quadratic_residue_11x13': lambda: QuadraticResidueGrid(max_n, image_size, 11, 13).render_primes(),
        'quadratic_residue_13x17': lambda: QuadraticResidueGrid(max_n, image_size, 13, 17).render_primes(),
        'sophie_germain': lambda: SophieGermainHighlight(max_n, image_size).render_primes(),
        'mersenne_proximity': lambda: MersenneProximityPlot(max_n, image_size).render_primes(),
        'prime_gap_histogram': lambda: PrimeGapHistogramPlot(max_n, image_size).render_primes(),
        'digit_sum_modular': lambda: DigitSumModularPlot(max_n, image_size).render_primes(),
        'fermat_residue': lambda: FermatResidueSpiral(max_n, image_size).render_primes(),
        'collatz_steps': lambda: CollatzStepsPlot(max_n, image_size).render_primes(),
        'primitive_root': lambda: PrimitiveRootPattern(max_n, image_size).render_primes(),
    }
