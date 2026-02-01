"""3D Ulam spiral generation and visualization."""

from typing import Optional
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(cache=True)
    def generate_3d_spiral_numba(size: int, start: int = 1) -> np.ndarray:
        """Generate 3D cubic spiral grid using Numba.

        The spiral starts at the center and expands outward in shells,
        filling each shell layer by layer (like an onion).
        """
        grid = np.zeros((size, size, size), dtype=np.int64)
        center = size // 2

        # Start at center
        x, y, z = center, center, center
        grid[z, y, x] = start

        n = 1  # Current number (relative to start)
        shell = 1  # Current shell number

        while n < size * size * size:
            # For each shell, we fill 6 faces of a cube
            # Shell radius determines how far from center

            # Move to start of new shell (one step in +x direction from previous shell)
            x += 1
            if x >= size or y >= size or z >= size or x < 0 or y < 0 or z < 0:
                break
            grid[z, y, x] = start + n
            n += 1
            if n >= size * size * size:
                break

            # Trace around the shell in a systematic way
            # Each shell has 6 faces, and we trace around them

            side_len = 2 * shell

            # Face 1: +x face, go -z direction
            for _ in range(side_len - 1):
                z -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Face 2: go -x direction
            for _ in range(side_len):
                x -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Face 3: go +z direction
            for _ in range(side_len):
                z += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Face 4: go +x direction
            for _ in range(side_len):
                x += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Face 5: go +y direction (moving up a layer)
            for _ in range(side_len):
                y += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Now trace the top layer
            # Go -z
            for _ in range(side_len):
                z -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Go -x
            for _ in range(side_len):
                x -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Go +z
            for _ in range(side_len):
                z += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break
            if n >= size * size * size:
                break

            # Go -y (back down through the middle layers)
            for layer in range(side_len - 1):
                y -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= size * size * size:
                    break

                # Trace around this layer
                # -z
                for _ in range(side_len):
                    z -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= size * size * size:
                        break
                if n >= size * size * size:
                    break

                # -x
                for _ in range(side_len):
                    x -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= size * size * size:
                        break
                if n >= size * size * size:
                    break

                # +z
                for _ in range(side_len):
                    z += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= size * size * size:
                        break
                if n >= size * size * size:
                    break

                # +x (but not the last one, that's the start of next shell)
                for _ in range(side_len - 1):
                    x += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= size * size * size:
                        break
                if n >= size * size * size:
                    break

            shell += 1

        return grid


def generate_3d_spiral_simple(size: int, start: int = 1) -> np.ndarray:
    """Generate 3D spiral using a simpler layer-by-layer approach.

    Each z-layer is a 2D Ulam spiral, with layers stacked.
    This creates a "stacked spirals" pattern.
    """
    grid = np.zeros((size, size, size), dtype=np.int64)

    n = start
    for z in range(size):
        # Generate 2D spiral for this layer
        layer = _generate_2d_spiral(size, n)
        grid[z] = layer
        n += size * size

    return grid


def _generate_2d_spiral(size: int, start: int) -> np.ndarray:
    """Generate a single 2D Ulam spiral layer."""
    grid = np.zeros((size, size), dtype=np.int64)

    cx = (size - 1) // 2
    cy = (size - 1) // 2

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    px, py = cx, cy
    direction = 0
    steps_in_direction = 1
    steps_taken = 0
    direction_changes = 0

    n = size * size
    for i in range(n):
        if 0 <= px < size and 0 <= py < size:
            grid[py, px] = start + i

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

    return grid


def compute_sieve(limit: int) -> np.ndarray:
    """Compute prime sieve."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime


def visualize_3d_primes(grid: np.ndarray, prime_mask: np.ndarray, output_path: Optional[str] = None):
    """Visualize 3D prime locations using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for 3D visualization")
        return

    # Get prime locations
    is_prime = prime_mask[grid]
    z_coords, y_coords, x_coords = np.where(is_prime)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot primes as points
    ax.scatter(x_coords, y_coords, z_coords, c='green', marker='.', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore[attr-defined]
    ax.set_title(f'3D Prime Distribution ({len(x_coords)} primes)')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_3d_patterns(size: int = 32):
    """Analyze prime patterns in 3D spiral."""
    print(f"Generating {size}x{size}x{size} 3D spiral...")

    # Use simple stacked spiral approach
    grid = generate_3d_spiral_simple(size, start=1)

    max_val = grid.max()
    print(f"Max value: {max_val:,}")

    print("Computing prime sieve...")
    prime_mask = compute_sieve(int(max_val) + 1)

    # Analyze
    is_prime = prime_mask[grid]
    total_primes = is_prime.sum()
    total_cells = size ** 3

    print(f"\n3D Spiral Statistics:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total primes: {total_primes:,}")
    print(f"  Prime density: {total_primes/total_cells*100:.2f}%")

    # Analyze by layer
    print(f"\nPrimes per z-layer:")
    for z in range(min(5, size)):
        layer_primes = prime_mask[grid[z]].sum()
        print(f"  z={z}: {layer_primes} primes")

    # Visualize
    visualize_3d_primes(grid, prime_mask, "output/3d_primes.png")

    return grid, prime_mask


if __name__ == "__main__":
    analyze_3d_patterns(size=32)
