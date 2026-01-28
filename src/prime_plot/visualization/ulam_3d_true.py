"""True 3D Ulam spiral - numbers wind outward from center in 3D space."""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(cache=True)
    def generate_true_3d_spiral(size: int, start: int = 1) -> np.ndarray:
        """Generate a true 3D spiral that expands outward from center.

        The spiral visits positions in expanding cubic shells,
        winding around each shell before moving to the next.

        Pattern: Start at center, then spiral outward visiting each
        cubic shell completely before moving to the next shell.
        """
        grid = np.zeros((size, size, size), dtype=np.int64)
        center = size // 2

        # Direction vectors for 3D movement
        # We'll use a pattern that visits faces of each shell

        x, y, z = center, center, center
        grid[z, y, x] = start
        n = 1

        shell = 1
        max_n = size * size * size

        while n < max_n:
            # For each shell, we trace around all 6 faces
            # Shell defines the "radius" from center

            if shell > center:
                break

            # Move to start of new shell (+x direction)
            x += 1
            if not (0 <= x < size and 0 <= y < size and 0 <= z < size):
                break
            grid[z, y, x] = start + n
            n += 1
            if n >= max_n:
                break

            # Side length for this shell
            side = 2 * shell

            # Trace the shell in a specific order:
            # 1. Go around in +y direction (front face)
            for _ in range(side - 1):
                y += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break
            if n >= max_n:
                break

            # 2. Go -x (left side of front)
            for _ in range(side):
                x -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break
            if n >= max_n:
                break

            # 3. Go -y (back)
            for _ in range(side):
                y -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break
            if n >= max_n:
                break

            # 4. Go +x (right side of back)
            for _ in range(side):
                x += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break
            if n >= max_n:
                break

            # Now go up through z levels
            for level in range(side - 1):
                # Go up one z level
                z += 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break

                # Trace around this z level (alternating direction each level)
                if level % 2 == 0:
                    # Go +y
                    for _ in range(side):
                        y += 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go -x
                    for _ in range(side):
                        x -= 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go -y
                    for _ in range(side):
                        y -= 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go +x (not all the way)
                    for _ in range(side - 1):
                        x += 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                else:
                    # Go -y
                    for _ in range(side):
                        y -= 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go +x
                    for _ in range(side):
                        x += 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go +y
                    for _ in range(side):
                        y += 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break
                    # Go -x (not all the way)
                    for _ in range(side - 1):
                        x -= 1
                        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                            grid[z, y, x] = start + n
                        n += 1
                        if n >= max_n:
                            break
                    if n >= max_n:
                        break

            if n >= max_n:
                break

            # Top face
            z += 1
            if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                grid[z, y, x] = start + n
            n += 1
            if n >= max_n:
                break

            # Trace top face
            if (side - 1) % 2 == 0:
                for _ in range(side):
                    y += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side):
                    x -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side):
                    y -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side - 1):
                    x += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
            else:
                for _ in range(side):
                    y -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side):
                    x += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side):
                    y += 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break
                if n >= max_n:
                    break
                for _ in range(side - 1):
                    x -= 1
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        grid[z, y, x] = start + n
                    n += 1
                    if n >= max_n:
                        break

            if n >= max_n:
                break

            # Go back down through z
            for level in range(side):
                z -= 1
                if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                    grid[z, y, x] = start + n
                n += 1
                if n >= max_n:
                    break

            shell += 1

        return grid


def generate_3d_spiral_helix(size: int, start: int = 1) -> np.ndarray:
    """Generate 3D spiral using a helix pattern expanding outward.

    Numbers are assigned based on distance from center, with ties
    broken by angle (creating a spiral pattern within each shell).
    """
    grid = np.zeros((size, size, size), dtype=np.int64)
    center = size // 2

    positions = []

    # Generate all positions with their spiral ordering value
    for z in range(size):
        for y in range(size):
            for x in range(size):
                # Distance from center (primary sort key)
                dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)

                # Angle in XY plane (secondary sort - creates spiral within shell)
                angle_xy = np.arctan2(y - center, x - center)

                # Z coordinate (tertiary sort)
                z_norm = (z - center) / (center + 0.001)

                # Combine into spiral ordering
                # Use a helix: as distance increases, angle increases
                helix_angle = angle_xy + dist * 0.5  # Spiral gets "wound" as it expands
                spiral_val = dist + 0.001 * helix_angle + 0.0001 * z_norm

                positions.append((spiral_val, x, y, z))

    # Sort by spiral value
    positions.sort(key=lambda p: p[0])

    # Assign numbers
    for i, (_, x, y, z) in enumerate(positions):
        grid[z, y, x] = start + i

    return grid


def generate_3d_cubic_spiral(size: int, start: int = 1) -> np.ndarray:
    """Generate proper 3D cubic spiral that visits all positions.

    Uses shell-by-shell approach where each cubic shell is filled
    before moving to the next.
    """
    grid = np.zeros((size, size, size), dtype=np.int64)
    center = size // 2

    # Create list of all positions sorted by Chebyshev distance (cubic shells)
    # then by a spiral pattern within each shell
    positions = []

    for z in range(size):
        for y in range(size):
            for x in range(size):
                # Chebyshev distance (defines cubic shells)
                shell = max(abs(x - center), abs(y - center), abs(z - center))

                # Within each shell, use angle to create spiral
                angle_xy = np.arctan2(y - center, x - center)
                angle_xz = np.arctan2(z - center, x - center)

                # Create ordering within shell
                # Combine angles to get consistent winding
                within_shell = (angle_xy + np.pi) * 1000 + (angle_xz + np.pi) * 10 + z

                positions.append((shell, within_shell, x, y, z))

    # Sort by shell first, then by position within shell
    positions.sort(key=lambda p: (p[0], p[1]))

    # Assign numbers
    for i, (_, _, x, y, z) in enumerate(positions):
        grid[z, y, x] = start + i

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


def visualize_3d_slices(grid: np.ndarray, prime_mask: np.ndarray, output_dir: str = "output"):
    """Save 2D slices of 3D prime distribution."""
    from PIL import Image
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True)

    is_prime = prime_mask[grid].astype(np.uint8)
    size = grid.shape[0]

    # Save XY slices at different Z levels
    for z_idx in [size//4, size//2, 3*size//4]:
        slice_img = is_prime[z_idx] * 255
        Image.fromarray(slice_img).save(f"{output_dir}/3d_true_slice_z{z_idx}.png")

    # Save XZ slice at Y center
    xz_slice = is_prime[:, size//2, :] * 255
    Image.fromarray(xz_slice).save(f"{output_dir}/3d_true_slice_y{size//2}.png")

    # Save YZ slice at X center
    yz_slice = is_prime[:, :, size//2] * 255
    Image.fromarray(yz_slice).save(f"{output_dir}/3d_true_slice_x{size//2}.png")

    print(f"Saved slices to {output_dir}/")


def analyze_true_3d_spiral(size: int = 32):
    """Analyze the true 3D spiral pattern."""
    print(f"Generating {size}x{size}x{size} true 3D cubic spiral...")

    # Use the cubic spiral (guaranteed to fill all positions)
    grid = generate_3d_cubic_spiral(size, start=1)

    # Check coverage
    unique_vals = np.unique(grid)
    expected = size ** 3
    print(f"Unique values: {len(unique_vals)} (expected: {expected})")
    print(f"Min value: {grid.min()}, Max value: {grid.max()}")

    # Check center
    center = size // 2
    print(f"Center value: {grid[center, center, center]}")

    max_val = int(grid.max())
    print(f"\nComputing prime sieve up to {max_val:,}...")
    prime_mask = compute_sieve(max_val + 1)

    is_prime = prime_mask[grid]
    total_primes = is_prime.sum()

    print(f"\n3D Spiral Statistics:")
    print(f"  Total cells: {size**3:,}")
    print(f"  Total primes: {total_primes:,}")
    print(f"  Prime density: {total_primes/size**3*100:.2f}%")

    # Analyze by distance from center
    print(f"\nPrime density by distance from center:")
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)

    for r in [2, 4, 6, 8, 10, 12, 14]:
        if r >= center:
            break
        mask = (dist >= r - 1) & (dist < r + 1)
        shell_primes = is_prime[mask].sum()
        shell_total = mask.sum()
        if shell_total > 0:
            print(f"  r={r}: {shell_primes}/{shell_total} = {shell_primes/shell_total*100:.2f}%")

    # Visualize
    visualize_3d_slices(grid, prime_mask)

    return grid, prime_mask


if __name__ == "__main__":
    analyze_true_3d_spiral(size=32)
