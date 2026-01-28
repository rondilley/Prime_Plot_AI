"""Klauber triangle visualization.

The Klauber triangle (Laurence Klauber, 1932) arranges integers in a triangular
pattern where each row n contains 2n-1 integers centered below the previous row.
Prime patterns appear as vertical and diagonal lines at 60-degree angles.
"""

from __future__ import annotations

import numpy as np

from prime_plot.core.sieve import prime_sieve_mask

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class KlauberTriangle:
    """Generator for Klauber triangle visualizations.

    In the Klauber triangle:
    - Row 1: [1]
    - Row 2: [2, 3, 4]
    - Row 3: [5, 6, 7, 8, 9]
    - Row n contains integers from n^2 - 2n + 2 to n^2

    The central column contains 1, 3, 7, 13, 21, ... (centered hexagonal numbers).

    Attributes:
        num_rows: Number of rows in the triangle.
    """

    def __init__(self, num_rows: int):
        """Initialize Klauber triangle generator.

        Args:
            num_rows: Number of rows to generate.

        Raises:
            ValueError: If num_rows < 1.
        """
        if num_rows < 1:
            raise ValueError(f"num_rows must be >= 1, got {num_rows}")

        self.num_rows = num_rows
        self._grid: np.ndarray | None = None

    @property
    def max_value(self) -> int:
        """Maximum integer value in the triangle."""
        return self.num_rows * self.num_rows

    @property
    def width(self) -> int:
        """Width of the triangle (bottom row)."""
        return 2 * self.num_rows - 1

    def row_start(self, row: int) -> int:
        """Get the first integer in a row (1-indexed).

        Args:
            row: Row number (1 to num_rows).

        Returns:
            First integer value in the row.
        """
        if row < 1:
            raise ValueError(f"Row must be >= 1, got {row}")
        return (row - 1) * (row - 1) + 1

    def row_end(self, row: int) -> int:
        """Get the last integer in a row (1-indexed).

        Args:
            row: Row number (1 to num_rows).

        Returns:
            Last integer value in the row.
        """
        return row * row

    def row_values(self, row: int) -> np.ndarray:
        """Get all integer values in a row.

        Args:
            row: Row number (1 to num_rows).

        Returns:
            Array of integers in the row.
        """
        return np.arange(self.row_start(row), self.row_end(row) + 1)

    def generate_grid(self) -> np.ndarray:
        """Generate 2D grid representation of the triangle.

        The triangle is embedded in a rectangular grid with zeros
        padding the non-triangle regions.

        Returns:
            2D array where grid[row, col] contains the integer at that
            position, or 0 for positions outside the triangle.
        """
        if self._grid is not None:
            return self._grid

        grid = np.zeros((self.num_rows, self.width), dtype=np.int64)

        for row in range(1, self.num_rows + 1):
            row_len = 2 * row - 1
            start_col = self.num_rows - row
            values = self.row_values(row)

            grid[row - 1, start_col:start_col + row_len] = values

        self._grid = grid
        return self._grid

    def render_primes(self, use_gpu: bool = False) -> np.ndarray:
        """Render triangle with primes marked as 1, others as 0.

        Args:
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D binary array (uint8) where 1 indicates prime.
        """
        grid = self.generate_grid()

        if self.max_value < 2:
            return np.zeros_like(grid, dtype=np.uint8)

        mask = prime_sieve_mask(self.max_value + 1, use_gpu=False)
        mask = np.insert(mask, 0, False)

        if use_gpu and HAS_CUPY:
            grid_gpu = cp.asarray(grid)
            mask_gpu = cp.asarray(mask)
            result = mask_gpu[grid_gpu].astype(cp.uint8)
            return cp.asnumpy(result)

        return mask[grid].astype(np.uint8)

    def render_scaled(
        self,
        scale: int = 1,
        use_gpu: bool = False
    ) -> np.ndarray:
        """Render triangle scaled up by integer factor.

        Args:
            scale: Scale factor for each cell.
            use_gpu: If True and CuPy available, use GPU acceleration.

        Returns:
            2D uint8 array scaled up.
        """
        prime_grid = self.render_primes(use_gpu=use_gpu)

        if scale == 1:
            return prime_grid * 255

        scaled = np.repeat(np.repeat(prime_grid, scale, axis=0), scale, axis=1)
        return scaled * 255

    def get_column_values(self, col: int) -> list[int]:
        """Get all non-zero values in a column.

        Args:
            col: Column index (0 to width-1).

        Returns:
            List of integer values in the column.
        """
        grid = self.generate_grid()
        column = grid[:, col]
        return [int(v) for v in column if v > 0]

    def get_central_column(self) -> np.ndarray:
        """Get values in the central column (centered hexagonal numbers).

        Returns:
            Array of central column values.
        """
        center = self.num_rows - 1
        grid = self.generate_grid()
        values = grid[:, center]
        return values[values > 0]

    def get_diagonal_values(self, direction: str = "left") -> list[list[int]]:
        """Get all diagonal lines in the triangle.

        Args:
            direction: "left" for / diagonals, "right" for \\ diagonals.

        Returns:
            List of lists, each containing values along one diagonal.
        """
        grid = self.generate_grid()
        diagonals = []

        if direction == "left":
            for start_col in range(self.width):
                diag = []
                row, col = 0, start_col
                while row < self.num_rows and col < self.width:
                    if grid[row, col] > 0:
                        diag.append(int(grid[row, col]))
                    row += 1
                    col += 1
                if diag:
                    diagonals.append(diag)

            for start_row in range(1, self.num_rows):
                diag = []
                row, col = start_row, 0
                while row < self.num_rows and col < self.width:
                    if grid[row, col] > 0:
                        diag.append(int(grid[row, col]))
                    row += 1
                    col += 1
                if diag:
                    diagonals.append(diag)

        else:
            for start_col in range(self.width):
                diag = []
                row, col = 0, start_col
                while row < self.num_rows and col >= 0:
                    if grid[row, col] > 0:
                        diag.append(int(grid[row, col]))
                    row += 1
                    col -= 1
                if diag:
                    diagonals.append(diag)

            for start_row in range(1, self.num_rows):
                diag = []
                row, col = start_row, self.width - 1
                while row < self.num_rows and col >= 0:
                    if grid[row, col] > 0:
                        diag.append(int(grid[row, col]))
                    row += 1
                    col -= 1
                if diag:
                    diagonals.append(diag)

        return diagonals


def generate_klauber_image(
    num_rows: int,
    scale: int = 1,
    use_gpu: bool = False
) -> np.ndarray:
    """Convenience function to generate Klauber triangle image.

    Args:
        num_rows: Number of rows in the triangle.
        scale: Scale factor for rendering.
        use_gpu: Use GPU acceleration if available.

    Returns:
        2D uint8 array suitable for image display.
    """
    triangle = KlauberTriangle(num_rows)
    return triangle.render_scaled(scale=scale, use_gpu=use_gpu)
