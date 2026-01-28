"""Tests for Ulam spiral functionality."""

import numpy as np
import pytest

from prime_plot.visualization.ulam import UlamSpiral, generate_ulam_image


class TestUlamSpiral:
    """Tests for UlamSpiral class."""

    def test_basic_creation(self):
        """Test basic spiral creation."""
        spiral = UlamSpiral(5)
        assert spiral.size == 5
        assert spiral.start == 1
        assert spiral.max_value == 25

    def test_custom_start(self):
        """Test spiral with custom starting value."""
        spiral = UlamSpiral(5, start=41)
        assert spiral.start == 41
        assert spiral.max_value == 65

    def test_generate_grid_center(self):
        """Test that grid has correct center value."""
        spiral = UlamSpiral(5)
        grid = spiral.generate_grid()

        center = spiral.size // 2
        assert grid[center, center] == 1

    def test_generate_grid_shape(self):
        """Test grid shape."""
        spiral = UlamSpiral(7)
        grid = spiral.generate_grid()

        assert grid.shape == (7, 7)

    def test_grid_contains_all_values(self):
        """Test that grid contains all expected values."""
        spiral = UlamSpiral(5)
        grid = spiral.generate_grid()

        values = set(grid.flatten())
        expected = set(range(1, 26))
        assert values == expected

    def test_render_primes_shape(self):
        """Test rendered primes shape."""
        spiral = UlamSpiral(10)
        primes = spiral.render_primes()

        assert primes.shape == (10, 10)
        assert primes.dtype == np.uint8

    def test_render_primes_binary(self):
        """Test that rendered primes are binary."""
        spiral = UlamSpiral(10)
        primes = spiral.render_primes()

        unique_values = set(primes.flatten())
        assert unique_values <= {0, 1}

    def test_prime_count(self):
        """Test approximate prime count in spiral."""
        spiral = UlamSpiral(100)
        primes = spiral.render_primes()

        # There are 1229 primes <= 10000
        prime_count = primes.sum()
        assert 1200 <= prime_count <= 1300

    def test_integer_to_coords_center(self):
        """Test coordinate conversion for center."""
        x, y = UlamSpiral.integer_to_coords(1)
        assert (x, y) == (0, 0)

    def test_integer_to_coords_adjacent(self):
        """Test coordinate conversion for adjacent values.

        Spiral goes: right, down, left, up (clockwise from center).
        """
        assert UlamSpiral.integer_to_coords(2) == (1, 0)   # right
        assert UlamSpiral.integer_to_coords(3) == (1, 1)   # down
        assert UlamSpiral.integer_to_coords(4) == (0, 1)   # left
        assert UlamSpiral.integer_to_coords(5) == (-1, 1)  # left

    def test_generate_coordinates_length(self):
        """Test coordinate arrays have correct length."""
        spiral = UlamSpiral(7)
        x, y = spiral.generate_coordinates()

        assert len(x) == 49
        assert len(y) == 49

    def test_diagonal_values(self):
        """Test diagonal extraction."""
        spiral = UlamSpiral(5)
        diag = spiral.get_diagonal_values("main")

        assert len(diag) == 5


class TestGenerateUlamImage:
    """Tests for generate_ulam_image function."""

    def test_basic_generation(self):
        """Test basic image generation."""
        image = generate_ulam_image(100)

        assert image.shape == (100, 100)
        assert image.dtype == np.uint8

    def test_values_are_0_or_255(self):
        """Test that values are 0 or 255."""
        image = generate_ulam_image(50)

        unique = set(image.flatten())
        assert unique <= {0, 255}


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_size_1(self):
        """Test spiral of size 1."""
        spiral = UlamSpiral(1)
        grid = spiral.generate_grid()

        assert grid.shape == (1, 1)
        assert grid[0, 0] == 1

    def test_invalid_size(self):
        """Test that invalid size raises error."""
        with pytest.raises(ValueError):
            UlamSpiral(0)

        with pytest.raises(ValueError):
            UlamSpiral(-5)

    def test_invalid_start(self):
        """Test that negative start raises error."""
        with pytest.raises(ValueError):
            UlamSpiral(5, start=-1)
