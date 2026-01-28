"""Tests for modular arithmetic visualizations."""

import numpy as np
import pytest

from prime_plot.visualization.modular import (
    ModularGrid,
    ModularClock,
    ModularMatrix,
    CageMatch,
    generate_modular_image,
)


class TestModularGrid:
    """Tests for ModularGrid class."""

    def test_basic_creation(self):
        """Test basic grid creation."""
        grid = ModularGrid(1000, mod1=6, mod2=6)
        assert grid.max_n == 1000
        assert grid.mod1 == 6
        assert grid.mod2 == 6

    def test_invalid_max_n(self):
        """Test invalid max_n raises error."""
        with pytest.raises(ValueError):
            ModularGrid(0)

    def test_invalid_modulus(self):
        """Test invalid modulus raises error."""
        with pytest.raises(ValueError):
            ModularGrid(100, mod1=1)
        with pytest.raises(ValueError):
            ModularGrid(100, mod2=0)

    def test_generate_density_grid(self):
        """Test density grid generation."""
        grid = ModularGrid(1000, mod1=6, mod2=6)
        density = grid.generate_density_grid()
        assert density.shape == (6, 6)
        assert density.dtype == np.int32

    def test_mod6_prime_pattern(self):
        """Test that primes > 3 only appear at specific mod 6 positions."""
        grid = ModularGrid(10000, mod1=6, mod2=6)
        density = grid.generate_density_grid()
        # Primes > 3 are only 1 or 5 mod 6
        # Positions (0,*), (2,*), (3,*), (4,*) should have very few primes
        # Only 2 and 3 can appear at positions other than 1,5
        # Check that (1,1), (1,5), (5,1), (5,5) have the most primes
        total = density.sum()
        corner_sum = density[1, 1] + density[1, 5] + density[5, 1] + density[5, 5]
        # These four positions should have > 90% of primes (excluding 2 and 3)
        assert corner_sum > total * 0.9

    def test_render(self):
        """Test render output."""
        grid = ModularGrid(1000, mod1=6, mod2=6)
        image = grid.render(scale=10)
        assert image.shape == (60, 60)
        assert image.dtype == np.uint8

    def test_analyze_residue_classes(self):
        """Test residue class analysis."""
        grid = ModularGrid(1000, mod1=6, mod2=6)
        analysis = grid.analyze_residue_classes()
        assert 'total_primes' in analysis
        assert 'active_classes' in analysis
        assert 'residues' in analysis
        assert analysis['total_primes'] > 0


class TestModularClock:
    """Tests for ModularClock class."""

    def test_basic_creation(self):
        """Test basic clock creation."""
        clock = ModularClock(1000, modulus=12, image_size=500)
        assert clock.max_n == 1000
        assert clock.modulus == 12
        assert clock.image_size == 500

    def test_invalid_modulus(self):
        """Test invalid modulus raises error."""
        with pytest.raises(ValueError):
            ModularClock(100, modulus=1)

    def test_invalid_image_size(self):
        """Test invalid image_size raises error."""
        with pytest.raises(ValueError):
            ModularClock(100, image_size=5)

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        clock = ModularClock(100, modulus=12)
        x, y = clock.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100

    def test_coordinates_cached(self):
        """Test that coordinates are cached."""
        clock = ModularClock(100, modulus=12)
        coords1 = clock.generate_coordinates()
        coords2 = clock.generate_coordinates()
        assert coords1 is coords2

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        clock = ModularClock(1000, modulus=12, image_size=200)
        image = clock.render_primes()
        assert image.shape == (200, 200)
        assert image.dtype == np.uint8

    def test_render_with_spokes(self):
        """Test render_with_spokes output."""
        clock = ModularClock(1000, modulus=12, image_size=200)
        image = clock.render_with_spokes()
        assert image.shape == (200, 200)
        # Should have more non-zero pixels due to spokes
        assert np.count_nonzero(image) > 0


class TestModularMatrix:
    """Tests for ModularMatrix class."""

    def test_basic_creation(self):
        """Test basic matrix creation."""
        matrix = ModularMatrix(1000, max_modulus=20)
        assert matrix.max_n == 1000
        assert matrix.max_modulus == 20

    def test_invalid_max_modulus(self):
        """Test invalid max_modulus raises error."""
        with pytest.raises(ValueError):
            ModularMatrix(100, max_modulus=1)

    def test_generate_matrix(self):
        """Test matrix generation."""
        matrix = ModularMatrix(1000, max_modulus=10)
        m = matrix.generate_matrix()
        # Shape is (max_modulus - 1, max_modulus) for moduli 2 to max_modulus
        assert m.shape == (9, 10)
        assert m.dtype == np.int32

    def test_matrix_row_sums(self):
        """Test that each row sums to total prime count."""
        matrix = ModularMatrix(1000, max_modulus=10)
        m = matrix.generate_matrix()
        # Each row should sum to same total (number of primes)
        from prime_plot.core.sieve import generate_primes
        total_primes = len(generate_primes(1000))
        for row in m:
            assert row.sum() == total_primes

    def test_render(self):
        """Test render output."""
        matrix = ModularMatrix(1000, max_modulus=10)
        image = matrix.render(scale=20)
        assert image.shape == (180, 200)  # (9*20, 10*20)

    def test_find_forbidden_residues(self):
        """Test finding forbidden residues."""
        matrix = ModularMatrix(10000, max_modulus=20)
        forbidden = matrix.find_forbidden_residues()
        # For large enough max_n, there should be no truly forbidden residues
        # (coprime residues with no primes)
        assert isinstance(forbidden, dict)


class TestCageMatch:
    """Tests for CageMatch class."""

    def test_basic_creation(self):
        """Test basic creation."""
        cage = CageMatch(1000, modulus=10, image_size=500)
        assert cage.max_n == 1000
        assert cage.modulus == 10
        assert cage.image_size == 500

    def test_invalid_modulus(self):
        """Test invalid modulus raises error."""
        with pytest.raises(ValueError):
            CageMatch(100, modulus=1)

    def test_fibonacci_mod_computed(self):
        """Test that Fibonacci mod sequence is computed."""
        cage = CageMatch(100, modulus=10)
        assert len(cage._fib_mod) > 0
        # All values should be in [0, modulus)
        assert all(0 <= v < 10 for v in cage._fib_mod)

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        cage = CageMatch(100, modulus=10)
        x, y = cage.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100
        # x values are Fibonacci mod, y values are linear mod
        assert all(0 <= v < 10 for v in x)
        assert all(0 <= v < 10 for v in y)

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        cage = CageMatch(1000, modulus=10, image_size=200)
        image = cage.render_primes()
        assert image.shape == (200, 200)

    def test_render_density(self):
        """Test density rendering."""
        cage = CageMatch(1000, modulus=10)
        image = cage.render_density(scale=20)
        assert image.shape == (200, 200)

    def test_pisano_period(self):
        """Test Pisano period calculation."""
        # Known Pisano periods
        test_cases = [
            (2, 3),
            (3, 8),
            (5, 20),
            (10, 60),
        ]
        for modulus, expected_period in test_cases:
            cage = CageMatch(100, modulus=modulus)
            period = cage.pisano_period()
            assert period == expected_period, f"Pisano period for mod {modulus} should be {expected_period}, got {period}"


class TestGenerateModularImage:
    """Tests for convenience function."""

    def test_grid_type(self):
        """Test grid type generation."""
        image = generate_modular_image(1000, viz_type="grid", modulus=6, image_size=120)
        assert image.shape[0] > 0
        assert image.shape[1] > 0

    def test_clock_type(self):
        """Test clock type generation."""
        image = generate_modular_image(1000, viz_type="clock", modulus=12, image_size=200)
        assert image.shape == (200, 200)

    def test_matrix_type(self):
        """Test matrix type generation."""
        image = generate_modular_image(1000, viz_type="matrix", modulus=10, image_size=200)
        assert image.shape[0] > 0

    def test_cage_type(self):
        """Test cage type generation."""
        image = generate_modular_image(1000, viz_type="cage", modulus=10, image_size=200)
        assert image.shape == (200, 200)

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(ValueError):
            generate_modular_image(1000, viz_type="invalid")
