"""Tests for Fibonacci spiral visualizations."""

import numpy as np
import pytest

from prime_plot.visualization.fibonacci import (
    FibonacciSpiral,
    ReverseFibonacciSpiral,
    FibonacciShellPlot,
    fibonacci_sequence,
    fibonacci_up_to,
    find_fibonacci_shell,
    generate_fibonacci_image,
    PHI,
)


class TestFibonacciSequence:
    """Tests for Fibonacci sequence utilities."""

    def test_fibonacci_sequence_empty(self):
        """Test empty sequence."""
        assert fibonacci_sequence(0) == []

    def test_fibonacci_sequence_one(self):
        """Test single element."""
        assert fibonacci_sequence(1) == [0]

    def test_fibonacci_sequence_two(self):
        """Test two elements."""
        assert fibonacci_sequence(2) == [0, 1]

    def test_fibonacci_sequence_ten(self):
        """Test first 10 Fibonacci numbers."""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert fibonacci_sequence(10) == expected

    def test_fibonacci_up_to(self):
        """Test Fibonacci numbers up to limit."""
        result = fibonacci_up_to(100)
        assert result == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    def test_fibonacci_up_to_exact(self):
        """Test when limit is exactly a Fibonacci number."""
        result = fibonacci_up_to(21)
        assert 21 in result
        assert 34 not in result

    def test_find_fibonacci_shell(self):
        """Test shell finding."""
        fib = [0, 1, 1, 2, 3, 5, 8, 13, 21]
        assert find_fibonacci_shell(0, fib) == 0
        # Note: fib[1] = fib[2] = 1, so shell for 1 is 2 (first shell where 1 < next)
        assert find_fibonacci_shell(1, fib) == 2  # 1 <= 1 < 2
        assert find_fibonacci_shell(4, fib) == 4  # 3 <= 4 < 5
        assert find_fibonacci_shell(7, fib) == 5  # 5 <= 7 < 8
        assert find_fibonacci_shell(15, fib) == 7  # 13 <= 15 < 21


class TestFibonacciSpiral:
    """Tests for FibonacciSpiral class."""

    def test_basic_creation(self):
        """Test basic spiral creation."""
        spiral = FibonacciSpiral(1000, image_size=500)
        assert spiral.max_n == 1000
        assert spiral.image_size == 500

    def test_invalid_max_n(self):
        """Test invalid max_n raises error."""
        with pytest.raises(ValueError):
            FibonacciSpiral(0)

    def test_invalid_image_size(self):
        """Test invalid image_size raises error."""
        with pytest.raises(ValueError):
            FibonacciSpiral(100, image_size=5)

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        spiral = FibonacciSpiral(100)
        x, y = spiral.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100

    def test_coordinates_cached(self):
        """Test that coordinates are cached."""
        spiral = FibonacciSpiral(100)
        coords1 = spiral.generate_coordinates()
        coords2 = spiral.generate_coordinates()
        assert coords1 is coords2

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        spiral = FibonacciSpiral(1000, image_size=200)
        image = spiral.render_primes()
        assert image.shape == (200, 200)
        assert image.dtype == np.uint8

    def test_primes_between_fibonacci(self):
        """Test prime counting between Fibonacci numbers."""
        spiral = FibonacciSpiral(100)
        results = spiral.primes_between_fibonacci()
        assert len(results) > 0
        # Each result is (F_n, F_{n+1}, prime_count)
        for f_n, f_n1, count in results:
            # Note: consecutive Fibonacci numbers can be equal (1, 1)
            assert f_n <= f_n1
            assert count >= 0


class TestReverseFibonacciSpiral:
    """Tests for ReverseFibonacciSpiral class."""

    def test_basic_creation(self):
        """Test basic spiral creation."""
        spiral = ReverseFibonacciSpiral(1000, image_size=500)
        assert spiral.max_n == 1000
        assert spiral.image_size == 500

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        spiral = ReverseFibonacciSpiral(100)
        x, y = spiral.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100

    def test_reverse_radius_decreases(self):
        """Test that radius decreases for larger n."""
        spiral = ReverseFibonacciSpiral(100)
        x, y = spiral.generate_coordinates()
        r_first = np.sqrt(x[0]**2 + y[0]**2)
        r_last = np.sqrt(x[-1]**2 + y[-1]**2)
        # First point should have larger radius than last
        assert r_first > r_last

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        spiral = ReverseFibonacciSpiral(1000, image_size=200)
        image = spiral.render_primes()
        assert image.shape == (200, 200)


class TestFibonacciShellPlot:
    """Tests for FibonacciShellPlot class."""

    def test_basic_creation(self):
        """Test basic creation."""
        plot = FibonacciShellPlot(1000, image_size=500)
        assert plot.max_n == 1000
        assert plot.image_size == 500

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        plot = FibonacciShellPlot(100)
        x, y = plot.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        plot = FibonacciShellPlot(1000, image_size=200)
        image = plot.render_primes()
        assert image.shape == (200, 200)

    def test_shell_statistics(self):
        """Test shell statistics calculation."""
        plot = FibonacciShellPlot(1000)
        stats = plot.shell_statistics()
        assert len(stats) > 0
        for s in stats:
            assert 'shell' in s
            assert 'start' in s
            assert 'end' in s
            assert 'size' in s
            assert 'primes' in s
            assert 'density' in s
            assert s['size'] == s['end'] - s['start']


class TestGenerateFibonacciImage:
    """Tests for convenience function."""

    def test_forward_type(self):
        """Test forward spiral generation."""
        image = generate_fibonacci_image(1000, image_size=200, spiral_type="forward")
        assert image.shape == (200, 200)

    def test_reverse_type(self):
        """Test reverse spiral generation."""
        image = generate_fibonacci_image(1000, image_size=200, spiral_type="reverse")
        assert image.shape == (200, 200)

    def test_shell_type(self):
        """Test shell plot generation."""
        image = generate_fibonacci_image(1000, image_size=200, spiral_type="shell")
        assert image.shape == (200, 200)

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(ValueError):
            generate_fibonacci_image(1000, spiral_type="invalid")


class TestGoldenRatio:
    """Tests for golden ratio constant."""

    def test_phi_value(self):
        """Test golden ratio value."""
        assert abs(PHI - (1 + np.sqrt(5)) / 2) < 1e-15

    def test_phi_property(self):
        """Test golden ratio property: phi^2 = phi + 1."""
        assert abs(PHI**2 - (PHI + 1)) < 1e-14
