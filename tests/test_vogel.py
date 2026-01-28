"""Tests for Vogel spiral visualization."""

import numpy as np
import pytest

from prime_plot.visualization.vogel import (
    VogelSpiral,
    generate_vogel_image,
    PHI,
    GOLDEN_ANGLE,
    GOLDEN_ANGLE_DEGREES,
)


class TestVogelSpiral:
    """Tests for VogelSpiral class."""

    def test_basic_creation(self):
        """Test basic spiral creation."""
        spiral = VogelSpiral(1000, image_size=500)
        assert spiral.max_n == 1000
        assert spiral.image_size == 500
        assert spiral.scaling == "sqrt"

    def test_scaling_modes(self):
        """Test different scaling modes."""
        for mode in ["sqrt", "linear", "log"]:
            spiral = VogelSpiral(100, scaling=mode)
            assert spiral.scaling == mode
            coords = spiral.generate_coordinates()
            assert len(coords[0]) == 100

    def test_invalid_max_n(self):
        """Test invalid max_n raises error."""
        with pytest.raises(ValueError):
            VogelSpiral(0)
        with pytest.raises(ValueError):
            VogelSpiral(-1)

    def test_invalid_image_size(self):
        """Test invalid image_size raises error."""
        with pytest.raises(ValueError):
            VogelSpiral(100, image_size=5)

    def test_invalid_scaling(self):
        """Test invalid scaling raises error."""
        with pytest.raises(ValueError):
            VogelSpiral(100, scaling="invalid")

    def test_generate_coordinates(self):
        """Test coordinate generation."""
        spiral = VogelSpiral(100)
        x, y = spiral.generate_coordinates()
        assert len(x) == 100
        assert len(y) == 100
        assert x.dtype == np.float64
        assert y.dtype == np.float64

    def test_coordinates_cached(self):
        """Test that coordinates are cached."""
        spiral = VogelSpiral(100)
        coords1 = spiral.generate_coordinates()
        coords2 = spiral.generate_coordinates()
        assert coords1 is coords2

    def test_render_primes_shape(self):
        """Test render_primes output shape."""
        spiral = VogelSpiral(1000, image_size=200)
        image = spiral.render_primes()
        assert image.shape == (200, 200)
        assert image.dtype == np.uint8

    def test_render_primes_values(self):
        """Test render_primes output values are 0 or 255."""
        spiral = VogelSpiral(1000, image_size=200)
        image = spiral.render_primes()
        unique = np.unique(image)
        assert all(v in [0, 255] for v in unique)

    def test_render_all_integers(self):
        """Test render_all_integers includes composites."""
        spiral = VogelSpiral(100, image_size=200)
        image = spiral.render_all_integers(prime_color=255, composite_color=64)
        unique = np.unique(image)
        # Should have at least 0 (background), 64 (composite), 255 (prime)
        assert len(unique) >= 2

    def test_count_primes_on_rays(self):
        """Test ray counting."""
        spiral = VogelSpiral(10000)
        counts = spiral.count_primes_on_rays(num_rays=10)
        assert len(counts) == 10
        assert all(isinstance(v, int) for v in counts.values())
        assert sum(counts.values()) > 0


class TestGoldenConstants:
    """Tests for golden ratio constants."""

    def test_phi_value(self):
        """Test golden ratio value."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_golden_angle_radians(self):
        """Test golden angle in radians."""
        assert abs(GOLDEN_ANGLE - 2.399963229728653) < 1e-10

    def test_golden_angle_degrees(self):
        """Test golden angle in degrees."""
        assert abs(GOLDEN_ANGLE_DEGREES - 137.5077640500378) < 1e-8


class TestGenerateVogelImage:
    """Tests for convenience function."""

    def test_basic_generation(self):
        """Test basic image generation."""
        image = generate_vogel_image(1000, image_size=200)
        assert image.shape == (200, 200)
        assert image.dtype == np.uint8

    def test_with_scaling(self):
        """Test generation with different scaling."""
        for scaling in ["sqrt", "linear", "log"]:
            image = generate_vogel_image(1000, image_size=100, scaling=scaling)
            assert image.shape == (100, 100)
