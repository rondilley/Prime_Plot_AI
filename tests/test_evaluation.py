"""Tests for evaluation framework."""

import numpy as np
import pytest

from prime_plot.evaluation.metrics import (
    PatternMetrics,
    calculate_line_density,
    calculate_diagonal_density,
    calculate_cluster_coherence,
    calculate_entropy,
    calculate_snr,
    calculate_sparsity,
    compute_all_metrics,
)
from prime_plot.evaluation.detectors import (
    detect_lines,
    detect_clusters,
    compute_fft_spectrum,
    compute_autocorrelation,
    detect_radial_patterns,
)
from prime_plot.evaluation.baseline import (
    generate_random_baseline,
    generate_density_matched_baseline,
    generate_radial_density_baseline,
    generate_local_density_baseline,
)
from prime_plot.evaluation.pipeline import (
    EvaluationPipeline,
    VisualizationResult,
)


class TestPatternMetrics:
    """Tests for PatternMetrics class."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = PatternMetrics(name="test")
        assert metrics.name == "test"
        assert metrics.line_density == 0.0

    def test_overall_score(self):
        """Test overall score calculation."""
        metrics = PatternMetrics(
            name="test",
            line_density=0.8,
            snr_linear=5.0,
            cluster_coherence=0.6,
        )
        score = metrics.overall_score()
        assert 0 <= score <= 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = PatternMetrics(name="test", line_density=0.5)
        d = metrics.to_dict()
        assert d['name'] == "test"
        assert d['line_density'] == 0.5
        assert 'overall_score' in d


class TestLineDensity:
    """Tests for line density calculation."""

    def test_empty_image(self):
        """Test with empty image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        score, details = calculate_line_density(image)
        assert score == 0.0

    def test_horizontal_line(self):
        """Test with horizontal line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50, :] = 255
        score, details = calculate_line_density(image, angles=[0, 90])
        assert score > 0

    def test_diagonal_line(self):
        """Test with diagonal line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            image[i, i] = 255
        score, details = calculate_line_density(image, angles=[45, 135])
        assert score > 0


class TestClusterCoherence:
    """Tests for cluster coherence calculation."""

    def test_empty_image(self):
        """Test with empty image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        score = calculate_cluster_coherence(image)
        assert score == 0.0

    def test_single_cluster(self):
        """Test with single cluster."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255
        score = calculate_cluster_coherence(image)
        assert score > 0

    def test_multiple_clusters(self):
        """Test with multiple clusters."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:20, 10:20] = 255
        image[10:20, 80:90] = 255
        image[80:90, 10:20] = 255
        image[80:90, 80:90] = 255
        score = calculate_cluster_coherence(image)
        assert score > 0


class TestEntropy:
    """Tests for entropy calculation."""

    def test_uniform_image(self):
        """Test with uniform image (low entropy)."""
        image = np.zeros((100, 100), dtype=np.uint8)
        ent = calculate_entropy(image)
        assert ent >= 0

    def test_random_image(self):
        """Test with random image (high entropy)."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        ent = calculate_entropy(image)
        assert ent > 0


class TestSNR:
    """Tests for SNR calculation."""

    def test_identical_images(self):
        """Test with identical signal and noise."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        snr_db, snr_linear = calculate_snr(image, image)
        # Should be low SNR when signal equals noise pattern
        assert isinstance(snr_db, float)
        assert isinstance(snr_linear, float)

    def test_without_baseline(self):
        """Test SNR calculation without baseline."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        snr_db, snr_linear = calculate_snr(image)
        assert isinstance(snr_db, float)


class TestSparsity:
    """Tests for sparsity calculation."""

    def test_empty_image(self):
        """Test empty image sparsity."""
        image = np.zeros((100, 100), dtype=np.uint8)
        sparsity = calculate_sparsity(image)
        assert sparsity == 0.0

    def test_full_image(self):
        """Test full image sparsity."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        sparsity = calculate_sparsity(image)
        assert sparsity == 1.0

    def test_half_image(self):
        """Test half-filled image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 255
        sparsity = calculate_sparsity(image)
        assert abs(sparsity - 0.5) < 0.01


class TestDetectors:
    """Tests for pattern detectors."""

    def test_detect_lines_empty(self):
        """Test line detection on empty image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        lines, score = detect_lines(image)
        assert len(lines) == 0
        assert score == 0.0

    def test_detect_clusters_empty(self):
        """Test cluster detection on empty image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        clusters, score = detect_clusters(image)
        assert len(clusters) == 0
        assert score == 0.0

    def test_fft_spectrum(self):
        """Test FFT spectrum computation."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        spectrum, score, peaks = compute_fft_spectrum(image)
        assert spectrum.shape[0] > 0
        assert isinstance(score, float)

    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        autocorr, score = compute_autocorrelation(image)
        assert autocorr.shape == image.shape
        assert isinstance(score, float)

    def test_radial_patterns(self):
        """Test radial pattern detection."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        ray_density, score = detect_radial_patterns(image)
        assert len(ray_density) == 36  # Default num_rays
        assert isinstance(score, float)


class TestBaseline:
    """Tests for baseline generation."""

    def test_random_baseline_shape(self):
        """Test random baseline shape."""
        baseline = generate_random_baseline((100, 100), density=0.1)
        assert baseline.shape == (100, 100)

    def test_random_baseline_density(self):
        """Test random baseline density."""
        density = 0.1
        baseline = generate_random_baseline((100, 100), density=density, seed=42)
        actual_density = (baseline > 0).mean()
        assert abs(actual_density - density) < 0.02

    def test_density_matched_baseline(self):
        """Test density-matched baseline."""
        reference = np.zeros((100, 100), dtype=np.uint8)
        reference[:30, :30] = 255  # ~9% density
        baseline = generate_density_matched_baseline(reference, seed=42)
        ref_density = (reference > 0).mean()
        baseline_density = (baseline > 0).mean()
        assert abs(ref_density - baseline_density) < 0.02

    def test_radial_density_baseline(self):
        """Test radial density baseline."""
        baseline = generate_radial_density_baseline(
            (100, 100), total_points=500, seed=42
        )
        assert baseline.shape == (100, 100)
        assert (baseline > 0).sum() <= 500

    def test_local_density_baseline(self):
        """Test local density baseline."""
        reference = np.zeros((100, 100), dtype=np.uint8)
        reference[25:75, 25:75] = 255
        baseline = generate_local_density_baseline(reference, window_size=25, seed=42)
        assert baseline.shape == reference.shape


class TestEvaluationPipeline:
    """Tests for evaluation pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=100, verbose=False)
        assert pipeline.max_n == 1000
        assert pipeline.image_size == 100
        assert len(pipeline.methods) > 0

    def test_add_method(self):
        """Test adding custom method."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=100, verbose=False)
        pipeline.add_method("custom", lambda: np.zeros((100, 100), dtype=np.uint8))
        assert "custom" in pipeline.methods

    def test_evaluate_single_method(self):
        """Test evaluating single method."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=100, verbose=False)
        result = pipeline.evaluate_method(
            "test",
            lambda: np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        )
        assert isinstance(result, VisualizationResult)
        assert result.name == "test"
        assert result.image.shape == (100, 100)

    def test_run_evaluation(self):
        """Test running full evaluation."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=50, verbose=False)
        # Only evaluate a couple methods to keep test fast
        results = pipeline.run_evaluation(['ulam', 'sacks'])
        assert len(results) == 2
        # Results should be sorted by score
        assert results[0].overall_score() >= results[1].overall_score()

    def test_get_ranking(self):
        """Test getting ranking."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=50, verbose=False)
        pipeline.run_evaluation(['ulam', 'sacks'])
        ranking = pipeline.get_ranking()
        assert len(ranking) == 2
        assert all(isinstance(name, str) and isinstance(score, float)
                  for name, score in ranking)

    def test_generate_report(self):
        """Test report generation."""
        pipeline = EvaluationPipeline(max_n=1000, image_size=50, verbose=False)
        pipeline.run_evaluation(['ulam'])
        report = pipeline.generate_report()
        assert 'config' in report
        assert 'ranking' in report
        assert 'methods' in report
        assert 'summary' in report


class TestVisualizationResult:
    """Tests for VisualizationResult class."""

    def test_creation(self):
        """Test result creation."""
        result = VisualizationResult(
            name="test",
            image=np.zeros((100, 100), dtype=np.uint8),
            metrics=PatternMetrics(name="test"),
        )
        assert result.name == "test"

    def test_overall_score(self):
        """Test overall score."""
        result = VisualizationResult(
            name="test",
            image=np.zeros((100, 100), dtype=np.uint8),
            metrics=PatternMetrics(name="test", line_density=0.5),
        )
        score = result.overall_score()
        assert isinstance(score, float)

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = VisualizationResult(
            name="test",
            image=np.zeros((100, 100), dtype=np.uint8),
            metrics=PatternMetrics(name="test"),
        )
        d = result.to_dict()
        assert d['name'] == "test"
        assert 'metrics' in d
        assert 'overall_score' in d
