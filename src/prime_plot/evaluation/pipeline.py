"""Evaluation pipeline for comparing prime visualization methods.

Orchestrates the full evaluation process:
1. Generate visualizations using all methods
2. Apply pattern detection algorithms
3. Compute quality metrics
4. Compare against random baselines
5. Rank methods by pattern quality
6. Generate comprehensive reports
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prime_plot.evaluation.metrics import (
    PatternMetrics,
    compute_all_metrics,
    calculate_snr,
)
from prime_plot.evaluation.detectors import (
    detect_lines,
    detect_clusters,
    compute_fft_spectrum,
    compute_autocorrelation,
    detect_radial_patterns,
)
from prime_plot.evaluation.baseline import (
    generate_density_matched_baseline,
    compute_baseline_statistics,
)


@dataclass
class VisualizationResult:
    """Result from evaluating a single visualization method.

    Attributes:
        name: Name of the visualization method.
        image: Generated image array.
        metrics: Computed pattern metrics.
        lines: Detected lines.
        clusters: Detected clusters.
        fft_score: FFT peak strength score.
        autocorr_score: Autocorrelation score.
        radial_score: Radial pattern score.
        baseline_comparison: Comparison with random baseline.
        generation_time: Time to generate visualization (seconds).
        analysis_time: Time to analyze patterns (seconds).
    """
    name: str
    image: np.ndarray
    metrics: PatternMetrics
    lines: list = field(default_factory=list)
    clusters: list = field(default_factory=list)
    fft_score: float = 0.0
    autocorr_score: float = 0.0
    radial_score: float = 0.0
    baseline_comparison: dict = field(default_factory=dict)
    generation_time: float = 0.0
    analysis_time: float = 0.0

    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return self.metrics.overall_score()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'metrics': self.metrics.to_dict(),
            'num_lines': len(self.lines),
            'num_clusters': len(self.clusters),
            'fft_score': self.fft_score,
            'autocorr_score': self.autocorr_score,
            'radial_score': self.radial_score,
            'baseline_comparison': self.baseline_comparison,
            'generation_time': self.generation_time,
            'analysis_time': self.analysis_time,
            'overall_score': self.overall_score(),
        }


class EvaluationPipeline:
    """Pipeline for evaluating and comparing visualization methods.

    Attributes:
        max_n: Maximum integer for visualizations.
        image_size: Image size for rendering.
        methods: Dictionary of visualization method factories.
        results: List of evaluation results.
    """

    def __init__(
        self,
        max_n: int = 100000,
        image_size: int = 500,
        verbose: bool = True
    ):
        """Initialize evaluation pipeline.

        Args:
            max_n: Maximum integer to include in visualizations.
            image_size: Size of rendered images.
            verbose: Print progress messages.
        """
        self.max_n = max_n
        self.image_size = image_size
        self.verbose = verbose
        self.methods: dict[str, Callable[[], np.ndarray]] = {}
        self.results: list[VisualizationResult] = []
        self._setup_default_methods()

    def _setup_default_methods(self):
        """Setup default visualization methods."""
        from prime_plot.visualization.ulam import UlamSpiral
        from prime_plot.visualization.sacks import SacksSpiral
        from prime_plot.visualization.klauber import KlauberTriangle
        from prime_plot.visualization.vogel import VogelSpiral
        from prime_plot.visualization.fibonacci import (
            FibonacciSpiral,
            ReverseFibonacciSpiral,
            FibonacciShellPlot,
        )
        from prime_plot.visualization.modular import (
            ModularClock,
            CageMatch,
        )

        # Calculate appropriate sizes
        ulam_size = int(np.sqrt(self.max_n)) + 1
        klauber_rows = int(np.sqrt(self.max_n * 2)) + 1

        self.methods = {
            'ulam': lambda: UlamSpiral(ulam_size).render_primes(),

            'sacks': lambda: SacksSpiral(
                self.max_n, self.image_size
            ).render_primes(point_size=1),

            'vogel_sqrt': lambda: VogelSpiral(
                self.max_n, self.image_size, scaling='sqrt'
            ).render_primes(point_size=1),

            'vogel_log': lambda: VogelSpiral(
                self.max_n, self.image_size, scaling='log'
            ).render_primes(point_size=1),

            'fibonacci_forward': lambda: FibonacciSpiral(
                min(self.max_n, 50000), self.image_size
            ).render_primes(point_size=2),

            'fibonacci_reverse': lambda: ReverseFibonacciSpiral(
                min(self.max_n, 50000), self.image_size
            ).render_primes(point_size=2),

            'fibonacci_shell': lambda: FibonacciShellPlot(
                min(self.max_n, 50000), self.image_size
            ).render_primes(point_size=2),

            'modular_clock_6': lambda: ModularClock(
                self.max_n, modulus=6, image_size=self.image_size
            ).render_primes(point_size=1),

            'modular_clock_30': lambda: ModularClock(
                self.max_n, modulus=30, image_size=self.image_size
            ).render_primes(point_size=1),

            'cage_match_10': lambda: CageMatch(
                self.max_n, modulus=10, image_size=self.image_size
            ).render_primes(point_size=2),
        }

        # Add Klauber only if size is reasonable
        if klauber_rows <= 1000:
            self.methods['klauber'] = lambda: KlauberTriangle(
                klauber_rows
            ).render_scaled(scale=max(1, self.image_size // klauber_rows))

    def add_method(self, name: str, generator: Callable[[], np.ndarray]):
        """Add a custom visualization method.

        Args:
            name: Name for this method.
            generator: Callable that returns a visualization image.
        """
        self.methods[name] = generator

    def evaluate_method(
        self,
        name: str,
        generator: Callable[[], np.ndarray]
    ) -> VisualizationResult:
        """Evaluate a single visualization method.

        Args:
            name: Method name.
            generator: Function that generates the visualization.

        Returns:
            VisualizationResult with all metrics.
        """
        if self.verbose:
            print(f"  Evaluating {name}...")

        # Generate visualization
        start_time = time.time()
        try:
            image = generator()
        except Exception as e:
            if self.verbose:
                print(f"    Error generating {name}: {e}")
            # Return empty result
            return VisualizationResult(
                name=name,
                image=np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                metrics=PatternMetrics(name=name),
            )
        generation_time = time.time() - start_time

        # Ensure image is 2D and uint8
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Generate baseline for comparison
        start_time = time.time()
        baseline = generate_density_matched_baseline(image, seed=42)

        # Compute base metrics
        metrics = compute_all_metrics(image, name, baseline)

        # Detect patterns
        lines, line_score = detect_lines(image)
        clusters, cluster_score = detect_clusters(image)
        _, fft_score, _ = compute_fft_spectrum(image)
        _, autocorr_score = compute_autocorrelation(image)
        _, radial_score = detect_radial_patterns(image)

        # Update metrics with detector scores
        metrics.fft_peak_strength = fft_score
        metrics.autocorr_score = autocorr_score

        # Compute SNR against baseline
        snr_db, snr_linear = calculate_snr(image, baseline)
        metrics.snr = snr_db
        metrics.snr_linear = snr_linear

        # Baseline comparison statistics
        baseline_stats = compute_baseline_statistics(image, num_baselines=5, seed=42)

        analysis_time = time.time() - start_time

        result = VisualizationResult(
            name=name,
            image=image,
            metrics=metrics,
            lines=lines,
            clusters=clusters,
            fft_score=fft_score,
            autocorr_score=autocorr_score,
            radial_score=radial_score,
            baseline_comparison=baseline_stats,
            generation_time=generation_time,
            analysis_time=analysis_time,
        )

        if self.verbose:
            print(f"    Score: {result.overall_score():.4f} "
                  f"(gen: {generation_time:.2f}s, analysis: {analysis_time:.2f}s)")

        return result

    def run_evaluation(self, methods: list[str] | None = None) -> list[VisualizationResult]:
        """Run evaluation on specified or all methods.

        Args:
            methods: List of method names to evaluate. None = all methods.

        Returns:
            List of VisualizationResult sorted by score.
        """
        if methods is None:
            methods = list(self.methods.keys())

        if self.verbose:
            print(f"Running evaluation on {len(methods)} methods...")
            print(f"  max_n: {self.max_n:,}")
            print(f"  image_size: {self.image_size}")

        self.results = []

        for name in methods:
            if name not in self.methods:
                if self.verbose:
                    print(f"  Warning: Unknown method '{name}', skipping")
                continue

            result = self.evaluate_method(name, self.methods[name])
            self.results.append(result)

        # Sort by overall score
        self.results.sort(key=lambda r: r.overall_score(), reverse=True)

        if self.verbose:
            print("\nResults (sorted by overall score):")
            for i, r in enumerate(self.results, 1):
                print(f"  {i}. {r.name}: {r.overall_score():.4f}")

        return self.results

    def get_ranking(self) -> list[tuple[str, float]]:
        """Get ranked list of methods by score.

        Returns:
            List of (method_name, score) tuples.
        """
        return [(r.name, r.overall_score()) for r in self.results]

    def get_best_method(self) -> VisualizationResult | None:
        """Get the best-performing visualization method.

        Returns:
            Best VisualizationResult or None if no results.
        """
        return self.results[0] if self.results else None

    def generate_report(self, output_dir: str | Path | None = None) -> dict:
        """Generate comprehensive evaluation report.

        Args:
            output_dir: Optional directory to save report and images.

        Returns:
            Report dictionary.
        """
        report = {
            'config': {
                'max_n': self.max_n,
                'image_size': self.image_size,
                'num_methods': len(self.results),
            },
            'ranking': self.get_ranking(),
            'methods': {r.name: r.to_dict() for r in self.results},
            'summary': self._generate_summary(),
        }

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save report JSON
            with open(output_dir / 'evaluation_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Save images
            from prime_plot.visualization.renderer import save_raw_image
            for result in self.results:
                save_raw_image(result.image, output_dir / f'{result.name}.png')

            if self.verbose:
                print(f"\nReport saved to {output_dir}/")

        return report

    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        if not self.results:
            return {}

        scores = [r.overall_score() for r in self.results]
        line_densities = [r.metrics.line_density for r in self.results]
        snrs = [r.metrics.snr for r in self.results]

        return {
            'best_method': self.results[0].name,
            'best_score': scores[0],
            'worst_method': self.results[-1].name,
            'worst_score': scores[-1],
            'score_range': max(scores) - min(scores),
            'avg_score': np.mean(scores),
            'avg_line_density': np.mean(line_densities),
            'avg_snr': np.mean(snrs),
            'methods_above_baseline': sum(1 for r in self.results if r.metrics.snr > 0),
        }


def run_full_evaluation(
    max_n: int = 100000,
    image_size: int = 500,
    output_dir: str | Path | None = None,
    verbose: bool = True
) -> dict:
    """Run full evaluation pipeline with default settings.

    Convenience function for quick evaluation.

    Args:
        max_n: Maximum integer for visualizations.
        image_size: Image size for rendering.
        output_dir: Optional directory to save results.
        verbose: Print progress messages.

    Returns:
        Evaluation report dictionary.
    """
    pipeline = EvaluationPipeline(max_n=max_n, image_size=image_size, verbose=verbose)
    pipeline.run_evaluation()
    return pipeline.generate_report(output_dir)


def compare_methods_at_scales(
    scales: list[int],
    methods: list[str] | None = None,
    verbose: bool = True
) -> dict[int, list[tuple[str, float]]]:
    """Compare visualization methods across different number scales.

    Tests how pattern quality changes as we look at larger primes.

    Args:
        scales: List of max_n values to test.
        methods: Specific methods to compare. None = all.
        verbose: Print progress.

    Returns:
        Dictionary mapping scale to ranking list.
    """
    results = {}

    for scale in scales:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating at scale: {scale:,}")
            print('='*50)

        pipeline = EvaluationPipeline(max_n=scale, verbose=verbose)
        pipeline.run_evaluation(methods)
        results[scale] = pipeline.get_ranking()

    # Print comparison summary
    if verbose:
        print("\n" + "="*50)
        print("SCALE COMPARISON SUMMARY")
        print("="*50)

        # Get all method names
        all_methods = set()
        for rankings in results.values():
            all_methods.update(name for name, _ in rankings)

        # Print table header
        print(f"{'Method':<25}", end='')
        for scale in scales:
            print(f"{scale:>12,}", end='')
        print()
        print("-" * (25 + 12 * len(scales)))

        # Print scores for each method
        for method in sorted(all_methods):
            print(f"{method:<25}", end='')
            for scale in scales:
                ranking = results[scale]
                score = next((s for n, s in ranking if n == method), None)
                if score is not None:
                    print(f"{score:>12.4f}", end='')
                else:
                    print(f"{'N/A':>12}", end='')
            print()

    return results
