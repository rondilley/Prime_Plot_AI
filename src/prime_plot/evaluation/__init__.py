"""Evaluation framework for comparing prime visualization methods.

This module provides tools to:
1. Generate visualizations using multiple methods
2. Detect and measure patterns in each visualization
3. Calculate signal-to-noise ratios
4. Rank visualization methods by pattern quality
5. Integrate with ML to find most learnable representations
"""

from prime_plot.evaluation.metrics import (
    PatternMetrics,
    calculate_line_density,
    calculate_cluster_coherence,
    calculate_entropy,
    calculate_snr,
)
from prime_plot.evaluation.detectors import (
    detect_lines,
    detect_clusters,
    compute_fft_spectrum,
    compute_autocorrelation,
)
from prime_plot.evaluation.baseline import (
    generate_random_baseline,
    generate_density_matched_baseline,
)
from prime_plot.evaluation.pipeline import (
    EvaluationPipeline,
    VisualizationResult,
    run_full_evaluation,
)

__all__ = [
    # Metrics
    "PatternMetrics",
    "calculate_line_density",
    "calculate_cluster_coherence",
    "calculate_entropy",
    "calculate_snr",
    # Detectors
    "detect_lines",
    "detect_clusters",
    "compute_fft_spectrum",
    "compute_autocorrelation",
    # Baseline
    "generate_random_baseline",
    "generate_density_matched_baseline",
    # Pipeline
    "EvaluationPipeline",
    "VisualizationResult",
    "run_full_evaluation",
]
