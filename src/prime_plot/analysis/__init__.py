"""Analysis tools for prime patterns and density."""

from prime_plot.analysis.patterns import (
    detect_diagonal_patterns,
    analyze_line_density,
    extract_high_density_regions,
)
from prime_plot.analysis.density import (
    compute_local_density,
    prime_density_map,
    radial_density_profile,
)
from prime_plot.analysis.enhanced_detection import (
    comprehensive_pattern_detection,
    evaluate_residual,
    create_gabor_filter_bank,
    detect_with_gabor_bank,
    detect_with_fft,
    detect_clusters,
    detect_voids_and_concentrations,
    enhanced_directional_detection,
    evaluate_residual_density,
    PatternDetectionResult,
    ResidualEvaluationResult,
)

__all__ = [
    # Original exports
    "detect_diagonal_patterns",
    "analyze_line_density",
    "extract_high_density_regions",
    "compute_local_density",
    "prime_density_map",
    "radial_density_profile",
    # Enhanced detection exports
    "comprehensive_pattern_detection",
    "evaluate_residual",
    "create_gabor_filter_bank",
    "detect_with_gabor_bank",
    "detect_with_fft",
    "detect_clusters",
    "detect_voids_and_concentrations",
    "enhanced_directional_detection",
    "evaluate_residual_density",
    "PatternDetectionResult",
    "ResidualEvaluationResult",
]
