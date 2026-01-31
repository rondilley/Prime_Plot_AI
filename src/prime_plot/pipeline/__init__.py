"""Prime prediction pipeline module."""

from prime_plot.pipeline.predictor import PrimePredictor
from prime_plot.pipeline.polynomial_search import (
    polynomial_first_search,
    brute_force_search,
    compare_methods,
    PrimeGeneratingPolynomial,
    PRIME_POLYNOMIALS,
    SearchResult,
)
from prime_plot.pipeline.residual_search import (
    stacked_search,
    modular_search,
    build_density_map,
    combined_mod_coords,
    ResidualSearchResult,
    DensityMap,
    compare_methods as compare_residual_methods,
)
from prime_plot.pipeline.frequency_analysis import (
    analyze_frequencies,
    multi_scale_analysis,
    test_period_predictive_power,
    FrequencyPeak,
    FrequencyAnalysisResult,
)

__all__ = [
    # Predictor
    'PrimePredictor',
    # Polynomial search
    'polynomial_first_search',
    'brute_force_search',
    'compare_methods',
    'PrimeGeneratingPolynomial',
    'PRIME_POLYNOMIALS',
    'SearchResult',
    # Residual pattern search
    'stacked_search',
    'modular_search',
    'build_density_map',
    'combined_mod_coords',
    'ResidualSearchResult',
    'DensityMap',
    'compare_residual_methods',
    # Frequency analysis
    'analyze_frequencies',
    'multi_scale_analysis',
    'test_period_predictive_power',
    'FrequencyPeak',
    'FrequencyAnalysisResult',
]
