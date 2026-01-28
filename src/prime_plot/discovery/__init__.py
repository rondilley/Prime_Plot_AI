"""Evolutionary discovery of novel prime visualization techniques.

This module implements genetic algorithms to automatically discover
coordinate mapping functions that create visualizations with high
predictive power for prime detection.
"""

from prime_plot.discovery.genome import VisualizationGenome
from prime_plot.discovery.parametric import ParametricVisualization
from prime_plot.discovery.evolutionary import (
    EvolutionaryDiscovery,
    EvolutionConfig,
    run_discovery,
)

__all__ = [
    "VisualizationGenome",
    "ParametricVisualization",
    "EvolutionaryDiscovery",
    "EvolutionConfig",
    "run_discovery",
]
