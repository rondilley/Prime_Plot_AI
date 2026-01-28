"""Visualization modules for prime number spirals and patterns."""

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
    ModularGrid,
    ModularClock,
    ModularMatrix,
    CageMatch,
)
from prime_plot.visualization.novel import (
    PyramidPlot,
    ConePlot,
    HexagonalSpiral,
    PrimeGapSpiral,
    PolynomialSpiral,
    DiagonalWave,
    LogarithmicSpiral,
    SquareRootSpiral,
    PrimeFactorSpiral,
    DoubleSpiralInterleave,
)
from prime_plot.visualization.novel_predictive import (
    TwinPrimeSpiral,
    QuadraticResidueGrid,
    SophieGermainHighlight,
    MersenneProximityPlot,
    PrimeGapHistogramPlot,
    DigitSumModularPlot,
    FermatResidueSpiral,
    CollatzStepsPlot,
    PrimitiveRootPattern,
)
from prime_plot.visualization.ulam_3d import (
    generate_3d_spiral_simple,
    visualize_3d_primes,
    analyze_3d_patterns,
)
from prime_plot.visualization.ulam_3d_true import (
    generate_3d_cubic_spiral,
    generate_3d_spiral_helix,
    visualize_3d_slices,
    analyze_true_3d_spiral,
)
from prime_plot.visualization.renderer import render_to_image, save_image

__all__ = [
    # Classic spiral visualizations
    "UlamSpiral",
    "SacksSpiral",
    "KlauberTriangle",
    "VogelSpiral",
    # Fibonacci-based
    "FibonacciSpiral",
    "ReverseFibonacciSpiral",
    "FibonacciShellPlot",
    # Modular arithmetic
    "ModularGrid",
    "ModularClock",
    "ModularMatrix",
    "CageMatch",
    # Novel visualizations
    "PyramidPlot",
    "ConePlot",
    "HexagonalSpiral",
    "PrimeGapSpiral",
    "PolynomialSpiral",
    "DiagonalWave",
    "LogarithmicSpiral",
    "SquareRootSpiral",
    "PrimeFactorSpiral",
    "DoubleSpiralInterleave",
    # Predictive visualizations
    "TwinPrimeSpiral",
    "QuadraticResidueGrid",
    "SophieGermainHighlight",
    "MersenneProximityPlot",
    "PrimeGapHistogramPlot",
    "DigitSumModularPlot",
    "FermatResidueSpiral",
    "CollatzStepsPlot",
    "PrimitiveRootPattern",
    # 3D visualizations
    "generate_3d_spiral_simple",
    "visualize_3d_primes",
    "analyze_3d_patterns",
    "generate_3d_cubic_spiral",
    "generate_3d_spiral_helix",
    "visualize_3d_slices",
    "analyze_true_3d_spiral",
    # Utilities
    "render_to_image",
    "save_image",
]
