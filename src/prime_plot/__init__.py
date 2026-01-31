"""Prime_Plot_AI - GPU/NPU-accelerated prime visualization and pattern recognition."""

__version__ = "0.1.0"

from prime_plot.core.sieve import generate_primes, is_prime, is_prime_array
from prime_plot.visualization.ulam import UlamSpiral
from prime_plot.visualization.sacks import SacksSpiral
from prime_plot.visualization.klauber import KlauberTriangle

__all__ = [
    "generate_primes",
    "is_prime",
    "is_prime_array",
    "UlamSpiral",
    "SacksSpiral",
    "KlauberTriangle",
]
