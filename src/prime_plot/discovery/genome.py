"""Genome representation for evolutionary visualization discovery.

A genome encodes parameters for a parametric coordinate mapping function
that transforms integers to 2D coordinates for visualization.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar
import json


# Parameter bounds as module constant
PARAM_BOUNDS: Dict[str, tuple] = {
    'r_const': (-10.0, 10.0),
    'r_sqrt': (0.0, 5.0),
    'r_lin': (-0.01, 0.01),
    'r_sin': (0.0, 5.0),
    'r_freq': (0.001, 1.0),
    't_const': (0.0, 2 * np.pi),
    't_sqrt': (0.0, 2 * np.pi),
    't_lin': (0.0, 0.1),
    't_mod': (0.0, 1.0),
    't_mod_base': (2.0, 50.0),
    'x_mod': (0.0, 10.0),
    'x_mod_base': (2.0, 100.0),
    'x_div': (0.0, 10.0),
    'x_div_base': (2.0, 1000.0),
    'y_mod': (0.0, 10.0),
    'y_mod_base': (2.0, 100.0),
    'y_div': (0.0, 10.0),
    'y_div_base': (2.0, 1000.0),
    'blend': (0.0, 1.0),
    'x_digit_sum': (0.0, 5.0),
    'y_digit_sum': (0.0, 5.0),
    'qr_mod': (0.0, 5.0),
    'qr_base': (3.0, 50.0),
}


@dataclass
class VisualizationGenome:
    """Genome encoding a parametric visualization.

    The coordinate mapping uses a hybrid polar/cartesian formula:

    Polar component (spiral-like patterns):
        r = r_const + r_sqrt*sqrt(n) + r_lin*n + r_sin*sin(r_freq*n)
        theta = t_const + t_sqrt*sqrt(n) + t_lin*n + t_mod*(n mod t_mod_base)
        x_polar = r * cos(theta)
        y_polar = r * sin(theta)

    Cartesian component (grid-like patterns):
        x_cart = x_mod*(n mod x_mod_base) + x_div*floor(n/x_div_base)
        y_cart = y_mod*(n mod y_mod_base) + y_div*floor(n/y_div_base)

    Final coordinates:
        x = blend * x_polar + (1-blend) * x_cart + x_noise*digit_sum(n)
        y = blend * y_polar + (1-blend) * y_cart + y_noise*digit_sum(n)

    This allows discovery of spirals, grids, and hybrid patterns.
    """

    # Polar/spiral parameters
    r_const: float = 0.0       # Constant radius offset
    r_sqrt: float = 1.0        # sqrt(n) coefficient for radius
    r_lin: float = 0.0         # Linear n coefficient for radius
    r_sin: float = 0.0         # Sinusoidal amplitude for radius
    r_freq: float = 0.1        # Frequency for sinusoidal radius component

    t_const: float = 0.0       # Constant angle offset
    t_sqrt: float = 1.0        # sqrt(n) coefficient for angle (Sacks-like)
    t_lin: float = 0.0         # Linear n coefficient for angle
    t_mod: float = 0.0         # Modular component coefficient
    t_mod_base: float = 6.0    # Base for modular angle component

    # Cartesian/grid parameters
    x_mod: float = 1.0         # Modular x component coefficient
    x_mod_base: float = 10.0   # Base for modular x
    x_div: float = 0.0         # Division-based x component
    x_div_base: float = 100.0  # Base for division x

    y_mod: float = 0.0         # Modular y component coefficient
    y_mod_base: float = 10.0   # Base for modular y
    y_div: float = 1.0         # Division-based y component
    y_div_base: float = 10.0   # Base for division y

    # Blending and noise
    blend: float = 1.0         # 1.0 = pure polar, 0.0 = pure cartesian
    x_digit_sum: float = 0.0   # Digit sum influence on x
    y_digit_sum: float = 0.0   # Digit sum influence on y

    # Quadratic residue influence (novel component)
    qr_mod: float = 0.0        # Quadratic residue modular component
    qr_base: float = 7.0       # Base for quadratic residue calculation

    # Fitness (not part of genome, set during evaluation)
    fitness: float = field(default=0.0, repr=False)
    generation: int = field(default=0, repr=False)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> 'VisualizationGenome':
        """Create a random genome."""
        if rng is None:
            rng = np.random.default_rng()

        params = {}
        for name, (low, high) in PARAM_BOUNDS.items():
            params[name] = rng.uniform(low, high)

        return cls(**params)

    @classmethod
    def from_preset(cls, preset: str) -> 'VisualizationGenome':
        """Create genome from a known visualization preset."""
        presets = {
            'sacks': cls(
                r_const=0, r_sqrt=1.0, r_lin=0, r_sin=0, r_freq=0,
                t_const=0, t_sqrt=2*np.pi, t_lin=0, t_mod=0, t_mod_base=6,
                blend=1.0,
            ),
            'ulam': cls(
                # Ulam is hard to represent parametrically, approximate with grid
                blend=0.0,
                x_mod=1.0, x_mod_base=2.0, x_div=1.0, x_div_base=2.0,
                y_mod=1.0, y_mod_base=2.0, y_div=1.0, y_div_base=2.0,
            ),
            'modular_6': cls(
                r_const=0, r_sqrt=0.5, r_lin=0, r_sin=0, r_freq=0,
                t_const=0, t_sqrt=0, t_lin=0, t_mod=2*np.pi/6, t_mod_base=6,
                blend=1.0,
            ),
            'modular_30': cls(
                r_const=0, r_sqrt=0.5, r_lin=0, r_sin=0, r_freq=0,
                t_const=0, t_sqrt=0, t_lin=0, t_mod=2*np.pi/30, t_mod_base=30,
                blend=1.0,
            ),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]

    def to_array(self) -> np.ndarray:
        """Convert genome to numpy array for genetic operations."""
        return np.array([
            self.r_const, self.r_sqrt, self.r_lin, self.r_sin, self.r_freq,
            self.t_const, self.t_sqrt, self.t_lin, self.t_mod, self.t_mod_base,
            self.x_mod, self.x_mod_base, self.x_div, self.x_div_base,
            self.y_mod, self.y_mod_base, self.y_div, self.y_div_base,
            self.blend, self.x_digit_sum, self.y_digit_sum,
            self.qr_mod, self.qr_base,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'VisualizationGenome':
        """Create genome from numpy array."""
        return cls(
            r_const=arr[0], r_sqrt=arr[1], r_lin=arr[2], r_sin=arr[3], r_freq=arr[4],
            t_const=arr[5], t_sqrt=arr[6], t_lin=arr[7], t_mod=arr[8], t_mod_base=arr[9],
            x_mod=arr[10], x_mod_base=arr[11], x_div=arr[12], x_div_base=arr[13],
            y_mod=arr[14], y_mod_base=arr[15], y_div=arr[16], y_div_base=arr[17],
            blend=arr[18], x_digit_sum=arr[19], y_digit_sum=arr[20],
            qr_mod=arr[21], qr_base=arr[22],
        )

    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        rng: Optional[np.random.Generator] = None
    ) -> 'VisualizationGenome':
        """Create mutated copy of genome."""
        if rng is None:
            rng = np.random.default_rng()

        arr = self.to_array()
        bounds_list = list(PARAM_BOUNDS.values())

        for i in range(len(arr)):
            if rng.random() < mutation_rate:
                low, high = bounds_list[i]
                range_size = high - low
                # Gaussian mutation
                arr[i] += rng.normal(0, mutation_strength * range_size)
                # Clip to bounds
                arr[i] = np.clip(arr[i], low, high)

        child = self.from_array(arr)
        child.generation = self.generation + 1
        return child

    @staticmethod
    def crossover(
        parent1: 'VisualizationGenome',
        parent2: 'VisualizationGenome',
        rng: Optional[np.random.Generator] = None
    ) -> 'VisualizationGenome':
        """Create child genome via crossover of two parents."""
        if rng is None:
            rng = np.random.default_rng()

        arr1 = parent1.to_array()
        arr2 = parent2.to_array()

        # Blend crossover with random weights per parameter
        weights = rng.random(len(arr1))
        child_arr = weights * arr1 + (1 - weights) * arr2

        child = VisualizationGenome.from_array(child_arr)
        child.generation = max(parent1.generation, parent2.generation) + 1
        return child

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'r_const': self.r_const,
            'r_sqrt': self.r_sqrt,
            'r_lin': self.r_lin,
            'r_sin': self.r_sin,
            'r_freq': self.r_freq,
            't_const': self.t_const,
            't_sqrt': self.t_sqrt,
            't_lin': self.t_lin,
            't_mod': self.t_mod,
            't_mod_base': self.t_mod_base,
            'x_mod': self.x_mod,
            'x_mod_base': self.x_mod_base,
            'x_div': self.x_div,
            'x_div_base': self.x_div_base,
            'y_mod': self.y_mod,
            'y_mod_base': self.y_mod_base,
            'y_div': self.y_div,
            'y_div_base': self.y_div_base,
            'blend': self.blend,
            'x_digit_sum': self.x_digit_sum,
            'y_digit_sum': self.y_digit_sum,
            'qr_mod': self.qr_mod,
            'qr_base': self.qr_base,
            'fitness': self.fitness,
            'generation': self.generation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VisualizationGenome':
        """Create from dictionary."""
        fitness = d.pop('fitness', 0.0)
        generation = d.pop('generation', 0)
        genome = cls(**{k: v for k, v in d.items() if k in PARAM_BOUNDS})
        genome.fitness = fitness
        genome.generation = generation
        return genome

    def describe(self) -> str:
        """Human-readable description of the visualization type."""
        parts = []

        if self.blend > 0.7:
            parts.append("spiral-dominant")
            if self.t_sqrt > 1.0:
                parts.append("Sacks-like")
            if self.t_mod > 0.1:
                parts.append(f"mod-{int(self.t_mod_base)}")
            if self.r_sin > 0.5:
                parts.append("oscillating")
        elif self.blend < 0.3:
            parts.append("grid-dominant")
            if self.x_mod > 0.5:
                parts.append(f"x-mod-{int(self.x_mod_base)}")
            if self.y_div > 0.5:
                parts.append(f"y-div-{int(self.y_div_base)}")
        else:
            parts.append("hybrid")

        if self.qr_mod > 0.5:
            parts.append(f"qr-{int(self.qr_base)}")

        if self.x_digit_sum > 0.5 or self.y_digit_sum > 0.5:
            parts.append("digit-sum")

        return " + ".join(parts) if parts else "mixed"
