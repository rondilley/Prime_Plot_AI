"""Evolutionary algorithm for visualization discovery.

Uses genetic algorithms to search the space of parametric visualizations
and find those with high predictive power for prime detection.
"""

from __future__ import annotations

import numpy as np
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from prime_plot.discovery.genome import VisualizationGenome
from prime_plot.discovery.parametric import ParametricVisualization
from prime_plot.evaluation.predictive_power import calculate_predictive_power


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary search."""

    # Population parameters
    population_size: int = 50
    elite_count: int = 5
    tournament_size: int = 3

    # Genetic operators
    mutation_rate: float = 0.15
    mutation_strength: float = 0.25
    crossover_rate: float = 0.7

    # Evolution parameters
    generations: int = 100
    stagnation_limit: int = 20  # Stop if no improvement for N generations

    # Evaluation parameters
    max_n: int = 10000
    image_size: int = 256

    # Output
    output_dir: Optional[Path] = None
    save_interval: int = 10
    verbose: bool = True

    # Seeding with known good visualizations
    seed_presets: List[str] = field(default_factory=lambda: ['sacks', 'modular_6'])


class EvolutionaryDiscovery:
    """Genetic algorithm for discovering prime visualizations."""

    def __init__(self, config: EvolutionConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.population: List[VisualizationGenome] = []
        self.best_genome: Optional[VisualizationGenome] = None
        self.best_fitness: float = 0.0
        self.generation: int = 0
        self.stagnation_count: int = 0

        self.history: Dict[str, List[Any]] = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
        }

        self.output_dir: Optional[Path] = None
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_population(self) -> None:
        """Create initial population."""
        self.population = []

        # Seed with known good presets
        for preset in self.config.seed_presets:
            try:
                genome = VisualizationGenome.from_preset(preset)
                self.population.append(genome)
            except ValueError:
                pass

        # Fill rest with random genomes
        while len(self.population) < self.config.population_size:
            genome = VisualizationGenome.random(self.rng)
            self.population.append(genome)

        if self.config.verbose:
            print(f"Initialized population with {len(self.population)} individuals")
            print(f"  Seeded presets: {self.config.seed_presets}")

    def evaluate_fitness(self, genome: VisualizationGenome) -> float:
        """Evaluate fitness of a genome using predictive power."""
        try:
            viz = ParametricVisualization(
                genome,
                max_n=self.config.max_n,
                image_size=self.config.image_size,
            )

            # Check for degenerate visualizations
            stats = viz.get_stats()
            if stats['is_degenerate']:
                return 0.001  # Very low fitness for degenerate cases

            # Render and evaluate
            image = viz.render_primes()

            # Calculate predictive power
            metrics = calculate_predictive_power(image)

            # Combine metrics into fitness score
            # Weight predictive_value highest, but also consider others
            fitness = (
                0.5 * metrics.predictive_value
                + 0.2 * metrics.information_gain
                + 0.2 * metrics.separation_score
                + 0.1 * min(metrics.density_variance * 10, 1.0)  # Cap variance contribution
            )

            return max(fitness, 0.001)

        except Exception as e:
            if self.config.verbose:
                print(f"    Evaluation error: {e}")
            return 0.001

    def evaluate_population(self) -> None:
        """Evaluate fitness of all individuals."""
        for i, genome in enumerate(self.population):
            if genome.fitness == 0:  # Only evaluate if not already evaluated
                genome.fitness = self.evaluate_fitness(genome)

            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"    Evaluated {i + 1}/{len(self.population)}", end='\r')

        if self.config.verbose:
            print(f"    Evaluated {len(self.population)}/{len(self.population)}")

    def select_parent(self) -> VisualizationGenome:
        """Tournament selection."""
        indices = self.rng.choice(
            len(self.population),
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        tournament: List[VisualizationGenome] = [self.population[int(i)] for i in indices]
        return max(tournament, key=lambda g: g.fitness)

    def calculate_diversity(self) -> float:
        """Calculate population diversity as average pairwise distance."""
        if len(self.population) < 2:
            return 0.0

        arrays = [g.to_array() for g in self.population]
        total_dist: float = 0.0
        count = 0

        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                total_dist += float(np.linalg.norm(arrays[i] - arrays[j]))
                count += 1

        return total_dist / count if count > 0 else 0.0

    def evolve_generation(self) -> None:
        """Create next generation."""
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_genome = self.population[0]
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        # Record history
        fitnesses = [g.fitness for g in self.population]
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['diversity'].append(self.calculate_diversity())

        # Create new population
        new_population = []

        # Elitism - keep best individuals
        for i in range(self.config.elite_count):
            elite = VisualizationGenome.from_array(self.population[i].to_array())
            elite.fitness = self.population[i].fitness
            elite.generation = self.generation
            new_population.append(elite)

        # Generate offspring
        while len(new_population) < self.config.population_size:
            if self.rng.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = VisualizationGenome.crossover(parent1, parent2, self.rng)
            else:
                # Clone parent
                parent = self.select_parent()
                child = VisualizationGenome.from_array(parent.to_array())

            # Mutate
            child = child.mutate(
                self.config.mutation_rate,
                self.config.mutation_strength,
                self.rng,
            )
            child.fitness = 0  # Reset fitness for re-evaluation
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def run(
        self,
        callback: Optional[Callable[[int, Optional[VisualizationGenome], float], None]] = None
    ) -> Optional[VisualizationGenome]:
        """Run evolutionary search.

        Args:
            callback: Optional function called each generation with
                     (generation, best_genome, best_fitness)

        Returns:
            Best discovered genome.
        """
        if self.config.verbose:
            print("=" * 60)
            print("EVOLUTIONARY VISUALIZATION DISCOVERY")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.generations}")
            print(f"Max N: {self.config.max_n}")
            print(f"Image size: {self.config.image_size}")
            print()

        # Initialize
        self.initialize_population()

        start_time = time.time()

        for gen in range(self.config.generations):
            gen_start = time.time()

            if self.config.verbose:
                print(f"Generation {gen + 1}/{self.config.generations}")

            # Evaluate
            self.evaluate_population()

            # Evolve
            self.evolve_generation()

            gen_time = time.time() - gen_start

            if self.config.verbose:
                print(f"  Best fitness: {self.best_fitness:.4f}")
                print(f"  Avg fitness: {self.history['avg_fitness'][-1]:.4f}")
                print(f"  Diversity: {self.history['diversity'][-1]:.2f}")
                print(f"  Time: {gen_time:.1f}s")
                if self.best_genome:
                    print(f"  Best type: {self.best_genome.describe()}")
                print()

            # Callback
            if callback:
                callback(gen, self.best_genome, self.best_fitness)

            # Save checkpoint
            if self.output_dir and (gen + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"gen_{gen + 1:04d}")

            # Check stagnation
            if self.stagnation_count >= self.config.stagnation_limit:
                if self.config.verbose:
                    print(f"Stopping early due to stagnation ({self.stagnation_count} generations)")
                break

        total_time = time.time() - start_time

        if self.config.verbose:
            print("=" * 60)
            print("DISCOVERY COMPLETE")
            print("=" * 60)
            print(f"Total time: {total_time:.1f}s")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Best genome type: {self.best_genome.describe() if self.best_genome else 'None'}")

        # Save final results
        if self.output_dir:
            self.save_checkpoint("final")
            self.save_results()

        return self.best_genome

    def save_checkpoint(self, name: str) -> None:
        """Save population checkpoint."""
        if not self.output_dir:
            return

        checkpoint = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'population': [g.to_dict() for g in self.population],
        }

        path = self.output_dir / f"checkpoint_{name}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def save_results(self) -> None:
        """Save final results and visualizations."""
        if not self.output_dir:
            return

        # Save history
        with open(self.output_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save top genomes
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        top_genomes = {
            'top_10': [g.to_dict() for g in self.population[:10]],
        }
        with open(self.output_dir / "top_genomes.json", 'w') as f:
            json.dump(top_genomes, f, indent=2)

        # Render best visualizations
        from prime_plot.visualization.renderer import save_raw_image

        for i, genome in enumerate(self.population[:5]):
            viz = ParametricVisualization(
                genome,
                max_n=self.config.max_n,
                image_size=self.config.image_size,
            )
            image = viz.render_primes()
            save_raw_image(image, self.output_dir / f"best_{i + 1}_fitness_{genome.fitness:.4f}.png")

    def load_checkpoint(self, path: Path) -> None:
        """Load population from checkpoint."""
        with open(path) as f:
            checkpoint = json.load(f)

        self.generation = checkpoint['generation']
        self.best_fitness = checkpoint['best_fitness']

        if checkpoint['best_genome']:
            self.best_genome = VisualizationGenome.from_dict(checkpoint['best_genome'])

        self.population = [
            VisualizationGenome.from_dict(g) for g in checkpoint['population']
        ]


def run_discovery(
    generations: int = 50,
    population_size: int = 30,
    max_n: int = 10000,
    image_size: int = 256,
    output_dir: str = "output/discovery",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Optional[VisualizationGenome]:
    """Run evolutionary discovery with default settings.

    Args:
        generations: Number of generations to evolve.
        population_size: Size of population.
        max_n: Maximum integer to include in visualizations.
        image_size: Size of rendered images.
        output_dir: Directory to save results.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Best discovered visualization genome.
    """
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        max_n=max_n,
        image_size=image_size,
        output_dir=Path(output_dir) if output_dir else None,
        verbose=verbose,
    )

    discovery = EvolutionaryDiscovery(config, seed=seed)
    return discovery.run()


if __name__ == "__main__":
    # Quick test
    best = run_discovery(
        generations=20,
        population_size=20,
        max_n=5000,
        image_size=128,
        output_dir="output/discovery_test",
        seed=42,
    )

    print("\nBest genome parameters:")
    if best is not None:
        for key, value in best.to_dict().items():
            if key not in ('fitness', 'generation'):
                print(f"  {key}: {value:.4f}")
    else:
        print("  No best genome found")
