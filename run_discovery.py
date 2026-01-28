"""Run evolutionary discovery of novel prime visualizations.

This script uses genetic algorithms to automatically discover
coordinate mapping functions that create visualizations with
high predictive power for prime detection.

Output is organized in output/runs/ with timestamped directories.
"""

import argparse
from pathlib import Path

from prime_plot.discovery import (
    EvolutionaryDiscovery,
    EvolutionConfig,
    VisualizationGenome,
    ParametricVisualization,
)
from prime_plot.evaluation.predictive_power import (
    calculate_predictive_power,
    compare_to_random_baseline,
)
from prime_plot.utils.run_manager import create_run, Run


def run_discovery(
    generations: int = 50,
    population: int = 40,
    seed: int = 42,
    max_n: int = 15000,
    image_size: int = 300,
    description: str = "full",
) -> VisualizationGenome:
    """Run evolutionary discovery with organized output.

    Args:
        generations: Number of generations to evolve
        population: Population size
        seed: Random seed for reproducibility
        max_n: Maximum number to include in visualization
        image_size: Size of visualization images
        description: Short description for run directory

    Returns:
        Best genome found
    """
    # Create run with configuration
    config_dict = {
        "generations": generations,
        "population_size": population,
        "seed": seed,
        "max_n": max_n,
        "image_size": image_size,
        "elite_count": max(2, population // 10),
        "tournament_size": 3,
        "mutation_rate": 0.2,
        "mutation_strength": 0.3,
        "crossover_rate": 0.7,
        "stagnation_limit": min(15, generations // 3),
        "seed_presets": ["sacks", "modular_6", "modular_30"],
    }

    run = create_run(
        run_type="discovery",
        description=description,
        config=config_dict,
        tags=["evolutionary", "visualization"],
    )

    print("=" * 70)
    print("EVOLUTIONARY PRIME VISUALIZATION DISCOVERY")
    print("=" * 70)
    print()
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print()
    print("Configuration:")
    print(f"  Generations: {generations}")
    print(f"  Population: {population}")
    print(f"  Seed: {seed}")
    print(f"  Max N: {max_n}")
    print(f"  Image size: {image_size}")
    print()

    run.log(f"Starting discovery with {generations} generations, population {population}")

    # Create evolution config
    config = EvolutionConfig(
        population_size=population,
        elite_count=config_dict["elite_count"],
        tournament_size=config_dict["tournament_size"],
        mutation_rate=config_dict["mutation_rate"],
        mutation_strength=config_dict["mutation_strength"],
        crossover_rate=config_dict["crossover_rate"],
        generations=generations,
        stagnation_limit=config_dict["stagnation_limit"],
        max_n=max_n,
        image_size=image_size,
        output_dir=run.checkpoints_dir,
        save_interval=max(1, generations // 5),
        verbose=True,
        seed_presets=config_dict["seed_presets"],
    )

    # Run evolution
    discovery = EvolutionaryDiscovery(config, seed=seed)
    best_genome = discovery.run()

    run.log(f"Evolution complete. Best fitness: {best_genome.fitness:.4f}")

    # Evaluate and save results
    print("\n" + "=" * 70)
    print("DETAILED EVALUATION OF BEST DISCOVERY")
    print("=" * 70)

    results = evaluate_and_save(run, best_genome, max_n, image_size, "best")

    # Also evaluate at larger scale
    if max_n < 50000:
        print("\n" + "-" * 70)
        print("Evaluation at larger scale (max_n=50000, image_size=500)")
        print("-" * 70)
        large_results = evaluate_and_save(run, best_genome, 50000, 500, "best_large_scale")
        results["large_scale"] = large_results

    # Save top genomes (population is sorted by fitness after run)
    top_genomes = sorted(discovery.population, key=lambda g: g.fitness, reverse=True)[:5]
    results["top_genomes"] = [
        {
            "rank": i + 1,
            "fitness": g.fitness,
            "description": g.describe(),
            "parameters": g.to_dict(),
        }
        for i, g in enumerate(top_genomes)
    ]

    # Save final results
    run.save_results(
        results,
        summary={
            "best_fitness": best_genome.fitness,
            "best_description": best_genome.describe(),
            "generations_run": generations,
        }
    )

    run.complete(
        status="completed",
        summary={
            "best_fitness": best_genome.fitness,
            "best_description": best_genome.describe(),
        }
    )

    print("\n" + "=" * 70)
    print(f"Run complete! Results saved to: {run.run_dir}")
    print("=" * 70)

    return best_genome


def evaluate_and_save(
    run: Run,
    genome: VisualizationGenome,
    max_n: int,
    image_size: int,
    name_prefix: str,
) -> dict:
    """Evaluate genome and save visualization.

    Args:
        run: Run object for saving outputs
        genome: Genome to evaluate
        max_n: Maximum number to include
        image_size: Image size
        name_prefix: Prefix for saved files

    Returns:
        Evaluation results dictionary
    """
    viz = ParametricVisualization(genome, max_n=max_n, image_size=image_size)
    image = viz.render_primes()

    print(f"\nGenome description: {genome.describe()}")
    print(f"Fitness: {genome.fitness:.4f}")

    # Get stats
    stats = viz.get_stats()
    print(f"\nVisualization stats:")
    print(f"  Prime pixels: {stats['prime_pixels']}")
    print(f"  Unique positions: {stats['unique_positions']}")
    print(f"  Coverage: {stats['coverage']:.4f}")

    # Predictive power metrics
    metrics = calculate_predictive_power(image)
    print(f"\nPredictive power metrics:")
    print(f"  Predictive value: {metrics.predictive_value:.4f}")
    print(f"  Information gain: {metrics.information_gain:.4f}")
    print(f"  Separation score: {metrics.separation_score:.4f}")
    print(f"  Density variance: {metrics.density_variance:.4f}")
    print(f"  High density regions: {metrics.num_high_density_regions}")
    print(f"  Low density regions: {metrics.num_low_density_regions}")

    # Compare to random baseline
    baseline = compare_to_random_baseline(image, num_samples=5)
    print(f"\nCompared to random baseline:")
    print(f"  Improvement ratio: {baseline['improvement_ratio']:.2f}x")
    print(f"  Z-score: {baseline['z_score']:.2f}")
    print(f"  Significantly better: {baseline['significantly_better']}")

    # Key parameters
    print(f"\nKey genome parameters:")
    print(f"  Blend (1=spiral, 0=grid): {genome.blend:.3f}")
    print(f"  Radius: const={genome.r_const:.2f}, sqrt={genome.r_sqrt:.3f}, sin={genome.r_sin:.3f}")
    print(f"  Theta: sqrt={genome.t_sqrt:.3f}, mod={genome.t_mod:.3f} (base {int(genome.t_mod_base)})")
    print(f"  Digit sum influence: x={genome.x_digit_sum:.3f}, y={genome.y_digit_sum:.3f}")
    print(f"  QR influence: {genome.qr_mod:.3f} (base {int(genome.qr_base)})")

    # Save visualization
    run.save_image(image, f"{name_prefix}_n{max_n}")
    run.log(f"Saved {name_prefix} visualization at scale {max_n}")

    return {
        "genome": genome.to_dict(),
        "description": genome.describe(),
        "fitness": genome.fitness,
        "max_n": max_n,
        "image_size": image_size,
        "stats": stats,
        "metrics": {
            "predictive_value": metrics.predictive_value,
            "information_gain": metrics.information_gain,
            "separation_score": metrics.separation_score,
            "density_variance": metrics.density_variance,
            "high_density_regions": metrics.num_high_density_regions,
            "low_density_regions": metrics.num_low_density_regions,
        },
        "baseline_comparison": baseline,
    }


def quick_test() -> VisualizationGenome:
    """Quick test of the discovery system."""
    print("Running quick test (10 generations, small population)...")
    return run_discovery(
        generations=10,
        population=15,
        seed=123,
        max_n=5000,
        image_size=128,
        description="quick_test",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolutionary discovery of prime visualizations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test instead of full discovery"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (default: 50)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=40,
        help="Population size (default: 40)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=15000,
        help="Maximum number to include (default: 15000)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="Visualization image size (default: 300)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="full",
        help="Short description for this run"
    )

    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        run_discovery(
            generations=args.generations,
            population=args.population,
            seed=args.seed,
            max_n=args.max_n,
            image_size=args.image_size,
            description=args.description,
        )
