"""Evolve linear genome parameters using genetic algorithm.

Goal: Find parameters that maximize prime detection while maintaining
scale-invariant coordinate behavior.

Output is organized in output/runs/ with timestamped directories.
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from copy import deepcopy
import random

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prime_plot.ml.models import create_model
from prime_plot.core.sieve import generate_primes_range


@dataclass
class LinearGenome:
    """Genome with linear/modular coordinate formulas."""

    # Radial components
    r_const: float = 1.0
    r_scale: float = 0.00001
    r_mod: float = 2.0
    r_mod_base: int = 97

    # Angular components
    t_const: float = 0.0
    t_mod: float = 1.0
    t_mod_base: int = 19

    # Grid components
    x_mod: float = 1.0
    x_mod_base: int = 37
    y_mod: float = 1.0
    y_mod_base: int = 41

    # Effects
    qr_base: int = 25
    qr_effect: float = 0.5
    digit_sum_effect: float = 0.3

    # Blend (0 = pure grid, 1 = pure polar)
    blend: float = 0.3

    # Fitness (computed)
    fitness: float = 0.0
    generation: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 0.2):
        """Create mutated copy."""
        child = deepcopy(self)

        # Float parameters
        float_params = [
            'r_const', 'r_scale', 'r_mod', 't_const', 't_mod',
            'x_mod', 'y_mod', 'qr_effect', 'digit_sum_effect', 'blend'
        ]

        for param in float_params:
            if random.random() < mutation_rate:
                val = getattr(child, param)
                delta = val * mutation_strength * random.gauss(0, 1)
                new_val = val + delta

                # Clamp specific parameters
                if param == 'blend':
                    new_val = max(0.0, min(1.0, new_val))
                elif param == 'r_scale':
                    new_val = max(1e-8, min(0.001, new_val))
                elif param in ['qr_effect', 'digit_sum_effect']:
                    new_val = max(0.0, min(2.0, new_val))
                elif param in ['r_mod', 't_mod', 'x_mod', 'y_mod']:
                    new_val = max(0.1, min(10.0, new_val))

                setattr(child, param, new_val)

        # Integer parameters (mod bases) - mutate to nearby primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]

        int_params = ['r_mod_base', 't_mod_base', 'x_mod_base', 'y_mod_base', 'qr_base']

        for param in int_params:
            if random.random() < mutation_rate:
                # Pick a random prime
                setattr(child, param, random.choice(primes))

        child.fitness = 0.0
        return child

    def crossover(self, other: 'LinearGenome') -> 'LinearGenome':
        """Create child from two parents."""
        child = LinearGenome()

        for field_name in self.__dataclass_fields__:
            if field_name in ['fitness', 'generation']:
                continue
            # Random choice from either parent
            if random.random() < 0.5:
                setattr(child, field_name, getattr(self, field_name))
            else:
                setattr(child, field_name, getattr(other, field_name))

        return child


def digit_sum(n: int) -> int:
    """Sum of digits."""
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total


def compute_coordinates(numbers: np.ndarray, genome: LinearGenome):
    """Compute coordinates using linear genome."""
    n = numbers.astype(np.float64)

    # Polar with linear/modular radial
    r = genome.r_const + genome.r_scale * n + genome.r_mod * (n % genome.r_mod_base)
    theta = genome.t_const + genome.t_mod * (n % genome.t_mod_base) * (2 * np.pi / genome.t_mod_base)

    x_polar = r * np.cos(theta)
    y_polar = r * np.sin(theta)

    # Grid (pure modular)
    x_grid = genome.x_mod * (n % genome.x_mod_base)
    y_grid = genome.y_mod * (n % genome.y_mod_base)

    # QR effect
    qr_vals = np.array([pow(int(num) % genome.qr_base, 2, genome.qr_base) for num in numbers])
    is_qr = (qr_vals < genome.qr_base // 2).astype(float)
    qr_offset = genome.qr_effect * (is_qr - 0.5)

    # Digit sum effect
    ds_vals = np.array([digit_sum(int(num)) % 9 for num in numbers])
    ds_offset = genome.digit_sum_effect * (ds_vals / 9 - 0.5)

    # Blend
    x = genome.blend * x_polar + (1 - genome.blend) * x_grid + qr_offset + ds_offset
    y = genome.blend * y_polar + (1 - genome.blend) * y_grid + qr_offset - ds_offset

    return x, y


def create_visualization(numbers: np.ndarray, genome: LinearGenome, block_size: int = 128):
    """Create input/target images."""
    primes = set(generate_primes_range(int(numbers.min()), int(numbers.max())))

    x, y = compute_coordinates(numbers, genome)
    valid = np.isfinite(x) & np.isfinite(y)

    if not valid.any():
        return None, None

    x_min, x_max = x[valid].min(), x[valid].max()
    y_min, y_max = y[valid].min(), y[valid].max()

    margin = 0.02
    x_norm = (x - x_min) / (x_max - x_min + 1e-10) * (1 - 2*margin) + margin
    y_norm = (y - y_min) / (y_max - y_min + 1e-10) * (1 - 2*margin) + margin

    input_img = np.zeros((block_size, block_size), dtype=np.float32)
    target_img = np.zeros((block_size, block_size), dtype=np.float32)

    for i, n in enumerate(numbers):
        if valid[i]:
            px = int(np.clip(x_norm[i] * (block_size - 1), 0, block_size - 1))
            py = int(np.clip(y_norm[i] * (block_size - 1), 0, block_size - 1))
            input_img[py, px] = 1.0
            if int(n) in primes:
                target_img[py, px] = 1.0

    return input_img, target_img


def evaluate_genome(genome: LinearGenome, n_samples: int = 20, block_size: int = 128) -> float:
    """Evaluate genome fitness by training a small model and measuring accuracy.

    Fitness combines:
    1. Model accuracy on prime detection
    2. Scale consistency (similar performance at different scales)
    """
    from torch.utils.data import Dataset, DataLoader

    # Generate samples at multiple scales for robustness
    scales = [
        (10_000_000, 50_000_000),      # 10M - 50M
        (100_000_000, 500_000_000),    # 100M - 500M
        (500_000_000, 1_500_000_000),  # 500M - 1.5B
    ]

    range_size = block_size * block_size * 4
    samples_per_scale = n_samples // len(scales)

    inputs = []
    targets = []

    for scale_min, scale_max in scales:
        centers = np.random.randint(scale_min, scale_max, size=samples_per_scale * 2)

        for center in centers:
            if len(inputs) >= n_samples:
                break

            start = max(2, center - range_size // 2)
            numbers = np.arange(start, start + range_size, dtype=np.int64)

            inp, tgt = create_visualization(numbers, genome, block_size)
            if inp is not None and tgt.sum() > 0:  # Ensure some primes
                inputs.append(inp)
                targets.append(tgt)

        if len(inputs) >= n_samples:
            break

    if len(inputs) < n_samples // 2:
        return 0.0  # Not enough valid samples

    inputs = np.stack(inputs[:n_samples])
    targets = np.stack(targets[:n_samples])

    # Quick training
    class SimpleDS(Dataset):
        def __init__(self, x, y):
            self.x = torch.from_numpy(x).unsqueeze(1)
            self.y = torch.from_numpy(y).unsqueeze(1)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.y[i]

    n_train = int(0.8 * len(inputs))
    train_ds = SimpleDS(inputs[:n_train], targets[:n_train])
    val_ds = SimpleDS(inputs[n_train:], targets[n_train:])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model('simple_unet', in_channels=1, out_channels=1, features=[16, 32, 64, 128])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Quick training (5 epochs)
    model.train()
    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    total_correct = 0
    total_pixels = 0
    prime_confidence = []
    composite_confidence = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = torch.sigmoid(model(x))

            pred = (out > 0.5).float()
            total_correct += (pred == y).sum().item()
            total_pixels += y.numel()

            # Track confidence for primes vs composites
            prime_mask = y > 0.5
            if prime_mask.any():
                prime_confidence.extend(out[prime_mask].cpu().numpy().tolist())
            composite_mask = (y < 0.5) & (x > 0.5)  # Composite positions with input
            if composite_mask.any():
                composite_confidence.extend(out[composite_mask].cpu().numpy().tolist())

    accuracy = total_correct / total_pixels

    # Compute confidence margin (how well it distinguishes primes from composites)
    if prime_confidence and composite_confidence:
        prime_conf = np.mean(prime_confidence)
        comp_conf = np.mean(composite_confidence)
        margin = prime_conf - comp_conf  # Positive = good discrimination
    else:
        margin = 0.0

    # Fitness combines accuracy and discrimination margin
    # Accuracy matters but discrimination is key
    fitness = 0.3 * accuracy + 0.7 * max(0, margin + 0.5)  # Shift margin to 0-1 range

    return float(fitness)


def evolve(
    population_size: int = 20,
    generations: int = 30,
    elite_count: int = 3,
    mutation_rate: float = 0.3,
    n_eval_samples: int = 20,
    description: str = "linear_genome",
):
    """Run genetic algorithm to find optimal linear genome."""
    from prime_plot.utils.run_manager import create_run

    # Create run with configuration
    config_dict = {
        "population_size": population_size,
        "generations": generations,
        "elite_count": elite_count,
        "mutation_rate": mutation_rate,
        "n_eval_samples": n_eval_samples,
        "scales": ["10M-50M", "100M-500M", "500M-1.5B"],
    }

    run = create_run(
        run_type="evolution",
        description=description,
        config=config_dict,
        tags=["linear-genome", "scale-invariant"],
    )

    print("=" * 70, flush=True)
    print("EVOLVING LINEAR GENOME PARAMETERS", flush=True)
    print("=" * 70, flush=True)
    print(f"Run ID: {run.metadata.run_id}", flush=True)
    print(f"Output: {run.run_dir}", flush=True)
    print(f"Population: {population_size}, Generations: {generations}", flush=True)
    print(f"Elite count: {elite_count}, Mutation rate: {mutation_rate}", flush=True)
    print(flush=True)

    run.log(f"Starting evolution with {generations} generations, population {population_size}")

    # Initialize population with diverse starting points
    population = []

    # Seed with known reasonable genome
    base = LinearGenome(
        r_const=1.0, r_scale=0.00001, r_mod=2.0, r_mod_base=97,
        t_const=0.0, t_mod=1.0, t_mod_base=19,
        x_mod=1.0, x_mod_base=37, y_mod=1.0, y_mod_base=41,
        qr_base=25, qr_effect=0.5, digit_sum_effect=0.3, blend=0.3
    )
    population.append(base)

    # Add variations
    for _ in range(population_size - 1):
        mutant = base.mutate(mutation_rate=0.5, mutation_strength=0.5)
        population.append(mutant)

    best_ever = None
    best_fitness_ever = 0.0

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}", flush=True)
        print("-" * 40, flush=True)

        # Evaluate fitness
        for i, genome in enumerate(population):
            if genome.fitness == 0.0:  # Not yet evaluated
                genome.fitness = evaluate_genome(genome, n_samples=n_eval_samples)
                genome.generation = gen + 1
            print(f"  Genome {i+1}: fitness = {genome.fitness:.4f}", flush=True)

        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best
        if population[0].fitness > best_fitness_ever:
            best_fitness_ever = population[0].fitness
            best_ever = deepcopy(population[0])
            print(f"  ** New best: {best_fitness_ever:.4f}")

        print(f"  Best this gen: {population[0].fitness:.4f}")
        print(f"  Avg fitness: {np.mean([g.fitness for g in population]):.4f}")

        if gen == generations - 1:
            break

        # Selection and reproduction
        new_population = []

        # Keep elite
        for i in range(elite_count):
            elite = deepcopy(population[i])
            new_population.append(elite)

        # Fill rest with offspring
        while len(new_population) < population_size:
            # Tournament selection
            def tournament(k=3):
                contestants = random.sample(population[:population_size//2], k)
                return max(contestants, key=lambda g: g.fitness)

            parent1 = tournament()
            parent2 = tournament()

            # Crossover
            child = parent1.crossover(parent2)

            # Mutate
            child = child.mutate(mutation_rate=mutation_rate)

            new_population.append(child)

        population = new_population

    # Final results
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nBest genome (fitness = {best_ever.fitness:.4f}):")

    for key, val in best_ever.to_dict().items():
        if key not in ['fitness', 'generation']:
            print(f"  {key}: {val}")

    # Save results using run manager
    results = {
        "best_genome": best_ever.to_dict(),
        "best_fitness": float(best_ever.fitness),
        "generations": generations,
        "population_size": population_size,
        "top_5": [g.to_dict() for g in sorted(population, key=lambda x: x.fitness, reverse=True)[:5]]
    }

    run.save_results(
        results,
        summary={
            "best_fitness": best_ever.fitness,
            "best_generation": best_ever.generation,
            "t_mod_base": best_ever.t_mod_base,
            "y_mod_base": best_ever.y_mod_base,
        }
    )

    # Save checkpoint
    run.save_checkpoint(results, "final", is_final=True)

    run.complete(
        status="completed",
        summary={
            "best_fitness": best_ever.fitness,
            "key_params": f"t_mod={best_ever.t_mod_base}, y_mod={best_ever.y_mod_base}",
        }
    )

    print(f"\nResults saved to {run.run_dir}")
    run.log(f"Evolution complete. Best fitness: {best_ever.fitness:.4f}")

    return best_ever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve linear genome parameters for scale-invariant prime visualization"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=15,
        help="Population size (default: 15)"
    )
    parser.add_argument(
        "--elite",
        type=int,
        default=2,
        help="Number of elite to keep (default: 2)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.35,
        help="Mutation rate (default: 0.35)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Evaluation samples per genome (default: 16)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="linear_genome",
        help="Short description for this run"
    )

    args = parser.parse_args()

    # Run evolution
    best_genome = evolve(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elite,
        mutation_rate=args.mutation_rate,
        n_eval_samples=args.samples,
        description=args.description,
    )
