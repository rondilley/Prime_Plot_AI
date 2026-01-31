"""Command-line interface for prime_plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_ulam(args: argparse.Namespace) -> int:
    """Generate Ulam spiral visualization."""
    from prime_plot.visualization.ulam import UlamSpiral
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating Ulam spiral: size={args.size}, start={args.start}")

    spiral = UlamSpiral(args.size, start=args.start)
    image = spiral.render_primes(use_gpu=args.gpu) * 255

    output = Path(args.output)
    save_raw_image(image, output)

    print(f"Saved to {output}")
    return 0


def cmd_sacks(args: argparse.Namespace) -> int:
    """Generate Sacks spiral visualization."""
    from prime_plot.visualization.sacks import SacksSpiral
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating Sacks spiral: max_n={args.max_n}, size={args.size}")

    spiral = SacksSpiral(args.max_n, image_size=args.size)
    image = spiral.render_primes(point_size=args.point_size, use_gpu=args.gpu)

    output = Path(args.output)
    save_raw_image(image, output)

    print(f"Saved to {output}")
    return 0


def cmd_klauber(args: argparse.Namespace) -> int:
    """Generate Klauber triangle visualization."""
    from prime_plot.visualization.klauber import KlauberTriangle
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating Klauber triangle: rows={args.rows}, scale={args.scale}")

    triangle = KlauberTriangle(args.rows)
    image = triangle.render_scaled(scale=args.scale, use_gpu=args.gpu)

    output = Path(args.output)
    save_raw_image(image, output)

    print(f"Saved to {output}")
    return 0


def cmd_vogel(args: argparse.Namespace) -> int:
    """Generate Vogel spiral visualization (golden angle)."""
    from prime_plot.visualization.vogel import VogelSpiral
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating Vogel spiral: max_n={args.max_n}, size={args.size}, scaling={args.scaling}")

    spiral = VogelSpiral(args.max_n, image_size=args.size, scaling=args.scaling)
    image = spiral.render_primes(point_size=args.point_size, use_gpu=args.gpu)

    output = Path(args.output)
    save_raw_image(image, output)

    print(f"Saved to {output}")
    return 0


def cmd_fibonacci(args: argparse.Namespace) -> int:
    """Generate Fibonacci-based spiral visualization."""
    from prime_plot.visualization.fibonacci import (
        FibonacciSpiral,
        ReverseFibonacciSpiral,
        FibonacciShellPlot,
    )
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating Fibonacci spiral: max_n={args.max_n}, type={args.type}, size={args.size}")

    if args.type == "forward":
        spiral = FibonacciSpiral(args.max_n, image_size=args.size)
    elif args.type == "reverse":
        spiral = ReverseFibonacciSpiral(args.max_n, image_size=args.size)
    elif args.type == "shell":
        spiral = FibonacciShellPlot(args.max_n, image_size=args.size)
    else:
        print(f"Unknown type: {args.type}")
        return 1

    image = spiral.render_primes(point_size=args.point_size, use_gpu=args.gpu)

    output = Path(args.output)
    save_raw_image(image, output)

    # Print statistics for shell type
    if args.type == "shell" and args.stats:
        stats = spiral.shell_statistics()
        print("\nFibonacci shell statistics:")
        print(f"{'Shell':>5} {'Start':>10} {'End':>10} {'Size':>8} {'Primes':>7} {'Density':>8}")
        for s in stats[:15]:
            print(f"{s['shell']:>5} {s['start']:>10} {s['end']:>10} {s['size']:>8} {s['primes']:>7} {s['density']:>8.4f}")

    print(f"Saved to {output}")
    return 0


def cmd_modular(args: argparse.Namespace) -> int:
    """Generate modular arithmetic visualization."""
    from prime_plot.visualization.modular import (
        ModularGrid,
        ModularClock,
        ModularMatrix,
        CageMatch,
    )
    from prime_plot.visualization.renderer import save_raw_image

    print(f"Generating modular {args.type}: max_n={args.max_n}, modulus={args.modulus}")

    if args.type == "grid":
        viz = ModularGrid(args.max_n, args.modulus, args.modulus2 or args.modulus)
        image = viz.render(scale=args.size // args.modulus)
        if args.stats:
            analysis = viz.analyze_residue_classes()
            print(f"\nActive residue classes: {analysis['active_classes']}/{analysis['total_classes']}")
            print("Top residues by prime count:")
            for r in analysis['residues'][:10]:
                print(f"  ({r['residue1']}, {r['residue2']}): {r['count']} primes ({r['fraction']*100:.1f}%)")
    elif args.type == "clock":
        viz = ModularClock(args.max_n, args.modulus, args.size)
        if args.spokes:
            image = viz.render_with_spokes(point_size=args.point_size)
        else:
            image = viz.render_primes(point_size=args.point_size, use_gpu=args.gpu)
    elif args.type == "matrix":
        viz = ModularMatrix(args.max_n, args.modulus)
        image = viz.render(scale=max(1, args.size // args.modulus))
        if args.stats:
            forbidden = viz.find_forbidden_residues()
            if forbidden:
                print("\nResidues with no primes (coprime to modulus):")
                for m, residues in forbidden.items():
                    print(f"  mod {m}: {residues}")
            else:
                print("\nNo forbidden residue classes found (expected).")
    elif args.type == "cage":
        viz = CageMatch(args.max_n, args.modulus, args.size)
        if args.density:
            image = viz.render_density(scale=args.size // args.modulus)
        else:
            image = viz.render_primes(point_size=args.point_size, use_gpu=args.gpu)
        if args.stats:
            period = viz.pisano_period()
            print(f"\nPisano period for mod {args.modulus}: {period}")
    else:
        print(f"Unknown type: {args.type}")
        return 1

    output = Path(args.output)
    save_raw_image(image, output)

    print(f"Saved to {output}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train prime pattern recognition model."""
    from prime_plot.ml.models import create_model
    from prime_plot.ml.dataset import create_dataloader
    from prime_plot.ml.train import train_model

    print(f"Training {args.model} model")
    print(f"  Block size: {args.block_size}")
    print(f"  Num blocks: {args.num_blocks}")
    print(f"  Epochs: {args.epochs}")

    model = create_model(args.model)

    train_loader = create_dataloader(
        block_size=args.block_size,
        num_blocks=int(args.num_blocks * 0.85),
        batch_size=args.batch_size,
        seed=42,
    )

    val_loader = create_dataloader(
        block_size=args.block_size,
        num_blocks=int(args.num_blocks * 0.15),
        batch_size=args.batch_size,
        seed=43,
        augment=False,
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze prime patterns in a spiral."""
    from prime_plot.visualization.ulam import UlamSpiral
    from prime_plot.analysis.patterns import detect_diagonal_patterns, extract_high_density_regions

    print(f"Analyzing Ulam spiral: size={args.size}")

    spiral = UlamSpiral(args.size)
    grid = spiral.generate_grid()
    prime_grid = spiral.render_primes()

    print("\nDetecting diagonal patterns...")
    patterns = detect_diagonal_patterns(grid, min_length=args.min_length, min_density=args.min_density)

    print(f"Found {len(patterns)} high-density diagonals:")
    for i, p in enumerate(patterns[:10]):
        print(f"  {i+1}. Density: {p.density:.3f}, Length: {p.length}, Primes: {p.prime_count}")

    print("\nExtracting high-density regions...")
    regions = extract_high_density_regions(prime_grid, window_size=32, min_density=args.min_density)

    print(f"Found {len(regions)} high-density regions:")
    for i, r in enumerate(regions[:10]):
        print(f"  {i+1}. Density: {r.density:.3f}, Position: ({r.x}, {r.y}), Primes: {r.prime_count}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate and compare visualization methods."""
    from prime_plot.evaluation.pipeline import (
        EvaluationPipeline,
        run_full_evaluation,
        compare_methods_at_scales,
    )

    if args.compare_scales:
        # Compare methods across different scales
        scales = [int(s) for s in args.scales.split(',')]
        print(f"Comparing visualization methods across scales: {scales}")

        results = compare_methods_at_scales(
            scales=scales,
            methods=args.methods.split(',') if args.methods else None,
            verbose=True,
        )

        return 0

    # Standard evaluation
    output_dir = Path(args.output) if args.output else None

    print(f"Running visualization evaluation...")
    print(f"  max_n: {args.max_n:,}")
    print(f"  image_size: {args.size}")

    if args.methods:
        methods = args.methods.split(',')
        print(f"  methods: {methods}")
    else:
        methods = None
        print(f"  methods: all")

    pipeline = EvaluationPipeline(
        max_n=args.max_n,
        image_size=args.size,
        verbose=True,
    )

    results = pipeline.run_evaluation(methods)

    # Generate report
    report = pipeline.generate_report(output_dir)

    # Print detailed results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Rank':<6}{'Method':<25}{'Score':<10}{'SNR(dB)':<10}{'Lines':<10}")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r.name:<25}{r.overall_score():<10.4f}"
              f"{r.metrics.snr:<10.2f}{r.metrics.line_density:<10.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = report['summary']
    print(f"Best method: {summary['best_method']} (score: {summary['best_score']:.4f})")
    print(f"Worst method: {summary['worst_method']} (score: {summary['worst_score']:.4f})")
    print(f"Score range: {summary['score_range']:.4f}")
    print(f"Methods above random baseline: {summary['methods_above_baseline']}/{len(results)}")

    if output_dir:
        print(f"\nResults saved to {output_dir}/")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search for primes using various optimized strategies."""
    import time
    import numpy as np

    # Handle benchmark mode (multiple scales, saves to output/runs/)
    if args.benchmark:
        return _run_search_benchmark(args)

    print(f"Prime Search: Finding {args.count} primes starting at {args.start:,}")
    print(f"Method: {args.method}")
    print("=" * 60)

    if args.method == "polynomial":
        from prime_plot.pipeline.polynomial_search import (
            polynomial_first_search,
            brute_force_search,
            PRIME_POLYNOMIALS,
        )

        if args.compare:
            print("\nComparing polynomial-first vs brute force...")
            print()

            t0 = time.perf_counter()
            poly_result = polynomial_first_search(
                args.start, args.count, seed=args.seed
            )
            poly_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            brute_result = brute_force_search(
                args.start, args.count, seed=args.seed
            )
            brute_time = (time.perf_counter() - t0) * 1000

            improvement = brute_result.tests_performed / poly_result.tests_performed

            print(f"Polynomial-first:")
            print(f"  Tests: {poly_result.tests_performed} "
                  f"({poly_result.polynomial_tests} poly + {poly_result.fallback_tests} fallback)")
            print(f"  Time:  {poly_time:.1f}ms")
            print()
            print(f"Brute force:")
            print(f"  Tests: {brute_result.tests_performed}")
            print(f"  Time:  {brute_time:.1f}ms")
            print()
            print(f"Improvement: {improvement:.2f}x fewer tests with polynomial-first")
        else:
            t0 = time.perf_counter()
            result = polynomial_first_search(
                args.start, args.count, seed=args.seed
            )
            elapsed = (time.perf_counter() - t0) * 1000

            print(f"\nFound {len(result.primes_found)} primes in {elapsed:.1f}ms")
            print(f"Tests performed: {result.tests_performed} "
                  f"({result.polynomial_tests} polynomial, {result.fallback_tests} fallback)")

            if args.verbose:
                print(f"\nPrimes found:")
                for i, p in enumerate(result.primes_found[:20]):
                    print(f"  {i+1}. {p:,}")
                if len(result.primes_found) > 20:
                    print(f"  ... and {len(result.primes_found) - 20} more")

        if args.list_polynomials:
            print("\nPrime-generating polynomials used:")
            for poly in PRIME_POLYNOMIALS:
                print(f"  {poly.name}: {poly.description}")

    elif args.method in ("residual", "stacked", "modular"):
        from prime_plot.pipeline.residual_search import (
            stacked_search,
            modular_search,
            brute_force_search,
            compare_methods as compare_residual_methods,
        )

        if args.compare:
            print("\nComparing residual pattern methods...")
            print()

            t0 = time.perf_counter()
            results = compare_residual_methods(args.start, args.count, seed=args.seed)
            elapsed = (time.perf_counter() - t0) * 1000

            brute = results['brute_force']
            mod = results['modular']
            stacked = results['stacked']

            print(f"Brute Force:")
            print(f"  Tests: {brute.tests_performed}")
            print()
            print(f"Modular (combined residues mod 7,11,13,17,19,23):")
            print(f"  Tests: {mod.tests_performed}")
            print(f"  Improvement: {brute.tests_performed/mod.tests_performed:.2f}x")
            print()
            print(f"Stacked (Modular + Gap Autocorrelation):")
            print(f"  Tests: {stacked.tests_performed}")
            print(f"  Improvement: {brute.tests_performed/stacked.tests_performed:.2f}x")
            print(f"  Gap skips: {stacked.gap_skips}")
            print()
            print(f"Total time: {elapsed:.1f}ms")
        else:
            t0 = time.perf_counter()
            if args.method == "modular":
                result = modular_search(args.start, args.count, seed=args.seed)
            else:  # stacked or residual
                result = stacked_search(args.start, args.count, seed=args.seed)
            elapsed = (time.perf_counter() - t0) * 1000

            print(f"\nFound {len(result.primes_found)} primes in {elapsed:.1f}ms")
            print(f"Tests performed: {result.tests_performed}")
            print(f"High-density hits: {result.high_density_hits}")
            if result.gap_skips > 0:
                print(f"Gap-guided skips: {result.gap_skips}")

            if args.verbose:
                print(f"\nPrimes found:")
                for i, p in enumerate(result.primes_found[:20]):
                    print(f"  {i+1}. {p:,}")
                if len(result.primes_found) > 20:
                    print(f"  ... and {len(result.primes_found) - 20} more")

    else:
        print(f"Unknown method: {args.method}")
        return 1

    return 0


def _run_search_benchmark(args: argparse.Namespace) -> int:
    """Run search benchmark across multiple scales, saving results to output/runs/."""
    import time
    import numpy as np
    from prime_plot.utils.run_manager import create_run
    from prime_plot.pipeline.residual_search import (
        stacked_search,
        modular_search,
        brute_force_search as residual_brute_force,
    )
    from prime_plot.pipeline.polynomial_search import (
        polynomial_first_search,
        brute_force_search as poly_brute_force,
    )

    # Parse scales
    scales = [int(s.strip()) for s in args.scales.split(',')]

    # Create run
    config = {
        'scales': scales,
        'target_primes': args.count,
        'seed': args.seed,
    }

    run = create_run(
        run_type='evaluation',
        description='search_benchmark',
        config=config,
        tags=['search', 'benchmark'],
    )

    print("=" * 60)
    print("PRIME SEARCH BENCHMARK")
    print("=" * 60)
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print(f"Scales: {scales}")
    print(f"Target primes: {args.count}")
    print()

    run.log(f"Starting benchmark with scales {scales}")

    results = {
        'scales': scales,
        'target_primes': args.count,
        'seed': args.seed,
        'by_scale': {},
    }

    all_improvements = {'polynomial': [], 'modular': [], 'stacked': []}

    try:
        for scale in scales:
            run.log(f"Testing scale {scale:,}")
            print(f"\nScale: {scale:,}")
            print("-" * 40)

            scale_results = {}

            # Brute force
            t0 = time.perf_counter()
            bf_result = residual_brute_force(scale, args.count, seed=args.seed)
            bf_time = (time.perf_counter() - t0) * 1000
            scale_results['brute_force'] = {
                'tests': bf_result.tests_performed,
                'time_ms': bf_time,
            }
            print(f"  Brute Force: {bf_result.tests_performed} tests ({bf_time:.1f}ms)")

            # Polynomial
            t0 = time.perf_counter()
            poly_result = polynomial_first_search(scale, args.count, seed=args.seed)
            poly_time = (time.perf_counter() - t0) * 1000
            poly_imp = bf_result.tests_performed / poly_result.tests_performed
            scale_results['polynomial'] = {
                'tests': poly_result.tests_performed,
                'time_ms': poly_time,
                'improvement': poly_imp,
            }
            all_improvements['polynomial'].append(poly_imp)
            print(f"  Polynomial:  {poly_result.tests_performed} tests ({poly_imp:.2f}x)")

            # Modular
            t0 = time.perf_counter()
            mod_result = modular_search(scale, args.count, seed=args.seed)
            mod_time = (time.perf_counter() - t0) * 1000
            mod_imp = bf_result.tests_performed / mod_result.tests_performed
            scale_results['modular'] = {
                'tests': mod_result.tests_performed,
                'time_ms': mod_time,
                'improvement': mod_imp,
            }
            all_improvements['modular'].append(mod_imp)
            print(f"  Modular:     {mod_result.tests_performed} tests ({mod_imp:.2f}x)")

            # Stacked
            t0 = time.perf_counter()
            stacked_result = stacked_search(scale, args.count, seed=args.seed)
            stacked_time = (time.perf_counter() - t0) * 1000
            stacked_imp = bf_result.tests_performed / stacked_result.tests_performed
            scale_results['stacked'] = {
                'tests': stacked_result.tests_performed,
                'time_ms': stacked_time,
                'improvement': stacked_imp,
            }
            all_improvements['stacked'].append(stacked_imp)
            print(f"  Stacked:     {stacked_result.tests_performed} tests ({stacked_imp:.2f}x)")

            results['by_scale'][str(scale)] = scale_results
            run.save_checkpoint({'scale': scale, 'results': scale_results}, f"scale_{scale}")

        # Summary
        summary = {
            'avg_polynomial': float(np.mean(all_improvements['polynomial'])),
            'avg_modular': float(np.mean(all_improvements['modular'])),
            'avg_stacked': float(np.mean(all_improvements['stacked'])),
        }
        results['summary'] = summary

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Average improvements over brute force:")
        print(f"  Polynomial: {summary['avg_polynomial']:.2f}x")
        print(f"  Modular:    {summary['avg_modular']:.2f}x")
        print(f"  Stacked:    {summary['avg_stacked']:.2f}x")

        run.save_results(results, summary=summary)
        run.complete(status='completed', summary=summary)
        run.log("Benchmark completed successfully")

        print()
        print(f"Results saved to: {run.run_dir}")
        return 0

    except Exception as e:
        run.log(f"Error: {e}", level="ERROR")
        run.complete(status='failed', summary={'error': str(e)})
        raise


def cmd_polynomial(args: argparse.Namespace) -> int:
    """Analyze prime-generating polynomials."""
    from prime_plot.core.polynomials import PrimePolynomial, FAMOUS_POLYNOMIALS, find_dense_polynomials

    if args.search:
        print(f"Searching for high-density polynomials...")
        print(f"  a: {args.a_range}, b: {args.b_range}, c: {args.c_range}")

        results = find_dense_polynomials(
            a_range=args.a_range,
            b_range=args.b_range,
            c_range=args.c_range,
            eval_range=(0, args.eval_range),
            min_density=args.min_density,
        )

        print(f"\nFound {len(results)} polynomials with density >= {args.min_density}:")
        for poly, density in results[:20]:
            print(f"  {poly} -> density: {density:.3f}")

    else:
        print("Famous prime-generating polynomials:\n")
        for name, poly in FAMOUS_POLYNOMIALS.items():
            density = poly.prime_density(0, 100)
            consecutive = poly.consecutive_prime_run(0)
            print(f"  {poly}")
            print(f"    Density (n=0..99): {density:.3f}")
            print(f"    Consecutive primes from n=0: {consecutive}")
            print()

    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Run evolutionary discovery of novel prime visualizations."""
    import numpy as np
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
    from prime_plot.utils.run_manager import create_run

    def _convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: _convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Config
    generations = 10 if args.quick else args.generations
    population = 15 if args.quick else args.population
    max_n = 5000 if args.quick else args.max_n
    image_size = 128 if args.quick else args.image_size

    config_dict = {
        "generations": generations,
        "population_size": population,
        "seed": args.seed,
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
        description=args.description,
        config=config_dict,
        tags=["evolutionary", "visualization"],
    )

    print("=" * 70)
    print("EVOLUTIONARY PRIME VISUALIZATION DISCOVERY")
    print("=" * 70)
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print(f"Generations: {generations}, Population: {population}")
    print(f"Max N: {max_n}, Image size: {image_size}")
    print()

    run.log(f"Starting discovery with {generations} generations, population {population}")

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
        images_dir=run.images_dir,
        save_interval=max(1, generations // 5),
        verbose=True,
        seed_presets=config_dict["seed_presets"],
    )

    discovery = EvolutionaryDiscovery(config, seed=args.seed)
    best_genome = discovery.run()

    run.log(f"Evolution complete. Best fitness: {best_genome.fitness:.4f}")

    # Evaluate best
    viz = ParametricVisualization(best_genome, max_n=max_n, image_size=image_size)
    image = viz.render_primes()
    stats = viz.get_stats()
    metrics = calculate_predictive_power(image)
    baseline = compare_to_random_baseline(image, num_samples=5)

    print(f"\nBest genome: {best_genome.describe()}")
    print(f"Fitness: {best_genome.fitness:.4f}")
    print(f"Predictive value: {metrics.predictive_value:.4f}")
    print(f"Improvement vs random: {baseline['improvement_ratio']:.2f}x")

    run.save_image(image, f"best_n{max_n}")

    # Save results
    top_genomes = sorted(discovery.population, key=lambda g: g.fitness, reverse=True)[:5]
    results = _convert_numpy_types({
        "best_genome": best_genome.to_dict(),
        "best_fitness": best_genome.fitness,
        "best_description": best_genome.describe(),
        "stats": stats,
        "metrics": {
            "predictive_value": metrics.predictive_value,
            "information_gain": metrics.information_gain,
            "separation_score": metrics.separation_score,
        },
        "baseline_comparison": baseline,
        "top_genomes": [{"rank": i+1, "fitness": float(g.fitness), "description": g.describe()}
                       for i, g in enumerate(top_genomes)],
    })

    run.save_results(results, summary={
        "best_fitness": float(best_genome.fitness),
        "best_description": best_genome.describe(),
    })
    run.complete(status="completed", summary={
        "best_fitness": best_genome.fitness,
        "best_description": best_genome.describe(),
    })

    print(f"\nResults saved to: {run.run_dir}")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """Evolve linear genome parameters for scale-invariant visualization."""
    import numpy as np
    import torch
    from dataclasses import dataclass, asdict
    from copy import deepcopy
    import random
    from prime_plot.ml.models import create_model
    from prime_plot.core.sieve import generate_primes_range
    from prime_plot.utils.run_manager import create_run
    from torch.utils.data import Dataset, DataLoader

    @dataclass
    class LinearGenome:
        r_const: float = 1.0
        r_scale: float = 0.00001
        r_mod: float = 2.0
        r_mod_base: int = 97
        t_const: float = 0.0
        t_mod: float = 1.0
        t_mod_base: int = 19
        x_mod: float = 1.0
        x_mod_base: int = 37
        y_mod: float = 1.0
        y_mod_base: int = 41
        qr_base: int = 25
        qr_effect: float = 0.5
        digit_sum_effect: float = 0.3
        blend: float = 0.3
        fitness: float = 0.0
        generation: int = 0

        def to_dict(self):
            return asdict(self)

        def mutate(self, mutation_rate=0.3, mutation_strength=0.2):
            child = deepcopy(self)
            float_params = ['r_const', 'r_scale', 'r_mod', 't_const', 't_mod',
                           'x_mod', 'y_mod', 'qr_effect', 'digit_sum_effect', 'blend']
            for param in float_params:
                if random.random() < mutation_rate:
                    val = getattr(child, param)
                    delta = val * mutation_strength * random.gauss(0, 1)
                    new_val = val + delta
                    if param == 'blend':
                        new_val = max(0.0, min(1.0, new_val))
                    elif param == 'r_scale':
                        new_val = max(1e-8, min(0.001, new_val))
                    setattr(child, param, new_val)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                     53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]
            for param in ['r_mod_base', 't_mod_base', 'x_mod_base', 'y_mod_base', 'qr_base']:
                if random.random() < mutation_rate:
                    setattr(child, param, random.choice(primes))
            child.fitness = 0.0
            return child

        def crossover(self, other):
            child = LinearGenome()
            for field_name in ['r_const', 'r_scale', 'r_mod', 'r_mod_base', 't_const', 't_mod',
                              't_mod_base', 'x_mod', 'x_mod_base', 'y_mod', 'y_mod_base',
                              'qr_base', 'qr_effect', 'digit_sum_effect', 'blend']:
                if random.random() < 0.5:
                    setattr(child, field_name, getattr(self, field_name))
                else:
                    setattr(child, field_name, getattr(other, field_name))
            return child

    def digit_sum(n):
        total = 0
        while n > 0:
            total += n % 10
            n //= 10
        return total

    def compute_coords(numbers, genome):
        n = numbers.astype(np.float64)
        r = genome.r_const + genome.r_scale * n + genome.r_mod * (n % genome.r_mod_base)
        theta = genome.t_const + genome.t_mod * (n % genome.t_mod_base) * (2 * np.pi / genome.t_mod_base)
        x_polar = r * np.cos(theta)
        y_polar = r * np.sin(theta)
        x_grid = genome.x_mod * (n % genome.x_mod_base)
        y_grid = genome.y_mod * (n % genome.y_mod_base)
        qr_vals = np.array([pow(int(num) % genome.qr_base, 2, genome.qr_base) for num in numbers])
        is_qr = (qr_vals < genome.qr_base // 2).astype(float)
        qr_offset = genome.qr_effect * (is_qr - 0.5)
        ds_vals = np.array([digit_sum(int(num)) % 9 for num in numbers])
        ds_offset = genome.digit_sum_effect * (ds_vals / 9 - 0.5)
        x = genome.blend * x_polar + (1 - genome.blend) * x_grid + qr_offset + ds_offset
        y = genome.blend * y_polar + (1 - genome.blend) * y_grid + qr_offset - ds_offset
        return x, y

    def create_viz(numbers, genome, block_size=128):
        primes = set(generate_primes_range(int(numbers.min()), int(numbers.max())))
        x, y = compute_coords(numbers, genome)
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

    def evaluate_genome(genome, n_samples=20, block_size=128):
        scales = [(10_000_000, 50_000_000), (100_000_000, 500_000_000), (500_000_000, 1_500_000_000)]
        range_size = block_size * block_size * 4
        samples_per_scale = n_samples // len(scales)
        inputs, targets = [], []
        for scale_min, scale_max in scales:
            centers = np.random.randint(scale_min, scale_max, size=samples_per_scale * 2)
            for center in centers:
                if len(inputs) >= n_samples:
                    break
                start = max(2, center - range_size // 2)
                numbers = np.arange(start, start + range_size, dtype=np.int64)
                inp, tgt = create_viz(numbers, genome, block_size)
                if inp is not None and tgt.sum() > 0:
                    inputs.append(inp)
                    targets.append(tgt)
            if len(inputs) >= n_samples:
                break
        if len(inputs) < n_samples // 2:
            return 0.0
        inputs = np.stack(inputs[:n_samples])
        targets = np.stack(targets[:n_samples])

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
        model.train()
        for _ in range(5):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        model.eval()
        prime_conf, comp_conf = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = torch.sigmoid(model(x))
                prime_mask = y > 0.5
                if prime_mask.any():
                    prime_conf.extend(out[prime_mask].cpu().numpy().tolist())
                composite_mask = (y < 0.5) & (x > 0.5)
                if composite_mask.any():
                    comp_conf.extend(out[composite_mask].cpu().numpy().tolist())
        if prime_conf and comp_conf:
            margin = np.mean(prime_conf) - np.mean(comp_conf)
        else:
            margin = 0.0
        return float(0.3 * 0.5 + 0.7 * max(0, margin + 0.5))

    # Run evolution
    config_dict = {
        "population_size": args.population,
        "generations": args.generations,
        "elite_count": args.elite,
        "mutation_rate": args.mutation_rate,
        "n_eval_samples": args.samples,
    }

    run = create_run(
        run_type="evolution",
        description=args.description,
        config=config_dict,
        tags=["linear-genome", "scale-invariant"],
    )

    print("=" * 70)
    print("EVOLVING LINEAR GENOME PARAMETERS")
    print("=" * 70)
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print()

    run.log(f"Starting evolution with {args.generations} generations")

    population = [LinearGenome()]
    for _ in range(args.population - 1):
        population.append(population[0].mutate(mutation_rate=0.5, mutation_strength=0.5))

    best_ever = None
    best_fitness_ever = 0.0

    for gen in range(args.generations):
        print(f"\nGeneration {gen + 1}/{args.generations}")
        print("-" * 40)

        for i, genome in enumerate(population):
            if genome.fitness == 0.0:
                genome.fitness = evaluate_genome(genome, n_samples=args.samples)
                genome.generation = gen + 1
            print(f"  Genome {i+1}: fitness = {genome.fitness:.4f}")

        population.sort(key=lambda g: g.fitness, reverse=True)

        if population[0].fitness > best_fitness_ever:
            best_fitness_ever = population[0].fitness
            best_ever = deepcopy(population[0])
            print(f"  ** New best: {best_fitness_ever:.4f}")

        print(f"  Best this gen: {population[0].fitness:.4f}")

        if gen == args.generations - 1:
            break

        new_pop = [deepcopy(population[i]) for i in range(args.elite)]
        while len(new_pop) < args.population:
            def tournament(k=3):
                contestants = random.sample(population[:args.population//2], min(k, len(population)//2))
                return max(contestants, key=lambda g: g.fitness)
            parent1, parent2 = tournament(), tournament()
            child = parent1.crossover(parent2).mutate(mutation_rate=args.mutation_rate)
            new_pop.append(child)
        population = new_pop

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nBest genome (fitness = {best_ever.fitness:.4f}):")
    for key, val in best_ever.to_dict().items():
        if key not in ['fitness', 'generation']:
            print(f"  {key}: {val}")

    results = {
        "best_genome": best_ever.to_dict(),
        "best_fitness": float(best_ever.fitness),
        "generations": args.generations,
        "top_5": [g.to_dict() for g in sorted(population, key=lambda x: x.fitness, reverse=True)[:5]]
    }

    run.save_results(results, summary={"best_fitness": best_ever.fitness})
    run.save_checkpoint(results, "final", is_final=True)
    run.complete(status="completed", summary={"best_fitness": best_ever.fitness})

    print(f"\nResults saved to {run.run_dir}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """List and manage experiment runs."""
    from prime_plot.utils.run_manager import get_run_manager

    manager = get_run_manager()

    if args.cleanup:
        print(f"Cleaning up runs, keeping {args.keep} most recent...")
        deleted = manager.cleanup_old_runs(
            keep_count=args.keep,
            run_type=args.type,
            dry_run=not args.force,
        )
        if deleted:
            action = "Deleted" if args.force else "Would delete"
            print(f"{action} {len(deleted)} runs:")
            for run_id in deleted:
                print(f"  - {run_id}")
        else:
            print("No runs to clean up")
        return 0

    runs = manager.list_runs(run_type=args.type, limit=args.limit)

    if not runs:
        print("No runs found")
        if args.type:
            print(f"(filtered by type: {args.type})")
        return 0

    print(f"{'Run ID':<55} {'Type':<12} {'Status':<10}")
    print("-" * 80)

    for r in runs:
        parts = r.metadata.run_id.split("_")
        summary = ""
        if r.metadata.summary:
            if "best_fitness" in r.metadata.summary:
                summary = f"fitness={r.metadata.summary['best_fitness']:.4f}"
            elif "avg_polynomial" in r.metadata.summary:
                summary = f"poly={r.metadata.summary['avg_polynomial']:.2f}x"

        print(f"{r.metadata.run_id:<55} {r.metadata.run_type:<12} {r.metadata.status:<10}")
        if summary:
            print(f"  -> {summary}")

    print(f"\nTotal: {len(runs)} runs shown")
    return 0


def cmd_frequency(args: argparse.Namespace) -> int:
    """Frequency domain analysis of prime distributions."""
    import numpy as np
    from prime_plot.pipeline.frequency_analysis import (
        analyze_frequencies,
        multi_scale_analysis,
        test_period_predictive_power,
    )
    from prime_plot.utils.run_manager import create_run

    # Parse scales
    scales = [int(s.strip()) for s in args.scales.split(',')]

    config = {
        'scales': scales,
        'window_size': args.window,
        'significance_threshold': args.threshold,
    }

    run = create_run(
        run_type='evaluation',
        description='frequency_analysis',
        config=config,
        tags=['frequency', 'fft', 'analysis'],
    )

    print("=" * 70)
    print("FREQUENCY DOMAIN ANALYSIS OF PRIME DISTRIBUTIONS")
    print("=" * 70)
    print(f"Run ID: {run.metadata.run_id}")
    print(f"Output: {run.run_dir}")
    print(f"Scales: {scales}")
    print(f"Window size: {args.window:,}")
    print(f"Significance threshold: {args.threshold} sigma")
    print()

    run.log(f"Starting frequency analysis at scales {scales}")

    try:
        results = multi_scale_analysis(
            scales=scales,
            window_size=args.window,
            significance_threshold=args.threshold,
        )

        # Print results for each scale
        for result in results['results']:
            print(f"\nScale: {result.scale:,}")
            print("-" * 50)
            print(f"  Noise floor: {result.noise_floor:.4f}")
            print(f"  Significant peaks: {len(result.significant_peaks)}")

            if result.known_periods:
                print(f"  Known moduli detected: {list(result.known_periods.keys())}")

            if result.unexplained_peaks:
                print(f"  Unexplained peaks: {len(result.unexplained_peaks)}")
                for i, peak in enumerate(result.unexplained_peaks[:5]):
                    print(f"    {i+1}. Period={peak.period:.2f}, Z={peak.z_score:.1f}")

        # Cross-scale analysis
        print()
        print("=" * 70)
        print("CROSS-SCALE ANALYSIS")
        print("=" * 70)

        if results['consistent_periods']:
            print(f"\nPeriods appearing at multiple scales:")
            for cp in results['consistent_periods'][:10]:
                print(f"  Period={cp['period']:.2f}: {cp['scales_found']} scales, "
                      f"avg Z={cp['avg_z_score']:.1f}")

            # Test predictive power of best consistent period
            if args.test_predictive:
                best_period = results['consistent_periods'][0]['period']
                print(f"\nTesting predictive power of period {best_period:.2f}...")

                test_result = test_period_predictive_power(
                    period=best_period,
                    start=scales[0],
                    test_size=args.window,
                    num_primes=100,
                )

                print(f"  Density variance: {test_result['density_variance']:.6f}")
                print(f"  Brute force tests: {test_result['brute_force_tests']}")
                print(f"  Guided tests: {test_result['guided_tests']}")
                print(f"  Improvement: {test_result['improvement']:.2f}x")

                results['predictive_test'] = test_result
        else:
            print("\nNo consistent unexplained periods found across scales.")

        # Summary
        summary = {
            'total_unexplained': results['summary']['total_unexplained'],
            'consistent_periods': results['summary']['consistent_across_scales'],
            'scales_analyzed': len(scales),
        }

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total unexplained peaks: {summary['total_unexplained']}")
        print(f"  Consistent across scales: {summary['consistent_periods']}")

        # Save results
        save_results = {
            'scales': scales,
            'window_size': args.window,
            'consistent_periods': results['consistent_periods'],
            'summary': summary,
            'by_scale': [
                {
                    'scale': r.scale,
                    'significant_peaks': len(r.significant_peaks),
                    'unexplained_peaks': [
                        {'period': p.period, 'z_score': p.z_score, 'magnitude': p.magnitude}
                        for p in r.unexplained_peaks[:10]
                    ],
                    'known_periods': r.known_periods,
                }
                for r in results['results']
            ],
        }

        if 'predictive_test' in results:
            save_results['predictive_test'] = results['predictive_test']

        run.save_results(save_results, summary=summary)
        run.complete(status='completed', summary=summary)

        print(f"\nResults saved to: {run.run_dir}")
        return 0

    except Exception as e:
        run.log(f"Error: {e}", level="ERROR")
        run.complete(status='failed', summary={'error': str(e)})
        raise


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prime number visualization and pattern recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ulam_parser = subparsers.add_parser("ulam", help="Generate Ulam spiral")
    ulam_parser.add_argument("--size", type=int, default=500, help="Spiral size")
    ulam_parser.add_argument("--start", type=int, default=1, help="Starting number")
    ulam_parser.add_argument("--output", "-o", default="ulam.png", help="Output file")
    ulam_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    sacks_parser = subparsers.add_parser("sacks", help="Generate Sacks spiral")
    sacks_parser.add_argument("--max-n", type=int, default=100000, help="Maximum integer")
    sacks_parser.add_argument("--size", type=int, default=1000, help="Image size")
    sacks_parser.add_argument("--point-size", type=int, default=1, help="Point radius")
    sacks_parser.add_argument("--output", "-o", default="sacks.png", help="Output file")
    sacks_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    klauber_parser = subparsers.add_parser("klauber", help="Generate Klauber triangle")
    klauber_parser.add_argument("--rows", type=int, default=200, help="Number of rows")
    klauber_parser.add_argument("--scale", type=int, default=2, help="Scale factor")
    klauber_parser.add_argument("--output", "-o", default="klauber.png", help="Output file")
    klauber_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    vogel_parser = subparsers.add_parser("vogel", help="Generate Vogel spiral (golden angle)")
    vogel_parser.add_argument("--max-n", type=int, default=100000, help="Maximum integer")
    vogel_parser.add_argument("--size", type=int, default=1000, help="Image size")
    vogel_parser.add_argument("--point-size", type=int, default=1, help="Point radius")
    vogel_parser.add_argument("--scaling", choices=["sqrt", "linear", "log"], default="sqrt", help="Radius scaling")
    vogel_parser.add_argument("--output", "-o", default="vogel.png", help="Output file")
    vogel_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    fib_parser = subparsers.add_parser("fibonacci", help="Generate Fibonacci-based spiral")
    fib_parser.add_argument("--max-n", type=int, default=10000, help="Maximum integer")
    fib_parser.add_argument("--size", type=int, default=1000, help="Image size")
    fib_parser.add_argument("--point-size", type=int, default=2, help="Point radius")
    fib_parser.add_argument("--type", choices=["forward", "reverse", "shell"], default="forward", help="Spiral type")
    fib_parser.add_argument("--stats", action="store_true", help="Print shell statistics")
    fib_parser.add_argument("--output", "-o", default="fibonacci.png", help="Output file")
    fib_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    mod_parser = subparsers.add_parser("modular", help="Generate modular arithmetic visualization")
    mod_parser.add_argument("--max-n", type=int, default=100000, help="Maximum integer")
    mod_parser.add_argument("--modulus", type=int, default=6, help="Primary modulus")
    mod_parser.add_argument("--modulus2", type=int, default=None, help="Secondary modulus (for grid)")
    mod_parser.add_argument("--size", type=int, default=1000, help="Image size")
    mod_parser.add_argument("--point-size", type=int, default=1, help="Point radius")
    mod_parser.add_argument("--type", choices=["grid", "clock", "matrix", "cage"], default="grid", help="Visualization type")
    mod_parser.add_argument("--spokes", action="store_true", help="Show guide spokes (clock only)")
    mod_parser.add_argument("--density", action="store_true", help="Show density grid (cage only)")
    mod_parser.add_argument("--stats", action="store_true", help="Print statistics")
    mod_parser.add_argument("--output", "-o", default="modular.png", help="Output file")
    mod_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate and compare visualization methods")
    eval_parser.add_argument("--max-n", type=int, default=100000, help="Maximum integer")
    eval_parser.add_argument("--size", type=int, default=500, help="Image size")
    eval_parser.add_argument("--methods", type=str, default=None,
                            help="Comma-separated list of methods to evaluate")
    eval_parser.add_argument("--output", "-o", default=None, help="Output directory for report")
    eval_parser.add_argument("--compare-scales", action="store_true",
                            help="Compare methods across multiple scales")
    eval_parser.add_argument("--scales", type=str, default="10000,50000,100000",
                            help="Comma-separated scales for comparison")

    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--model", default="simple_unet", help="Model type")
    train_parser.add_argument("--block-size", type=int, default=256, help="Image block size")
    train_parser.add_argument("--num-blocks", type=int, default=350, help="Number of blocks")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze prime patterns")
    analyze_parser.add_argument("--size", type=int, default=500, help="Spiral size")
    analyze_parser.add_argument("--min-length", type=int, default=10, help="Minimum diagonal length")
    analyze_parser.add_argument("--min-density", type=float, default=0.3, help="Minimum density")

    search_parser = subparsers.add_parser("search", help="Search for primes using optimized strategies")
    search_parser.add_argument("--start", type=int, default=1000000, help="Starting point for search")
    search_parser.add_argument("--count", type=int, default=100, help="Number of primes to find")
    search_parser.add_argument("--method", choices=["polynomial", "residual", "stacked", "modular"],
                              default="stacked",
                              help="Search method: polynomial (2x improvement), "
                                   "modular (1.4x), stacked (1.46x, default)")
    search_parser.add_argument("--compare", action="store_true", help="Compare methods")
    search_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Show found primes")
    search_parser.add_argument("--list-polynomials", action="store_true",
                              help="List polynomials used (polynomial method only)")
    search_parser.add_argument("--benchmark", action="store_true",
                              help="Run benchmark across multiple scales (saves to output/runs/)")
    search_parser.add_argument("--scales", type=str, default="100000,500000,1000000",
                              help="Comma-separated scales for benchmark mode")

    poly_parser = subparsers.add_parser("polynomial", help="Analyze polynomials")
    poly_parser.add_argument("--search", action="store_true", help="Search for new polynomials")
    poly_parser.add_argument("--a-range", type=int, nargs=2, default=[1, 10], help="Range for a")
    poly_parser.add_argument("--b-range", type=int, nargs=2, default=[-50, 50], help="Range for b")
    poly_parser.add_argument("--c-range", type=int, nargs=2, default=[1, 100], help="Range for c")
    poly_parser.add_argument("--eval-range", type=int, default=100, help="Range to evaluate")
    poly_parser.add_argument("--min-density", type=float, default=0.3, help="Minimum density")

    # Evolutionary discovery
    discover_parser = subparsers.add_parser("discover", help="Evolutionary discovery of visualizations")
    discover_parser.add_argument("--quick", action="store_true", help="Quick test (10 generations)")
    discover_parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    discover_parser.add_argument("--population", type=int, default=40, help="Population size")
    discover_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    discover_parser.add_argument("--max-n", type=int, default=15000, help="Max number in visualization")
    discover_parser.add_argument("--image-size", type=int, default=300, help="Image size")
    discover_parser.add_argument("--description", type=str, default="discovery", help="Run description")

    # Linear genome evolution
    evolve_parser = subparsers.add_parser("evolve", help="Evolve linear genome parameters")
    evolve_parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    evolve_parser.add_argument("--population", type=int, default=15, help="Population size")
    evolve_parser.add_argument("--elite", type=int, default=2, help="Elite count")
    evolve_parser.add_argument("--mutation-rate", type=float, default=0.35, help="Mutation rate")
    evolve_parser.add_argument("--samples", type=int, default=16, help="Evaluation samples")
    evolve_parser.add_argument("--description", type=str, default="linear_genome", help="Run description")

    # Run management
    runs_parser = subparsers.add_parser("runs", help="List and manage experiment runs")
    runs_parser.add_argument("--type", type=str, help="Filter by run type")
    runs_parser.add_argument("--limit", type=int, default=20, help="Max runs to show")
    runs_parser.add_argument("--cleanup", action="store_true", help="Clean up old runs")
    runs_parser.add_argument("--keep", type=int, default=10, help="Runs to keep when cleaning")
    runs_parser.add_argument("--force", action="store_true", help="Actually delete (default is dry-run)")

    # Frequency domain analysis
    freq_parser = subparsers.add_parser("frequency", help="FFT analysis of prime distributions")
    freq_parser.add_argument("--scales", type=str, default="1000000,10000000,100000000",
                            help="Comma-separated starting points to analyze")
    freq_parser.add_argument("--window", type=int, default=100000,
                            help="Analysis window size")
    freq_parser.add_argument("--threshold", type=float, default=5.0,
                            help="Significance threshold (z-score)")
    freq_parser.add_argument("--test-predictive", action="store_true",
                            help="Test predictive power of discovered periods")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "ulam": cmd_ulam,
        "sacks": cmd_sacks,
        "klauber": cmd_klauber,
        "vogel": cmd_vogel,
        "fibonacci": cmd_fibonacci,
        "modular": cmd_modular,
        "evaluate": cmd_evaluate,
        "train": cmd_train,
        "analyze": cmd_analyze,
        "polynomial": cmd_polynomial,
        "search": cmd_search,
        "discover": cmd_discover,
        "evolve": cmd_evolve,
        "runs": cmd_runs,
        "frequency": cmd_frequency,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
