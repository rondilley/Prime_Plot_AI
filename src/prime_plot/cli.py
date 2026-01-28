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

    poly_parser = subparsers.add_parser("polynomial", help="Analyze polynomials")
    poly_parser.add_argument("--search", action="store_true", help="Search for new polynomials")
    poly_parser.add_argument("--a-range", type=int, nargs=2, default=[1, 10], help="Range for a")
    poly_parser.add_argument("--b-range", type=int, nargs=2, default=[-50, 50], help="Range for b")
    poly_parser.add_argument("--c-range", type=int, nargs=2, default=[1, 100], help="Range for c")
    poly_parser.add_argument("--eval-range", type=int, default=100, help="Range to evaluate")
    poly_parser.add_argument("--min-density", type=float, default=0.3, help="Minimum density")

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
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
