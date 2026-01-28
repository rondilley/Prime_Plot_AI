"""Quick start example for prime_plot.

Run this script to generate sample visualizations and test the installation.
"""

from pathlib import Path


def main():
    print("Prime Plot - Quick Start Demo")
    print("=" * 50)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("\n1. Generating Ulam spiral (500x500)...")
    from prime_plot.visualization.ulam import UlamSpiral
    from prime_plot.visualization.renderer import save_raw_image

    spiral = UlamSpiral(500)
    ulam_image = spiral.render_primes() * 255
    save_raw_image(ulam_image, output_dir / "ulam_spiral.png")
    print(f"   Saved to {output_dir / 'ulam_spiral.png'}")

    print("\n2. Generating Sacks spiral (100k integers)...")
    from prime_plot.visualization.sacks import SacksSpiral

    sacks = SacksSpiral(100_000, image_size=800)
    sacks_image = sacks.render_primes(point_size=1)
    save_raw_image(sacks_image, output_dir / "sacks_spiral.png")
    print(f"   Saved to {output_dir / 'sacks_spiral.png'}")

    print("\n3. Generating Klauber triangle (200 rows)...")
    from prime_plot.visualization.klauber import KlauberTriangle

    triangle = KlauberTriangle(200)
    klauber_image = triangle.render_scaled(scale=2)
    save_raw_image(klauber_image, output_dir / "klauber_triangle.png")
    print(f"   Saved to {output_dir / 'klauber_triangle.png'}")

    print("\n4. Analyzing prime-generating polynomials...")
    from prime_plot.core.polynomials import FAMOUS_POLYNOMIALS

    for name, poly in FAMOUS_POLYNOMIALS.items():
        density = poly.prime_density(0, 100)
        consecutive = poly.consecutive_prime_run(0)
        print(f"   {name}: density={density:.3f}, consecutive={consecutive}")

    print("\n5. Testing prime sieve performance...")
    import time
    from prime_plot.core.sieve import generate_primes, count_primes

    start = time.perf_counter()
    primes = generate_primes(10_000_000)
    elapsed = time.perf_counter() - start

    print(f"   Generated {len(primes):,} primes up to 10M in {elapsed:.3f}s")
    print(f"   First 10: {primes[:10].tolist()}")
    print(f"   Last 10: {primes[-10:].tolist()}")

    print("\n6. Computing prime density analysis...")
    from prime_plot.analysis.density import segment_density_analysis

    analysis = segment_density_analysis(1, 100_000, segment_size=10_000)
    print("   Segment | Observed | Theoretical | Ratio")
    print("   " + "-" * 45)
    for i in range(min(5, len(analysis["segments"]))):
        print(f"   {analysis['segments'][i]:>7,} | "
              f"{analysis['densities'][i]:.4f}   | "
              f"{analysis['theoretical'][i]:.4f}      | "
              f"{analysis['ratio'][i]:.3f}")

    print("\n7. Detecting diagonal patterns...")
    from prime_plot.analysis.patterns import detect_diagonal_patterns

    small_spiral = UlamSpiral(200)
    grid = small_spiral.generate_grid()
    patterns = detect_diagonal_patterns(grid, min_length=20, min_density=0.35)

    print(f"   Found {len(patterns)} high-density diagonals")
    if patterns:
        top = patterns[0]
        print(f"   Best: density={top.density:.3f}, length={top.length}, primes={top.prime_count}")

    print("\n" + "=" * 50)
    print("Demo complete. Check the 'output' folder for images.")
    print("\nNext steps:")
    print("  - Run 'prime-plot --help' to see CLI options")
    print("  - Try 'prime-plot ulam --size 1000' for larger spirals")
    print("  - Explore the ML training with 'prime-plot train --epochs 10'")


if __name__ == "__main__":
    main()
