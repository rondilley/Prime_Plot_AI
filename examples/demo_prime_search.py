"""Demonstrate visualization-guided prime search.

This script tests the core hypothesis:
    Can visualization patterns guide the search for unknown primes,
    reducing the number of primality tests needed?

We compare three approaches:
1. Sequential search (baseline)
2. Random search (baseline)
3. Visualization-guided search (using trained model)

If the visualization captures real prime structure, the guided search
should find primes with fewer tests than random search.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prime_plot.pipeline.prime_search import (
    VisualizationGuidedSearch,
    compare_search_methods,
)


def main():
    print("=" * 70)
    print("VISUALIZATION-GUIDED PRIME SEARCH DEMONSTRATION")
    print("=" * 70)
    print()
    print("Hypothesis: Visualization patterns can guide prime search,")
    print("reducing the number of primality tests needed to find primes.")
    print()

    # Test at different scales
    test_cases = [
        (100_000, 10_000, 10, "100K"),
        (1_000_000, 50_000, 10, "1M"),
        (10_000_000, 100_000, 10, "10M"),
        (100_000_000, 200_000, 10, "100M"),
    ]

    results = []

    for start_n, range_size, target_primes, label in test_cases:
        print("-" * 70)
        print(f"Test: Find {target_primes} primes starting near {label}")
        print(f"Range: {start_n:,} to {start_n + range_size:,}")
        print("-" * 70)

        result = compare_search_methods(start_n, range_size, target_primes)
        results.append((label, result))

        print(f"\nSequential search:")
        print(f"  Tests performed: {result['sequential']['tests']:,}")
        print(f"  Time: {result['sequential']['time']:.3f}s")
        print(f"  First prime: {result['sequential']['primes'][0]:,}")

        print(f"\nRandom search:")
        print(f"  Tests performed: {result['random']['tests']:,}")
        print(f"  Time: {result['random']['time']:.3f}s")

        print(f"\nVisualization-guided search:")
        print(f"  Tests performed: {result['guided']['tests']:,}")
        print(f"  Time: {result['guided']['time']:.3f}s")
        print(f"  First prime: {result['guided']['primes'][0]:,}")

        print(f"\nEfficiency:")
        print(f"  Guided vs Sequential: {result['guided_vs_sequential']} fewer tests")

        # Calculate actual efficiency
        seq_tests = result['sequential']['tests']
        guided_tests = result['guided']['tests']
        if guided_tests < seq_tests:
            savings = (seq_tests - guided_tests) / seq_tests * 100
            print(f"  Tests saved: {savings:.1f}%")
        else:
            overhead = (guided_tests - seq_tests) / seq_tests * 100
            print(f"  Overhead: +{overhead:.1f}% (guided required more tests)")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("| Scale | Sequential | Random | Guided | Efficiency |")
    print("|-------|------------|--------|--------|------------|")

    for label, result in results:
        seq = result['sequential']['tests']
        rand = result['random']['tests']
        guided = result['guided']['tests']
        eff = result['guided_vs_sequential']
        print(f"| {label:>5} | {seq:>10,} | {rand:>6,} | {guided:>6,} | {eff:>10} |")

    print()
    print("Analysis:")
    print("-" * 70)

    # Analyze whether guided search helps
    wins = sum(1 for _, r in results if r['guided']['tests'] < r['sequential']['tests'])
    total = len(results)

    if wins > total / 2:
        print(f"Visualization-guided search outperformed sequential in {wins}/{total} tests.")
        print("The visualization patterns DO help focus the search.")
    else:
        print(f"Visualization-guided search only won {wins}/{total} tests.")
        print("At these scales, sequential search may be competitive because")
        print("prime density is still relatively high.")

    print()
    print("Key insight: The true value emerges at VERY large scales where:")
    print("  1. Prime density is much lower (primes are rarer)")
    print("  2. Primality testing is expensive (Miller-Rabin rounds)")
    print("  3. The visualization can identify high-density regions")
    print()
    print("For extremely large primes (cryptographic scale), focusing search")
    print("on high-confidence regions could provide significant savings.")


if __name__ == '__main__':
    main()
