"""Demo script for the prime prediction pipeline.

Shows how to use the PrimePredictor to find primes in various ranges.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prime_plot.pipeline import PrimePredictor


def main():
    print("=" * 70)
    print("PRIME PREDICTION PIPELINE DEMO")
    print("=" * 70)

    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = PrimePredictor()
    print(f"  Device: {predictor.device}")
    print(f"  Block size: {predictor.block_size}")
    print(f"  Genome: {predictor.genome.describe()}")

    # Demo 1: Predict in a small range
    print("\n" + "-" * 70)
    print("DEMO 1: Small Range Prediction (1000-2000)")
    print("-" * 70)

    result = predictor.predict_range(1000, 2000, confidence_threshold=0.5)
    print(result.summary())

    print("\n  Sample predictions:")
    for pred in result.predictions[:10]:
        status = "CORRECT" if pred.is_actual_prime else "FALSE POSITIVE"
        print(f"    {pred.number}: conf={pred.confidence:.3f} [{status}]")

    # Demo 2: Large scale prediction
    print("\n" + "-" * 70)
    print("DEMO 2: Large Scale Prediction (1M range)")
    print("-" * 70)

    result = predictor.predict_range(1_000_000, 1_010_000, confidence_threshold=0.5)
    print(result.summary())

    print("\n  Top 10 highest confidence predictions:")
    sorted_preds = sorted(result.predictions, key=lambda p: -p.confidence)
    for pred in sorted_preds[:10]:
        status = "CORRECT" if pred.is_actual_prime else "FALSE POSITIVE"
        print(f"    {pred.number}: conf={pred.confidence:.3f} [{status}]")

    # Demo 3: Find primes near a target
    print("\n" + "-" * 70)
    print("DEMO 3: Find Primes Near 10 Million")
    print("-" * 70)

    primes_near_10M = predictor.find_primes_near(10_000_000, window=5000, confidence_threshold=0.7)
    print(f"  Found {len(primes_near_10M)} predicted primes near 10M")

    actual_primes = [p for p in primes_near_10M if p.is_actual_prime]
    print(f"  Actual primes: {len(actual_primes)}")
    print(f"  False positives: {len(primes_near_10M) - len(actual_primes)}")

    print("\n  Primes closest to 10M:")
    sorted_by_distance = sorted(primes_near_10M, key=lambda p: abs(p.number - 10_000_000))
    for pred in sorted_by_distance[:5]:
        if pred.is_actual_prime:
            distance = pred.number - 10_000_000
            print(f"    {pred.number} (distance: {distance:+d})")

    # Demo 4: Batch prediction at multiple scales
    print("\n" + "-" * 70)
    print("DEMO 4: Batch Prediction at Multiple Scales")
    print("-" * 70)

    ranges = [
        (100_000, 100_500),
        (1_000_000, 1_000_500),
        (10_000_000, 10_000_500),
        (50_000_000, 50_000_500),
    ]

    results = predictor.predict_batch(ranges, confidence_threshold=0.5)

    print("\n  Results by scale:")
    for result in results:
        scale = result.start_n
        print(f"    {scale:>12,}: Acc={result.accuracy*100:.1f}%, "
              f"Prec={result.precision*100:.1f}%, "
              f"Rec={result.recall*100:.1f}%, "
              f"F1={result.f1*100:.1f}%")

    # Demo 5: Density map
    print("\n" + "-" * 70)
    print("DEMO 5: Prime Density Map")
    print("-" * 70)

    density_map = predictor.get_prime_density_map(1_000_000, 1_100_000, grid_size=10)
    print("  Predicted prime density map (10x10 grid):")
    print("  (Values show average confidence in each region)")

    for row in density_map:
        row_str = " ".join(f"{v:.2f}" for v in row)
        print(f"    {row_str}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nThe prediction pipeline successfully:")
    print("  1. Loads the evolutionarily discovered visualization genome")
    print("  2. Loads the trained U-Net model")
    print("  3. Renders integer ranges using the discovered coordinate mapping")
    print("  4. Runs inference to get confidence scores")
    print("  5. Maps predictions back to candidate prime numbers")

    avg_f1 = sum(r.f1 for r in results) / len(results)
    print(f"\nAverage F1 score across all scales: {avg_f1*100:.1f}%")

    # Save demo results
    demo_results = {
        'small_range': {
            'range': [1000, 2000],
            'accuracy': result.accuracy,
            'total_predicted': result.total_predicted,
        },
        'batch_results': [
            {
                'range': [r.start_n, r.end_n],
                'accuracy': r.accuracy,
                'f1': r.f1,
            }
            for r in results
        ],
        'average_f1': avg_f1,
    }

    output_path = Path("output/prediction_demo_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(demo_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
