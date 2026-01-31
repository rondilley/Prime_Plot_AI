"""Frequency domain analysis of prime distributions.

Look for periodic patterns in primes beyond known modular structure.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass
from prime_plot.core.sieve import generate_primes_range


class FrequencyPeak(NamedTuple):
    """A detected frequency peak."""
    frequency: float
    period: float
    magnitude: float
    phase: float
    z_score: float  # How many std devs above noise floor


@dataclass
class FrequencyAnalysisResult:
    """Results from frequency domain analysis."""
    scale: int
    window_size: int
    significant_peaks: List[FrequencyPeak]
    known_periods: Dict[int, float]  # modulus -> magnitude
    unexplained_peaks: List[FrequencyPeak]
    noise_floor: float
    strongest_unexplained: Optional[FrequencyPeak]


def create_prime_indicator(start: int, size: int) -> np.ndarray:
    """Create binary array: 1 where prime, 0 otherwise."""
    primes = set(generate_primes_range(start, start + size))
    indicator = np.zeros(size, dtype=np.float64)
    for i in range(size):
        if (start + i) in primes:
            indicator[i] = 1.0
    return indicator


def create_residual_indicator(start: int, size: int) -> Tuple[np.ndarray, List[int]]:
    """Create prime indicator with mod-6 density removed.

    Only look at numbers that are 1 or 5 mod 6 (the only places primes > 3 can be).
    This removes the dominant mod-6 pattern from the signal.
    """
    primes = set(generate_primes_range(start, start + size))

    # Count positions that are coprime to 6
    coprime_positions = []
    for i in range(size):
        n = start + i
        if n % 6 == 1 or n % 6 == 5:
            coprime_positions.append(i)

    # Create indicator only for coprime positions
    indicator = np.zeros(len(coprime_positions), dtype=np.float64)
    for j, i in enumerate(coprime_positions):
        if (start + i) in primes:
            indicator[j] = 1.0

    # Remove mean (DC component) to center signal
    indicator = indicator - indicator.mean()

    return indicator, coprime_positions


def analyze_frequencies(
    start: int,
    window_size: int = 100_000,
    significance_threshold: float = 5.0,  # z-score threshold
) -> FrequencyAnalysisResult:
    """Analyze frequency content of prime distribution.

    Args:
        start: Starting number
        window_size: Size of analysis window
        significance_threshold: Z-score threshold for peak detection

    Returns:
        FrequencyAnalysisResult with detected peaks
    """
    # Create residual indicator (mod-6 removed)
    indicator, positions = create_residual_indicator(start, window_size)

    if len(indicator) < 100:
        raise ValueError(f"Not enough data points: {len(indicator)}")

    # Apply FFT
    fft_result = np.fft.rfft(indicator)
    magnitudes = np.abs(fft_result)
    phases = np.angle(fft_result)

    # Frequency bins
    n = len(indicator)
    freqs = np.fft.rfftfreq(n)

    # Estimate noise floor (median of magnitudes, excluding DC and very low freq)
    noise_magnitudes = magnitudes[10:]  # Skip first 10 bins
    noise_floor = np.median(noise_magnitudes)
    noise_std = np.std(noise_magnitudes)

    # Find significant peaks
    significant_peaks = []
    for i in range(1, len(magnitudes)):  # Skip DC
        if freqs[i] > 0:
            z_score = (magnitudes[i] - noise_floor) / (noise_std + 1e-10)
            if z_score > significance_threshold:
                period = 1.0 / freqs[i] if freqs[i] > 0 else float('inf')
                peak = FrequencyPeak(
                    frequency=freqs[i],
                    period=period,
                    magnitude=magnitudes[i],
                    phase=phases[i],
                    z_score=z_score,
                )
                significant_peaks.append(peak)

    # Sort by magnitude
    significant_peaks.sort(key=lambda p: p.magnitude, reverse=True)

    # Check which peaks correspond to known moduli
    # After removing mod-6, we still expect peaks at mod-5, mod-7, mod-11, etc.
    known_moduli = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    known_periods = {}

    for mod in known_moduli:
        # The period in our coprime-to-6 sequence for mod-m
        # is roughly m * (6/2) / gcd considerations
        # This is approximate - we check if any peak is close
        expected_freq = 1.0 / mod
        for peak in significant_peaks:
            if abs(peak.frequency - expected_freq) < 0.01 or abs(peak.period - mod) < 1.0:
                known_periods[mod] = peak.magnitude
                break

    # Find unexplained peaks (not near any known modulus)
    unexplained_peaks = []
    for peak in significant_peaks:
        is_explained = False
        for mod in known_moduli:
            # Check if period is close to modulus or simple multiple/fraction
            for mult in [0.5, 1, 2, 3, 4]:
                if abs(peak.period - mod * mult) < 1.0:
                    is_explained = True
                    break
            if is_explained:
                break

        if not is_explained and peak.period > 2 and peak.period < window_size / 10:
            unexplained_peaks.append(peak)

    return FrequencyAnalysisResult(
        scale=start,
        window_size=window_size,
        significant_peaks=significant_peaks[:20],  # Top 20
        known_periods=known_periods,
        unexplained_peaks=unexplained_peaks[:10],  # Top 10 unexplained
        noise_floor=noise_floor,
        strongest_unexplained=unexplained_peaks[0] if unexplained_peaks else None,
    )


def multi_scale_analysis(
    scales: List[int],
    window_size: int = 100_000,
    significance_threshold: float = 5.0,
) -> Dict[str, any]:
    """Run frequency analysis at multiple scales to find consistent patterns.

    Args:
        scales: List of starting points to analyze
        window_size: Size of each analysis window
        significance_threshold: Z-score threshold

    Returns:
        Dictionary with analysis results and cross-scale patterns
    """
    results = []
    all_unexplained = []

    for scale in scales:
        try:
            result = analyze_frequencies(scale, window_size, significance_threshold)
            results.append(result)

            for peak in result.unexplained_peaks:
                all_unexplained.append({
                    'scale': scale,
                    'period': peak.period,
                    'magnitude': peak.magnitude,
                    'z_score': peak.z_score,
                })
        except Exception as e:
            print(f"  Error at scale {scale}: {e}")

    # Look for periods that appear at multiple scales
    if all_unexplained:
        periods = [p['period'] for p in all_unexplained]

        # Cluster similar periods
        period_clusters = {}
        for item in all_unexplained:
            period = item['period']
            # Find or create cluster
            found_cluster = None
            for cluster_period in period_clusters:
                if abs(period - cluster_period) / cluster_period < 0.1:  # Within 10%
                    found_cluster = cluster_period
                    break

            if found_cluster:
                period_clusters[found_cluster].append(item)
            else:
                period_clusters[period] = [item]

        # Find clusters that appear at multiple scales
        consistent_periods = []
        for period, items in period_clusters.items():
            scales_found = set(item['scale'] for item in items)
            if len(scales_found) >= 2:  # Appears at 2+ scales
                avg_period = np.mean([item['period'] for item in items])
                avg_z = np.mean([item['z_score'] for item in items])
                consistent_periods.append({
                    'period': avg_period,
                    'scales_found': len(scales_found),
                    'total_occurrences': len(items),
                    'avg_z_score': avg_z,
                })

        consistent_periods.sort(key=lambda x: x['scales_found'], reverse=True)
    else:
        consistent_periods = []

    return {
        'scales': scales,
        'window_size': window_size,
        'results': results,
        'all_unexplained_peaks': all_unexplained,
        'consistent_periods': consistent_periods,
        'summary': {
            'total_unexplained': len(all_unexplained),
            'consistent_across_scales': len(consistent_periods),
        }
    }


def test_period_predictive_power(
    period: float,
    start: int,
    test_size: int = 50_000,
    num_primes: int = 100,
) -> Dict[str, float]:
    """Test if a detected period has predictive power for finding primes.

    Args:
        period: The period to test
        start: Starting point
        test_size: Range to test in
        num_primes: Number of primes to find

    Returns:
        Dictionary with test results
    """
    from prime_plot.pipeline.residual_search import brute_force_search

    primes = set(generate_primes_range(start, start + test_size))

    # Create phase bins based on period
    num_bins = max(2, min(20, int(period)))
    bin_counts = np.zeros(num_bins)
    bin_totals = np.zeros(num_bins)

    for n in range(start, start + test_size):
        if n % 6 == 1 or n % 6 == 5:  # Only coprime to 6
            phase = (n / period) % 1.0
            bin_idx = int(phase * num_bins) % num_bins
            bin_totals[bin_idx] += 1
            if n in primes:
                bin_counts[bin_idx] += 1

    # Calculate density per bin
    densities = bin_counts / (bin_totals + 1e-10)

    # Find high and low density bins
    mean_density = densities.mean()
    high_bins = np.where(densities > mean_density * 1.1)[0]  # 10% above mean
    low_bins = np.where(densities < mean_density * 0.9)[0]   # 10% below mean

    # Test: search for primes prioritizing high-density phase bins
    # vs brute force
    bf_result = brute_force_search(start, num_primes, seed=42)

    # Guided search using phase
    guided_tests = 0
    guided_found = 0
    candidates = []

    for n in range(start, start + test_size):
        if n % 6 == 1 or n % 6 == 5:
            phase = (n / period) % 1.0
            bin_idx = int(phase * num_bins) % num_bins
            priority = densities[bin_idx]
            candidates.append((n, priority))

    # Sort by priority (high density first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    for n, _ in candidates:
        guided_tests += 1
        if n in primes:
            guided_found += 1
            if guided_found >= num_primes:
                break

    improvement = bf_result.tests_performed / guided_tests if guided_tests > 0 else 0

    return {
        'period': period,
        'num_bins': num_bins,
        'density_variance': float(np.var(densities)),
        'density_range': float(densities.max() - densities.min()),
        'high_density_bins': len(high_bins),
        'low_density_bins': len(low_bins),
        'brute_force_tests': bf_result.tests_performed,
        'guided_tests': guided_tests,
        'improvement': improvement,
    }


def deep_frequency_search(
    scales: List[int],
    window_size: int = 500_000,
    test_all_periods: bool = True,
) -> Dict[str, any]:
    """Deep search for exploitable frequency patterns.

    Tests ALL significant peaks for predictive power, not just unexplained ones.
    """
    from prime_plot.pipeline.residual_search import brute_force_search

    results = {
        'scales': scales,
        'window_size': window_size,
        'period_tests': [],
        'best_period': None,
        'best_improvement': 0.0,
    }

    # Collect all periods to test
    all_periods = set()

    for scale in scales:
        try:
            result = analyze_frequencies(scale, window_size, significance_threshold=3.0)

            # Test all significant peaks
            for peak in result.significant_peaks:
                if 3 < peak.period < 1000:  # Reasonable range
                    all_periods.add(round(peak.period, 1))

        except Exception as e:
            print(f"  Error at scale {scale}: {e}")

    print(f"\nTesting {len(all_periods)} candidate periods for predictive power...")

    # Test each period
    test_scale = scales[0]
    for period in sorted(all_periods):
        test_result = test_period_predictive_power(
            period=period,
            start=test_scale,
            test_size=min(100_000, window_size),
            num_primes=100,
        )

        results['period_tests'].append({
            'period': period,
            'improvement': test_result['improvement'],
            'density_variance': test_result['density_variance'],
        })

        if test_result['improvement'] > results['best_improvement']:
            results['best_improvement'] = test_result['improvement']
            results['best_period'] = period

        if test_result['improvement'] > 1.05:
            print(f"  Period {period:.1f}: {test_result['improvement']:.2f}x improvement")

    # Sort by improvement
    results['period_tests'].sort(key=lambda x: x['improvement'], reverse=True)

    return results
