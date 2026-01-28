"""Prime search optimization using visualization-based candidate prioritization.

This module implements the core hypothesis: visualization patterns can guide
the search for unknown primes by identifying regions with higher prime density.

The approach:
1. Given a target range (potentially beyond sieve capability)
2. Compute visualization coordinates for candidate numbers
3. Use the trained model to score each candidate's "prime likelihood"
4. Prioritize high-scoring candidates for actual primality testing
5. Measure efficiency gain vs random/sequential search

This addresses the fundamental goal: reduce cycles needed to find large primes.
"""

from __future__ import annotations

import torch
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from prime_plot.discovery.genome import VisualizationGenome
from prime_plot.discovery.parametric import ParametricVisualization
from prime_plot.ml.models import create_model


def miller_rabin_test(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test.

    Args:
        n: Number to test
        k: Number of rounds (higher = more accurate)

    Returns:
        True if probably prime, False if definitely composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses to test
    def check(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    # Test with k random witnesses
    import random
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if not check(a):
            return False
    return True


@dataclass
class SearchResult:
    """Results from a prime search."""
    range_start: int
    range_end: int
    primes_found: List[int]
    candidates_tested: int
    total_candidates: int
    search_time: float
    tests_saved_percent: float
    method: str


class VisualizationGuidedSearch:
    """Use visualization patterns to guide prime number search.

    The key insight: if the visualization creates regions where primes
    cluster, we can compute which candidates fall in high-density regions
    WITHOUT actually knowing if they're prime. Then prioritize testing those.
    """

    def __init__(
        self,
        genome_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load genome
        if genome_path is None:
            genome_path = Path("output/discovery/top_genomes.json")

        with open(genome_path) as f:
            data = json.load(f)

        if "top_10" in data:
            self.genome = VisualizationGenome.from_dict(data["top_10"][0])
        elif "best_genome" in data:
            self.genome = VisualizationGenome.from_dict(data["best_genome"])
        else:
            self.genome = VisualizationGenome.from_dict(data)

        # Load model
        if model_path is None:
            model_path = Path("checkpoints/discovered/best_model.pt")

        self.model = create_model(
            'simple_unet',
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256],
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def compute_candidate_scores(
        self,
        candidates: np.ndarray,
        block_size: int = 128,
    ) -> np.ndarray:
        """Compute prime-likelihood scores for candidates.

        This function creates a DENSE visualization (all integers in range)
        and uses the model to predict prime locations. Scores are then
        extracted for the candidate positions.

        The key insight: the model was trained on dense visualizations
        showing ALL numbers. It learns which REGIONS have primes, not
        which individual points.

        Args:
            candidates: Array of integers to score
            block_size: Size of visualization block

        Returns:
            Array of scores (0-1) for each candidate
        """
        # Get range bounds from candidates
        min_n = int(candidates.min())
        max_n = int(candidates.max()) + 1

        # Create dense array of ALL integers in the range (not just candidates)
        all_numbers = np.arange(min_n, max_n, dtype=np.int64)

        # Create visualization context
        viz = ParametricVisualization(
            self.genome,
            max_n=max_n,
            image_size=block_size,
        )

        # Compute coordinates for ALL numbers (dense input)
        x_all, y_all = viz.compute_coordinates(all_numbers)
        valid_all = np.isfinite(x_all) & np.isfinite(y_all)

        if not valid_all.any():
            return np.zeros(len(candidates))

        x_valid = x_all[valid_all]
        y_valid = y_all[valid_all]

        x_min, x_max = x_valid.min(), x_valid.max()
        y_min, y_max = y_valid.min(), y_valid.max()
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        margin = 0.02

        # Create DENSE input image with all numbers
        dense_image = np.zeros((block_size, block_size), dtype=np.float32)
        x_norm_all = (x_all - x_min) / x_range * (1 - 2*margin) + margin
        y_norm_all = (y_all - y_min) / y_range * (1 - 2*margin) + margin

        for i in range(len(all_numbers)):
            if valid_all[i]:
                px = int(np.clip(x_norm_all[i] * (block_size - 1), 0, block_size - 1))
                py = int(np.clip(y_norm_all[i] * (block_size - 1), 0, block_size - 1))
                dense_image[py, px] = 1.0

        # Run model on dense image to get confidence map
        image_tensor = torch.from_numpy(dense_image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            confidence_map = torch.sigmoid(output).cpu().numpy()[0, 0]

        # Now compute coordinates for candidates and extract their scores
        x_cand, y_cand = viz.compute_coordinates(candidates)
        valid_cand = np.isfinite(x_cand) & np.isfinite(y_cand)

        x_norm_cand = (x_cand - x_min) / x_range * (1 - 2*margin) + margin
        y_norm_cand = (y_cand - y_min) / y_range * (1 - 2*margin) + margin

        # Extract scores for each candidate
        scores = np.zeros(len(candidates))
        for i in range(len(candidates)):
            if valid_cand[i]:
                px = int(np.clip(x_norm_cand[i] * (block_size - 1), 0, block_size - 1))
                py = int(np.clip(y_norm_cand[i] * (block_size - 1), 0, block_size - 1))
                scores[i] = confidence_map[py, px]

        return scores

    def search_with_prioritization(
        self,
        start_n: int,
        end_n: int,
        target_primes: int = 10,
        primality_test: Callable[[int], bool] = miller_rabin_test,
    ) -> SearchResult:
        """Search for primes using visualization-guided prioritization.

        Args:
            start_n: Start of search range
            end_n: End of search range
            target_primes: Stop after finding this many primes
            primality_test: Function to test primality

        Returns:
            SearchResult with found primes and efficiency metrics
        """
        start_time = time.time()

        # Filter to odd candidates only (basic optimization)
        if start_n % 2 == 0:
            start_n += 1
        candidates = np.arange(start_n, end_n, 2)

        # Score all candidates
        scores = self.compute_candidate_scores(candidates)

        # Sort by score (highest first)
        sorted_indices = np.argsort(-scores)

        # Test in priority order
        primes_found = []
        tests_performed = 0

        for idx in sorted_indices:
            candidate = int(candidates[idx])
            tests_performed += 1

            if primality_test(candidate):
                primes_found.append(candidate)
                if len(primes_found) >= target_primes:
                    break

        search_time = time.time() - start_time
        total_candidates = len(candidates)
        tests_saved = (total_candidates - tests_performed) / total_candidates * 100

        return SearchResult(
            range_start=start_n,
            range_end=end_n,
            primes_found=primes_found,
            candidates_tested=tests_performed,
            total_candidates=total_candidates,
            search_time=search_time,
            tests_saved_percent=tests_saved,
            method="visualization_guided",
        )

    def search_sequential(
        self,
        start_n: int,
        end_n: int,
        target_primes: int = 10,
        primality_test: Callable[[int], bool] = miller_rabin_test,
    ) -> SearchResult:
        """Baseline: sequential search without guidance.

        Tests candidates in order until target primes found.
        """
        start_time = time.time()

        if start_n % 2 == 0:
            start_n += 1

        primes_found = []
        tests_performed = 0
        total_candidates = (end_n - start_n) // 2

        for candidate in range(start_n, end_n, 2):
            tests_performed += 1

            if primality_test(candidate):
                primes_found.append(candidate)
                if len(primes_found) >= target_primes:
                    break

        search_time = time.time() - start_time
        tests_saved = (total_candidates - tests_performed) / total_candidates * 100

        return SearchResult(
            range_start=start_n,
            range_end=end_n,
            primes_found=primes_found,
            candidates_tested=tests_performed,
            total_candidates=total_candidates,
            search_time=search_time,
            tests_saved_percent=tests_saved,
            method="sequential",
        )

    def search_random(
        self,
        start_n: int,
        end_n: int,
        target_primes: int = 10,
        primality_test: Callable[[int], bool] = miller_rabin_test,
        seed: int = 42,
    ) -> SearchResult:
        """Baseline: random search without guidance.

        Tests candidates in random order until target primes found.
        """
        start_time = time.time()
        np.random.seed(seed)

        if start_n % 2 == 0:
            start_n += 1
        candidates = np.arange(start_n, end_n, 2)
        np.random.shuffle(candidates)

        primes_found = []
        tests_performed = 0

        for candidate in candidates:
            tests_performed += 1

            if primality_test(int(candidate)):
                primes_found.append(int(candidate))
                if len(primes_found) >= target_primes:
                    break

        search_time = time.time() - start_time
        total_candidates = len(candidates)
        tests_saved = (total_candidates - tests_performed) / total_candidates * 100

        return SearchResult(
            range_start=start_n,
            range_end=end_n,
            primes_found=primes_found,
            candidates_tested=tests_performed,
            total_candidates=total_candidates,
            search_time=search_time,
            tests_saved_percent=tests_saved,
            method="random",
        )


def compare_search_methods(
    start_n: int,
    range_size: int,
    target_primes: int = 10,
) -> dict:
    """Compare visualization-guided vs baseline search methods.

    Args:
        start_n: Start of search range
        range_size: Size of range to search
        target_primes: Number of primes to find

    Returns:
        Dictionary with comparison results
    """
    end_n = start_n + range_size

    searcher = VisualizationGuidedSearch()

    # Run all methods
    guided_result = searcher.search_with_prioritization(start_n, end_n, target_primes)
    sequential_result = searcher.search_sequential(start_n, end_n, target_primes)
    random_result = searcher.search_random(start_n, end_n, target_primes)

    # Calculate efficiency
    guided_efficiency = sequential_result.candidates_tested / guided_result.candidates_tested
    random_efficiency = sequential_result.candidates_tested / random_result.candidates_tested

    return {
        'range': (start_n, end_n),
        'target_primes': target_primes,
        'guided': {
            'tests': guided_result.candidates_tested,
            'time': guided_result.search_time,
            'primes': guided_result.primes_found,
        },
        'sequential': {
            'tests': sequential_result.candidates_tested,
            'time': sequential_result.search_time,
            'primes': sequential_result.primes_found,
        },
        'random': {
            'tests': random_result.candidates_tested,
            'time': random_result.search_time,
            'primes': random_result.primes_found,
        },
        'guided_vs_sequential': f"{guided_efficiency:.2f}x",
        'guided_vs_random': f"{guided_efficiency / random_efficiency:.2f}x" if random_efficiency > 0 else "N/A",
    }
