"""Multi-visualization dataset for prime pattern recognition.

Uses top-performing visualization methods based on predictive power evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List, Optional, Dict, Any

from prime_plot.core.sieve import prime_sieve_mask, generate_primes
from prime_plot.visualization.ulam import UlamSpiral
from prime_plot.visualization.modular import ModularClock
from prime_plot.visualization.novel_predictive import (
    TwinPrimeSpiral, QuadraticResidueGrid, DigitSumModularPlot,
    PrimitiveRootPattern, FermatResidueSpiral, CollatzStepsPlot
)


class MultiVisualizationDataset(Dataset):
    """Dataset using multiple visualization methods for training.

    Generates samples from top-performing visualization methods based on
    predictive power evaluation results.

    Args:
        block_size: Size of each image block.
        num_blocks: Number of blocks to generate per method.
        start_range: (min, max) range for starting values.
        methods: List of method names to use (default: top performers).
        transform: Optional transform to apply.
        seed: Random seed for reproducibility.
    """

    AVAILABLE_METHODS = [
        'twin_prime_spiral',
        'primitive_root',
        'modular_clock_6',
        'modular_clock_30',
        'digit_sum_modular',
        'quadratic_residue_7x11',
        'fermat_residue',
        'collatz_steps',
        'ulam',
    ]

    def __init__(
        self,
        block_size: int = 256,
        num_blocks: int = 100,
        start_range: tuple[int, int] = (1, 1_000_000),
        methods: Optional[List[str]] = None,
        transform: Callable | None = None,
        seed: int | None = None,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.start_range = start_range
        self.transform = transform

        # Default to top 5 methods by predictive power
        if methods is None:
            methods = [
                'twin_prime_spiral',
                'primitive_root',
                'modular_clock_6',
                'digit_sum_modular',
                'quadratic_residue_7x11',
            ]
        self.methods = methods

        if seed is not None:
            np.random.seed(seed)

        # Generate random starting values for each sample
        total_samples = num_blocks * len(methods)
        self.starts = np.random.randint(
            start_range[0],
            start_range[1],
            size=total_samples,
        )

        # Assign methods to samples
        self.sample_methods = []
        for i in range(total_samples):
            method_idx = i % len(methods)
            self.sample_methods.append(methods[method_idx])

        self._cache: Dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.starts)

    def _generate_visualization(
        self,
        method: str,
        max_n: int,
        image_size: int
    ) -> np.ndarray:
        """Generate visualization using specified method."""

        if method == 'twin_prime_spiral':
            viz = TwinPrimeSpiral(max_n, image_size)
            return viz.render_primes()

        elif method == 'primitive_root':
            prim_root_viz = PrimitiveRootPattern(max_n, image_size)
            return prim_root_viz.render_primes()

        elif method == 'modular_clock_6':
            clock6_viz = ModularClock(max_n, 6, image_size)
            return clock6_viz.render_primes(point_size=1)

        elif method == 'modular_clock_30':
            clock30_viz = ModularClock(max_n, 30, image_size)
            return clock30_viz.render_primes(point_size=1)

        elif method == 'digit_sum_modular':
            ds_viz = DigitSumModularPlot(max_n, image_size)
            return ds_viz.render_primes()

        elif method == 'quadratic_residue_7x11':
            qr_viz = QuadraticResidueGrid(max_n, image_size, 7, 11)
            return qr_viz.render_primes()

        elif method == 'fermat_residue':
            fermat_viz = FermatResidueSpiral(max_n, image_size)
            return fermat_viz.render_primes()

        elif method == 'collatz_steps':
            collatz_viz = CollatzStepsPlot(max_n, image_size)
            return collatz_viz.render_primes()

        elif method == 'ulam':
            ulam_size = int(np.sqrt(max_n)) + 1
            ulam_viz = UlamSpiral(min(ulam_size, image_size))
            return ulam_viz.render_primes() * 255

        else:
            raise ValueError(f"Unknown method: {method}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx in self._cache:
            image, label, method_id = self._cache[idx]
        else:
            start = int(self.starts[idx])
            method = self.sample_methods[idx]

            # Calculate max_n based on block_size squared
            max_n = start + self.block_size * self.block_size

            # Generate visualization
            image = self._generate_visualization(method, max_n, self.block_size)

            # Normalize image
            if image.max() > 0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)

            # For label: create prime density map
            # Use the same image as label (primes are marked)
            label = image.copy()

            # Method ID for potential multi-task learning
            method_id = self.methods.index(method)

            # Cache if small enough
            if len(self._cache) < 50:
                self._cache[idx] = (image, label, method_id)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Add channel dimension
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        if len(label.shape) == 2:
            label = label[np.newaxis, ...]

        return torch.from_numpy(image), torch.from_numpy(label)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        return {
            "index": idx,
            "start": int(self.starts[idx]),
            "method": self.sample_methods[idx],
            "block_size": self.block_size,
        }


class RegionalPredictionDataset(Dataset):
    """Dataset for regional prime density prediction.

    Instead of pixel-wise prediction, this predicts prime density
    for regions of the visualization - matching the predictive power metric.

    Args:
        block_size: Size of each visualization.
        num_samples: Number of samples to generate.
        grid_size: Number of regions per axis for density labels.
        methods: List of visualization methods to use.
        start_range: Range for starting values.
        seed: Random seed.
    """

    def __init__(
        self,
        block_size: int = 256,
        num_samples: int = 500,
        grid_size: int = 8,
        methods: Optional[List[str]] = None,
        start_range: tuple[int, int] = (1, 1_000_000),
        seed: int | None = None,
    ):
        self.block_size = block_size
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.start_range = start_range

        if methods is None:
            methods = [
                'quadratic_residue_7x11',
                'twin_prime_spiral',
                'modular_clock_6',
            ]
        self.methods = methods

        if seed is not None:
            np.random.seed(seed)

        self.starts = np.random.randint(
            start_range[0], start_range[1], size=num_samples
        )
        self.sample_methods = [
            methods[i % len(methods)] for i in range(num_samples)
        ]

        self._multi_dataset = MultiVisualizationDataset(
            block_size=block_size,
            num_blocks=1,
            start_range=start_range,
            methods=methods,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get base visualization
        image, _ = self._multi_dataset[idx % len(self._multi_dataset)]

        # Calculate regional densities as labels
        img_np = image.numpy().squeeze()
        h, w = img_np.shape
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        densities = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                region = img_np[y1:y2, x1:x2]
                densities[i, j] = region.mean()

        # Normalize densities to 0-1 range
        if densities.max() > 0:
            densities = densities / densities.max()

        return image, torch.from_numpy(densities).unsqueeze(0)


def create_multi_dataloader(
    block_size: int = 256,
    num_blocks: int = 100,
    batch_size: int = 8,
    num_workers: int = 0,
    methods: Optional[List[str]] = None,
    start_range: tuple[int, int] = (1, 1_000_000),
    seed: int | None = None,
) -> DataLoader:
    """Create DataLoader for multi-visualization training.

    Args:
        block_size: Size of each image block.
        num_blocks: Blocks per method.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        methods: Visualization methods to use.
        start_range: Range for starting values.
        seed: Random seed.

    Returns:
        Configured DataLoader.
    """
    dataset = MultiVisualizationDataset(
        block_size=block_size,
        num_blocks=num_blocks,
        start_range=start_range,
        methods=methods,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
