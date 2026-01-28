"""Dataset generation for prime pattern recognition training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from prime_plot.visualization.ulam import UlamSpiral
from prime_plot.core.sieve import prime_sieve_mask


class PrimeSpiralDataset(Dataset):
    """Dataset of Ulam spiral image blocks for training.

    Generates blocks from different regions of the Ulam spiral,
    with labels indicating prime/composite for each pixel.

    Args:
        block_size: Size of each image block.
        num_blocks: Number of blocks to generate.
        start_range: (min, max) range for spiral starting values.
        transform: Optional transform to apply to images.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        block_size: int = 256,
        num_blocks: int = 350,
        start_range: tuple[int, int] = (1, 500_000_000),
        transform: Callable | None = None,
        seed: int | None = None,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.start_range = start_range
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)

        self.starts = np.random.randint(
            start_range[0],
            start_range[1],
            size=num_blocks,
        )

        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx in self._cache:
            image, label = self._cache[idx]
        else:
            start = int(self.starts[idx])
            spiral = UlamSpiral(self.block_size, start=start)

            grid = spiral.generate_grid()

            max_val = int(grid.max())
            mask = prime_sieve_mask(max_val + 1)

            label = mask[grid].astype(np.float32)

            image = (grid > 0).astype(np.float32)

            if len(self._cache) < 100:
                self._cache[idx] = (image, label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        return image_tensor, label_tensor

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a specific sample."""
        return {
            "index": idx,
            "start": int(self.starts[idx]),
            "block_size": self.block_size,
            "max_value": int(self.starts[idx]) + self.block_size ** 2 - 1,
        }


class PrimeSpiralImageDataset(Dataset):
    """Dataset using pre-rendered spiral images with prime labels.

    For large-scale training, pre-render spirals to disk and use this
    dataset for efficient loading.

    Args:
        image_dir: Directory containing spiral images.
        label_dir: Directory containing label images.
        transform: Optional transform to apply.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path | None = None,
        transform: Callable | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else self.image_dir / "labels"
        self.transform = transform

        self.image_files = sorted(self.image_dir.glob("*.npy"))

        if not self.image_files:
            self.image_files = sorted(self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]

        if image_path.suffix == ".npy":
            image = np.load(image_path).astype(np.float32)
        else:
            from PIL import Image
            image = np.array(Image.open(image_path)).astype(np.float32) / 255.0

        label_path = self.label_dir / image_path.name
        if label_path.exists():
            if label_path.suffix == ".npy":
                label = np.load(label_path).astype(np.float32)
            else:
                from PIL import Image
                label = np.array(Image.open(label_path)).astype(np.float32) / 255.0
        else:
            label = image.copy()

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        if len(label.shape) == 2:
            label = label[np.newaxis, ...]

        return torch.from_numpy(image), torch.from_numpy(label)


class RandomAugmentation:
    """Random augmentation for spiral images.

    Applies random flips and rotations that preserve the
    mathematical structure of the visualization.
    """

    def __init__(self, p_flip: float = 0.5, p_rotate: float = 0.5):
        self.p_flip = p_flip
        self.p_rotate = p_rotate

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p_flip:
            image = np.fliplr(image).copy()

        if np.random.random() < self.p_flip:
            image = np.flipud(image).copy()

        if np.random.random() < self.p_rotate:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()

        return image


def create_dataloader(
    block_size: int = 256,
    num_blocks: int = 350,
    batch_size: int = 8,
    num_workers: int = 0,
    start_range: tuple[int, int] = (1, 500_000_000),
    augment: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader for prime spiral training.

    Args:
        block_size: Size of each image block.
        num_blocks: Total number of blocks.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        start_range: Range for spiral starting values.
        augment: Apply random augmentation.
        seed: Random seed.

    Returns:
        Configured DataLoader.
    """
    transform = RandomAugmentation() if augment else None

    dataset = PrimeSpiralDataset(
        block_size=block_size,
        num_blocks=num_blocks,
        start_range=start_range,
        transform=transform,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def generate_training_data(
    output_dir: str | Path,
    num_samples: int = 1000,
    block_size: int = 256,
    start_range: tuple[int, int] = (1, 500_000_000),
    seed: int = 42,
) -> None:
    """Pre-generate training data to disk.

    Args:
        output_dir: Directory to save generated data.
        num_samples: Number of samples to generate.
        block_size: Size of each block.
        start_range: Range for starting values.
        seed: Random seed.
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    starts = np.random.randint(start_range[0], start_range[1], size=num_samples)

    from tqdm import tqdm

    for i, start in enumerate(tqdm(starts, desc="Generating training data")):
        spiral = UlamSpiral(block_size, start=int(start))
        grid = spiral.generate_grid()

        image = (grid > 0).astype(np.uint8)

        max_val = int(grid.max())
        mask = prime_sieve_mask(max_val + 1)
        label = mask[grid].astype(np.uint8)

        np.save(image_dir / f"sample_{i:06d}.npy", image)
        np.save(label_dir / f"sample_{i:06d}.npy", label)

    metadata = {
        "num_samples": num_samples,
        "block_size": block_size,
        "start_range": start_range,
        "seed": seed,
        "starts": starts.tolist(),
    }

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
