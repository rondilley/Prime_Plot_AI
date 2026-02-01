"""Prime prediction pipeline using the discovered visualization and trained model.

This pipeline:
1. Takes a range of integers
2. Renders them using the evolutionarily discovered visualization
3. Runs the trained ML model for inference
4. Maps predictions back to candidate prime numbers
5. Outputs predicted primes with confidence scores
"""

from __future__ import annotations

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from prime_plot.discovery.genome import VisualizationGenome
from prime_plot.discovery.parametric import ParametricVisualization
from prime_plot.ml.models import create_model
from prime_plot.core.sieve import prime_sieve_mask


@dataclass
class PrimePrediction:
    """A predicted prime number with confidence."""
    number: int
    confidence: float
    is_actual_prime: Optional[bool] = None

    def __repr__(self):
        actual = f", actual={self.is_actual_prime}" if self.is_actual_prime is not None else ""
        return f"PrimePrediction({self.number}, conf={self.confidence:.3f}{actual})"


@dataclass
class PredictionResult:
    """Results from a prediction run."""
    start_n: int
    end_n: int
    predictions: List[PrimePrediction]
    total_predicted: int
    high_confidence_count: int
    processing_time: float

    # If ground truth available
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Prediction Results for range [{self.start_n:,}, {self.end_n:,})",
            f"  Total predictions: {self.total_predicted:,}",
            f"  High confidence (>0.9): {self.high_confidence_count:,}",
            f"  Processing time: {self.processing_time:.2f}s",
        ]
        if self.accuracy is not None:
            lines.append(f"  Accuracy: {self.accuracy*100:.2f}%")
        if self.precision is not None:
            lines.append(f"  Precision: {self.precision*100:.2f}%")
        if self.recall is not None:
            lines.append(f"  Recall: {self.recall*100:.2f}%")
        if self.f1 is not None:
            lines.append(f"  F1 Score: {self.f1*100:.2f}%")
        return "\n".join(lines)


class PrimePredictor:
    """Pipeline for predicting prime numbers using ML.

    Uses the evolutionarily discovered visualization method
    and trained U-Net model to identify likely primes in a
    given range of integers.
    """

    def __init__(
        self,
        genome_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        block_size: int = 128,
    ):
        """Initialize the prediction pipeline.

        Args:
            genome_path: Path to genome JSON file. Uses default if None.
            model_path: Path to model checkpoint. Uses default if None.
            device: 'cuda', 'cpu', or None for auto-detect.
            block_size: Size of image blocks for processing.
        """
        self.block_size = block_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load genome
        self.genome = self._load_genome(genome_path)

        # Load model
        self.model = self._load_model(model_path)

    def _load_genome(self, genome_path: Optional[Path]) -> VisualizationGenome:
        """Load the visualization genome."""
        if genome_path is None:
            genome_path = Path("output/discovery/top_genomes.json")

        with open(genome_path) as f:
            data = json.load(f)

        if "top_10" in data:
            genome_dict = data["top_10"][0]
        elif "genome" in data:
            genome_dict = data["genome"]
        else:
            genome_dict = data

        return VisualizationGenome.from_dict(genome_dict)

    def _load_model(self, model_path: Optional[Path]):
        """Load the trained model."""
        if model_path is None:
            model_path = Path("checkpoints/discovered/best_model.pt")

        model = create_model(
            'simple_unet',
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256],
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def _render_range(self, start_n: int, end_n: int) -> Tuple[np.ndarray, dict]:
        """Render a range of integers to visualization image.

        Returns:
            image: The rendered visualization
            coord_map: Dictionary mapping (x,y) pixel coords to n values
        """
        # Create visualization for this range
        viz = ParametricVisualization(
            self.genome,
            max_n=end_n,
            image_size=self.block_size,
        )

        # Get coordinates for all numbers in range
        n_values = np.arange(start_n, end_n)
        x_coords, y_coords = viz.compute_coordinates(n_values)

        # Normalize coordinates
        valid = np.isfinite(x_coords) & np.isfinite(y_coords)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        n_valid = n_values[valid]

        if len(x_coords) == 0:
            return np.zeros((self.block_size, self.block_size), dtype=np.float32), {}

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        margin = 0.02
        x_norm = (x_coords - x_min) / x_range * (1 - 2*margin) + margin
        y_norm = (y_coords - y_min) / y_range * (1 - 2*margin) + margin

        px = (x_norm * (self.block_size - 1)).astype(np.int32)
        py = (y_norm * (self.block_size - 1)).astype(np.int32)

        px = np.clip(px, 0, self.block_size - 1)
        py = np.clip(py, 0, self.block_size - 1)

        # Create coordinate mapping (for reverse lookup)
        coord_map: dict[tuple[int, int], list[int]] = {}
        for i in range(len(px)):
            key = (px[i], py[i])
            if key not in coord_map:
                coord_map[key] = []
            coord_map[key].append(int(n_valid[i]))

        # Get prime mask and render
        prime_mask = prime_sieve_mask(end_n)
        image = np.zeros((self.block_size, self.block_size), dtype=np.float32)

        for i in range(len(px)):
            n = n_valid[i]
            if n < len(prime_mask) and prime_mask[n]:
                image[py[i], px[i]] = 1.0

        return image, coord_map

    def predict_range(
        self,
        start_n: int,
        end_n: int,
        confidence_threshold: float = 0.5,
        verify_with_sieve: bool = True,
    ) -> PredictionResult:
        """Predict primes in the given range.

        Args:
            start_n: Start of range (inclusive)
            end_n: End of range (exclusive)
            confidence_threshold: Minimum confidence for prediction
            verify_with_sieve: If True, compare predictions with actual primes

        Returns:
            PredictionResult with predictions and metrics
        """
        import time
        start_time = time.time()

        # Render visualization
        image, coord_map = self._render_range(start_n, end_n)

        # Prepare for model
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            confidence_map = torch.sigmoid(output).cpu().numpy()[0, 0]

        # Extract predictions from confidence map
        predictions = []
        predicted_numbers = set()

        for (x, y), n_list in coord_map.items():
            conf = float(confidence_map[y, x])
            if conf >= confidence_threshold:
                for n in n_list:
                    if n not in predicted_numbers:
                        predicted_numbers.add(n)
                        predictions.append(PrimePrediction(
                            number=n,
                            confidence=conf,
                        ))

        # Sort by number
        predictions.sort(key=lambda p: p.number)

        # Verify with sieve if requested
        if verify_with_sieve:
            prime_mask = prime_sieve_mask(end_n)
            actual_primes = set(np.where(prime_mask[start_n:end_n])[0] + start_n)

            for pred in predictions:
                pred.is_actual_prime = pred.number in actual_primes

            # Calculate metrics
            tp = sum(1 for p in predictions if p.is_actual_prime)
            fp = sum(1 for p in predictions if not p.is_actual_prime)
            fn = len(actual_primes) - tp

            total = end_n - start_n
            tn = total - tp - fp - fn

            acc_val = (tp + tn) / total if total > 0 else 0.0
            prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0.0

            accuracy: Optional[float] = acc_val
            precision: Optional[float] = prec_val
            recall: Optional[float] = rec_val
            f1: Optional[float] = f1_val
        else:
            accuracy = None
            precision = None
            recall = None
            f1 = None

        processing_time = time.time() - start_time

        return PredictionResult(
            start_n=start_n,
            end_n=end_n,
            predictions=predictions,
            total_predicted=len(predictions),
            high_confidence_count=sum(1 for p in predictions if p.confidence > 0.9),
            processing_time=processing_time,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def predict_batch(
        self,
        ranges: List[Tuple[int, int]],
        confidence_threshold: float = 0.5,
        verify_with_sieve: bool = True,
    ) -> List[PredictionResult]:
        """Predict primes for multiple ranges.

        Args:
            ranges: List of (start, end) tuples
            confidence_threshold: Minimum confidence
            verify_with_sieve: Verify against actual primes

        Returns:
            List of PredictionResult objects
        """
        return [
            self.predict_range(start, end, confidence_threshold, verify_with_sieve)
            for start, end in ranges
        ]

    def find_primes_near(
        self,
        target: int,
        window: int = 1000,
        confidence_threshold: float = 0.7,
    ) -> List[PrimePrediction]:
        """Find predicted primes near a target number.

        Args:
            target: Center of search window
            window: Half-width of search window
            confidence_threshold: Minimum confidence

        Returns:
            List of predicted primes near target
        """
        start_n = max(2, target - window)
        end_n = target + window

        result = self.predict_range(
            start_n, end_n,
            confidence_threshold=confidence_threshold,
            verify_with_sieve=True,
        )

        return result.predictions

    def get_prime_density_map(
        self,
        start_n: int,
        end_n: int,
        grid_size: int = 20,
    ) -> np.ndarray:
        """Get predicted prime density map for visualization.

        Args:
            start_n: Start of range
            end_n: End of range
            grid_size: Number of cells in density grid

        Returns:
            2D array of predicted prime densities
        """
        # Render visualization
        image, coord_map = self._render_range(start_n, end_n)

        # Get confidence map
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            confidence_map = torch.sigmoid(output).cpu().numpy()[0, 0]

        # Downsample to density grid
        cell_size = self.block_size // grid_size
        density_map = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                cell = confidence_map[
                    i*cell_size:(i+1)*cell_size,
                    j*cell_size:(j+1)*cell_size
                ]
                density_map[i, j] = np.mean(cell)

        return density_map
