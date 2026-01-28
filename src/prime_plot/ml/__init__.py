"""Machine learning models for prime pattern recognition."""

from prime_plot.ml.models import PrimeUNet, create_model
from prime_plot.ml.models_3d import SimpleUNet3D
from prime_plot.ml.dataset import PrimeSpiralDataset, create_dataloader
from prime_plot.ml.train import Trainer, train_model

__all__ = [
    # 2D models
    "PrimeUNet",
    "create_model",
    # 3D models
    "SimpleUNet3D",
    # Dataset
    "PrimeSpiralDataset",
    "create_dataloader",
    # Training
    "Trainer",
    "train_model",
]
