"""Training utilities for prime pattern recognition models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        batch_size: Training batch size.
        device: Device to train on ("cuda", "cpu", "npu").
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Steps between logging.
        eval_interval: Epochs between evaluation.
        early_stopping_patience: Epochs without improvement before stopping.
    """

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_interval: int = 10
    eval_interval: int = 1
    early_stopping_patience: int = 10


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance.

    Downweights easy examples and focuses on hard ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)

        # Alpha weighting for positive class
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """BCE loss with class weighting for imbalanced data."""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        return nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class WeightedCombinedLoss(nn.Module):
    """Combined weighted BCE, Focal, and Dice loss for imbalanced data."""

    def __init__(
        self,
        pos_weight: float = 10.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.3,
        focal_weight: float = 0.7,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class Trainer:
    """Training manager for prime pattern models.

    Args:
        model: The neural network model.
        config: Training configuration.
        loss_fn: Loss function (defaults to CombinedLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        loss_fn: nn.Module | None = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.loss_fn = loss_fn or CombinedLoss()

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)

        for images, labels in pbar:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        num_batches = 0

        for images, labels in dataloader:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": total_correct / max(total_pixels, 1),
        }

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: Path) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Training history dictionary.
        """
        print(f"Training on {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["loss"]
                self.history["val_loss"].append(val_loss)

                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

                self.scheduler.step(val_loss)

                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch, is_best)

                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

        return self.history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
) -> dict[str, list[float]]:
    """Convenience function for training a model.

    Args:
        model: Neural network model.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to train on.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        Training history.
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir=Path(checkpoint_dir),
    )

    trainer = Trainer(model, config)
    return trainer.fit(train_loader, val_loader)


@torch.no_grad()
def predict(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Run inference on a single image.

    Args:
        model: Trained model.
        image: Input image tensor (C, H, W) or (H, W).
        device: Device for inference.
        threshold: Classification threshold.

    Returns:
        Binary prediction tensor.
    """
    model.eval()
    model = model.to(device)

    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    output = model(image)
    pred = (torch.sigmoid(output) > threshold).float()

    return pred.squeeze()
