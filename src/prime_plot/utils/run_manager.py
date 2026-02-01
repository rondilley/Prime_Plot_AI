"""Run management utilities for organizing experiment outputs.

Provides consistent naming, timestamping, and organization for all
experiment runs including discovery, training, and evaluation.

Directory Structure:
    output/
        runs/
            YYYYMMDD_HHMMSS_<run_type>_<description>/
                config.json       # Full configuration
                results.json      # Final results with metadata
                checkpoints/      # Intermediate saves
                images/           # Generated visualizations
                models/           # Trained model weights
                logs/             # Training/evolution logs

Run Types:
    - discovery: Evolutionary visualization discovery
    - evolution: Linear genome evolution
    - training: Model training runs
    - evaluation: Method evaluation/comparison
    - inference: Inference/prediction runs
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class RunMetadata:
    """Metadata for a run."""
    run_id: str
    run_type: str
    description: str
    created_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    parent_run: Optional[str] = None  # For runs that build on previous runs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunMetadata':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RunManager:
    """Manages experiment runs with consistent organization."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize run manager.

        Args:
            base_dir: Base directory for outputs. Defaults to ./output
        """
        if base_dir is None:
            # Find project root (contains src/)
            current = Path.cwd()
            while current != current.parent:
                if (current / "src" / "prime_plot").exists():
                    base_dir = current / "output"
                    break
                current = current.parent
            else:
                base_dir = Path("output")

        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        run_type: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_run: Optional[str] = None,
    ) -> 'Run':
        """Create a new run with timestamped directory.

        Args:
            run_type: Type of run (discovery, evolution, training, evaluation, inference)
            description: Short description (used in directory name, no spaces)
            config: Configuration dictionary to save
            tags: Optional tags for categorization
            parent_run: Optional parent run ID this builds on

        Returns:
            Run object for managing this run
        """
        # Create timestamped run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for directory name
        safe_desc = description.replace(" ", "_").replace("/", "-")[:30]
        run_id = f"{timestamp}_{run_type}_{safe_desc}"

        # Create directory structure
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)

        # Create metadata
        metadata = RunMetadata(
            run_id=run_id,
            run_type=run_type,
            description=description,
            created_at=datetime.now().isoformat(),
            config=config,
            tags=tags or [],
            parent_run=parent_run,
        )

        # Save initial metadata
        run = Run(run_dir, metadata)
        run._save_metadata()

        # Save config separately for easy access
        if config:
            run.save_config(config)

        return run

    def get_run(self, run_id: str) -> Optional['Run']:
        """Get an existing run by ID.

        Args:
            run_id: The run ID (directory name)

        Returns:
            Run object or None if not found
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None

        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = RunMetadata.from_dict(json.load(f))
        else:
            # Create minimal metadata for legacy runs
            metadata = RunMetadata(
                run_id=run_id,
                run_type="unknown",
                description="Legacy run",
                created_at="unknown",
            )

        return Run(run_dir, metadata)

    def list_runs(
        self,
        run_type: Optional[str] = None,
        limit: int = 20,
        include_completed: bool = True,
    ) -> List['Run']:
        """List runs, most recent first.

        Args:
            run_type: Filter by run type
            limit: Maximum number of runs to return
            include_completed: Include completed runs

        Returns:
            List of Run objects
        """
        runs: list['Run'] = []

        if not self.runs_dir.exists():
            return runs

        # Sort by directory name (which includes timestamp)
        run_dirs = sorted(self.runs_dir.iterdir(), reverse=True)

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue

            run = self.get_run(run_dir.name)
            if run is None:
                continue

            # Apply filters
            if run_type and run.metadata.run_type != run_type:
                continue
            if not include_completed and run.metadata.status == "completed":
                continue

            runs.append(run)

            if len(runs) >= limit:
                break

        return runs

    def get_latest_run(self, run_type: Optional[str] = None) -> Optional['Run']:
        """Get the most recent run.

        Args:
            run_type: Filter by run type

        Returns:
            Most recent Run or None
        """
        runs = self.list_runs(run_type=run_type, limit=1)
        return runs[0] if runs else None

    def cleanup_old_runs(
        self,
        keep_count: int = 10,
        run_type: Optional[str] = None,
        dry_run: bool = True,
    ) -> List[str]:
        """Remove old runs, keeping the most recent.

        Args:
            keep_count: Number of runs to keep
            run_type: Only clean up runs of this type
            dry_run: If True, just report what would be deleted

        Returns:
            List of run IDs that were/would be deleted
        """
        runs = self.list_runs(run_type=run_type, limit=1000)
        to_delete = runs[keep_count:]

        deleted = []
        for run in to_delete:
            if not dry_run:
                shutil.rmtree(run.run_dir)
            deleted.append(run.metadata.run_id)

        return deleted


class Run:
    """Represents a single experiment run."""

    def __init__(self, run_dir: Path, metadata: RunMetadata):
        self.run_dir = Path(run_dir)
        self.metadata = metadata

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def images_dir(self) -> Path:
        return self.run_dir / "images"

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to config.json."""
        self.metadata.config = config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        self._save_metadata()

    def save_results(self, results: Dict[str, Any], summary: Optional[Dict[str, Any]] = None):
        """Save results to results.json.

        Args:
            results: Full results dictionary
            summary: Optional summary for quick reference
        """
        # Add metadata to results
        results["_run_id"] = self.metadata.run_id
        results["_created_at"] = self.metadata.created_at
        results["_completed_at"] = datetime.now().isoformat()

        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        if summary:
            self.metadata.summary = summary
            self._save_metadata()

    def save_checkpoint(
        self,
        data: Dict[str, Any],
        name: str,
        is_final: bool = False,
    ):
        """Save a checkpoint.

        Args:
            data: Checkpoint data
            name: Checkpoint name (e.g., "gen_0010", "epoch_050")
            is_final: If True, also save as "final"
        """
        filename = f"checkpoint_{name}.json"
        with open(self.checkpoints_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

        if is_final:
            with open(self.checkpoints_dir / "checkpoint_final.json", "w") as f:
                json.dump(data, f, indent=2)

    def save_image(self, image_data, name: str, format: str = "png"):
        """Save an image.

        Args:
            image_data: Image array or PIL Image
            name: Image name (without extension)
            format: Image format (png, jpg)
        """
        from PIL import Image
        import numpy as np

        filepath = self.images_dir / f"{name}.{format}"

        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                image_data = (image_data * 255).astype(np.uint8)
            if len(image_data.shape) == 2:
                img = Image.fromarray(image_data, mode='L')
            else:
                img = Image.fromarray(image_data)
            img.save(filepath)
        elif hasattr(image_data, 'save'):
            # PIL Image
            image_data.save(filepath)
        else:
            raise ValueError(f"Unknown image type: {type(image_data)}")

        return filepath

    def save_model(self, model, name: str):
        """Save a PyTorch model.

        Args:
            model: PyTorch model or state dict
            name: Model name (without extension)
        """
        import torch

        filepath = self.models_dir / f"{name}.pth"

        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), filepath)
        else:
            torch.save(model, filepath)

        return filepath

    def log(self, message: str, level: str = "INFO"):
        """Append to run log.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"

        with open(self.logs_dir / "run.log", "a") as f:
            f.write(log_line)

    def complete(self, status: str = "completed", summary: Optional[Dict[str, Any]] = None):
        """Mark run as complete.

        Args:
            status: Final status (completed, failed, cancelled)
            summary: Optional summary of results
        """
        self.metadata.status = status
        self.metadata.completed_at = datetime.now().isoformat()
        if summary:
            self.metadata.summary = summary
        self._save_metadata()

    def get_image_path(self, name: str, format: str = "png") -> Path:
        """Get path for an image file.

        Args:
            name: Image name (without extension)
            format: Image format

        Returns:
            Path to image file
        """
        return self.images_dir / f"{name}.{format}"

    def __repr__(self) -> str:
        return f"Run({self.metadata.run_id}, type={self.metadata.run_type}, status={self.metadata.status})"


# Convenience functions for quick access
_default_manager: Optional[RunManager] = None


def get_run_manager() -> RunManager:
    """Get the default run manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = RunManager()
    return _default_manager


def create_run(
    run_type: str,
    description: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Run:
    """Create a new run (convenience function).

    Args:
        run_type: Type of run
        description: Short description
        config: Configuration dictionary
        **kwargs: Additional arguments to RunManager.create_run

    Returns:
        Run object
    """
    return get_run_manager().create_run(run_type, description, config, **kwargs)


def get_latest_run(run_type: Optional[str] = None) -> Optional[Run]:
    """Get the most recent run (convenience function)."""
    return get_run_manager().get_latest_run(run_type)


def list_runs(run_type: Optional[str] = None, limit: int = 20) -> List[Run]:
    """List runs (convenience function)."""
    return get_run_manager().list_runs(run_type=run_type, limit=limit)
