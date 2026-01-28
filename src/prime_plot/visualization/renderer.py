"""Common rendering utilities for prime visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def render_to_image(
    data: np.ndarray,
    colormap: str = "binary",
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    show_axes: bool = False,
) -> "matplotlib.figure.Figure":
    """Render numpy array to matplotlib figure.

    Args:
        data: 2D array to visualize.
        colormap: Matplotlib colormap name.
        figsize: Figure size in inches (width, height).
        title: Optional title for the figure.
        show_axes: Whether to show axis labels and ticks.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        aspect = data.shape[1] / data.shape[0]
        figsize = (10, int(10 / aspect))

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data, cmap=colormap, interpolation="nearest")

    if title:
        ax.set_title(title)

    if not show_axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def save_image(
    data: np.ndarray,
    path: str | Path,
    colormap: str = "binary",
    dpi: int = 100,
    **kwargs,
) -> None:
    """Save visualization to image file.

    Args:
        data: 2D array to visualize.
        path: Output file path (supports PNG, PDF, SVG, etc.).
        colormap: Matplotlib colormap name.
        dpi: Resolution in dots per inch.
        **kwargs: Additional arguments passed to render_to_image.
    """
    fig = render_to_image(data, colormap=colormap, **kwargs)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    import matplotlib.pyplot as plt
    plt.close(fig)


def save_raw_image(
    data: np.ndarray,
    path: str | Path,
) -> None:
    """Save raw array data as PNG without matplotlib.

    This is faster for large images and preserves exact pixel values.

    Args:
        data: 2D uint8 array.
        path: Output PNG file path.
    """
    from PIL import Image

    if data.dtype != np.uint8:
        if data.max() <= 1:
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

    img = Image.fromarray(data, mode="L")
    img.save(path)


def create_animation_frames(
    spiral_class,
    start_size: int,
    end_size: int,
    num_frames: int,
    **spiral_kwargs,
) -> list[np.ndarray]:
    """Generate animation frames showing spiral growth.

    Args:
        spiral_class: UlamSpiral, SacksSpiral, or KlauberTriangle class.
        start_size: Starting size parameter.
        end_size: Ending size parameter.
        num_frames: Number of frames to generate.
        **spiral_kwargs: Additional arguments for spiral constructor.

    Returns:
        List of numpy arrays for each frame.
    """
    sizes = np.linspace(start_size, end_size, num_frames, dtype=int)
    sizes = np.unique(sizes)

    frames = []
    for size in sizes:
        spiral = spiral_class(size, **spiral_kwargs)
        frame = spiral.render_primes()
        frames.append(frame)

    return frames


def overlay_polynomial(
    base_image: np.ndarray,
    spiral,
    polynomial,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Overlay polynomial curve on spiral visualization.

    Args:
        base_image: Grayscale base image.
        spiral: Spiral object with generate_coordinates method.
        polynomial: PrimePolynomial or callable f(n) -> int.
        color: RGB color for the polynomial curve.

    Returns:
        RGB image with polynomial highlighted.
    """
    if len(base_image.shape) == 2:
        rgb = np.stack([base_image] * 3, axis=-1)
    else:
        rgb = base_image.copy()

    if hasattr(spiral, "size"):
        max_n = spiral.size * spiral.size
    elif hasattr(spiral, "max_n"):
        max_n = spiral.max_n
    else:
        max_n = spiral.max_value

    x_coords, y_coords = spiral.generate_coordinates()

    for n in range(max_n):
        if hasattr(polynomial, "evaluate"):
            val = polynomial.evaluate(n)
        else:
            val = polynomial(n)

        if 1 <= val <= max_n:
            idx = val - 1
            if idx < len(x_coords):
                x, y = int(x_coords[idx]), int(y_coords[idx])
                if 0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]:
                    rgb[y, x] = color

    return rgb


def composite_visualizations(
    images: list[np.ndarray],
    labels: list[str],
    cols: int = 2,
) -> "matplotlib.figure.Figure":
    """Create composite figure with multiple visualizations.

    Args:
        images: List of 2D arrays to display.
        labels: Labels for each image.
        cols: Number of columns in the grid.

    Returns:
        Matplotlib Figure with all images arranged in a grid.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        ax.imshow(img, cmap="binary", interpolation="nearest")
        ax.set_title(label)
        ax.axis("off")

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig
