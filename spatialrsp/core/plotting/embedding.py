"""Plot a 2D embedding with background and one or more foreground overlays,
and force the plotting window to be square in data space.

This module takes background coordinates, a dictionary of labeled foreground
coordinates, and optionally a vantage point (with label/color), and draws
everything on a Matplotlib Axes (creating one if none is provided).
"""

from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def plot_embedding_overlay(
    bg_coords: np.ndarray,
    fg_coords_dict: Dict[str, np.ndarray],
    vantage_point: Optional[np.ndarray] = None,
    vantage_point_label: Optional[str] = None,
    vantage_point_color: str = "red",
    ax: Optional[Axes] = None,
    title: str = "UMAP Embedding",
) -> None:
    """
    Plot a 2D embedding with background points, multiple foreground clusters,
    and an optional “vantage point.” Ensures that the final plotting window
    is square (equal data scaling on both axes).

    Args:
        bg_coords (np.ndarray):
            Array of shape (N_bg, 2) containing background (x, y) points.
        fg_coords_dict (Dict[str, np.ndarray]):
            A mapping from label → array of shape (N_fg_i, 2) for each foreground
            cluster. Each value is a NumPy array of 2D coordinates.
        vantage_point (Optional[np.ndarray], optional):
            If provided, a length-2 array [x, y] marking a special “vantage point”
            to highlight. Default is None (no vantage point plotted).
        vantage_point_label (Optional[str], optional):
            If `vantage_point` is not None, this string will appear in the legend
            for that point. Default is None, in which case the legend label will be
            “Vantage Point.”
        vantage_point_color (str, optional):
            Color used to draw the vantage point marker. Default is "red".
        ax (Optional[Axes], optional):
            If provided, draw on this existing Axes. Otherwise, a new square Axes
            (6x6 inches) is created. Default is None.
        title (str, optional):
            Title for the Axes. Default is "UMAP Embedding".

    Returns:
        None: This function draws directly onto Matplotlib's current figure/axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        bg_coords[:, 0],
        bg_coords[:, 1],
        c="lightgray",
        s=1,
        label="Background",
    )

    colors = ["red", "green", "orange", "blue", "purple"]
    for i, (label, coords) in enumerate(fg_coords_dict.items()):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=1,
            alpha=0.6,
            label=label,
            color=colors[i % len(colors)],
        )

    if vantage_point is not None:
        ax.scatter(
            vantage_point[0],
            vantage_point[1],
            marker="x",
            color=vantage_point_color,
            label=vantage_point_label or "Vantage Point",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=14)

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.5)

    all_coords = [bg_coords] + list(fg_coords_dict.values())
    if vantage_point is not None:
        # reshape vantage_point from (2,) → (1, 2) so it can stack
        all_coords.append(vantage_point.reshape(1, 2))

    all_coords = np.vstack(all_coords)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    half_side = max(x_span, y_span) / 2

    if vantage_point is not None:
        cx, cy = vantage_point[0], vantage_point[1]
    else:
        cx = (x_max + x_min) / 2
        cy = (y_max + y_min) / 2

    ax.set_xlim(cx - half_side, cx + half_side)
    ax.set_ylim(cy - half_side, cy + half_side)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
