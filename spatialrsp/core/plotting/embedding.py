import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_overlay(
    bg_coords: np.ndarray,
    fg_coords_dict: dict[str, np.ndarray],
    vantage_point: np.ndarray = None,
    vantage_point_label: str = None,
    vantage_point_color: str = "red",
    ax: plt.Axes = None,
    title: str = "UMAP Embedding",
) -> None:
    """Plot a 2D embedding with background and one or more foreground overlays,
    and force the plotting window to be a square in data space.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c="lightgray", s=1, label="Background")

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
        all_coords.append(vantage_point.reshape(1, 2))

    all_coords = np.vstack(all_coords)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    half_side = max(x_span, y_span) / 2

    if vantage_point is not None:
        cx, cy = vantage_point
    else:
        cx = (x_max + x_min) / 2
        cy = (y_max + y_min) / 2

    ax.set_xlim(cx - half_side, cx + half_side)
    ax.set_ylim(cy - half_side, cy + half_side)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
