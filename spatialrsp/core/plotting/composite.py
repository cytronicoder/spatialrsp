import os

import matplotlib.pyplot as plt
import numpy as np

from .embedding import plot_embedding_overlay
from .rsp_curves import plot_rsp_curve


def plot_rsp_composite(
    bg_coords: np.ndarray,
    fg_coords_dict: dict[str, np.ndarray],
    angle_range: np.ndarray,
    fg_rsp_dict: dict[str, np.ndarray],
    bg_curve: np.ndarray,
    expected_fg_curve: np.ndarray = None,
    vantage_point: np.ndarray = None,
    vantage_point_label: str = None,
    vantage_point_color: str = "red",
    title: str = "UMAP Embedding",
    save_path: str = None,
) -> None:
    """Create a composite plot with UMAP + RSP curve side-by-side.

    Args:
        bg_coords (np.ndarray): Background points.
        fg_coords_dict (dict): Label → foreground coordinates.
        angle_range (np.ndarray): Angular bins.
        fg_rsp_dict (dict): Label → foreground RSP curves.
        bg_curve (np.ndarray): Background RSP curve.
        expected_fg_curve (np.ndarray, optional): Expected foreground (absolute mode).
        vantage_point (np.ndarray, optional): Vantage point coordinates.
        vantage_point_label (str, optional): Vantage point label.
        vantage_point_color (str, optional): Vantage point color.
        title (str, optional): Plot title.
        save_path (str, optional): Output path to save figure.
    """
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="polar")

    plot_embedding_overlay(
        bg_coords,
        fg_coords_dict,
        ax=ax0,
        vantage_point=vantage_point,
        vantage_point_label=vantage_point_label,
        vantage_point_color=vantage_point_color,
    )

    plot_rsp_curve(
        angle_range,
        fg_rsp_dict,
        bg_curve,
        expected_fg_curve=expected_fg_curve,
        ax=ax1,
    )

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("UMAP Embedding and RSP Curve", fontsize=16)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
