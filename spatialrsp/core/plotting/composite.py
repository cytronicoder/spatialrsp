"""Create a composite figure showing a UMAP embedding next to its RSP curve.

This module overlays a background and multiple foreground clusters in a
2D embedding (e.g. UMAP) and plots their RSP curves side-by-side in a
polar subplot.
"""

import os
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from .embedding import plot_embedding_overlay
from .rsp_curves import plot_rsp_curve


def plot_rsp_composite(
    bg_coords: np.ndarray,
    fg_coords_dict: Dict[str, np.ndarray],
    angle_range: np.ndarray,
    fg_rsp_dict: Dict[str, np.ndarray],
    bg_curve: np.ndarray,
    *,
    expected_fg_curve: Optional[np.ndarray] = None,
    vantage_point: Optional[np.ndarray] = None,
    vantage_point_label: Optional[str] = None,
    vantage_point_color: str = "red",
    title: str = "UMAP Embedding",
    save_path: Optional[str] = None,
) -> None:
    """
    Create a composite plot with UMAP embedding (left) and RSP curves (right).

    Args:
        bg_coords (np.ndarray):
            2D coordinates of background points (shape: [N_bg, 2]).
        fg_coords_dict (Dict[str, np.ndarray]):
            Mapping from label → 2D coordinates of foreground points
            (each array has shape [N_fg_label, 2]).
        angle_range (np.ndarray):
            1D array of angular bin midpoints (e.g. size M, for the RSP curve).
        fg_rsp_dict (Dict[str, np.ndarray]):
            Mapping from label → 1D array of RSP values (length M) for each
            foreground cluster.
        bg_curve (np.ndarray):
            1D array of background RSP values (length M).
        expected_fg_curve (Optional[np.ndarray], optional):
            If provided, a 1D “expected” RSP curve (length M).  Default: None.
        vantage_point (Optional[np.ndarray], optional):
            If provided, the 2D coordinates (shape [2,]) of a special “vantage
            point” to highlight on the UMAP.  Default: None.
        vantage_point_label (Optional[str], optional):
            Label to print for the vantage point in the embedding.  Default: None.
        vantage_point_color (str, optional):
            Color to use for the vantage point marker.  Default: "red".
        title (str, optional):
            Figure-level title.  Default: "UMAP Embedding".
        save_path (Optional[str], optional):
            If provided, save the figure at this path (e.g. "out/fig.png").
            Default: None (i.e. do not save).
    """
    fig = plt.figure(figsize=(12, 6))

    # Left subplot: UMAP embedding overlay
    ax0 = fig.add_subplot(1, 2, 1)

    # Only pass vantage_point‐related kwargs if vantage_point and its label exist.
    if (vantage_point is not None) and (vantage_point_label is not None):
        plot_embedding_overlay(
            bg_coords,
            fg_coords_dict,
            ax=ax0,
            vantage_point=vantage_point,
            vantage_point_label=vantage_point_label,
            vantage_point_color=vantage_point_color,
        )
    else:
        # Call without vantage_point/vantage_point_label
        plot_embedding_overlay(
            bg_coords,
            fg_coords_dict,
            ax=ax0,
        )

    # Right subplot: RSP curves (polar)
    ax1 = fig.add_subplot(1, 2, 2, projection="polar")

    # Only pass expected_fg_curve if it’s not None
    if expected_fg_curve is not None:
        plot_rsp_curve(
            angle_range,
            fg_rsp_dict,
            bg_curve,
            expected_fg_curve=expected_fg_curve,
            ax=ax1,
        )
    else:
        plot_rsp_curve(
            angle_range,
            fg_rsp_dict,
            bg_curve,
            ax=ax1,
        )

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("UMAP Embedding and RSP Curve", fontsize=16)

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    # Function returns None implicitly
