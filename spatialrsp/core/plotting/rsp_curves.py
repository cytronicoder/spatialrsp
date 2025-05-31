"""Plot RSP angular curves (single or multiple) in polar coordinates.

This module provides two functions:
- `plot_rsp_curve`: plots multiple foreground RSP curves alongside a background
  and optional expected curve on a single polar Axes.
- `plot_single_rsp_curve`: plots one foreground RSP curve with background and
  optional expected curve on a single polar Axes.
"""

from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def plot_rsp_curve(
    angle_range: np.ndarray,
    fg_curves: Dict[str, np.ndarray],
    bg_curve: np.ndarray,
    expected_fg_curve: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = "RSP Curve",
) -> None:
    """
    Plot RSP angular curves in polar coordinates.

    Args:
        angle_range (np.ndarray):
            1D array of angular values (radians). Shape: (M,).
        fg_curves (Dict[str, np.ndarray]):
            Mapping from label → 1D array of RSP values (length M) for each
            foreground cluster.
        bg_curve (np.ndarray):
            1D array of background RSP values (length M).
        expected_fg_curve (Optional[np.ndarray], optional):
            If provided, a 1D “expected” foreground curve (length M). Default: None.
        ax (Optional[Axes], optional):
            A Matplotlib polar Axes to draw on. If None, a new one is created.
            Default: None.
        title (str, optional):
            Title for the polar plot. Default: "RSP Curve".

    Returns:
        None: Draws directly onto the Matplotlib Axes (or a newly created one).
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    # Close the angle and curve loops by appending the first element to the end
    angle_closed = np.append(angle_range, angle_range[0])
    bg_closed = np.append(bg_curve, bg_curve[0])

    ax.plot(
        angle_closed,
        bg_closed,
        linestyle="dotted",
        color="black",
        label="Background",
    )

    if expected_fg_curve is not None:
        expected_closed = np.append(expected_fg_curve, expected_fg_curve[0])
        ax.plot(
            angle_closed,
            expected_closed,
            color="gray",
            label="Expected FG",
        )

    for label, curve in fg_curves.items():
        closed_curve = np.append(curve, curve[0])
        # If expected_fg_curve exists, draw FG in red; otherwise let Matplotlib choose color
        curve_color = "red" if expected_fg_curve is not None else None
        ax.plot(
            angle_closed,
            closed_curve,
            label=label,
            color=curve_color,
            linewidth=2,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()


def plot_single_rsp_curve(
    angle_range: np.ndarray,
    fg_curve: np.ndarray,
    bg_curve: np.ndarray,
    expected_fg_curve: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    label: str = "Foreground",
) -> None:
    """
    Plot a single RSP curve in polar coordinates, alongside background and
    optional expected curve.

    Args:
        angle_range (np.ndarray):
            1D array of angular values (radians). Shape: (M,).
        fg_curve (np.ndarray):
            1D array of foreground RSP values (length M).
        bg_curve (np.ndarray):
            1D array of background RSP values (length M).
        expected_fg_curve (Optional[np.ndarray], optional):
            If provided, a 1D “expected” foreground curve (length M). Default: None.
        ax (Optional[Axes], optional):
            A Matplotlib polar Axes to draw on. If None, a new one is created.
            Default: None.
        label (str, optional):
            Legend label for the foreground curve. Default: "Foreground".

    Returns:
        None: Draws directly onto the Matplotlib Axes (or a newly created one).
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    angle_closed = np.append(angle_range, angle_range[0])
    bg_closed = np.append(bg_curve, bg_curve[0])
    fg_closed = np.append(fg_curve, fg_curve[0])

    ax.plot(
        angle_closed,
        bg_closed,
        linestyle="dotted",
        color="black",
        label="Background",
    )
    ax.plot(
        angle_closed,
        fg_closed,
        color="red",
        linewidth=2,
        label=label,
    )

    if expected_fg_curve is not None:
        expected_closed = np.append(expected_fg_curve, expected_fg_curve[0])
        ax.plot(
            angle_closed,
            expected_closed,
            color="gray",
            label="Expected FG",
        )

    ax.set_title("RSP Curve", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
