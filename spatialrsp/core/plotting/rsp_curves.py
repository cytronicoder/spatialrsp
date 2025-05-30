import matplotlib.pyplot as plt
import numpy as np


def plot_rsp_curve(
    angle_range: np.ndarray,
    fg_curves: dict[str, np.ndarray],
    bg_curve: np.ndarray,
    expected_fg_curve: np.ndarray = None,
    ax: plt.Axes = None,
    title: str = "RSP Curve",
) -> None:
    """Plot RSP angular curves in polar coordinates.

    Args:
        angle_range (np.ndarray): Angular values in radians.
        fg_curves (dict): Label → RSP values.
        bg_curve (np.ndarray): Background curve.
        expected_fg_curve (np.ndarray, optional): Expected foreground (for absolute mode).
        ax (plt.Axes, optional): Polar axis. New one if None.
        title (str): Title of the plot.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    angle_closed = np.append(angle_range, angle_range[0])
    bg_closed = np.append(bg_curve, bg_curve[0])
    ax.plot(
        angle_closed, bg_closed, linestyle="dotted", color="black", label="Background"
    )

    if expected_fg_curve is not None:
        expected_closed = np.append(expected_fg_curve, expected_fg_curve[0])
        ax.plot(angle_closed, expected_closed, color="gray", label="Expected FG")

    for label, curve in fg_curves.items():
        closed_curve = np.append(curve, curve[0])
        color = "red" if expected_fg_curve is not None else None
        ax.plot(angle_closed, closed_curve, label=label, color=color, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()


def plot_single_rsp_curve(
    angle_range: np.ndarray,
    fg_curve: np.ndarray,
    bg_curve: np.ndarray,
    expected_fg_curve: np.ndarray = None,
    ax: plt.Axes = None,
    label: str = "Foreground",
) -> None:
    """Plot a single RSP curve in polar coordinates.

    Args:
        angle_range (np.ndarray): Angular values in radians.
        fg_curve (np.ndarray): Foreground RSP values.
        bg_curve (np.ndarray): Background RSP values.
        expected_fg_curve (np.ndarray, optional): Expected foreground (for absolute mode).
        ax (plt.Axes, optional): Polar axis. New one if None.
        label (str): Label for foreground curve.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    angle_closed = np.append(angle_range, angle_range[0])
    bg_closed = np.append(bg_curve, bg_curve[0])
    fg_closed = np.append(fg_curve, fg_curve[0])

    ax.plot(
        angle_closed, bg_closed, linestyle="dotted", color="black", label="Background"
    )
    ax.plot(angle_closed, fg_closed, color="red", linewidth=2, label=label)

    if expected_fg_curve is not None:
        expected_closed = np.append(expected_fg_curve, expected_fg_curve[0])
        ax.plot(angle_closed, expected_closed, color="gray", label="Expected FG")

    ax.set_title("RSP Curve", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
