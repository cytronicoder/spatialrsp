"""Collection of plotting utilities: boxplots, barplots, and scatter plots.

This module provides:
- `plot_coverage_boxplot`: Boxplot of foreground coverage across conditions.
- `plot_rsp_comparison_barplot`: Barplot comparing two metrics with optional significance stars.
- `plot_expression_scatter`: Scatter plot of paired values with optional color labeling.
"""

from typing import Optional, List, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def plot_coverage_boxplot(
    foregrounds: List[List[int]],
    background_sizes: List[int],
    labels: List[str],
    ax: Optional[Axes] = None,
) -> None:
    """
    Boxplot of foreground coverage across conditions.

    For each condition i, `foregrounds[i]` is a list of foreground sizes,
    and `background_sizes[i]` is the background size. This plots the
    ratio (fg_size / background_size) for each foreground within each condition.

    Args:
        foregrounds (List[List[int]]):
            A list of length C (number of conditions).  Each element is a
            list of ints, representing the sizes of each foreground group
            in that condition.
        background_sizes (List[int]):
            A list of length C giving the background size for each condition.
        labels (List[str]):
            A list of length C of string labels for each condition (used on
            the x-axis).
        ax (Optional[Axes], optional):
            A Matplotlib Axes to draw on. If None, a new Axes is created.
            Default is None.
    """
    # Compute coverage ratios: for condition i, divide each fg size by background_sizes[i]
    coverage: List[List[float]] = [
        [fg_size / max(background_sizes[i], 1) for fg_size in foregrounds[i]]
        for i in range(len(labels))
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Draw the boxplot; do not pass `labels=` here to avoid Pylance complaints
    ax.boxplot(coverage, patch_artist=True)

    # Manually set the x-tick labels after creating the boxplot
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_ylabel("Foreground Coverage")
    ax.set_title("Foreground Coverage Across Conditions")


def plot_rsp_comparison_barplot(
    means: List[float],
    errors: List[float],
    labels: Tuple[str, str] = ("A1", "A2"),
    title: str = "Fold Change from Control",
    colors: Tuple[str, str] = ("pink", "orange"),
    stars: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
) -> None:
    """
    Barplot of metric comparison (e.g., A1 vs A2) with optional significance stars.

    Args:
        means (List[float]):
            A list of length 2 containing the two mean values to plot.
        errors (List[float]):
            A list of length 2 containing the standard errors for each bar.
        labels (Tuple[str, str], optional):
            A tuple of two strings giving the x-axis labels for the bars.
            Default: ("A1", "A2").
        title (str, optional):
            Title for the plot. Default: "Fold Change from Control".
        colors (Tuple[str, str], optional):
            A tuple of two color names/hex strings for the bars. Default: ("pink", "orange").
        stars (Optional[List[str]], optional):
            If provided, a length-2 list of significance stars (e.g., ["*", "ns"]).
            These will be drawn above each bar. Default is None (no stars).
        ax (Optional[Axes], optional):
            A Matplotlib Axes to draw on. If None, a new Axes is created.
            Default is None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    x = np.arange(len(means))  # array([0, 1])
    bar_width = 0.5

    bar_rects = ax.bar(
        x,
        means,
        yerr=errors,
        width=bar_width,
        color=colors,
        capsize=5,
    )

    # Draw a horizontal reference line at y = 1
    ax.axhline(1, color="black", linestyle="--", linewidth=1)

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_ylabel("Fold Change (from control)", fontsize=14)
    ax.set_title(title, fontsize=14)

    # If `stars` is provided, draw each star above the corresponding bar
    if stars is not None:
        for rect, star in zip(bar_rects, stars):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.02,
                star,
                ha="center",
                va="bottom",
                fontsize=16,
                color="black",
            )


def plot_expression_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[Sequence] = None,
    xlabel: str = "Gene A",
    ylabel: str = "Gene B",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    alpha: float = 0.5,
) -> None:
    """
    Scatter plot of expression or metric pairs, with optional color-coding.

    Args:
        x (np.ndarray):
            1D array of X values.
        y (np.ndarray):
            1D array of Y values.
        labels (Optional[Sequence], optional):
            If provided, an array-like of the same length as `x` and `y` to
            color-code each point. If None, all points are plotted in black.
            Default is None.
        xlabel (str, optional):
            Label for the X-axis. Default: "Gene A".
        ylabel (str, optional):
            Label for the Y-axis. Default: "Gene B".
        title (Optional[str], optional):
            Title for the plot. If None, no title is set. Default is None.
        ax (Optional[Axes], optional):
            A Matplotlib Axes to draw on. If None, a new Axes is created.
            Default is None.
        alpha (float, optional):
            Transparency for plotted points. Default: 0.5.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if labels is not None:
        # Color-coded scatter
        scatter = ax.scatter(x, y, c=labels, s=5, alpha=alpha, cmap="viridis")
        plt.colorbar(scatter, ax=ax)
    else:
        # All points in black
        ax.scatter(x, y, s=5, alpha=alpha, color="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
