"""Compute the scaled area under a histogram's CDF, with optional coverage scaling.

This module takes a histogram (counts and bin edges) and returns both the area
under its cumulative distribution function (CDF) and the CDF array itself.
If run in “absolute” mode with a provided coverage value, the scaled area
is multiplied by that coverage ratio.
"""

from typing import Optional, Tuple

import numpy as np


def compute_area_under_cdf(
    hist: np.ndarray,
    bins: np.ndarray,
    coverage: Optional[float] = None,
    mode: str = "absolute",
) -> Tuple[float, np.ndarray]:
    """
    Compute the scaled area under the CDF of a histogram.

    Args:
        hist (np.ndarray):
            1D array of histogram counts (length N_bins-1).
        bins (np.ndarray):
            1D array of bin edge values (length N_bins).
        coverage (Optional[float], optional):
            Foreground-to-background ratio. Only used when mode == "absolute".
            If None (default), no coverage scaling is applied.
        mode (str, optional):
            Either "absolute" or "relative". In "absolute" mode, if coverage is
            provided, the computed area is multiplied by that coverage factor.
            Default is "absolute".

    Returns:
        Tuple[float, np.ndarray]:
            - The first element is the (possibly scaled) area under the CDF.
            - The second element is the CDF array (length N_bins-1).
    """
    # Compute the CDF by cumulative sum of histogram counts
    cdf = np.cumsum(hist)

    # Compute bin centers for integration
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Basic trapezoidal integration of the CDF over bin centers
    area = np.trapezoid(cdf, x=bin_centers)

    # Normalize by window width to get a density-like measure
    window_width = bins[-1] - bins[0]
    scaled_area = area * (2.0 / window_width)

    # In absolute mode, apply coverage scaling if provided
    if mode == "absolute" and (coverage is not None):
        scaled_area *= coverage

    return scaled_area, cdf
