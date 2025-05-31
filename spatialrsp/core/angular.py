"""Compute angular bias areas from one or two foreground distributions
and a background distribution, using a sliding angular window.

This function returns, for each center angle in a specified scanning range,
the normalized area under the CDF curve. In “absolute” mode, it also computes
an expected foreground area. In “relative” mode, it returns areas for two
foregrounds and background.
"""

from typing import Optional, Tuple

import numpy as np

from spatialrsp.core.cdf import compute_area_under_cdf


def compute_angular_area(
    theta_fg1: np.ndarray,
    theta_bg: np.ndarray,
    scanning_window: float,
    resolution: int,
    scanning_range: np.ndarray,
    *,
    theta_fg2: Optional[np.ndarray] = None,
    mode: str = "absolute",
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Compute angular bias curves from foreground/background angular distributions.

    Args:
        theta_fg1 (np.ndarray):
            1D array of foreground-1 angular values (radians). Shape: (N_fg1,).
        theta_bg (np.ndarray):
            1D array of background angular values (radians). Shape: (N_bg,).
        scanning_window (float):
            Width of the angular scanning window (in radians).
        resolution (int):
            Number of histogram bins within the scanning window.
        scanning_range (np.ndarray):
            1D array of center angles (radians) over which to scan. Shape: (M,).
        theta_fg2 (Optional[np.ndarray], optional):
            If provided (for “relative” mode), 1D array of foreground-2 angular
            values (radians). Default is None.
        mode (str, optional):
            Either "absolute" or "relative". In "absolute" mode, returns
            (fg1_area, expected_fg1_area, bg_area). In "relative" mode, returns
            (fg1_area, fg2_area, bg_area). Default is "absolute".

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
            - If mode == "absolute":
                (abs_area_fg1, abs_area_expected_fg, abs_area_bg)
                where
                  abs_area_fg1:   shape (M,) array of observed FG1 areas
                  abs_area_expected_fg: shape (M,) array of expected FG1 areas
                  abs_area_bg:    shape (M,) array of background areas
            - If mode == "relative" and theta_fg2 is not None:
                (abs_area_fg1, abs_area_fg2, abs_area_bg)
                  abs_area_fg1: shape (M,) observed FG1 areas
                  abs_area_fg2: shape (M,) observed FG2 areas
                  abs_area_bg:  shape (M,) background areas
    """
    bins = np.linspace(-scanning_window / 2, scanning_window / 2, resolution + 1)
    coverage = len(theta_fg1) / len(theta_bg)

    num_centers = len(scanning_range)
    abs_area_fg1 = np.zeros(num_centers)
    abs_area_bg = np.zeros(num_centers)

    # In “relative” mode, we need fg2; in “absolute” mode, we need expected_fg
    abs_area_fg2: Optional[np.ndarray]
    abs_area_expected_fg: Optional[np.ndarray]

    if theta_fg2 is not None:
        # Relative mode: compute fg2 areas
        abs_area_fg2 = np.zeros(num_centers)
        abs_area_expected_fg = None
    else:
        # Absolute mode: compute expected fg1 areas
        abs_area_fg2 = None
        abs_area_expected_fg = np.zeros(num_centers)

    eps = np.finfo(float).eps

    for i, center in enumerate(scanning_range):
        rel_theta_fg1 = ((theta_fg1 - center + np.pi) % (2 * np.pi)) - np.pi
        rel_theta_bg = ((theta_bg - center + np.pi) % (2 * np.pi)) - np.pi

        mask_fg1 = np.abs(rel_theta_fg1) <= (scanning_window / 2)
        mask_bg = np.abs(rel_theta_bg) <= (scanning_window / 2)

        hist_fg1_obs, _ = np.histogram(rel_theta_fg1[mask_fg1], bins=bins)
        hist_bg_obs, _ = np.histogram(rel_theta_bg[mask_bg], bins=bins)

        hist_fg1_obs = hist_fg1_obs.astype(float)
        hist_fg1_obs[hist_fg1_obs == 0] = eps
        hist_bg_obs = hist_bg_obs.astype(float)
        hist_bg_obs[hist_bg_obs == 0] = eps

        area_bg, _ = compute_area_under_cdf(hist_bg_obs, bins, coverage, mode)
        area_fg1, _ = compute_area_under_cdf(hist_fg1_obs, bins, coverage, mode)

        abs_area_fg1[i] = np.sqrt(area_fg1 / area_bg)
        abs_area_bg[i] = np.sqrt(
            area_bg / area_bg
        )  # always 1.0 but included for consistency

        if (mode == "absolute") and (abs_area_expected_fg is not None):
            hist_expected_fg = hist_fg1_obs * coverage
            area_exp_fg, _ = compute_area_under_cdf(
                hist_expected_fg, bins, coverage, mode
            )
            abs_area_expected_fg[i] = np.sqrt(area_exp_fg / area_bg)

        if theta_fg2 is not None:
            # Re-center FG2
            rel_theta_fg2 = ((theta_fg2 - center + np.pi) % (2 * np.pi)) - np.pi
            mask_fg2 = np.abs(rel_theta_fg2) <= (scanning_window / 2)
            hist_fg2_obs, _ = np.histogram(rel_theta_fg2[mask_fg2], bins=bins)
            hist_fg2_obs = hist_fg2_obs.astype(float)
            hist_fg2_obs[hist_fg2_obs == 0] = eps

            area_fg2, _ = compute_area_under_cdf(hist_fg2_obs, bins, coverage, mode)
            # abs_area_fg2 must not be None if theta_fg2 is not None
            assert abs_area_fg2 is not None
            abs_area_fg2[i] = np.sqrt(area_fg2 / area_bg)

    if theta_fg2 is not None:
        # relative mode
        return abs_area_fg1, abs_area_fg2, abs_area_bg  # type: ignore

    # absolute mode
    return abs_area_fg1, abs_area_expected_fg, abs_area_bg
