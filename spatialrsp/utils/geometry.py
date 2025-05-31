from typing import Union
import numpy as np


def shift_angles(angles: np.ndarray, center_angle: float) -> np.ndarray:
    """Shift angles to center around `center_angle` in [-π, π)."""
    return (angles - center_angle + np.pi) % (2 * np.pi) - np.pi


def within_window(
    theta: np.ndarray, center_theta: float, window_width: float, verbose: bool = False
) -> Union[bool, np.ndarray]:
    """Check if angles lie within a specified angular window.

    Args:
        theta (np.ndarray): Angles to check (radians).
        center_theta (float): Window center (radians).
        window_width (float): Angular width (radians).
        verbose (bool): Enable debug output.

    Returns:
        np.ndarray | bool: Boolean array if input is array; bool if scalar.
    """
    if not 0 < window_width <= 2 * np.pi:
        raise ValueError("window_width must be in (0, 2π].")
    delta = np.abs((theta - center_theta + np.pi) % (2 * np.pi) - np.pi)
    result = delta <= window_width / 2
    if verbose:
        if hasattr(theta, "shape"):
            print(
                f"[within_window] {np.sum(result)} / {len(theta)} angles "
                f"within ±{window_width/2:.2f} radians."
            )
        else:
            print(f"[within_window] Angle {theta:.2f} within window: {result}")
    return result
