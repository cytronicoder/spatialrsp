"""Preprocessing utilities for spatialRSP.

This module contains functions for:
  - Converting various data types (AnnData, SciPy sparse, pandas) to dense arrays
  - Splitting features and labels from AnnData
  - Normalizing feature matrices
  - Selecting a centroid-based vantage point
"""

from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import anndata as ad


def _get_dense_array(
    data: Any,
    key: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Convert `data` (AnnData, sparse matrix, pandas, or array-like) into a 2D numpy.ndarray.

    Args:
        data: One of:
          - An AnnData: extract `.X` if key is "X" or None; otherwise `.obsm[key]` or `.obs[key]`.
          - A SciPy sparse matrix (csr_matrix, csc_matrix, coo_matrix).
          - A pandas Series or DataFrame.
          - An array-like (e.g. numpy ndarray, H5Array, CSRDataset, etc.).
        key: If `data` is AnnData, this selects which slot to extract ("X", a key in
             `.obsm`, or a column in `.obs`).
        verbose: If True, prints information about the conversion.

    Returns:
        A 2D numpy.ndarray.
    """
    # 1) If it's an AnnData, extract the requested slot
    if isinstance(data, ad.AnnData):
        if key is None or key == "X":
            arr = data.X
        else:
            if key in data.obsm:
                arr = data.obsm[key]
            elif key in data.obs:
                arr = data.obs[key]
            else:
                raise KeyError(f"Key '{key}' not found in AnnData.")
    else:
        arr = data

    # 2) If it's a supported SciPy sparse type, convert to dense
    if isinstance(arr, (csr_matrix, csc_matrix, coo_matrix)):
        dense = arr.toarray()
        if verbose:
            print(f"[INFO] Converted sparse to dense with shape {dense.shape}.")
        return dense

    # 3) If it's a pandas Series or DataFrame, convert to numpy
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        dense = arr.to_numpy()
        if verbose:
            print(f"[INFO] Converted pandas to numpy with shape {dense.shape}.")
        return dense

    # 4) Otherwise, assume it’s array-like; attempt to convert to ndarray
    try:
        dense = np.asarray(arr)
    except Exception as e:
        raise ValueError(
            f"Cannot convert object of type {type(arr)} to numpy array: {e!r}"
        ) from e

    # 5) Ensure 2D: if 1D, reshape to (n, 1)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    if dense.ndim < 2:
        raise ValueError(f"Resulting array has {dense.ndim} dims; expected >= 2 dims.")

    if verbose:
        print(f"[INFO] Using dense ndarray with shape {dense.shape}.")

    return dense


def split_features_labels(
    adata: ad.AnnData,
    feature_key: str = "X",
    label_key: str = "y",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an AnnData containing features and labels, return (x_array, y_array).

    Features come from `adata.X` or `adata.obsm[feature_key]`. Labels come from
    `adata.obs[label_key]`.

    Args:
        adata: AnnData object.
        feature_key: "X" or a key in `.obsm` where the feature matrix resides.
        label_key: Column name in `.obs` containing labels.
        verbose: If True, prints shapes of x and y.

    Returns:
        x_array (ndarray of shape (n_obs, n_vars)), y_array (1D ndarray of length n_obs)
    """
    # 1) Extract raw features
    if feature_key == "X":
        raw = adata.X
    else:
        if feature_key not in adata.obsm:
            raise KeyError(f"Feature key '{feature_key}' not present in .obsm.")
        raw = adata.obsm[feature_key]

    x_array = _get_dense_array(raw, key=None, verbose=verbose)

    # 2) Extract labels
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not present in .obs.")
    y_raw = adata.obs[label_key]
    if isinstance(y_raw, pd.Series):
        y_array = y_raw.to_numpy()
    else:
        y_array = np.asarray(y_raw)

    if y_array.ndim != 1:
        raise ValueError(f"Labels array must be 1D; got shape {y_array.shape}.")
    if len(y_array) != x_array.shape[0]:
        raise ValueError(
            f"Number of labels {len(y_array)} != number of samples {x_array.shape[0]}"
        )

    if verbose:
        print(
            f"[INFO] Split into x with shape {x_array.shape} and y with length {y_array.shape[0]}."
        )

    return x_array, y_array


def normalize_features(
    features: np.ndarray,
    method: str = "minmax",
    verbose: bool = False,
) -> np.ndarray:
    """
    Normalize the feature matrix according to the specified method.

    Supported methods:
      - "minmax": scale each column to [0, 1]
      - "zscore": subtract mean and divide by standard deviation

    Args:
        features (np.ndarray): 2D array (n_samples, n_features).
        method (str): "minmax" or "zscore".
        verbose (bool): If True, prints summary.

    Returns:
        features_norm (np.ndarray): Normalized feature matrix.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D array for features; got shape {features.shape}.")

    if method == "minmax":
        mins = np.min(features, axis=0)
        maxs = np.max(features, axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0  # avoid division by zero
        features_norm = (features - mins) / denom
        if verbose:
            print("[INFO] Applied min-max scaling.")
        return features_norm

    elif method == "zscore":
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0, ddof=1)
        stds[stds == 0] = 1.0  # avoid division by zero
        features_norm = (features - means) / stds
        if verbose:
            print("[INFO] Applied z-score normalization.")
        return features_norm

    else:
        raise ValueError(f"Unsupported normalization method '{method}'.")


def select_vantage_point(
    coords: np.ndarray,
    _labels: np.ndarray,
    bg_mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Compute the centroid of background samples to use as a vantage point.

    Args:
        coords (np.ndarray): 2D array (n_samples, 2) of coordinates.
        _labels (np.ndarray): 1D array of labels (length n_samples).
        bg_mask (Optional[np.ndarray]): Boolean mask selecting background rows. If None, use all.
        verbose (bool): If True, prints the centroid.

    Returns:
        (x_center, y_center): Coordinates of the vantage point.
    """
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must have at least 2 columns; got shape {coords.shape}."
        )

    if bg_mask is None:
        subset = coords
    else:
        if bg_mask.dtype != bool or len(bg_mask) != coords.shape[0]:
            raise ValueError(
                "bg_mask must be a boolean array of length equal to n_samples."
            )
        subset = coords[bg_mask]

    if subset.shape[0] == 0:
        raise ValueError("Background mask selects zero samples.")

    center = np.mean(subset, axis=0)
    if verbose:
        print(f"[INFO] Vantage point = ({center[0]:.4f}, {center[1]:.4f})")

    return float(center[0]), float(center[1])
