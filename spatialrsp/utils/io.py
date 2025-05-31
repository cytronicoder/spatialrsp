"""Utilities for extracting 2D UMAP coordinates and computing polar angles
from AnnData objects, with proper type handling for pandas, NumPy, and sparse data.
"""

from typing import Dict, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import anndata as ad

from .transform import cartesian_to_polar


def load_coords_and_angles(
    datasets: Dict[str, ad.AnnData],
    vp: Union[Sequence[float], np.ndarray, pd.Series],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For each dataset, extract 2D UMAP coordinates and compute polar angles
    relative to a vantage point.

    Args:
        datasets (Dict[str, ad.AnnData]):
            Mapping of condition names to AnnData objects. Each AnnData must
            have a 'X_umap' entry under .obsm (which may be a sparse matrix,
            a NumPy array, a pandas DataFrame, or a pandas Series).
        vp (Union[Sequence[float], np.ndarray, pd.Series]):
            Vantage point in UMAP space (length 2). May be a list/tuple,
            a NumPy array of shape (2,), or a pandas Series of length 2.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            - coords_2d (Dict[str, np.ndarray]): 2D Cartesian coordinates
              (shape: (n_cells, 2)) for each state.
            - polars (Dict[str, np.ndarray]): Angular coordinates (theta)
              for each cell in each state (shape: (n_cells,)).
    """
    # Convert vp to a numpy array if it's a pandas Series or other sequence
    if isinstance(vp, pd.Series):
        vp_array = vp.to_numpy()
    else:
        vp_array = np.asarray(vp)

    if vp_array.ndim != 1 or vp_array.size != 2:
        raise ValueError(
            f"vantage point `vp` must be length-2, got shape {vp_array.shape}"
        )
    vp_tuple: Tuple[float, float] = (float(vp_array[0]), float(vp_array[1]))

    coords_2d: Dict[str, np.ndarray] = {}
    polars: Dict[str, np.ndarray] = {}

    for state, adata_obj in datasets.items():
        coords_raw = adata_obj.obsm.get("X_umap")
        if coords_raw is None:
            raise KeyError(f"AnnData for state '{state}' has no 'X_umap' in .obsm")

        # 1) If it's a scipy sparse matrix, convert to dense
        if isinstance(coords_raw, spmatrix):
            coords_array = coords_raw.toarray()
        # 2) If it's a pandas DataFrame or Series, get its numpy values
        elif isinstance(coords_raw, (pd.DataFrame, pd.Series)):
            coords_array = coords_raw.to_numpy()
        # 3) Otherwise, assume it can be converted into an ndarray
        else:
            coords_array = np.asarray(coords_raw)

        if coords_array.ndim == 1:
            coords_array = coords_array.reshape(-1, 1)
        if coords_array.ndim < 2:
            raise ValueError(
                f"'X_umap' for state '{state}' must have at least 2 dimensions, got {coords_array.ndim}"
            )

        # Take only the first two columns (x, y)
        coords_xy = coords_array[:, :2].astype(float)

        # Compute polar angles around the vantage point; ignore radial distances (_)
        theta, _ = cartesian_to_polar(coords_xy, vp_tuple, verbose=False)

        coords_2d[state] = coords_xy
        polars[state] = theta

    return coords_2d, polars
