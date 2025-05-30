from pathlib import Path
from typing import Literal

import pandas as pd
import anndata as ad
from scipy import sparse

from .transform import cartesian_to_polar


def load_data(
    file_path: str,
    sep: str = "\t",
    index_col: int = 0,
    dtype: Literal["float32", "float64"] = "float32",
    sparse_output: bool = True,
    verbose: bool = False,
) -> ad.AnnData:
    """
    Load data from a file into an AnnData object.
    Supports memory-mapped CSV files and H5AD files.

    Args:
        file_path (str): Path to the input file.
        sep (str): Separator for CSV files. Default is tab ('\t').
        index_col (int): Column to use as index in the DataFrame.
            Default is 0.
        dtype (Literal["float32", "float64"]): Data type for the DataFrame.
            Default is 'float32'.
        sparse_output (bool): If True, returns a sparse matrix; otherwise, a dense matrix.
            Default is True.
        verbose (bool): If True, prints additional information during loading.
            Default is False.

    Returns:
        ad.AnnData: An AnnData object containing the loaded data.
    """
    path = Path(file_path)

    if path.suffix == ".h5ad":
        if verbose:
            print(f"[↓] Memory-mapping AnnData from {path}")
        return ad.read_h5ad(str(path))

    if verbose:
        print(f"[INFO] Fast CSV read of {path} with dtype {dtype} and sep '{sep}'")

    df = pd.read_csv(
        file_path,
        sep=sep,
        index_col=index_col,
        engine="pyarrow",
        dtype=dtype,
        low_memory=False,
        memory_map=True,
    )

    # build AnnData directly from the numpy block
    block = df.values.T  # shape = (n_obs, n_vars)
    X = sparse.csr_matrix(block) if sparse_output else block

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.columns),
        var=pd.DataFrame(index=df.index),
    )

    if verbose:
        print(f"[✓] Loaded AnnData with shape {adata.shape}")

    return adata


def save_data(
    adata: ad.AnnData,
    file_path: str,
    compression: Literal["gzip", "lzf"] | None = "lzf",
    verbose: bool = False,
) -> ad.AnnData:
    """
    Save an AnnData object to a file in H5AD format.
    Supports gzip and lzf compression.

    Args:
        adata (ad.AnnData): AnnData object to save.
        file_path (str): Path to the output file.
        compression (Literal["gzip", "lzf"] | None): Compression method.
            Default is 'lzf'. Set to None for no compression.
        verbose (bool): If True, prints additional information during saving.
            Default is False.

    Returns:
        ad.AnnData: The saved AnnData object.
    """
    if verbose:
        print(f"[↑] Writing AnnData to {file_path} with {compression} compression")
    adata.write_h5ad(file_path, compression=compression)
    if verbose:
        print("[✓] File saved.")
    return adata


def load_coords_and_angles(datasets, vp):
    """
    For each dataset, extract 2D UMAP coordinates and compute polar angles relative to a vantage point.

    Args:
        datasets (dict[str, ad.AnnData]): Mapping of condition names to AnnData objects.
        vp (array-like): Vantage point in UMAP space, shape (2,).

    Returns:
        tuple: A tuple containing:
            - coords_2d (dict[str, np.ndarray]): 2D Cartesian coordinates for each state.
            - polars (dict[str, np.ndarray]): Angular coordinates (theta) for each cell in each state.
    """
    coords_2d = {}
    polars = {}
    for state, ad in datasets.items():
        coords = ad.obsm["X_umap"]

        # Convert sparse to dense if needed
        if sparse.issparse(coords):
            coords = coords.toarray()

        # Use only first two dimensions
        coords = coords[:, :2].astype(float)

        # Compute polar angle around vp; discard radial distances (_)
        theta, _ = cartesian_to_polar(coords, vp, verbose=False)
        coords_2d[state] = coords
        polars[state] = theta

    return coords_2d, polars
