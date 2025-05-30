from typing import Optional, Tuple

import numpy as np
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy import sparse

from spatialrsp.utils import cartesian_to_polar, sigfigs


def _get_dense_X(adata: ad.AnnData, verbose: bool = False) -> np.ndarray:
    """Convert adata.X to dense format if needed.

    Args:
        adata (AnnData): Annotated data object.
        verbose (bool): Verbose output toggle.

    Returns:
        np.ndarray: Dense data matrix.
    """
    if verbose:
        print("[INFO] Converting adata.X to dense format if needed.")
    if not isinstance(adata.X, np.ndarray):
        dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        if verbose:
            print(f"[INFO] Converted to dense array with shape: {dense.shape}")
        return dense
    if verbose:
        print("[INFO] adata.X is already dense. Copying array.")
    return adata.X.copy()


def quality_control(
    adata: ad.AnnData,
    min_nonzero: float = 10,
    max_dropout: float = 0.5,
    verbose: bool = False,
) -> ad.AnnData:
    """Filter observations with low expression or high dropout.

    Args:
        adata (AnnData): Annotated data object.
        min_nonzero (float): Minimum nonzero values per observation.
        max_dropout (float): Maximum allowed dropout rate.
        verbose (bool): Verbose output toggle.

    Returns:
        AnnData: Filtered data object.
    """
    if verbose:
        print("[INFO] Starting quality control...")

    X_dense = _get_dense_X(adata, verbose)
    nonzero_counts = (X_dense != 0).sum(axis=1)
    dropout_rate = 1 - (nonzero_counts / X_dense.shape[1])
    valid = (nonzero_counts >= min_nonzero) & (dropout_rate <= max_dropout)

    if verbose:
        print(f"[✓] Retained {np.sum(valid)} / {adata.n_obs} observations.")
    return adata[valid].copy()


def normalize_data(adata: ad.AnnData, verbose: bool = False) -> ad.AnnData:
    """Normalize and log-transform expression data.

    Args:
        adata (AnnData): Annotated data object.
        verbose (bool): Verbose output toggle.

    Returns:
        AnnData: Normalized data object.
    """
    if verbose:
        print("[INFO] Normalizing data...")

    X_dense = _get_dense_X(adata, verbose)
    totals = X_dense.sum(axis=1)
    median_total = np.median(totals)

    if verbose:
        print(f"[INFO] Median total count = {sigfigs(median_total, 3)}")

    scaled = (X_dense.T / totals).T * median_total
    adata.X = np.log1p(scaled)

    if verbose:
        print(f"[✓] Normalization complete. Shape: {adata.X.shape}")
    return adata


def reduce_dimensionality(
    adata: ad.AnnData,
    method: str = "PCA",
    n_components: int = 50,
    verbose: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Compute low-dimensional embedding using PCA, UMAP, or t-SNE.

    Args:
        adata (AnnData): Annotated data object.
        method (str): Method name ("PCA", "UMAP", or "TSNE").
        n_components (int): Number of components.
        verbose (bool): Verbose output toggle.
        **kwargs: Extra parameters for the reduction method.

    Returns:
        AnnData: Data object with embedding in `obsm["X_emb"]`.
    """
    if verbose:
        print(f"[INFO] Running {method.upper()} with {n_components} components...")

    X_dense = _get_dense_X(adata, verbose)
    n_samples = X_dense.shape[0]
    method_upper = method.upper()

    if method_upper == "PCA":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method_upper == "UMAP":
        kwargs["n_neighbors"] = min(kwargs.get("n_neighbors", 15), n_samples - 1)
        reducer = UMAP(n_components=n_components, **kwargs)
    elif method_upper in {"TSNE", "T-SNE"}:
        kwargs["perplexity"] = min(kwargs.get("perplexity", 30), n_samples - 1)
        reducer = TSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    embedding = reducer.fit_transform(X_dense)
    adata.obsm["X_emb"] = embedding

    if verbose:
        print(f"[✓] Embedding shape: {embedding.shape}")
    return adata


def update_h2ad_embedding(
    adata: ad.AnnData,
    embedding: np.ndarray,
    key: str = "X_emb",
    verbose: bool = False,
) -> ad.AnnData:
    """Update or insert embedding into AnnData object.

    Args:
        adata (AnnData): Annotated data object.
        embedding (np.ndarray): Embedding matrix.
        key (str): Key in `obsm` to store embedding.
        verbose (bool): Verbose output toggle.

    Returns:
        AnnData: Updated object.
    """
    if embedding.shape[0] != adata.n_obs:
        raise ValueError("Embedding rows must match number of observations.")

    adata.obsm[key] = embedding
    if verbose:
        print(f"[✓] Updated embedding '{key}' with shape {embedding.shape}")
    return adata


def select_vantage_point(
    adata: ad.AnnData,
    embedding_key: str = "X_umap",
    bg_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Compute a centroid-based vantage point for polar conversion.

    Args:
        adata (AnnData): Annotated data object.
        embedding_key (str): Key for 2D embedding in `obsm`.
        bg_indices (np.ndarray, optional): Indices to subset the centroid.
        verbose (bool): Verbose output toggle.

    Returns:
        tuple[float, float]: (x, y) coordinates of the vantage point.
    """
    embedding = adata.obsm.get(embedding_key)
    if embedding is None:
        raise ValueError(f"Embedding '{embedding_key}' not found.")

    subset = embedding[bg_indices] if bg_indices is not None else embedding
    vantage_point = tuple(np.mean(subset, axis=0))

    if verbose:
        print(f"[INFO] Selected vantage point: {vantage_point}")
    return vantage_point


def polar_transform(
    adata: ad.AnnData,
    embedding_key: str = "X_umap",
    polar_key: str = "X_polar",
    vantage_point: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Convert 2D embedding to polar coordinates.

    Args:
        adata (AnnData): Annotated data object.
        embedding_key (str): Key for input 2D embedding.
        polar_key (str): Key to store polar embedding.
        vantage_point (tuple[float, float], optional): Origin for conversion.
        verbose (bool): Verbose output toggle.

    Returns:
        AnnData: Updated object with polar coordinates.
    """
    if verbose:
        print(f"[INFO] Converting '{embedding_key}' to polar coordinates...")

    embedding = adata.obsm.get(embedding_key)
    if embedding is None:
        raise ValueError(f"Embedding '{embedding_key}' not found.")

    if vantage_point is None:
        vantage_point = select_vantage_point(adata, embedding_key, verbose)

    theta, r = cartesian_to_polar(embedding, vantage_point=vantage_point)
    polar_coords = np.column_stack((theta, r))
    adata.obsm[polar_key] = polar_coords

    if verbose:
        print(
            f"[✓] Stored polar coordinates in '{polar_key}' with shape {polar_coords.shape}"
        )

    return adata


def select_top(expr: np.ndarray, pct: float) -> np.ndarray:
    """
    Select the top fraction of values in an expression array.

    Args:
        expr (np.ndarray): 1D array of expression levels.
        pct (float): Fraction between 0 and 1 indicating the proportion of top cells to select.

    Returns:
        mask (np.ndarray): Boolean mask of the same length as expr, True for the top pct cells.
    """
    N = expr.size
    k = int(np.floor(pct * N))

    # If pct is so small that k==0, return all False
    if k <= 0:
        return np.zeros(N, bool)

    # Partial sort to get indices of top-k without full sort
    idx = np.argpartition(expr, -k)[-k:]
    mask = np.zeros(N, bool)
    mask[idx] = True

    return mask


def extract_expr(adata, gene):
    """
    Retrieve a 1D array of expression values for a single gene from an AnnData.

    Args:
        adata (AnnData): Annotated data object containing expression data.
        gene (str): Gene name or index to extract.

    Returns:
        np.ndarray: 1D array of expression values for the specified gene.
    """
    m = adata[:, gene].X

    if sparse.issparse(m):
        return m.toarray().ravel()

    return np.asarray(m).ravel()
