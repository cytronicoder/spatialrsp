import numpy as np


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Perform Benjamini-Hochberg FDR correction on an array of p-values.

    Parameters
    ----------
    pvals : np.ndarray
        Array of raw p-values.

    Returns
    -------
    np.ndarray
        Array of FDR-adjusted q-values (same shape as pvals), clipped to 1.0.
    """
    p = np.asarray(pvals)
    n = len(p)

    # Rank the p-values in ascending order
    order = np.argsort(p)
    rank = np.empty(n, int)
    rank[order] = np.arange(1, n + 1)

    # Compute the BH-adjusted values
    q = p * n / rank

    # Enforce monotonicity by taking cumulative minimum from largest to smallest
    q = np.minimum.accumulate(q[::-1])[::-1]

    return np.minimum(q, 1.0)
