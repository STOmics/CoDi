import numpy as np
import scipy


def preprocess(adata):
    """Perform count normalization"""
    # avoid any kind of preprocessing if counts are not integers
    if scipy.sparse.issparse(adata.X):
        sum_0_cell_counts = sum(adata.X[0].toarray().flatten())
    else:
        sum_0_cell_counts = sum(adata.X[0])
    if not sum_0_cell_counts.is_integer():
        return
    expression = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    if expression.dtype != np.float32:
        expression = expression.astype(np.float32)
    expression[expression == 0] = np.nan
    expression = np.log10(expression)
    min_vec = np.nanmin(expression, axis=1)
    max_vec = np.nanmax(expression, axis=1)
    expression = (expression.T - min_vec).T
    expression = (expression.T / (max_vec - min_vec + 10**-6)).T
    # distinction between zeros obtained with scaling vs original zeros
    expression[expression == 0] += 10**-9
    np.nan_to_num(expression, 0)
    adata.X = scipy.sparse.csr_matrix(expression)
