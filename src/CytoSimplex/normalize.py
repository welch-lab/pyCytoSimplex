from sklearn.preprocessing import normalize
from anndata import AnnData


def row_normalize(x):
    """
        Normalize the rows of a matrix by l1 norm.
        Parameters
        ----------
        x : AnnData, where x.X will be normalized.
            numpy.ndarray, or scipy.sparse.csr_matrix, The matrix to be
                normalized.

        Returns
        -------
        if x is AnnData, x.X will be normalized in place.
        else, return the normalized matrix.
    """
    if (isinstance(x, AnnData)):
        x.X = normalize(x.X, norm='l1', axis=1)
    else:
        x = normalize(x, norm='l1', axis=1)
        return x
