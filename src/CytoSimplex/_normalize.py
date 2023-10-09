from sklearn.preprocessing import normalize
from anndata import AnnData
from typing import Union
from scipy.sparse import csr_matrix
import numpy as np


def row_normalize(x: Union[AnnData, np.ndarray, csr_matrix]
                  ) -> Union[AnnData, np.ndarray, csr_matrix]:
    """
        Element-wise normalizes each row of a matrix by its l1 norm.

        Parameters
        ----------
        x
            The data to be normalized. When using an :class:`anndata.AnnData`,
            `x.X` will be normalized in place.

        Returns
        -------
            Normalized data of the same class.
    """
    if (isinstance(x, AnnData)):
        x.X = normalize(x.X, norm='l1', axis=1)
    else:
        x = normalize(x, norm='l1', axis=1)
        return x
