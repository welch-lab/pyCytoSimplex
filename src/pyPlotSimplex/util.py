from anndata import AnnData
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

SQRT3 = 3**0.5
SQRT6 = 6**0.5
TRIANGLE_VERTICES = pd.DataFrame({'x': [0, 0.5, 1],
                                  'y': [0, SQRT3/2, 0]})
TETRA_VERTICES = pd.DataFrame({'x': [-1/SQRT3, 2/SQRT3, -1/SQRT3, 0],
                               'y': [-1,       0,       1,        0],
                               'z': [0,        0,       0,        2*SQRT6/3]})


def _check_cluster_vertices(x, cluster_var, vertices, n=None):
    """Check if the cluster_var and vertices are valid.
    Parameters
    ----------
    x : AnnData, numpy.ndarray, or scipy.sparse.csr_matrix
    cluster_var : str, list, or pd.Series
        The cluster assignment of each observation.
        When str, the name of the cluster variable in adata.obs.
        When list or pd.Series, the length must equal to adata.n_obs or
            matrix.shape[0].
    vertices : list or dict
        The terminal specifications of the simplex.
        When list, the length must equal to n if n specified and all elements
            must be in cluster_var.
        When dist, served as grouped specifications of the terminals, meaning
            each terminal stands for a group of clusters. The number of the
            keys must equal to n if n specified, and all elements of each value
            (list) must be in cluster_var.
    n : int
        The number of vertices.

    Returns
    -------
    cluster_var : pd.Series
        The cluster assignment of each observation.
    vertices : dict
        The terminal specifications of the simplex.
    """
    if n is not None:
        if len(vertices) > n:
            print(f"WARNING: Detected more than {n} vertices "
                  f"({len(vertices)}). Using the first {n} vertices.")
            if isinstance(vertices, list):
                vertices = vertices[:n]
            elif isinstance(vertices, dict):
                keys = list(vertices.keys())[:n]
                vertices = {key: vertices[key] for key in keys}
        if len(vertices) < n:
            raise ValueError(f"Must specify {n} vertices.")

    # Obtain the number of observations. and check input type.
    if isinstance(x, AnnData):
        nobs = x.n_obs
        mat = x.X.copy()
    elif isinstance(x, np.ndarray):
        nobs = x.shape[0]
        mat = x
    elif isinstance(x, csr_matrix):
        nobs = x.shape[0]
        mat = x
    else:
        raise TypeError(
            "The object must be an AnnData, numpy.ndarray, or "
            "scipy.sparse.csr_matrix."
        )

    # Check cluster_var type and coerce to pd.Series.
    if isinstance(cluster_var, str):
        cluster_var = x.obs[cluster_var]
    elif isinstance(cluster_var, list):
        cluster_var = pd.Series(cluster_var)
    if len(cluster_var) != nobs:
        raise ValueError(
            "The length of the cluster_var must match the number of "
            "observations."
        )
    # Set category of cluster_var to be categorical.
    cluster_var = cluster_var.astype("category")

    if isinstance(vertices, list):
        # Check list type vertices and should all exist in the categories of
        # cluster_var.
        if not all(vertex in cluster_var.cat.categories
                   for vertex in vertices):
            raise ValueError("All vertices must be in cluster_var.")
        grouping = cluster_var
    elif isinstance(vertices, dict):
        grouping = _map_category(cluster_var, vertices)
        vertices = list(vertices.keys())
    else:
        raise TypeError("The vertices must be a list or dict.")

    return mat, grouping, vertices, cluster_var


def _map_category(x, mapping):
    """
    x: pd.Series dtype="category"
    mapping: A dict with {group_name1: [original existing categories],
                          group_name2: one existing category,
                          .....}.
    """
    map_rev = {}
    for key, value in mapping.items():
        if isinstance(value, str):
            if value not in x.cat.categories:
                raise ValueError(f"{value} is not in the categories of "
                                 "`cluster_var`.")
            map_rev[value] = key
        elif isinstance(value, int):
            map_rev[value] = key
        else:
            for v in value:
                if v not in x.cat.categories:
                    raise ValueError(f"{v} is not in the categories of "
                                     "`cluster_var`.")
                map_rev[v] = key
    for category in x.cat.categories:
        if category not in map_rev:
            map_rev[category] = category
    x = x.map(map_rev)
    x = x.astype("category")
    return x
