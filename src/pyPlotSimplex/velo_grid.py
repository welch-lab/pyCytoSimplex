from .util import TRIANGLE_VERTICES, TETRA_VERTICES
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def aggregate_vertex_velo(velo, cluster_var, vertices):
    if not isinstance(velo, csr_matrix):
        velo = csr_matrix(velo)

    velo_mat = pd.DataFrame(index=cluster_var.index, columns=vertices)
    for i, v in enumerate(vertices):
        velo_mat[v] = velo[cluster_var == v,:].mean(axis=0).T

    # Normalize each row to sum to 1.
    velo_mat = velo_mat.apply(lambda y: (y+1e-8)/(y+1e-8).sum(), axis=1)
    return velo_mat

def aggregate_grid_velo(sim_mat, velo_mat, n_grid=10, radius=0.1):
    n = sim_mat.shape[1]
    if n == 3:
        simplex = TRIANGLE_VERTICES
    elif n == 4:
        simplex = TETRA_VERTICES
    dim_range = np.array([simplex.max(0), simplex.min(0)])
    max_range = (dim_range[0] - dim_range[1]).max()
    win = max_range/n_grid
    seg = win/2

    # Get centroid coordinate on each dimension first
    dim_grid = [np.arange(dim_range[1,i]+seg, dim_range[0,i], win)
                for i in range(simplex.shape[1])]
    # Then expand to space by combining all possible combinations, which derives the
    # cartesian coordinates of the grid centroids.
    gird_cart = np.array(np.meshgrid(*dim_grid)).reshape(simplex.shape[1], -1)
    grid_cart = pd.DataFrame(gird_cart.T, columns=simplex.columns)

    # Select those that fall into the simplex space with barycentric coordinates.
    grid_bary = _cart2bary(simplex, grid_cart)
    grid_bary = pd.DataFrame(grid_bary, columns=sim_mat.columns)
    grid_selection = (grid_bary >= -1e-10).all(1) & (grid_bary <= 1+1e-10).all(1)
    grid_cart = grid_cart.loc[grid_selection]

    # Convert simplex barycentric coord of cells to cartesien coord (2D space)
    cell_cart = pd.DataFrame(np.dot(sim_mat, simplex), index=sim_mat.index,
                             columns=simplex.columns)

    # Aggregate velocity by grid, comparing cell cartesian coordinate
    grid_velo = pd.DataFrame(0, index=grid_cart.index, columns=velo_mat.columns)
    for i in range(grid_cart.shape[0]):
        cell_sel = pd.Series(True, index=cell_cart.index)
        for j in range(cell_cart.shape[1]):
            gt = (cell_cart.iloc[:,j] > (grid_cart.iloc[i,j] - seg))
            lt = (cell_cart.iloc[:,j] < (grid_cart.iloc[i,j] + seg))
            cell_sel = cell_sel & gt & lt
        if cell_sel.sum() > 4:
            grid_velo.iloc[i,:] = velo_mat.loc[cell_sel,:].mean(axis=0)
    grid_to_keep = grid_velo.sum(1) > 0
    grid_velo = grid_velo.loc[grid_to_keep,:]
    grid_cart = grid_cart.loc[grid_to_keep,:]

    grid_bary = _cart2bary(simplex, grid_cart)
    grid_bary = pd.DataFrame(grid_bary, index=grid_cart.index, columns=sim_mat.columns)

    # Get the arrow end points
    arrow_vec = []
    for i, v in enumerate(grid_velo.columns):
        arrow_end_cart = _get_arrow_end(grid_cart, np.array(simplex.loc[i]),
                                   grid_velo.iloc[:,i]*radius)
        arrow_end_bary = _cart2bary(simplex, arrow_end_cart)
        arrow_vec.append(
            pd.DataFrame(arrow_end_bary - grid_bary,
                         index=grid_cart.index,
                         columns=sim_mat.columns)
        )

    return grid_bary, arrow_vec


def _cart2bary(X, P):
    # X: simplex cartesian coordinates, vertices on row and dimensions on column
    # P: sample points, each row is a sample and each column is a dimension
    # return: barycentric coordinates of P
    M = P.shape[0]
    N = P.shape[1]
    X1 = X.iloc[:N, :] - np.dot(np.ones((N, 1)), X.iloc[N:N+1, :])
    Beta = np.dot((P - np.array([X.iloc[N, :] for i in range(M)])),
                  np.linalg.pinv(X1))
    Beta = np.concatenate((Beta, np.array([1-Beta.sum(1)]).T), axis=1)
    return Beta

def _get_arrow_end(starts, target, length):
    """
    starts: np.array, each row is a start point, each column is the coordinate
            on one dimension.
    target: 1D array, the target point. number of elements must equal to the
            number of columns in starts.
    length: vector of arrow lengths, each element corresponds to one start point
    """
    V = np.array([target for i in range(starts.shape[0])])
    vec_SV = V - starts
    len_SV = np.sqrt(np.sum(vec_SV**2, axis=1))
    length[length > len_SV] = len_SV[length > len_SV]
    vec_SA = vec_SV.div(len_SV, axis=0).mul(length, axis=0)
    A = starts + vec_SA
    return A
