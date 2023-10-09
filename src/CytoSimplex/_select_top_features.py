from ._util import _check_cluster_vertices
from ._normalize import row_normalize
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy import stats
from anndata import AnnData
from typing import Union, Optional


def select_top_features(
        x: Union[AnnData, np.ndarray, csr_matrix],
        cluster_var: Union[str, list, pd.Series],
        vertices: Union[list, dict],
        n_top: int = 30,
        processed: bool = False,
        lfc_thresh: float = 0.1,
        return_stats: bool = False,
        feature_names: Optional[str] = None
        ) -> Union[list, pd.DataFrame]:
    """
    Select top features for each vertices based on the wilcoxon test.

    Parameters
    ----------
    x
        The matrix of gene expression, where each row is a cell and each column
        is a gene. Recommended to be full size raw counts. (i.e. not
        log-transformed or normalized and not only for highly variable genes)
        When given :class:`anndata.AnnData`, `x.X` will be used.
    cluster_var
        The cluster assignment of each single cell. If `x` is an
        :class:`anndata.AnnData`, `cluster_var` can be a `str` that specifies
        the name of the cluster variable in `x.obs`. `list` or
        :class:`pandas.Series` is accepted in all cases, and the length must
        equal to `x.shape[0]`.
    vertices
        The terminal specifications. Depending on the type of simplex to be
        visualized in downstream, the number of vertices (n) should be
        determined by users. e.g. 3 elements for a 2-simplex (ternary simplex /
        triangle). Acceptable input include:

        - A :class:`list` of n :class:`str` that exist in the categories of
          `cluster_var`.
        - A :class:`dict` of n keys. The keys are presented as customizable
          vertex names. The corresponding value for each key can be either a
          :class:`str` for a single cluster, or a :class:`list` of :class:`str`
          for grouped vertex of multiple clusters.
    n_top
        The number of top features to select for each vertex.
    processed
        Whether the input matrix is already processed. If `False`, the input
        matrix will be log transformed and row normalized. If `True`, the input
        matrix will be directly used to calculate the rank-sum statistics. And
        logFC will be calculated assuming that the input matrix is
        log-transformed.
    lfc_thresh
        The log fold change threshold to select up-regulated genes.
    return_stats
        Whether to return the full statistics of all clusters and all features
        instead of only returning the selected top features by default.
    feature_names
        The names of the features in the matrix. If None, the feature names
        will be the index of the matrix.

    Returns
    -------
    selected : :class:`list`, when `return_stats=False`.
        The list of selected features. Maximum length is
        `n_top * len(vertices)` when enough features can pass the
        threshold.
    stats : :class:`pandas.DataFrame`, when `return_stats=True`.
        The statistics of the wilcoxon test, with `n_groups * n_features` rows.
        Columns are 'group', 'avgExpr', 'logFC', 'ustat', 'auc', 'pval',
        'padj', 'pct_in', 'pct_out' and 'feature'.

    Examples
    --------
        >>> import CytoSimplex as csx
        >>> import scanpy as sc
        >>> adata = sc.read(
        ...     filename="test.h5ad",
        ...     backup_url="https://figshare.com/ndownloader/files/41034857"
        ... )
        >>> vertices = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
        ...             'RE': ['Reticular_1', 'Reticular_2'],
        ...             'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
        >>> gene = csx.select_top_features(adata, "cluster", vertices)
        >>> gene[:8]
        ['Nrk', 'Eps8l2', 'Mfi2', 'Fam101a', 'Scin', 'Sox5', 'Fbln7', 'Edil3']
        >>> stats = csx.select_top_features(adata, "cluster", vertices, return_stats=True)
            group    avgExpr     logFC   ustat     auc          pval          padj  pct_in  pct_out           feature
        0         CH   0.000000  0.000000  5000.0  0.5000           NaN           NaN     0.0      0.0               Rp1
        1         CH   0.000000  0.000000  5000.0  0.5000           NaN           NaN     0.0      0.0             Sox17
        2         CH   8.413918  5.771032  6948.0  0.6948  7.674938e-08  9.017050e-07    64.0     19.0            Mrpl15
        3         CH   4.627888  2.507528  5894.0  0.5894  4.888534e-03  1.529972e-02    36.0     15.5            Lypla1
        4         CH   0.256851  0.256851  5100.0  0.5100  4.999579e-02  1.110755e-01     2.0      0.0           Gm37988
        ...      ...        ...       ...     ...     ...           ...           ...     ...      ...               ...
        141696    RE   2.269271  0.260934  5105.0  0.5105  7.154050e-01  1.000000e+00    16.0     14.5              PISD
        141697    RE   2.593425 -0.267436  4968.0  0.4968  9.268655e-01  1.000000e+00    18.0     22.0             DHRSX
        141698    RE   0.000000 -0.185628  4925.0  0.4925  3.973749e-01  9.619204e-01     0.0      1.5    CAAA01147332.1
        141699    RE  16.347951  3.563943  7373.0  0.7373  2.048452e-07  1.117704e-05   100.0     80.0    tdT-WPRE_trans
        141700    RE   0.000000  0.000000  5000.0  0.5000           NaN           NaN     0.0      0.0  CreER-WPRE_trans
    """
    if isinstance(x, AnnData):
        feature_names = x.var_names
    else:
        if feature_names is None:
            feature_names = np.arange(x.shape[1])
        else:
            assert len(feature_names) == x.shape[1], \
                "feature_names must have the same length as the number of " \
                "columns in X."
    mat, grouping, vertices, cluster_var = \
        _check_cluster_vertices(x, cluster_var, vertices)
    if not processed:
        mat = row_normalize(mat)
        mat = np.log1p(1e10*mat)
    if (grouping.value_counts() > 0).sum() < 2:
        raise ValueError("Must have at least 2 non-empty groups defined.")
    stats = wilcoxon(mat, grouping)
    stats['feature'] = np.tile(feature_names, len(grouping.cat.categories))
    if return_stats:
        return stats
    stats = stats[stats['group'].isin(vertices)]
    stats = stats[stats['logFC'] > lfc_thresh]
    selected = []
    for i, sub in stats.groupby('group', sort=False):
        sub = sub.sort_values(['padj', 'logFC'], ascending=[True, False])
        selected.extend(sub['feature'].values[:n_top])
    return selected


def wilcoxon(X, y):
    """
    X - n cells by m features sparse matrix
    y - n cells categorical vector
    """
    # Calculate Primary Statistics
    if isinstance(X, csr_matrix):
        rank_mat, ties = _ranking_csr(X)
        # rankdata(X.toarray().T, axis=1)
    else:
        rank_mat = rankdata(X.T, axis=1)
        ties = [_count_ties(rank_mat[i]) for i in range(rank_mat.shape[0])]
    # rank_mat: m features by n cells
    # Get mann-whitney u stats
    group_size = np.array([y.value_counts(sort=False)]).T
    ustats = _compute_ustats(rank_mat, y, group_size)
    # Get p-values

    n1n2 = group_size * (X.shape[0] - group_size)
    auc = ustats / n1n2
    pvals = _compute_pval(ustats, ties, X.shape[0], n1n2)
    fdr = _compute_fdr(pvals)

    # Calculate Auxiliary Statistics
    group_sums = np.zeros((len(y.cat.categories), X.shape[1]))
    for i, cat in enumerate(y.cat.categories):
        group_sums[i, :] = X[y == cat, :].sum(axis=0)
    group_nnz = np.zeros((len(y.cat.categories), X.shape[1]))
    for i, cat in enumerate(y.cat.categories):
        group_nnz[i, :] = (X[y == cat, :] > 0).sum(axis=0)
    group_mean = group_sums / group_size
    group_pct_in = group_nnz / group_size
    group_pct_out = group_nnz.sum(0) - group_nnz
    group_pct_out = group_pct_out / (X.shape[0] - group_size)
    lfc = group_mean - (X.sum(0) - group_sums) / (X.shape[0] - group_size)
    if isinstance(X, csr_matrix):
        lfc_flat = np.array(lfc.flatten())[0]
    else:
        lfc_flat = lfc.flatten()
    final = pd.DataFrame({
        'group': np.repeat(y.cat.categories, X.shape[1]),
        'avgExpr': group_mean.flatten(),
        'logFC': lfc_flat,
        'ustat': ustats.flatten(),
        'auc': auc.flatten(),
        'pval': pvals.flatten(),
        'padj': fdr.flatten(),
        'pct_in': 100*group_pct_in.flatten(),
        'pct_out': 100*group_pct_out.flatten()
    })
    return final


def _count_ties(rank_mat):
    """
    rank_mat - 1D array, length n cells
    """
    _, counts = np.unique(rank_mat, return_counts=True)
    return counts[counts > 1]


def _compute_ustats(Rank, y, group_size):
    """
    Rank       - m features x n cells matrix of ranking
    y          - n cells categorical pd.Series
    group_size - n groups x 1 array of group sizes
    """
    sums = np.zeros((len(y.cat.categories), Rank.shape[0]))
    for i, cat in enumerate(y.cat.categories):
        sums[i, :] = Rank[:, y == cat].sum(axis=1).T
    rhs = group_size * (group_size + 1) / 2
    if isinstance(Rank, np.ndarray):
        ustat = sums - rhs
    else:
        # Count for the rank sums of zeros, because ranking for csr matrix
        # does not count for zeros.
        group_nnz = np.zeros((len(y.cat.categories), Rank.shape[0]))
        for i, cat in enumerate(y.cat.categories):
            group_nnz[i, :] = Rank[:, y == cat].getnnz(axis=1)
        group_n_zero = group_size - group_nnz
        zero_ranks = (Rank.shape[1] - np.diff(Rank.indptr) + 1)/2
        ustat = group_n_zero * zero_ranks + sums - rhs
    return ustat


def _compute_pval(ustats, ties, N, n1n2):
    z = ustats - np.array([n1n2/2]).T
    z = z - np.sign(z)/0.5
    x1 = N ** 3 - N
    x2 = 1 / (12 * (N ** 2 - N))
    rhs = np.array([x1 - np.sum(tvals ** 3 - tvals) for tvals in ties]) * x2
    usigma = np.dot(n1n2, np.array([rhs]))
    usigma = np.sqrt(usigma)
    z = z / usigma
    pvals = 2 * stats.norm.cdf(-np.abs(z))
    return pvals[0]


def _compute_fdr(p_vals):
    ranked_p_values = rankdata(p_vals, axis=1)
    fdr = p_vals * p_vals.shape[1] / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


def _ranking_csr(X):
    """
    X - n cells by m features sparse matrix of gene expression
    This function only counts for the ranking of nonzero values in the
        csr_matrix.
    """
    rank_csr = X.copy()
    # Create a transposed row major csr matrix of ranks that has the same
    # sparse structure. So each row is a feaeture and diff(indptr) helps easily
    # locate nonzero values in each row.
    rank_csr = csr_matrix(rank_csr.T)
    all_ties = []
    nz_data_idx = 0
    # Vector of number of nonzero values per row
    nnz_per_row = np.diff(rank_csr.indptr)
    for nnz in nnz_per_row:
        # nnz: number of nonzero values in the row
        nz_slice = slice(nz_data_idx, nz_data_idx+nnz)
        rank = rankdata(rank_csr.data[nz_slice], method='average')
        n_zero = rank_csr.shape[1] - nnz
        rank += n_zero
        ties = [n_zero]
        ties.extend(np.unique(rank, return_counts=True)[1])
        all_ties.append(np.array(ties))
        # Inserting the rank values into the data array of the csr matrix
        # bc using [row,col] index creates copies of the matrix and is slow
        rank_csr.data[nz_slice] = rank
        nz_data_idx += nnz
    return rank_csr, all_ties
