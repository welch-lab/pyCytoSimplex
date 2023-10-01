from .util import _check_cluster_vertices
from .normalize import row_normalize
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy import stats
from anndata import AnnData


def select_top_features(
        X,
        cluster_var,
        vertices,
        n_top=30,
        processed=False,
        lfc_thresh=0.1,
        return_stats=False,
        feature_names=None
        ):
    """
    Select top features for each vertices based on the wilcoxon test.

    Parameters
    ----------
    X : AnnData, numpy.ndarray, or scipy.sparse.csr_matrix
        The matrix of gene expression. Recommended to be full size raw counts.
            (i.e. not log transformed or normalized and not only for highly
            variable genes)
        When AnnData, `X.X` will be used.
        Matrix like object must be n_obs by n_vars.
    cluster_var : str, list, or pd.Series
        The cluster assignment of each observation.
        When str, `X` has to be an AnnData object and `cluster_var` is the name
            of the cluster variable in `X.obs`.
        When list or pd.Series, the length must equal to `X.n_obs` or
            `X.shape[0]`.
    vertices : list or dict.
        The terminal specifications of the simplex.
        When list, each element must exist in the categories of `cluster_var`.
        When dict, served as grouped specifications of the terminals, meaning
            each terminal stands for one or more clusters. The values can be
            either a str for a single cluster or a list of multiple clusters.
            All clusters in the values will be treated as one cluster in the
            marker detection.
    n_top : int, optional (default: 30)
        The number of top features to select for each vertex.
    processed : bool, optional (default: False)
        Whether the input matrix is already processed.
        If True, the input matrix will be log transformed and row normalized.
        If False, the input matrix will be directly used to calculate the rank
            sum statistics. And logFC will be calculated assuming that the
            input matrix is log transformed.
    lfc_thresh : float, optional (default: 0.1)
        The log fold change threshold to select up-regulated genes.
    return_stats : bool, optional (default: False)
        Whether to return the full statistics of all clusters and all features
            instead of only returning the selected top features by default.
    feature_names : list, optional (default: None)
        The names of the features in the matrix. If None, the feature names
            will be the index of the matrix.

    Returns
    -------
    selected : list, when `return_stats=False`.
        The list of selected features. Maximum length is
            `n_top * len(vertices)` when enough features can pass the
            threshold.
    stats : pd.DataFrame, when `return_stats=True`.
        The statistics of the wilcoxon test, with `n_groups * n_features` rows.
            Columns are 'group', 'avgExpr', 'logFC', 'ustat', 'auc', 'pval',
            'padj', 'pct_in', 'pct_out' and 'feature'.
    """
    if isinstance(X, AnnData):
        feature_names = X.var_names
    else:
        if feature_names is None:
            feature_names = np.arange(X.shape[1])
        else:
            assert len(feature_names) == X.shape[1], \
                "feature_names must have the same length as the number of " \
                "columns in X."
    mat, grouping, vertices, cluster_var = \
        _check_cluster_vertices(X, cluster_var, vertices)
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
