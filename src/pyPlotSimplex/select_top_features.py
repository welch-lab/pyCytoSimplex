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
        return_stats=False):
    if isinstance(X, AnnData):
        feature_names = X.var_names
    else:
        feature_names = np.arange(X.shape[1])
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
        rank_mat = rankdata(X.toarray().T, axis=1)
    else:
        rank_mat = rankdata(X.T, axis=1)
    # rank_mat: m features by n cells
    # Get mann-whitney u stats
    group_size = np.array([y.value_counts(sort=False)]).T
    ustats = _compute_ustats(rank_mat.T, y, group_size)
    # Get p-values
    ties = [_count_ties(rank_mat[i]) for i in range(rank_mat.shape[0])]
    n1n2 = group_size * (X.shape[0] - group_size)
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
    final = pd.DataFrame({
        'group': np.repeat(y.cat.categories, X.shape[1]),
        'avgExpr': group_mean.flatten(),
        'logFC': np.array(lfc.flatten())[0],
        'ustat': ustats.flatten(),
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


def _compute_ustats(X, y, group_size):
    """
    X - n cells by m features sparse matrix
    y - n cells categorical vector
    """
    sums = np.zeros((len(y.cat.categories), X.shape[1]))
    for i, cat in enumerate(y.cat.categories):
        sums[i, :] = X[y == cat, :].sum(axis=0)
    rhs = group_size * (group_size + 1) / 2
    ustat = sums - rhs
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
