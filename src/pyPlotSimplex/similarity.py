from .util import _check_cluster_vertices
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from sklearn.feature_selection import r_regression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calc_sim(mat, cluster_var, vertices, method="euclidean", sigma=0.08,
             scale=True, force=False):
    if mat.shape[1] > 500 and not force:
        raise ValueError(f"Detected more than 500 ({mat.shape[1]}) features in "
                         "input matrix. Calculation will be slow and result will"
                         " be affected. Selection on features is recommended. "
                         "Use `force=True` to continue.")

    # For each vertices, calculate the mean of the observations in the corresponding clusters.
    centroids = np.zeros((len(vertices), mat.shape[1]))
    for i, v in enumerate(vertices):
        centroids[i, :] = mat[cluster_var == v,:].mean(axis=0)

    # Calculate the similarity matrix.
    if "euclidean".startswith(method):
        dis = euclidean_distances(mat, centroids)
        dis = pd.DataFrame(dis, columns=vertices)
        dis = dis.apply(lambda y: y/y.sum(), axis=1)
        sim = dis.apply(lambda y: y.apply(lambda y: np.exp(-y**2/sigma)))
    elif "cosine".startswith(method):
        dis = _converted_cosine(mat, centroids)
        dis = pd.DataFrame(dis, columns=vertices)
        dis = dis.apply(lambda y: y/y.sum(), axis=1)
        sim = dis.apply(lambda y: y.apply(lambda y: np.exp(-y**2/sigma)))
    elif "pearson".startswith(method):
        sim = pairwise_corr(mat, centroids, method="pearson")
        sim = pd.DataFrame(sim, columns=vertices)
    elif "spearman".startswith(method):
        sim = pairwise_corr(mat, centroids, method="spearman")
        sim = pd.DataFrame(sim, columns=vertices)
    else:
        raise ValueError("The method must be one of 'euclidean', 'cosine', "
                         "'pearson', or 'spearman'.")

    # Scale the similarity matrix by vertices.
    if scale:
        sim = sim.apply(lambda y: (y - y.min())/(y.max() - y.min()), axis=0)

    # Normalize the similarity matrix by observations.
    sim = sim.apply(lambda y: y/y.sum(), axis=1)
    sim.index = cluster_var.index
    return sim

def pairwise_corr(x, centroids, method="pearson"):
    if method == "pearson":
        corr = np.array([r_regression(x.T, centroids[i])
                         for i in range(centroids.shape[0])]).T
    else:
        if isinstance(x, np.ndarray):
            corr = spearmanr(x, centroids, axis=1)[0]
        elif isinstance(x, csr_matrix):
            corr = spearmanr(x.toarray(), centroids, axis=1)[0]
        corr = corr[:x.shape[0], -centroids.shape[0]:]
    return corr

def _converted_cosine(x, centroids):
    cosine_sim = cosine_similarity(x, centroids)
    tol = 1e-12
    cosine_sim[cosine_sim < -(1 - tol)] = -1
    cosine_sim[cosine_sim > 1 - tol] = 1
    cosine_sim = np.arccos(cosine_sim) * 180 / np.pi
    return cosine_sim
