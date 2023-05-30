from .util import _check_cluster_vertices
from .normalize import row_normalize
from .similarity import calc_sim
from .velo_grid import aggregate_vertex_velo, aggregate_grid_velo
from .util import TETRA_VERTICES
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def plot_quaternary(
          X,
          cluster_var,
          vertices,
          features=None,
          velo_graph=None,
          save_fig=False,
          fig_path="plot_ternary.png",
          fig_size=None,
          processed=False,
          method="euclidean",
          force=False,
          sigma=0.08,
          scale=True,
          title=None,
          split_cluster=False,
          cluster_title=True,
          dot_color="#8E8E8EFF",
          n_velogrid=10,
          radius=0.08,
          dot_size=0.6,
          vertex_colors=["#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"],
          vertex_label_size=12,
          gridline_alpha=0.4,
          arrow_linewidth=0.004
          ):
    """
    Create ternary plot that shows the similarity between each single cell and
    the three vertices of a simplex which represents specified clusters.

    Parameters
    ----------
    X : AnnData, numpy.ndarray, or scipy.sparse.csr_matrix
        The expression matrix of the single cells. Each row represents a single
        cell and each column represents a gene.
    cluster_var : str, list, or pd.Series
        The cluster assignment of each single cell.
        Only when `X` is AnnData, can be a str that specifies the name of the
        cluster variable in `X.obs`. list or pd.Series is accepted in all
        cases, and the length must equal to `X.shape[0]`.
    vertices : list or dict, where the length must equal to 4.
        The terminal specifications of the simplex.
        When list, each element must exist in the categories of `cluster_var`.
        When dict, served as grouped specifications of the terminals, meaning
            each terminal stands for one or more clusters. The values can be
            either a str for a single cluster or a list of multiple clusters.
    features : list, optional
        The features to be used in the calculation of similarity. If None, all
        features will be used.
    velo_graph : csr_matrix, str
        The velocity graph of the single cells.
        When csr_matrix, it should be a sparse matrix with shape (n_obs, n_obs)
        When `X` is an AnnData object, `velo_graph` can be a str which is a key
            in `X.uns` that specifies a sparse matrix with shape `(n_obs,
            n_obs)`.
    save_fig : bool, optional
        Whether to save the figure. Default: False.
    fig_path : str, optional
        The path to save the figure. Default: "plot_ternary.png".
    fig_size : tuple of two numbers, optional
        The size of the figure. The first number for width and the second for
        height. Default: None.
    processed : bool, optional
        Whether the input matrix is already processed. When `False`, a raw
            count matrix (with integers) is recommended and preprocessing of
            library-size normalization, scaled log1p transformation will happen
            internally. If `True`, the input matrix will directly be used for
            computing the similarity. Default: False.
    method : str, choose from "euclidean", "cosine", "pearson", or "spearman".
        The method to calculate the similarity. Default: "euclidean". When
            "euclidean" or "cosine", the similarity is converted from the
            distance with a Gaussian kernel. When "pearson" or "spearman", the
            similarity is derived as the correlation is.
    force : bool
        Whether to force the calculation when the number of features exceeds
            500. Default: False.
    sigma : float
        The sigma parameter in the Gaussian kernel. Default: 0.08.
    scale : bool
        Whether to scale the similarity matrix by vertices. Default: True.
    title : str
        The title of the plot. Default: None.
    split_cluster : bool
        Whether to split the plot by clusters. If False (default), all cells
            will be plotted in one plot. If True, the cells will be split by
            clusters and each cluster will be plotted in one subplot.
    cluster_title : bool
        Whether to show the cluster name as the title of each subplot when
        `split_cluster=True`. Default: True.
    dot_color : str
        The color of the dots. Default: "#8E8E8EFF".
    n_velogrid : int
        The number of square grids, along the bottom side of triangle, to
            aggregate the velocity of cells falling into each grid. Default:
            10.
    radius : float
        Scaling factor of aggregated velocity to get the arrow length. Default:
            0.08.
    dot_size : float
        The size of the dots. Default: 0.6.
    vertex_colors : list of three str
        The colors of the vertex labels, grid lines, axis labels and arrows.
            Default: `["#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"]`,
            respectively for left, top, and right vertices.
    vertex_label_size : int
        The size of the vertex labels. Default: 12.
    gridline_alpha : float
        The alpha of the gridlines. Default: 0.4.
    arrow_linewidth : float
        The linewidth of the arrows. Default: 0.004.

    Returns
    -------
    None. Figure will be shown or saved.
    """
    mat, grouping, vertices, original_cluster = \
        _check_cluster_vertices(X, cluster_var, vertices, n=4)
    if not processed:
        mat = row_normalize(mat)
        mat *= 10000
        mat = np.log1p(mat)
    if features is not None:
        if isinstance(X, AnnData):
            features = X.var_names.isin(features)
        mat = mat[:, features]

    sim_mat = calc_sim(mat=mat, cluster_var=grouping, vertices=vertices,
                       method=method, force=force, sigma=sigma, scale=scale)
    if velo_graph is not None:
        if isinstance(velo_graph, str):
            velo_graph = X.uns[velo_graph]
        velo_mat = aggregate_vertex_velo(velo_graph, grouping, vertices)
    else:
        velo_mat = None

    if not split_cluster:
        if fig_size is None:
            fig_size = (6, 6)
        fig = plt.figure(figsize=fig_size)
        ax = plt.subplot(projection="3d")
        _add_quaternary_subplot(ax, sim_mat, velo_mat=velo_mat,
                                dot_color=dot_color, n_velogrid=n_velogrid,
                                radius=radius, dot_size=dot_size,
                                vertex_colors=vertex_colors, title=title,
                                vertex_label_size=vertex_label_size,
                                gridline_alpha=gridline_alpha,
                                arrow_linewidth=arrow_linewidth)
    else:
        cats = original_cluster.cat.categories
        subplot_nrow = math.floor(math.sqrt(len(cats)+1))
        subplot_ncol = math.ceil((len(cats)+1) / subplot_nrow)
        if fig_size is None:
            fig_size = (2.5*subplot_ncol+3, 2*subplot_nrow+2)
        fig = plt.figure(figsize=fig_size)
        fig.subplots_adjust(left=0.075, right=0.925, wspace=0.3)

        for i, cluster in enumerate(cats):
            if velo_mat is not None:
                velo_mat_sub = velo_mat.loc[original_cluster == cluster, :]
            else:
                velo_mat_sub = None
            ax = fig.add_subplot(subplot_nrow, subplot_ncol, i + 1,
                                 projection="3d")
            _add_quaternary_subplot(
                ax,
                sim_mat.loc[original_cluster == cluster, :],
                velo_mat=velo_mat_sub,
                dot_color=dot_color, n_velogrid=n_velogrid,
                radius=radius, dot_size=dot_size,
                vertex_colors=vertex_colors,
                vertex_label_size=vertex_label_size,
                gridline_alpha=gridline_alpha,
                arrow_linewidth=arrow_linewidth
            )
            if cluster_title:
                ax.set_title(cluster, pad=10)
        ax = fig.add_subplot(subplot_nrow, subplot_ncol, i + 2,
                             projection="3d")
        _add_quaternary_subplot(ax, sim_mat, velo_mat=velo_mat,
                                dot_size=dot_size, dot_color=dot_color,
                                vertex_colors=vertex_colors,
                                vertex_label_size=vertex_label_size,
                                gridline_alpha=gridline_alpha,
                                arrow_linewidth=arrow_linewidth)
        if cluster_title:
            ax.set_title("All cells", pad=10)
        fig.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
    else:
        plt.show()


def _add_quaternary_subplot(
        ax,
        sim_mat,
        velo_mat=None,
        title=None,
        n_velogrid=10,
        radius=0.08,
        dot_size=0.6,
        dot_color="#8E8E8EFF",
        vertex_colors=["#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"],
        vertex_label_size=12,
        gridline_alpha=0.4,
        arrow_linewidth=0.004
):
    simplex = TETRA_VERTICES
    segment_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for pair in segment_pairs:
        ax.plot(simplex.iloc[pair, 0], simplex.iloc[pair, 1],
                simplex.iloc[pair, 2], color='black', lw=1)
    for i in range(4):
        ax.text(simplex.iloc[i, 0], simplex.iloc[i, 1], simplex.iloc[i, 2],
                s=sim_mat.columns[i], c=vertex_colors[i])
    sim_cart = pd.DataFrame(np.dot(sim_mat, simplex), index=sim_mat.index,
                            columns=simplex.columns)
    ax.scatter(sim_cart['x'], sim_cart['y'], sim_cart['z'], s=dot_size,
               c=dot_color, alpha=0.8, edgecolors='none')
    if velo_mat is not None:
        grid, arrow = aggregate_grid_velo(sim_mat, velo_mat, n_grid=n_velogrid,
                                          radius=radius)
        # TODO, see velo_grid.py
    ax.set_axis_off()
