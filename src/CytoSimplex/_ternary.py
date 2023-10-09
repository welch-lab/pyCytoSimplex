from ._similarity import calc_sim
from ._normalize import row_normalize
from ._util import _check_cluster_vertices, TRIANGLE_VERTICES
from ._velo_grid import aggregate_vertex_velo, aggregate_grid_velo, _cart2bary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
import math
from anndata import AnnData
from typing import Union, Optional, Tuple
from scipy.sparse import csr_matrix


def plot_ternary(
          x: Union[AnnData, np.ndarray, csr_matrix],
          cluster_var: Union[str, list, pd.Series],
          vertices: Union[list, dict],
          features: Optional[list[str]] = None,
          velo_graph: Optional[Union[csr_matrix, str]] = None,
          save_fig: bool = False,
          fig_path: str = "plot_ternary.png",
          fig_size: Optional[Tuple[int, int]] = None,
          processed: bool = False,
          method: str = "euclidean",
          force: bool = False,
          sigma: float = 0.08,
          scale: bool = True,
          title: Optional[str] = None,
          split_cluster: bool = False,
          cluster_title: bool = True,
          dot_color: str = "#8E8E8EFF",
          n_velogrid: int = 10,
          radius: float = 0.08,
          dot_size: float = 0.6,
          vertex_colors: list[str, str, str] = ["#3B4992FF", "#EE0000FF",
                                                "#008B45FF"],
          vertex_label_size: int = 12,
          gridline_alpha: float = 0.4,
          axis_text_show: bool = True,
          arrow_linewidth: float = 0.004
          ) -> None:
    """
    Create ternary plot that shows the similarity between each single cell and
    the three vertices of a simplex (equilateral triangle) which represents
    specified clusters. Velocity information can be added to the plot.

    Parameters
    ----------
    x
        Object that has the expression matrix of the single cells. Each row
        represents a single cell and each column represents a gene.
    cluster_var
        The cluster assignment of each single cell. If `x` is an
        :class:`anndata.AnnData`, `cluster_var` can be a `str` that specifies
        the name of the cluster variable in `x.obs`. `list` or
        :class:`pandas.Series` is accepted in all cases, and the length must
        equal to `x.shape[0]`.
    vertices
        The terminal specifications. Should have exactly 3 elements for the
        2-simplex (ternary simplex / triangle). Acceptable input include:

        - A :class:`list` of 3 :class:`str` that exist in the categories of
          `cluster_var`.
        - A :class:`dict` of 3 keys. The keys are presented as customizable
          vertex names. The corresponding value for each key can be either a
          :class:`str` for a single cluster, or a :class:`list` of :class:`str`
          for grouped vertex of multiple clusters.
    features
        The features to be used in the calculation of similarity. It is
        recommended to derive this list with
        :func:`~CytoSimplex.select_top_features`. Default uses all features.
    velo_graph
        The velocity graph of the single cells, presented as an n-by-n sparse
        matrix. When `x` is :class:`anndata.AnnData`, `velo_graph` can be a
        :class:`str` to be used as a key to retrieve the appropiate data from
        `x.uns`. By default, no velocity information will be added to the plot.
    save_fig
        Whether to save the figure.
    fig_path
        The path to save the figure.
    fig_size
        The size of the figure. The first number for width and the second for
        height.
    processed
        Whether the input matrix is already processed. When `False`, a raw
        count matrix (with integers) is recommended and preprocessing of
        library-size normalization, scaled log1p transformation will happen
        internally. If `True`, the input matrix will directly be used for
        computing the similarity.
    method
        Choose from `"euclidean"`, `"cosine"`, `"pearson"` and `"spearman"`.
        When `"euclidean"` or `"cosine"`, the similarity is converted from the
        distance with a Gaussian kernel and the argument `sigma` would be
        applied. When `"pearson"` or `"spearman"`, the similarity is derived as
        the correlation is.
    force
        Whether to force the calculation when the number of features exceeds
        500.
    sigma
        The sigma parameter in the Gaussian kernel that converts the distance
        metrics into similarity.
    scale
        Whether to scale the similarity matrix by vertices.
    title
        The title of the plot. Not used when `split_cluster=True`.
    split_cluster
        Whether to split the plot by clusters. If `False`, all cells will be
        plotted in one plot. If `True`, the cells will be split by clusters and
        each cluster will be plotted in one subplot.
    cluster_title
        Whether to show the cluster name as the title of each subplot when
        `split_cluster=True`.
    dot_color
        The color of the dots.
    n_velogrid
        The number of square grids, along the bottom side of the triangle, to
        aggregate the velocity of cells falling into each grid.
    radius
        Scaling factor of aggregated velocity to get the arrow length.
    dot_size
        The size of the dots.
    vertex_colors
        The colors of the vertex labels, grid lines, axis labels and arrows.
    vertex_label_size
        The size of the vertex labels.
    axis_text_show
        Whether to show the text along the axis.
    gridline_alpha
        The alpha of the gridlines.
    arrow_linewidth
        The linewidth of the arrows.

    Returns
    -------
        Figure will be shown or saved.

    Examples
    --------

    .. plot::
        :context: close-figs

        import CytoSimplex as csx
        import scanpy as sc
        adata = sc.read(
            filename="test.h5ad",
            backup_url="https://figshare.com/ndownloader/files/41034857"
        )
        vertices = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
                    'RE': ['Reticular_1', 'Reticular_2'],
                    'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
        gene = csx.select_top_features(adata, "cluster", vertices)
        csx.plot_ternary(adata, "cluster", vertices, gene)
        csx.plot_ternary(adata, "cluster", vertices, gene, velo_graph='velo')
    """
    # Useless call of mpltern function to avoid warning.
    mpltern_np_ver = mpltern.version("numpy")
    del mpltern_np_ver
    mat, grouping, vertices, original_cluster = \
        _check_cluster_vertices(x, cluster_var, vertices, n=3)
    if not processed:
        mat = row_normalize(mat)
        mat *= 10000
        mat = np.log1p(mat)
    if features is not None:
        if isinstance(x, AnnData):
            features = x.var_names.isin(features)
        mat = mat[:, features]

    sim_mat = calc_sim(mat=mat, cluster_var=grouping, vertices=vertices,
                       method=method, force=force, sigma=sigma, scale=scale)
    if velo_graph is not None:
        if isinstance(velo_graph, str):
            velo_graph = x.uns[velo_graph]
        velo_mat = aggregate_vertex_velo(velo_graph, grouping, vertices)
    else:
        velo_mat = None

    if not split_cluster:
        if fig_size is None:
            fig_size = (6, 6)
        fig = plt.figure(figsize=fig_size)
        ax = plt.subplot(projection='ternary')
        _add_ternary_subplot(ax, sim_mat, velo_mat=velo_mat,
                             dot_color=dot_color, n_velogrid=n_velogrid,
                             radius=radius, dot_size=dot_size,
                             vertex_colors=vertex_colors, title=title,
                             vertex_label_size=vertex_label_size,
                             gridline_alpha=gridline_alpha,
                             axis_text_show=axis_text_show,
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
                                 projection='ternary')
            _add_ternary_subplot(
                ax, sim_mat.loc[original_cluster == cluster, :],
                velo_mat=velo_mat_sub,
                dot_color=dot_color, n_velogrid=n_velogrid,
                radius=radius, dot_size=dot_size,
                vertex_colors=vertex_colors,
                vertex_label_size=vertex_label_size,
                gridline_alpha=gridline_alpha,
                axis_text_show=axis_text_show,
                arrow_linewidth=arrow_linewidth
            )
            if cluster_title:
                ax.set_title(cluster, pad=10)
        ax = fig.add_subplot(subplot_nrow, subplot_ncol, i + 2,
                             projection='ternary')
        _add_ternary_subplot(ax, sim_mat, velo_mat=velo_mat,
                             dot_size=dot_size, dot_color=dot_color,
                             vertex_colors=vertex_colors,
                             vertex_label_size=vertex_label_size,
                             gridline_alpha=gridline_alpha,
                             axis_text_show=axis_text_show,
                             arrow_linewidth=arrow_linewidth)
        if cluster_title:
            ax.set_title("All cells", pad=10)
        fig.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
    else:
        plt.show()


def _add_ternary_subplot(
        ax,
        sim_mat,
        velo_mat=None,
        title=None,
        n_velogrid=10,
        radius=0.08,
        dot_size=0.6,
        dot_color="#8E8E8EFF",
        vertex_colors=["#3B4992FF", "#EE0000FF", "#008B45FF"],
        vertex_label_size=12,
        gridline_alpha=0.4,
        axis_text_show=True,
        arrow_linewidth=0.004
        ):
    l_label = sim_mat.columns[0]
    t_label = sim_mat.columns[1]
    r_label = sim_mat.columns[2]
    ax.tick_params(labelrotation='horizontal', direction='in')
    ax.grid(axis='l', which='both', linestyle='--',
            color=vertex_colors[0], alpha=gridline_alpha)
    ax.grid(axis='t', which='both', linestyle='--',
            color=vertex_colors[1], alpha=gridline_alpha)
    ax.grid(axis='r', which='both', linestyle='--',
            color=vertex_colors[2], alpha=gridline_alpha)

    ax.scatter(sim_mat[t_label], sim_mat[l_label], sim_mat[r_label],
               s=dot_size, color=dot_color)

    if velo_mat is not None:
        grid_cart, arrow_vec = aggregate_grid_velo(sim_mat, velo_mat,
                                                   n_grid=n_velogrid,
                                                   radius=radius)
        if grid_cart.shape[0] > 0:
            grid_bary = _cart2bary(TRIANGLE_VERTICES, grid_cart)
            grid_bary = pd.DataFrame(grid_bary, index=grid_cart.index,
                                     columns=sim_mat.columns)
            for i, vec in enumerate(arrow_vec):
                arrow_end_bary = _cart2bary(TRIANGLE_VERTICES, vec)
                arrow_end_bary = pd.DataFrame(arrow_end_bary,
                                              index=vec.index,
                                              columns=sim_mat.columns)
                arrow_distence_bary = arrow_end_bary - grid_bary
                arrow_select = np.sqrt((arrow_distence_bary**2).sum(1)) > 1e-2
                if arrow_select.sum() > 0:
                    ax.quiver(grid_bary.loc[arrow_select, t_label],
                              grid_bary.loc[arrow_select, l_label],
                              grid_bary.loc[arrow_select, r_label],
                              arrow_end_bary.loc[arrow_select, t_label],
                              arrow_end_bary.loc[arrow_select, l_label],
                              arrow_end_bary.loc[arrow_select, r_label],
                              color=vertex_colors[i],
                              width=arrow_linewidth)

    ax.tick_params(axis='l', colors=vertex_colors[0])
    ax.tick_params(axis='t', colors=vertex_colors[1])
    ax.tick_params(axis='r', colors=vertex_colors[2])

    if not axis_text_show:
        ax.laxis.set_ticklabels([])
        ax.taxis.set_ticklabels([])
        ax.raxis.set_ticklabels([])

    ax.tick_params(tick1On=axis_text_show, tick2On=axis_text_show)

    ax.set_llabel(l_label, color=vertex_colors[0],
                  fontsize=vertex_label_size)
    ax.set_tlabel(t_label, color=vertex_colors[1],
                  fontsize=vertex_label_size)
    ax.set_rlabel(r_label, color=vertex_colors[2],
                  fontsize=vertex_label_size)

    ax.set_title(title)
