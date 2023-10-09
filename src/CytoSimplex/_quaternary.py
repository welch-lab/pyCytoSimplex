from ._util import _check_cluster_vertices
from ._normalize import row_normalize
from ._similarity import calc_sim
from ._velo_grid import aggregate_vertex_velo, aggregate_grid_velo
from ._util import TETRA_VERTICES
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import Union, Optional, Tuple
from scipy.sparse import csr_matrix


def plot_quaternary(
          x: Union[AnnData, np.ndarray, csr_matrix],
          cluster_var: Union[str, list, pd.Series],
          vertices: Union[list, dict],
          features: Optional[list[str]] = None,
          velo_graph: Optional[Union[csr_matrix, str]] = None,
          save_fig: bool = False,
          fig_path: str = "plot_quaternary.png",
          fig_size: Optional[Tuple[int, int]] = None,
          processed: bool = False,
          method: str = "euclidean",
          force: bool = False,
          sigma: float = 0.05,
          scale: bool = True,
          title: Optional[str] = None,
          split_cluster: bool = False,
          cluster_title: bool = True,
          dot_color: str = "#8E8E8EFF",
          n_velogrid: int = 10,
          radius: float = 0.16,
          dot_size: float = 0.6,
          vertex_colors: list[str, str, str, str] = ["#3B4992FF", "#EE0000FF",
                                                     "#008B45FF", "#631879FF"],
          vertex_label_size: int = 12,
          arrow_linewidth: float = 1.0
          ) -> None:
    """
    Create ternary plot that shows the similarity between each single cell and
    the four vertices of a simplex (tetrahedron) which represents specified
    clusters. Velocity information can be added to the plot.

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
        The terminal specifications. Should have exactly 4 elements for the
        3-simplex (quaternary simplex / tetrahedron). Acceptable input include:

        - A :class:`list` of 4 :class:`str` that exist in the categories of
          `cluster_var`.
        - A :class:`dict` of 4 keys. The keys are presented as customizable
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
    arrow_linewidth
        The linewidth of the arrows.

    Returns
    -------
        Figure will be shown or saved.

    Examples
    --------
    The 3-simplex (i.e. simplex in 3D, which is a tetrahedron) visualized by
    Matplotlib is indeed interactive in users interactive session.

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
                    'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
                    'ORT': ['ORT_1', 'ORT_2']}
        gene = csx.select_top_features(adata, "cluster", vertices)
        csx.plot_quaternary(adata, "cluster", vertices, gene)
        csx.plot_quaternary(adata, "cluster", vertices, gene, velo_graph='velo')
    """
    mat, grouping, vertices, original_cluster = \
        _check_cluster_vertices(x, cluster_var, vertices, n=4)
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
        ax = plt.subplot(projection="3d")
        _add_quaternary_subplot(ax, sim_mat, velo_mat=velo_mat,
                                dot_color=dot_color, n_velogrid=n_velogrid,
                                radius=radius, dot_size=dot_size,
                                vertex_colors=vertex_colors, title=title,
                                vertex_label_size=vertex_label_size,
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
        radius=0.16,
        dot_size=0.6,
        dot_color="#8E8E8EFF",
        vertex_colors=["#3B4992FF", "#EE0000FF", "#008B45FF", "#631879FF"],
        vertex_label_size=12,
        arrow_linewidth=1
):
    simplex = TETRA_VERTICES
    segment_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for pair in segment_pairs:
        ax.plot(simplex.iloc[pair, 0], simplex.iloc[pair, 1],
                simplex.iloc[pair, 2], color='black', lw=0.5)
    for i in range(4):
        ax.text(simplex.iloc[i, 0], simplex.iloc[i, 1], simplex.iloc[i, 2],
                s=sim_mat.columns[i], c=vertex_colors[i],
                fontsize=vertex_label_size)
    sim_cart = pd.DataFrame(np.dot(sim_mat, simplex), index=sim_mat.index,
                            columns=simplex.columns)
    ax.scatter(sim_cart['x'], sim_cart['y'], sim_cart['z'], s=dot_size,
               c=dot_color, alpha=0.8, edgecolors='none')
    if velo_mat is not None:
        grid_cart, arrow_vec = aggregate_grid_velo(sim_mat, velo_mat,
                                                   n_grid=n_velogrid,
                                                   radius=radius)
        for i, vec in enumerate(arrow_vec):
            arrow_dist = vec - grid_cart
            ax.quiver(grid_cart.loc[:, 'x'],
                      grid_cart.loc[:, 'y'],
                      grid_cart.loc[:, 'z'],
                      arrow_dist.loc[:, 'x'],
                      arrow_dist.loc[:, 'y'],
                      arrow_dist.loc[:, 'z'],
                      color=vertex_colors[i],
                      lw=arrow_linewidth)
    ax.set_axis_off()
    ax.set_title(title, pad=10)
