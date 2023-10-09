from ._similarity import calc_sim
from ._normalize import row_normalize
from ._util import _check_cluster_vertices
import numpy as np
import matplotlib.pyplot as plt
import mpltern
import math
from anndata import AnnData
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
import pandas as pd
from typing import Optional, Union, Tuple


def plot_binary(
          x: Union[AnnData, np.ndarray, csr_matrix],
          cluster_var: Union[str, list, pd.Series],
          vertices: Union[list, dict],
          features: Optional[list[str]] = None,
          save_fig: bool = False,
          fig_path: str = "plot_binary.png",
          fig_size: Optional[Tuple[int, int]] = None,
          processed: bool = False,
          method: str = "euclidean",
          force: bool = False,
          sigma: float = 0.08,
          scale: bool = True,
          title: Optional[str] = None,
          split_cluster: bool = False,
          cluster_title: bool = True,
          dot_color: bool = "#8E8E8EFF",
          dot_size: float = 0.6,
          vertex_colors: list[str, str] = ["#3B4992FF", "#EE0000FF"],
          vertex_label_size: int = 12,
          gridline_alpha: float = 0.4,
          ) -> None:
    """
    Create binary plot that shows the similarity between each single cell and
    the two vertices of a simplex (two-ended line) which represents specified
    clusters. The simplex is conceptually placed horizontally and while the
    y-axis value is jittered for clarity. Additionally, a density curve of the
    similarity distribution is plotted on the top of the plot. Adding the
    velocity information is not supported.

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
        The terminal specifications. Should have exactly 2 elements for the
        1-simplex (binary simplex / ended line). Acceptable input include:

        - A :class:`list` of 2 :class:`str` that exist in the categories of
          `cluster_var`.
        - A :class:`dict` of 2 keys. The keys are presented as customizable
          vertex names. The corresponding value for each key can be either a
          :class:`str` for a single cluster, or a :class:`list` of :class:`str`
          for grouped vertex of multiple clusters.
    features
        The features to be used in the calculation of similarity. It is
        recommended to derive this list with
        :func:`~CytoSimplex.select_top_features`. Default uses all features.
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
    dot_size
        The size of the dots.
    vertex_colors
        The colors of the vertex labels, grid lines, axis labels and arrows.
    vertex_label_size
        The size of the vertex labels.
    gridline_alpha
        The alpha of the gridlines.

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
        vertices = {"OS": ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
                    "RE": ["Reticular_1", "Reticular_2"]}
        gene = csx.select_top_features(adata, "cluster", vertices)
        csx.plot_binary(adata, "cluster", vertices, gene)

    """
    # Useless call of mpltern function to avoid warning.
    mpltern_np_ver = mpltern.version("numpy")
    del mpltern_np_ver
    mat, grouping, vertices, original_cluster = \
        _check_cluster_vertices(x, cluster_var, vertices, n=2)
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

    if not split_cluster:
        if fig_size is None:
            fig_size = (8, 4)
        fig = plt.figure(figsize=fig_size)
        ax = plt.subplot()
        _add_binary_subplot(ax, sim_mat,
                            dot_color=dot_color, dot_size=dot_size,
                            vertex_colors=vertex_colors, title=title,
                            vertex_label_size=vertex_label_size,
                            gridline_alpha=gridline_alpha)
    else:
        cats = original_cluster.cat.categories
        subplot_nrow = math.floor(math.sqrt(len(cats)+1))
        subplot_ncol = math.ceil((len(cats)+1) / subplot_nrow)
        if fig_size is None:
            fig_size = (3*subplot_ncol+3, 2*subplot_nrow)
        fig = plt.figure(figsize=fig_size)
        fig.subplots_adjust(left=0.075, right=0.925, wspace=0.3)

        for i, cluster in enumerate(cats):
            ax = fig.add_subplot(subplot_nrow, subplot_ncol, i + 1)
            _add_binary_subplot(
                ax, sim_mat.loc[original_cluster == cluster, :],
                dot_color=dot_color, dot_size=dot_size,
                vertex_colors=vertex_colors,
                vertex_label_size=vertex_label_size,
                gridline_alpha=gridline_alpha
            )
            if cluster_title:
                ax.set_title(cluster, pad=10)
        ax = fig.add_subplot(subplot_nrow, subplot_ncol, i + 2)
        _add_binary_subplot(ax, sim_mat,
                            dot_size=dot_size, dot_color=dot_color,
                            vertex_colors=vertex_colors,
                            vertex_label_size=vertex_label_size,
                            gridline_alpha=gridline_alpha)
        if cluster_title:
            ax.set_title("All cells", pad=10)
        fig.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
    else:
        plt.show()


def _add_binary_subplot(
        ax,
        sim_mat,
        title=None,
        dot_size=0.6,
        dot_color="#8E8E8EFF",
        vertex_colors=["#3B4992FF", "#EE0000FF"],
        vertex_label_size=12,
        gridline_alpha=0.4
        ):
    y = np.random.uniform(0, 1, sim_mat.shape[0])
    x = sim_mat.iloc[:, 0]
    v1, v2 = sim_mat.columns
    # Draw vertical grid lines with dash line
    for i in range(1, 5):
        ax.plot([i/5, i/5], [0, 1], color="grey",
                linestyle="--", alpha=gridline_alpha, zorder=1)

    ax.scatter(x, y, s=dot_size, c=dot_color, zorder=2)
    # Hide all spines.
    ax.set_frame_on(False)
    # Hide y ticks and labels.
    ax.set_yticks([])
    # Add top x ticks and labels in reverse order
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    # Add histogram density curve of `x`
    density = gaussian_kde(x)
    x_grid = np.linspace(0, 1, 100)
    ax.plot(x_grid, density(x_grid)/density(x_grid).max(), color='black',
            zorder=3)
    # Draw an arrow from (0, 0) to (1, 0).
    ax.arrow(0, 0, 1.03, 0, color=vertex_colors[0], linewidth=3,
             head_width=0.02, head_length=0.02, length_includes_head=True,
             zorder=10)
    # Draw an arrow from (1, 1) to (0, 1).
    ax.arrow(1, 1, -1.03, 0, color=vertex_colors[1], linewidth=3,
             head_width=0.02, head_length=0.02, length_includes_head=True,
             zorder=10)
    # Add vertex labels.
    ax.text(1.06, 0, v1, ha="left", va="center",
            color=vertex_colors[0], fontsize=vertex_label_size)
    ax.text(-0.06, 1, v2, ha="right", va="center",
            color=vertex_colors[1], fontsize=vertex_label_size)
    # Add title.
    if title is not None:
        ax.set_title(title, pad=4)
