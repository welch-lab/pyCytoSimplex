from .similarity import calc_sim
from .normalize import row_normalize
from .util import _check_cluster_vertices
import numpy as np
import matplotlib.pyplot as plt
import mpltern
import math
from anndata import AnnData
from scipy.stats import gaussian_kde


def plot_binary(
          X,
          cluster_var,
          vertices,
          features=None,
          save_fig=False,
          fig_path="plot_binary.png",
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
          vertex_colors=["#3B4992FF", "#EE0000FF"],
          vertex_label_size=12,
          gridline_alpha=0.4,
          ):
    """
    Create binary plot that shows the similarity between each single cell and
    the two vertices of a simplex (two-ended line) which represents specified
    clusters. The simplex is conceptually placed horizontally and while the
    y-axis value is jittered for clarity. Additionally, a density curve of the
    similarity distribution is plotted on the top of the plot. Adding the
    velocity information is not supported.

    Parameters
    ----------
    X : AnnData, numpy.ndarray, or scipy.sparse.csr_matrix
        The expression matrix of the single cells. Each row represents a single
            cell and each column represents a gene.
    cluster_var : str, list, or pd.Series
        The cluster assignment of each single cell.
        Only when `X` is AnnData, can be a str that specifies the name of the
            cluster variable in `X.obs`.
        list or pd.Series is accepted in all cases, and the length must equal
            to `X.shape[0]`.
    vertices : list or dict, where the length must equal to 2.
        The terminal specifications of the simplex.
        When list, each element must exist in the categories of `cluster_var`.
        When dict, served as grouped specifications of the terminals, meaning
            each terminal stands for one or more clusters. The values can be
            either a str for a single cluster or a list of multiple clusters.
    features : list, optional
        The features to be used in the calculation of similarity. If None, all
            features will be used.
    save_fig : bool, optional
        Whether to save the figure. Default: False.
    fig_path : str, optional
        The path to save the figure. Default: "plot_binary.png".
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
        The method to calculate the similarity. Default: "euclidean".
        When "euclidean" or "cosine", the similarity is converted from the
            distance with a Gaussian kernel.
        When "pearson" or "spearman", the similarity is derived as the
            correlation is.
    force : bool
        Whether to force the calculation when the number of features exceeds
            500. Default: False.
    sigma : float
        The sigma parameter in the Gaussian kernel. Default: 0.08.
    scale : bool
        Whether to scale the similarity matrix by vertices. Default: True.
    title : str
        The title of the plot. Not used when `split_cluster=True`. Default:
            None.
    split_cluster : bool
        Whether to split the plot by clusters. If False (default), all cells
            will be plotted in one plot. If True, the cells will be split by
            clusters and each cluster will be plotted in one subplot.
    cluster_title : bool
        Whether to show the cluster name as the title of each subplot when
            `split_cluster=True`. Default: True.
    dot_color : str
        The color of the dots. Default: "#8E8E8EFF".
    dot_size : float
        The size of the dots. Default: 0.6.
    vertex_colors : list of two str
        The colors of the vertex labels, grid lines, axis labels and arrows.
            Default: `["#3B4992FF", "#EE0000FF"]`, respectively
            for right and left vertices.
    vertex_label_size : int
        The size of the vertex labels. Default: 12.
    gridline_alpha : float
        The alpha of the gridlines. Default: 0.4.

    Returns
    -------
    None. Figure will be shown or saved.
    """
    # Useless call of mpltern function to avoid warning.
    mpltern_np_ver = mpltern.version("numpy")
    del mpltern_np_ver
    mat, grouping, vertices, original_cluster = \
        _check_cluster_vertices(X, cluster_var, vertices, n=2)
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
