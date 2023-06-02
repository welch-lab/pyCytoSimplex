import scanpy as sc
import numpy as np
import pandas as pd
import pytest
from pyPlotSimplex import row_normalize, select_top_features, \
    plot_binary, plot_ternary, plot_quaternary
import matplotlib.pyplot as plt
from unittest.mock import patch

adata = sc.read(filename='test.h5ad',
                backup_url="https://figshare.com/ndownloader/files/41034857")

VT_BIN = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2']}
VT_TER = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
VT_QUA = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
          'ORT': ['ORT_1', 'ORT_2']}

# PREPROCESSING TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_normalize():
    adata_test = adata.copy()
    x1 = row_normalize(adata_test.X)
    row_normalize(adata_test)
    assert np.allclose(x1.sum(1), 1)


def test_feature_selection():
    adata_test = adata.copy()
    gene = select_top_features(adata_test, "cluster", VT_TER)
    assert len(gene) == 90

    gene2 = select_top_features(adata_test.X, adata_test.obs.cluster,
                                VT_TER, feature_names=adata_test.var_names)
    assert gene == gene2

    gene3 = select_top_features(adata_test.X.toarray(),
                                adata_test.obs.cluster.tolist(),
                                VT_TER)
    assert (gene == adata_test.var_names[gene3]).all()

    stats = select_top_features(adata_test, "cluster",
                                ["Osteoblast_1", "Reticular_1"],
                                return_stats=True)
    assert type(stats) == pd.DataFrame


# BINARY PLOT TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GENE_BIN = select_top_features(adata, "cluster", VT_BIN)


def test_failed_binary():
    with pytest.raises(TypeError, match="The object must be an AnnData, "):
        plot_binary("Hi", "cluster", VT_TER, GENE_BIN)
    with pytest.raises(ValueError, match="The length of the cluster_var must"):
        plot_binary(adata, ['a', 'b', 'c'], VT_TER, GENE_BIN)
    with pytest.raises(ValueError, match="All vertices must be in cluster"):
        plot_binary(adata, "cluster", ["a", "b", "c"], GENE_BIN)
    with pytest.raises(TypeError, match="The vertices must be a list or"):
        plot_binary(adata, "cluster", 'yo', GENE_BIN)
    with pytest.raises(ValueError, match="hello is not in the categories of"):
        plot_binary(adata, "cluster", {'a': ['hello', 'Osteoblast_1'],
                                       'b': 'Reticular_1'}, GENE_BIN)
    with pytest.raises(ValueError, match="hello is not in the categories of"):
        plot_binary(adata, "cluster", {'a': ['Osteoblast_1'],
                                       'b': 'hello'}, GENE_BIN)
    with pytest.raises(ValueError, match="Detected more than 500"):
        plot_binary(adata, "cluster", VT_BIN)
    with pytest.raises(ValueError, match="The method must be one of"):
        plot_binary(adata, "cluster", VT_BIN, GENE_BIN, method='hello')


@patch('matplotlib.pyplot.show')
def test_binary_single(mock_show):
    adata_test = adata.copy()
    plot_binary(adata_test, "cluster",
                {'a': 'Osteoblast_1',
                 'b': 'Reticular_1'},
                GENE_BIN, title="title")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == 'title'


@patch('matplotlib.pyplot.show')
def test_binary_split(mock_show):
    adata_test = adata.copy()
    plot_binary(adata_test, "cluster", VT_BIN, GENE_BIN, split_cluster=True)
    fig = plt.gcf()
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1


# TERNARY PLOT TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GENE_TER = select_top_features(adata, "cluster", VT_TER)


def test_failed_ternary():
    adata_test = adata.copy()
    with pytest.raises(ValueError, match="Must specify 3 vertices."):
        plot_ternary(adata_test, "cluster", VT_BIN, GENE_TER)


@patch('matplotlib.pyplot.show')
def test_stdout_ternary(mock_show, capfd):
    adata_test = adata.copy()
    plot_ternary(adata_test, "cluster", VT_QUA, GENE_TER)
    out, err = capfd.readouterr()
    assert "WARNING: Detected more than 3 vertices" in out
    plot_ternary(adata_test, "cluster",
                 ["Osteoblast_1", "Reticular_1", "Chondrocyte_1", "ORT_1"],
                 GENE_TER)
    out, err = capfd.readouterr()
    assert "WARNING: Detected more than 3 vertices" in out


@patch('matplotlib.pyplot.show')
def test_ternary_single_wo_velo(mock_show):
    adata_test = adata.copy()
    plot_ternary(adata_test, "cluster", VT_TER, GENE_TER)
    fig = plt.gcf()
    assert fig.axes[0].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_single_w_velo(mock_show):
    adata_test = adata.copy()
    plot_ternary(adata_test, "cluster", VT_TER, GENE_TER, velo_graph='velo',
                 method="cos")
    fig = plt.gcf()
    assert fig.axes[0].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_split_wo_velo(mock_show):
    adata_test = adata.copy()
    plot_ternary(adata_test, "cluster", VT_TER, GENE_TER, split_cluster=True,
                 method='p')
    fig = plt.gcf()
    assert fig.axes[2].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_split_w_velo(mock_show):
    adata_test = adata.copy()
    plot_ternary(adata_test.X.toarray(), adata_test.obs.cluster,
                 VT_TER, adata_test.var_names.isin(GENE_TER),
                 velo_graph=adata_test.uns['velo'],
                 split_cluster=True, method='sp')
    fig = plt.gcf()
    assert fig.axes[3].get_llabel() == 'OS'
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1


# QUATERNARY PLOT TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GENE_QUA = select_top_features(adata, "cluster", VT_QUA)


@patch('matplotlib.pyplot.show')
def test_quaternary_single_wo_velo(mock_show):
    adata_test = adata.copy()
    plot_quaternary(adata_test, "cluster", VT_QUA, GENE_QUA, title="title")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == 'title'


@patch('matplotlib.pyplot.show')
def test_quaternary_single_w_velo(mock_show):
    adata_test = adata.copy()
    plot_quaternary(adata_test, "cluster", VT_QUA, GENE_QUA, velo_graph='velo',
                    method="cos")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == ''


@patch('matplotlib.pyplot.show')
def test_quaternary_split_wo_velo(mock_show):
    adata_test = adata.copy()
    plot_quaternary(adata_test, "cluster", VT_QUA, GENE_QUA,
                    split_cluster=True, method='p')
    fig = plt.gcf()
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1


@patch('matplotlib.pyplot.show')
def test_quaternary_split_w_velo(mock_show):
    adata_test = adata.copy()
    plot_quaternary(adata_test, "cluster", VT_QUA, GENE_QUA, velo_graph='velo',
                    split_cluster=True, method='sp')
    fig = plt.gcf()
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1
