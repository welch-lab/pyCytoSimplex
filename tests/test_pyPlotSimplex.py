import scanpy as sc
import numpy as np
import pandas as pd
import pytest
from src.pyPlotSimplex import row_normalize, select_top_features, \
    plot_ternary, plot_quaternary
import matplotlib.pyplot as plt
from unittest.mock import patch

adata = sc.read_h5ad("tests/bmmc_rna_small.h5ad")


def test_normalize():
    adata_test = adata.copy()
    x1 = row_normalize(adata_test.X)
    row_normalize(adata_test)
    assert np.allclose(x1.sum(1), 1)


def test_feature_selection():
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': 'Chondrocyte_1'}
    gene = select_top_features(adata_test, "cluster", vt)
    assert len(gene) == 90

    gene2 = select_top_features(adata_test.X.toarray(), adata_test.obs.cluster,
                                vt, feature_names=adata_test.var_names)
    assert gene == gene2

    gene3 = select_top_features(adata_test.X.toarray(), adata_test.obs.cluster,
                                vt)
    assert (gene == adata_test.var_names[gene3]).all()

    stats = select_top_features(adata_test, "cluster", vt, return_stats=True)
    assert type(stats) == pd.DataFrame


@patch('matplotlib.pyplot.show')
def test_ternary_single_wo_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_ternary(adata_test, "cluster", vt, gene)
    fig = plt.gcf()
    assert fig.axes[0].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_single_w_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_ternary(adata_test, "cluster", vt, gene, velo_graph='velo',
                 method="cos")
    fig = plt.gcf()
    assert fig.axes[0].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_split_wo_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_ternary(adata_test, "cluster", vt, gene, split_cluster=True,
                 method='p')
    fig = plt.gcf()
    assert fig.axes[2].get_llabel() == 'OS'


@patch('matplotlib.pyplot.show')
def test_ternary_split_w_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_ternary(adata_test, "cluster", vt, gene, velo_graph='velo',
                 split_cluster=True, method='sp')
    fig = plt.gcf()
    assert fig.axes[3].get_llabel() == 'OS'
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1


@patch('matplotlib.pyplot.show')
def test_quaternary_single_wo_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
          'ORT': ['ORT_1', 'ORT_2']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_quaternary(adata_test, "cluster", vt, gene, title="title")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == 'title'


@patch('matplotlib.pyplot.show')
def test_quaternary_single_w_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
          'ORT': ['ORT_1', 'ORT_2']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_quaternary(adata_test, "cluster", vt, gene, velo_graph='velo',
                    method="cos")
    fig = plt.gcf()
    assert fig.axes[0].get_title() == ''


@patch('matplotlib.pyplot.show')
def test_quaternary_split_wo_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
          'ORT': ['ORT_1', 'ORT_2']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_quaternary(adata_test, "cluster", vt, gene, split_cluster=True,
                    method='p')
    fig = plt.gcf()
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1


@patch('matplotlib.pyplot.show')
def test_quaternary_split_w_velo(mock_show):
    adata_test = adata.copy()
    vt = {'OS': ["Osteoblast_1", "Osteoblast_2", "Osteoblast_3"],
          'RE': ['Reticular_1', 'Reticular_2'],
          'CH': ['Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3'],
          'ORT': ['ORT_1', 'ORT_2']}
    gene = select_top_features(adata_test, "cluster", vt)
    plot_quaternary(adata_test, "cluster", vt, gene, velo_graph='velo',
                    split_cluster=True, method='sp')
    fig = plt.gcf()
    assert len(fig.axes) == len(adata_test.obs.cluster.cat.categories) + 1
