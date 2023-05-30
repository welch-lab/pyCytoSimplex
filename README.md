<img src="https://github.com/mvfki/scPlotSimplex/raw/main/man/figures/logo.png" width="120">

# pyPlotSimplex

"pyPlotSimplex" is a Python module that creates simplex plot showing similarity between single-cells and terminals represented by clusters of cells.
RNA velocity can be added as another layer of information.

For R users, we have an R package [scPlotSimplex](https://github.com/mvfki/scPlotSimplex) that provides the same functionalities.

## Installation

For latest developmental version, please make sure that you have a Python (>=3.7) installation in your current environment first. And then run the following command in a shell:

1. Download this package

```shell
git clone xxx
cd pyPlotSimplex
```

2. Make sure that all required dependencies are available and install as needed

```shell
pip install -r requirments.txt
```

3. Install this package

```shell
python setup.py install
```

## Quick Start

Assume that users have a well annotated dataset, contained in an `AnnData` object that is stored as an `.h5ad` file locally. Then users can load the data into Python environment using `scanpy` package:

```python
import pyPlotSimplex as ps
import scanpy as sc

adata = sc.read_h5ad("path/to/your/AnnData.h5ad")
```

Users will need to determine the terminals basing on their prior knowledge. For example, if users are interested in brain development, they may want to use the terminally differentiated cell types as terminals, such as oligodendrocytes, fibroblasts, etc. A cell type assignment has to be available in `adata.obs` or provided externally by users. Users can create a `list` for terminal specification. Alternatively, a `dict` with altered terminal names can be used. For example, for creating ternary plot (3 terminals forming a triangle):

```python
# Create a list of 3 terminal cell types
terminals = ["Oligodendrocytes", "Fibroblasts", "Neurons"]
# Or create a dict of 3 terminal cell types
terminals = {"OL": "Oligodendrocytes", "FI": "Fibroblasts", "NE": "Neurons"}
```

With a `dict` terminal specification, users can also group multiple cell types as one terminal:

```python
terminals = {"OL": ["Oligodendrocytes_1", "Oligodendrocytes_2"],
             "FI": ["Fibroblasts_1", "Fibroblasts_2"],
             "NE": ["Neurons_1", "Neurons_2", "Neurons_3"]}
```

For high dimensional single-cell data (e.g. scRNAseq), reducing the dimensionality by selecting top differentially expressed genes for each terminal cluster is recommended. By default, we recommend selecting 30 top differentially expressed genes for each terminal:

```python
genes = ps.select_top_features(adata, 'cell_type', terminals, n_top=30)
print(len(genes))
```

```
90
```

Then users can create a simplex plot using the following command:

```python
ps.plot_ternary(adata, 'cell_type', terminals, genes)
```
