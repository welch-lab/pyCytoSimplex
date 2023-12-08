=======
Example
=======

This package use simplex barycentric coordinate approach to assist exploration in the similarity between single cells
between selected cell clusters. We denote a number (2-4) of selected clusters, or groups of clusters, as vertices.
We calculate the similarity between each single cell and the average point of each vertex. By normalizing the similarity
between each single cell and all specified vertices to a unit sum, we can derive the barycentric coordinates for each single cell.
Visualization method for binary (ended line), ternary (equilateral triangle) and quaternary (tetrahedron) simplex are developed.
The main plotting functions are :func:`~CytoSimplex.plot_binary`, :func:`~CytoSimplex.plot_ternary` and :func:`~CytoSimplex.plot_quaternary`, respectively.

Test Dataset
^^^^^^^^^^^^

The dataset used in all example can be found on `figshare <https://figshare.com/ndownloader/files/41034857>`_.
This is a small subset of publicly available single cell RNA-seq data from the human bone marrow mononuclear cells (BMMC)
generated in our previous study [1]_. The data is stored in a ``.h5ad`` file format, which can be read by the
`AnnData <https://anndata.readthedocs.io/en/stable/index.html>`_ package. Users can load the dataset following the code below.

.. code-block:: python

    import CytoSimplex as csx
    import anndata
    adata = anndata.read(filename='test.h5ad',
                         backup_url="https://figshare.com/ndownloader/files/41034857")

The contents included in the ``adata`` object include:

- ``adata.X``: A CSR sparse matrix of 250 rows (cells) by 20,243 columns (genes) containing the raw gene count matrix.
- ``adata.obs["cluster"]``: A categorical annotation of cell clusters. There are totally 12 clusters which can be viewed via ``adata.obs["cluster"].cat.categories``.
  Namingly, 'Chondrocyte_1', 'Chondrocyte_2', 'Chondrocyte_3', 'OCT_Stem', 'ORT_1', 'ORT_2', 'ORT_3', 'Osteoblast_1', 'Osteoblast_2', 'Osteoblast_3', 'Reticular_1' and 'Reticular_2'.
- ``adata.uns["velo"]``: A 250 by 250 CSR sparse matrix of velocity neighbor graph. This is derived with `veloVAE <https://github.com/welch-lab/VeloVAE>`_ [2]_.
  We will have more introduction to this in detailed examples.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   select_top_features
   ternary
   binary
   quaternary

.. [1] Matsushita, Y., Liu, J., Chu, A.K.Y. et al. Bone marrow endosteal stem cells dictate active osteogenesis and aggressive tumorigenesis. Nat Commun 14, 2383 (2023). https://doi.org/10.1038/s41467-023-38034-2
.. [2] Gu, Y., Blaauw, D., Welch, J. D.. Bayesian Inference of RNA Velocity from Multi-Lineage Single-Cell Data. bioRxiv (2022). https://doi.org/10.1101/2022.07.08.499381
