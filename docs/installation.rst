======================
Installation
======================


Prerequisites
=============

To use this package, users will need to have Python pre-installed on their machine. This package is developed and tested with Python 3.10 environment. `Python can be installed following the instructions here <https://www.python.org/downloads/>`_.

(Optional) Conda environment / virtual environment
--------------------------------------------------

It has been a best practice for research involving Python-facilitated analyses to have environments created to keep track of module versions and hence keep the analyses reproducible. Although it is not mandatory for the this module in order to make it work, users can opt to create an environment with the following steps:

Conda environment
^^^^^^^^^^^^^^^^^

To create a conda environment:

.. code-block:: bash

      conda create --name cytosimplex

If users decide to use a conda environment, it must be activated with:

.. code-block:: bash

      conda activate cytosimplex

Python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

If users choose not to work with conda, a Python virtual environment, alternatively, can be created with:

.. code-block:: bash

      # If you have never tried a virtual environment before, you'll need to install the dependency first
      python -m venv cytosimplex
      # This line creates the virtual environment
      python -m venv cytosimplex

Here, a folder called ``"cytosimplex"`` is created as to store the necessary libraries for the environment. To activate it, different platforms has different approaches.

- On Unix or MacOS, using the bash shell: ``source cytosimplex/bin/activate``
- On Unix or MacOS, using the csh shell: ``source cytosimplex/bin/activate.csh``
- On Unix or MacOS, using the fish shell: ``source cytosimplex/bin/activate.fish``
- On Windows using the Command Prompt: ``cytosimplex\Scripts\activate.bat``
- On Windows using PowerShell: ``cytosimplex\Scripts\Activate.ps1``

Dependencies
------------

CytoSimplex depends on the following packages:
`scikit-learn <https://scikit-learn.org/stable/>`_,
`numpy <https://numpy.org/>`_,
`anndata <https://anndata.readthedocs.io/en/latest/>`_,
`scipy <https://scipy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`mpltern <https://mpltern.readthedocs.io/en/latest/>`_,
`matplotlib <https://matplotlib.org/>`_,
`scanpy <https://scanpy.readthedocs.io/en/stable/>`_.
All of these can be installed using ``pip``, and should automatically be installed when you install CytoSimplex with ``pip``. So ideally, users do not need to install these packages manually.

Installing CytoSimplex
======================

This Python module can be locally installed with the following command in shell/Command line tool:

.. code-block:: bash

      pip install git+https://github.com/welch-lab/pyCytoSimplex.git

This command automatically fetches the latest version in this GitHub repository and installs it in the current active environment.

This module will be submitted to PyPI for easier installation in the future.

If you have any problems, please `open an issue on GitHub <https://github.com/mvfki/pyCytoSimplex/issues/new>`_.
