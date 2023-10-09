from sphinx.application import Sphinx


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CytoSimplex'
copyright = '2023, Yichen Wang, Jialin Liu, Joshua D. Welch'
author = 'Yichen Wang, Jialin Liu, Joshua D. Welch'
release = '0.1.0'
repository_url = "https://github.com/mvfki/pyCytoSimplex"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
nitpick_ignore = [('py:class', 'type')]
needs_sphinx = '4.0'  # Nicer param docs


def setup(app: Sphinx):
    """App setup hook."""
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        True,
    )


extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    # 'git_ref',  # needs to be before scanpydoc.rtd_github_links
    'sphinx_design',
    'sphinxext.opengraph',
    'sphinx.ext.viewcode',
    'nbsphinx'
]
napoleon_attr_annotations = True
# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False
# api_dir = HERE / 'api'  # function_images
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
}
# html_css_files = ["css/override.css"]
html_show_sphinx = False
# html_logo = 'images/logo_bright.svg'
html_theme_options = {
   "logo": {
      "image_light": "images/logo_bright.png",
      "image_dark": "images/logo_dark.png",
   }
}
html_title = "CytoSimplex"
html_static_path = ['_static']
html_css_files = ["css/override.css"]
master_doc = 'source/index'

intersphinx_mapping = dict(
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    matplotlib=('https://matplotlib.org/stable/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    pytest=('https://docs.pytest.org/en/latest/', None),
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
)
