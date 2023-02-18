# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

import sphinx
import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('../qnm_filter'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QNM Filter'
copyright = '2023, Sizheng Ma'
author = 'Sizheng Ma'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "recommonmark",
]

autosummary_generate = True

autodoc_docstring_signature = True

# --------------------------------------------------------------------

templates_path = ['_templates']
exclude_patterns = []

if sphinx.version_info < (1, 8):
    source_parsers = {
        ".md": CommonMarkParser,
    }
    source_suffix = [".rst", ".md"]
else:
    source_suffix = {
        ".rst": "restructuredtext",
        ".txt": "markdown",
        ".md": "markdown",
    }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
