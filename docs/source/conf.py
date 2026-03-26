import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # apackage root

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nsEVDx"
copyright = "2025, nkafle, cimeier"
author = "nkafle, cimeier"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Extensions
extensions = [
    "sphinx.ext.autodoc",  # generates API docs from docstrings
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinx_autodoc_typehints",  # show type hints
    "sphinx_copybutton",  # copy code button
    "sphinx.ext.viewcode",  # links to source code
    "sphinx_design",  # for better layout and design of docs
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Optional: control docstring format
autodoc_typehints = "description"
