"""Sphinx configuration for the ``infl_ens`` documentation build.

This file is consumed by Sphinx to produce the HTML site that is published
to GitHub Pages by ``.github/workflows/docs.yml``. It enables

- :mod:`sphinx.ext.autodoc` and :mod:`sphinx.ext.autosummary` (with
  ``:recursive:``) so the API reference stays in sync with the package
  source without per-module RST stubs needing to be hand-maintained.
- :mod:`myst_parser` so ``README.md`` and ``structure.md`` can be pulled
  into the docs verbatim.
- :mod:`sphinx.ext.mathjax` to render the ``:math:`...``` directives that
  pervade the package docstrings (per AGENTS.md §2).

Heavy optional dependencies (``torch``, ``transformers``, ``datasets``,
``peft``, ``trl``, ``sentence_transformers``, ...) are listed in
:data:`autodoc_mock_imports` so the GitHub-Pages runner does not have to
install a full ML stack just to render docstrings.

:see: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# ``docs/`` sits at the repo root next to ``src/``; expose the package so
# autodoc can import ``infl_ens`` without an editable install.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "infl_ens"
author = "infl_ens contributors"
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001 (Sphinx convention)
release = "0.1.0"
version = release

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "myst_parser",
]

# Generate ``_autosummary/*.rst`` stub pages on every build so newly-added
# modules appear in the sidebar without manual RST edits.
autosummary_generate = True
autosummary_imported_members = False

# Match the in-file ordering of public symbols used by ``structure.md``.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}

# Modules that the package imports but that the docs runner should not
# have to install. Autodoc replaces each entry with a stub object so the
# rest of the module still parses.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers",
    "huggingface_hub",
    "sklearn",
    "scipy",
    "matplotlib",
    "pandas",
    "yaml",
    "hydra",
    "omegaconf",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# MyST (Markdown) configuration
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "dollarmath",   # ``$...$`` and ``$$...$$`` math, used in README.md
    "amsmath",      # ``\begin{align}`` blocks
    "deflist",
    "colon_fence",
    "linkify",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "infl_ens"
html_short_title = "infl_ens"
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    "source_repository": "",  # populated by GitHub Actions if desired
    "source_branch": "main",
    "source_directory": "docs/",
}

# Sphinx-copybutton: strip prompts when users copy code blocks.
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

# Render TODO blocks (off in production builds; flip for review builds).
todo_include_todos = False
