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
install a full ML stack just to render docstrings. ``autodoc_mock_imports``
only mocks *foreign* packages, however — ``infl_ens`` itself must be
importable. The :func:`_locate_package_root` helper probes several
candidate locations to handle both the canonical ``src/infl_ens/``
layout and a flat ``infl_ens/`` layout.

:see: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# ``conf.py`` lives in ``<repo>/docs/`` next to ``<repo>/src/infl_ens/``.
# Sphinx imports this file before autodoc runs, so any ``sys.path`` edits
# made here are visible to ``autosummary :recursive:``.
HERE = os.path.dirname(os.path.abspath(__file__))


def _locate_package_root(pkg_name: str = "infl_ens") -> Optional[str]:
    """Return the directory that should be prepended to ``sys.path``.

    Tries several common layouts (in order) and returns the first one
    that actually contains ``<root>/<pkg_name>``:

    1. ``<docs>/../src``      (canonical ``src/infl_ens/`` layout)
    2. ``<docs>/..``          (flat ``infl_ens/`` at the repo root)
    3. ``<cwd>/src``          (CI runners that ``cd`` into the repo root)
    4. ``<cwd>``              (same, flat layout)

    Each candidate is logged so the GitHub-Actions build log shows
    exactly what was tried — useful when the build fails because the
    layout on disk differs from what ``structure.md`` describes.

    :param pkg_name: Package directory name to look for.
    :type pkg_name: str
    :returns: Absolute path to prepend to ``sys.path``, or ``None`` if
              no candidate worked.
    :rtype: str | None
    """
    candidates: List[str] = [
        os.path.abspath(os.path.join(HERE, "..", "src")),
        os.path.abspath(os.path.join(HERE, "..")),
        os.path.abspath(os.path.join(os.getcwd(), "src")),
        os.path.abspath(os.getcwd()),
    ]
    print(f"[conf.py] looking for package {pkg_name!r}")
    seen: set[str] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        has_pkg = os.path.isdir(os.path.join(root, pkg_name))
        print(f"[conf.py]   tried {root!r} (has {pkg_name}/: {has_pkg})")
        if has_pkg:
            return root
    return None


_PKG_ROOT = _locate_package_root()
if _PKG_ROOT is not None:
    sys.path.insert(0, _PKG_ROOT)
    print(f"[conf.py] using {_PKG_ROOT!r} for autodoc imports")
else:
    # Don't raise: let Sphinx fail with its own (clearer) ImportError
    # message inside autosummary so the offending module is visible.
    print(
        "[conf.py] WARNING: could not locate 'infl_ens' on any candidate "
        "path; autodoc will fail. Check that the repo really contains "
        "'src/infl_ens/' or 'infl_ens/' with an __init__.py.",
        file=sys.stderr,
    )

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

# Foreign modules the package imports but that the docs runner should
# not have to install. Autodoc replaces each entry with a stub object
# so the rest of the module still parses.
#
# IMPORTANT: do not add ``infl_ens`` or its subpackages here — mocking
# them would defeat the entire docs build. ``infl_ens`` must be
# importable via the ``sys.path`` insertion performed above.
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
