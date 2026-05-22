"""Sphinx configuration for the ``infl_ens`` documentation build.

This file is consumed by Sphinx to produce the HTML site that is
published to GitHub Pages by ``.github/workflows/docs.yml``. The build
runs on a slim Linux runner that does **not** have the heavy ML stack
(``torch``, ``transformers``, ``datasets``, ...) installed. Three
mechanisms cooperate to make autosummary work in that environment:

1. **Pre-emptive ``sys.modules`` mocks** (this file, ``_install_mocks``).
   Installing :class:`unittest.mock.MagicMock` shims before Sphinx
   loads any extension guarantees that subsequent ``import torch`` (or
   any other listed name) inside ``infl_ens`` succeeds. This sidesteps
   a known autosummary/autodoc ordering issue where
   ``autodoc_mock_imports`` is configured *after* ``autosummary`` has
   already begun importing the documented modules.
2. **``autodoc_mock_imports``** (set below). Still configured so that
   later autodoc passes also see the same mock list; redundant in
   well-behaved cases but harmless.
3. **Explicit preflight import** (``_preflight_imports``). Imports
   every subpackage at conf-load time and prints the *real* traceback
   to stderr if anything fails. This is critical because Sphinx's
   ``import_by_name`` wraps the underlying error and re-raises with a
   synthetic ``"no module named X"`` message that hides the cause.

:see: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Step 1 — install MagicMock shims BEFORE anything else.
#
# This must run before any ``import`` that might transitively pull in
# one of the heavy deps. Doing it at the very top of ``conf.py`` is
# safe because Sphinx imports this file before initialising any
# extension.
# ---------------------------------------------------------------------------
_MOCK_MODULES: List[str] = [
    # ML stack
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "torch.optim",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers",
    "huggingface_hub",
    # Numerical / tabular
    "sklearn",
    "scipy",
    "scipy.stats",
    "scipy.linalg",
    "matplotlib",
    "matplotlib.pyplot",
    "pandas",
    # Config / orchestration
    "yaml",
    "hydra",
    "omegaconf",
    "tqdm",
]


def _install_mocks(names: List[str]) -> None:
    """Insert :class:`unittest.mock.MagicMock` shims into ``sys.modules``.

    Once present in ``sys.modules``, subsequent ``import name`` calls
    return the mock without ever touching the filesystem, so missing
    third-party packages no longer break the doc build.

    :param names: Module names to mock.
    :type names: list[str]
    """
    from unittest.mock import MagicMock

    class _Mock(MagicMock):
        """Mock that returns more mocks for *any* attribute access.

        Standard :class:`MagicMock` already does this, but subclassing
        gives a clearer ``repr`` in tracebacks and lets us extend later
        (e.g. to support ``class Foo(mock.SomeBase)`` inheritance via
        ``__mro_entries__`` if the need arises).
        """

        @classmethod
        def __getattr__(cls, attr: str) -> "MagicMock":
            return MagicMock()

    for name in names:
        if name not in sys.modules:
            sys.modules[name] = _Mock()


_install_mocks(_MOCK_MODULES)


# ---------------------------------------------------------------------------
# Step 2 — locate the ``infl_ens`` package on the filesystem.
# ---------------------------------------------------------------------------
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
    exactly what was tried.

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
    print(
        "[conf.py] WARNING: could not locate 'infl_ens' on any candidate "
        "path; autodoc will fail. Check that the repo really contains "
        "'src/infl_ens/' or 'infl_ens/' with an __init__.py.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Step 3 — preflight import every documented subpackage.
#
# Sphinx's :func:`sphinx.ext.autosummary.import_by_name` wraps ImportError
# and re-raises it as a synthetic ``"no module named X"`` message that
# discards the original traceback. By performing the imports here
# explicitly and printing the real traceback, we make the actual root
# cause visible in the GitHub Actions build log instead of the misleading
# autosummary wrapper.
# ---------------------------------------------------------------------------
_SUBPACKAGES_TO_PREFLIGHT: List[str] = [
    "infl_ens",
    "infl_ens.data",
    "infl_ens.data.benchmarks",
    "infl_ens.inflgame",
    "infl_ens.inflgame.router",
    "infl_ens.training",
    "infl_ens.utils",
]


def _preflight_imports(names: List[str]) -> None:
    """Import every name; print the full traceback on the first failure.

    Failures are reported to stderr (so they're prefixed with ``error::``
    in GitHub Actions log groups) but **not** raised — letting Sphinx
    proceed means the build still produces a diagnostic page, and the
    user sees both the preflight traceback and the autosummary error
    side by side.

    :param names: Fully-qualified module names to import in order.
    :type names: list[str]
    """
    print()
    print("[conf.py] preflight: importing every infl_ens subpackage")
    for name in names:
        try:
            mod = __import__(name, fromlist=["_"])
        except Exception:  # noqa: BLE001 — we want any failure surfaced
            print(
                f"[conf.py]   {name:32s} FAIL — full traceback follows",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            print(
                f"[conf.py] preflight aborted at {name!r}. Fix this import "
                f"before the doc build can succeed.",
                file=sys.stderr,
            )
            return
        # ``__file__`` is ``None`` for PEP 420 namespace packages. Every
        # subpackage in this codebase is documented as a real package
        # (with an ``__init__.py``) in ``structure.md``, so a namespace
        # package here almost certainly means an ``__init__.py`` is
        # missing on disk. Flag it loudly.
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            print(
                f"[conf.py]   {name:32s} NAMESPACE PACKAGE — "
                f"__init__.py is probably missing (autosummary will fail)",
                file=sys.stderr,
            )
        else:
            print(f"[conf.py]   {name:32s} OK   ({mod_file})")
    print("[conf.py] preflight: all subpackages imported cleanly")
    print()


_preflight_imports(_SUBPACKAGES_TO_PREFLIGHT)


# ---------------------------------------------------------------------------
# Step 4 — attribute-surface check.
#
# Importing a module is necessary but not sufficient: autosummary also
# needs the documented re-exports to be present as *attributes* on each
# subpackage. The structure.md document spells out which names each
# ``__init__.py`` should re-export; this check verifies the live module
# matches. Missing re-exports are reported as warnings (not failures)
# so the docs still build with cross-references to canonical locations,
# but the build log makes the gap obvious.
# ---------------------------------------------------------------------------
_EXPECTED_REEXPORTS: dict[str, tuple[str, ...]] = {
    "infl_ens.data": (
        "TraitSpace",
        "build_trait_space",
        "position_from_corpus",
        "SentenceTransformerEncoder",
        "HuggingFaceEncoder",
    ),
    "infl_ens.data.benchmarks": (
        "BenchmarkSplit",
        "load_beavertails",
        "load_halueval",
        "build_safety_trait_space",
        "LearnedAxis",
    ),
    "infl_ens.inflgame.router": (
        "RouterAgent",
        "InfluencerRouter",
        "allocation_weights",
        "expected_utilities",
        "empirical_utility",
        "strategic_routing_weights",
        "utility_gradient",
    ),
    "infl_ens.training": (
        "RouterTrainingConfig",
        "train_router_positions",
        # SFTTrainingConfig / sft_train_agent are lazy via __getattr__
        # and intentionally not listed here.
    ),
    "infl_ens.utils": (
        "weighted_mean",
        "weighted_covariance",
        "gaussian_stability_threshold",
    ),
}


def _check_attribute_surface(expected: dict) -> None:
    """Warn for each missing re-export named in ``expected``.

    For each ``(module_name, attrs)`` pair, import the module and
    verify that every attribute in ``attrs`` is accessible via
    :func:`getattr`. Missing attributes are reported individually so
    the build log shows exactly which ``__init__.py`` needs which
    re-export added.

    :param expected: Mapping of fully-qualified module name to a tuple
                     of attribute names that should be re-exported.
    :type expected: dict[str, tuple[str, ...]]
    """
    print()
    print("[conf.py] attribute-surface check (compared against structure.md)")
    total_missing = 0
    for mod_name, attrs in expected.items():
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except Exception:  # noqa: BLE001
            # Already reported by the import preflight; skip silently.
            continue
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            total_missing += len(missing)
            print(
                f"[conf.py]   {mod_name}: missing re-exports "
                f"{missing} — autosummary tables that depend on these "
                f"will fall back to canonical-path cross-references.",
                file=sys.stderr,
            )
        else:
            print(f"[conf.py]   {mod_name:32s} all {len(attrs)} re-exports present")
    if total_missing:
        print(
            f"[conf.py] {total_missing} missing re-exports total. "
            f"The docs still build (the API pages use canonical-path "
            f"xrefs), but adding the listed names to each __init__.py "
            f"will silence the warning.",
            file=sys.stderr,
        )
    else:
        print("[conf.py] all expected re-exports present on every subpackage")
    print()


_check_attribute_surface(_EXPECTED_REEXPORTS)


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

autosummary_generate = True
autosummary_imported_members = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}

# Kept in sync with ``_MOCK_MODULES`` above. ``autodoc_mock_imports``
# alone is not enough — see the docstring at the top of this file — but
# we set it so that any later autodoc machinery sees the same list.
autodoc_mock_imports = list({m.split(".")[0] for m in _MOCK_MODULES})

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Warning categories suppressed below the ``-W`` threshold. We keep
# ``-W --keep-going`` in the build command so a *genuine* import error
# still fails the workflow, but the following warning classes are
# known-cosmetic byproducts of design choices documented elsewhere:
#
# - ``ref.python``  — autosummary's recursive mode happily documents
#   re-exported symbols at both their canonical location and at the
#   re-export path (e.g. ``BenchmarkSplit`` appears under both
#   ``infl_ens.data.benchmarks`` and
#   ``infl_ens.data.benchmarks.base``). That ambiguity makes Sphinx
#   emit "more than one target found for cross-reference" on every
#   docstring xref. The duplication is intentional — re-exports are
#   the package's public surface (AGENTS.md §4) — so we silence the
#   warning rather than break the re-export pattern.
#
# - ``myst.xref_missing`` — ``README.md`` is included in the docs via
#   ``.. include:: ../README.md``, and any plain Markdown links
#   (``[text](path.html)``) that don't resolve to a Sphinx doc target
#   raise this warning. Suppressed so a README aimed at GitHub readers
#   doesn't have to be rewritten for Sphinx.
suppress_warnings = [
    "ref.python",
    "myst.xref_missing",
]


# ---------------------------------------------------------------------------
# MyST (Markdown) configuration
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
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
    "source_repository": "",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

todo_include_todos = False
