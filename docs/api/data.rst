:mod:`infl_ens.data`
====================

Trait-space construction, sentence-embedding wrappers, and benchmark
dataset loaders.

Top-level re-exports
--------------------

The following symbols are available directly off ``infl_ens.data``
(see ``src/infl_ens/data/__init__.py``). Fully qualified names are used
in the autosummary block because Sphinx's bare-name lookup does not
respect ``currentmodule`` in autosummary contexts; see
``docs/conf.py`` for the broader explanation.

.. autosummary::
   :nosignatures:

   infl_ens.data.TraitSpace
   infl_ens.data.build_trait_space
   infl_ens.data.position_from_corpus
   infl_ens.data.SentenceTransformerEncoder
   infl_ens.data.HuggingFaceEncoder

Submodules
----------

Recursive autosummary picks up every public submodule automatically and
generates one stub page per module under ``_autosummary/``. Adding a new
file to ``src/infl_ens/data/`` is enough — no doc edits required.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.data.trait_space
   infl_ens.data.encoders
   infl_ens.data.benchmarks
