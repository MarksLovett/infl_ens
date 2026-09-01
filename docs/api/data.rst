:mod:`infl_ens.data`
====================

Trait-space construction, the Hugging Face encoder, benchmark loaders
and downloaders, and the persisted train/val/test splits.

Top-level re-exports
--------------------

The following symbols are available directly off ``infl_ens.data`` (see
``src/infl_ens/data/__init__.py``). Each link jumps to the **canonical**
definition:

- :class:`~infl_ens.data.trait_space.TraitSpace`
- :func:`~infl_ens.data.trait_space.build_trait_space`
- :func:`~infl_ens.data.trait_space.position_from_corpus`
- :class:`~infl_ens.data.encoders.HuggingFaceEncoder`
- :class:`~infl_ens.data.trait_normalize.QuantileNormalizer`

The encoder is selected by config, not by a library default:
:func:`~infl_ens.data.encoders.make_encoder` reads ``trait_space.encoder``
(the model id, part of the cache fingerprint) and the top-level
``encoder`` block (constructor keyword arguments) — see
``configs/encoders/``.

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.data.encoders
   infl_ens.data.trait_space
   infl_ens.data.trait_space_cache
   infl_ens.data.trait_normalize
   infl_ens.data.position_blend
   infl_ens.data.splits
   infl_ens.data.download
   infl_ens.data.benchmarks
