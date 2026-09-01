infl_ens
========

``infl_ens`` is a research package implementing the **influencer's game**
(Lovett & Fu, 2024) and extending it to align small language models as
the learning agents. The router routes queries to candidate models via
the proportional-allocation rule

.. math::

   G_i(\mathbf{x}, b) = \frac{f_i(x_i, b)}{\sum_{j=1}^{N} f_j(x_j, b)}

with a multivariate-Gaussian influence kernel on a trait space learned
from labelled safety benchmarks.

This site is built from the package source on every push to ``main`` by
``.github/workflows/docs.yml`` and published to GitHub Pages.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   pipeline
   configs
   structure

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/config
   api/data
   api/inflgame
   api/training
   api/evaluation
   api/figures
   api/pipeline
   api/utils

Where to start
--------------

- New to the package? Read :doc:`getting_started` — installation, the
  one-command pipeline, and the design rules (positions reflect
  *observed* capability, fixed :math:`\sigma`, picking :math:`\sigma`
  relative to :math:`\sigma_0^*`).
- Want to change an experiment? :doc:`configs` explains the layered YAML
  fragments (encoder, data, model, arms, experiment) and the trait-space
  cache contract.
- Running on the GPU host? :doc:`pipeline` documents every stage and the
  ``scripts/run_on_doob.sh`` wrapper.
- Looking for a specific symbol? Jump to the API reference and use the
  search box (top-left).
- Want a file-by-file map of the repository? See :doc:`structure`.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
