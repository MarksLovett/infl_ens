"""Registry and configuration dispatch for behavioral benchmark loaders."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from infl_ens.data.behavioral.base import BehavioralSuite
from infl_ens.data.behavioral.bipia import load_bipia
from infl_ens.data.behavioral.confaide import load_confaide
from infl_ens.data.behavioral.harmbench import load_harmbench
from infl_ens.data.behavioral.ifeval import load_ifeval
from infl_ens.data.behavioral.strongreject import load_strongreject
from infl_ens.data.behavioral.truthfulqa import load_truthfulqa
from infl_ens.data.behavioral.xstest import load_xstest

BehavioralLoader = Callable[[Mapping[str, Any]], BehavioralSuite]

BEHAVIORAL_LOADERS: dict[str, BehavioralLoader] = {
    "harmbench": load_harmbench,
    "strongreject": load_strongreject,
    "xstest": load_xstest,
    "truthfulqa": load_truthfulqa,
    "confaide": load_confaide,
    "bipia": load_bipia,
    "ifeval": load_ifeval,
}


def load_behavioral_suite(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load one behavioral suite from an experiment mapping.

    :param entry: Mapping containing at least ``kind`` and ``path``.
    :type entry: Mapping[str, Any]
    :returns: Loaded suite.
    :rtype: BehavioralSuite
    :raises ValueError: If ``kind`` is unknown.
    """
    kind = str(entry.get("kind", ""))
    if kind not in BEHAVIORAL_LOADERS:
        raise ValueError(
            f"unknown behavioral suite kind {kind!r}; known: {sorted(BEHAVIORAL_LOADERS)}",
        )
    return BEHAVIORAL_LOADERS[kind](entry)


def load_behavioral_suites(
    entries: Sequence[Mapping[str, Any]],
) -> list[BehavioralSuite]:
    """Load behavioral suites in declared experiment order.

    :param entries: Suite config mappings.
    :type entries: Sequence[Mapping[str, Any]]
    :returns: Loaded suites.
    :rtype: list[BehavioralSuite]
    """
    suites = [load_behavioral_suite(entry) for entry in entries]
    names = [suite.name for suite in suites]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate behavioral suite names: {names}")
    return suites


__all__ = ["BEHAVIORAL_LOADERS", "load_behavioral_suite", "load_behavioral_suites"]
