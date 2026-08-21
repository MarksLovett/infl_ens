#!/usr/bin/env python3
"""Print within_merge L2 at final round for a closed-loop run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.training.pool_dynamics import agent_pairwise_geometry  # noqa: E402

GROUPS = [
    ("merge-harm", ["clone-0", "clone-1"]),
    ("merge-hallucination", ["clone-2", "clone-3"]),
    ("merge-privacy", ["clone-4", "clone-5"]),
    ("merge-overrefusal", ["clone-6", "clone-7"]),
    ("merge-policy", ["clone-8", "clone-9"]),
]
NAMES = [f"clone-{i}" for i in range(10)]

hist = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pos = np.stack([np.asarray(hist[-1]["positions"][n]) for n in NAMES])
g = agent_pairwise_geometry(pos, NAMES, merge_groups=GROUPS)
print("round", hist[-1]["round"])
print("within_merge_l2", g["within_merge_l2"])
print("mean_pairwise_l2", g["mean_pairwise_l2"])
