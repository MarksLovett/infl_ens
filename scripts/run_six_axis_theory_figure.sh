#!/usr/bin/env bash
# Build split manifest, paired theory (n12), and 70/10/20 slice figure only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"

echo "--- rebuild 70/10/20 split manifest ---"
"${PY}" scripts/build_six_axis_split.py \
  --config configs/benchmark/router/six_axis_pair_merge_split_r24.yaml \
  --output data/splits/six_axis_seed0.json

echo "--- paired theory Nash (n12, lr=0.05) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/six_axis_theory_n12.yaml

echo "--- fixed_positions ---"
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
from infl_ens.utils.agent_init import co_locate_theory_pairs

out_dir = Path("results/six_axis_theory_n12")
payload = json.loads((out_dir / "positions.json").read_text(encoding="utf-8"))
names = list(payload["positions"].keys())
pos = np.asarray([payload["positions"][n] for n in names], dtype=float)
paired = co_locate_theory_pairs(pos, names)
fixed = {"positions": {names[i]: paired[i].tolist() for i in range(len(names))}}
(out_dir / "fixed_positions.json").write_text(json.dumps(fixed, indent=2), encoding="utf-8")
print(f"wrote {out_dir / 'fixed_positions.json'}")
PY

echo "--- theory layout ---"
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
from infl_ens.training.pool_dynamics import classify_layout, pairwise_spread

payload = json.loads(Path("results/six_axis_theory_n12/positions.json").read_text())
pos = np.asarray(list(payload["positions"].values()), dtype=float)
print(f"layout={classify_layout(pos)} spread={pairwise_spread(pos):.4f}")
for name, vec in payload["positions"].items():
    print(f"  {name}: argmax axis {int(np.argmax(vec))} = {vec[np.argmax(vec)]:.3f}")
PY

echo "--- harm-axis decorrelation check ---"
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
from infl_ens.training.__main__ import _load_splits, _load_yaml
from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space

cfg = _load_yaml(Path("configs/benchmark/router/six_axis_safety.yaml"))
splits = _load_splits(cfg)
space = build_or_load_safety_trait_space(cfg, splits)
prompts = [p for s in splits for p in s.prompts]
coords = space.project(prompts)
corr = np.corrcoef(coords.T)
off = corr[np.triu_indices(corr.shape[0], k=1)]
harm_off = [corr[0, j] for j in range(1, corr.shape[0])]
policy_idx = list(space.axis_labels).index("policy_violation")
policy_off = [corr[policy_idx, j] for j in range(corr.shape[0]) if j != policy_idx]
print(f"mean |off-diag corr| = {np.mean(np.abs(off)):.4f}")
print("harm vs other axes:")
for j, label in enumerate(space.axis_labels[1:], start=1):
    print(f"  harm vs {label}: {corr[0, j]:+.4f}")
print("policy_violation vs other axes:")
for j, label in enumerate(space.axis_labels):
    if j == policy_idx:
        continue
    print(f"  policy vs {label}: {corr[policy_idx, j]:+.4f}")
inject_idx = list(space.axis_labels).index("injection")
print("injection vs other axes:")
for j, label in enumerate(space.axis_labels):
    if j == inject_idx:
        continue
    print(f"  injection vs {label}: {corr[inject_idx, j]:+.4f}")
PY

echo "--- slice figure ---"
"${PY}" scripts/plot_benchmark_space_heatmaps.py \
  --config configs/benchmark/router/six_axis_safety.yaml \
  --positions-json results/six_axis_theory_n12/positions.json \
  --split-manifest data/splits/six_axis_seed0.json \
  --density-mode empirical \
  --pairwise-bins 48 \
  --dpi 300 \
  --mass-norm per_panel_power \
  --smooth-sigma 2.5 \
  --vmax-percentile 90 \
  --scatter-alpha 0.15 \
  --max-scatter 5000 \
  --output-stem scripts/figures/six_axis_resource_slices \
  --title "Six-axis trait space (harm+policy+injection stretch=4.0, 70/10/20 slices)"

echo "done"
