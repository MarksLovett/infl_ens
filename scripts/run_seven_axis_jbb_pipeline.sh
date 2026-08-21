#!/usr/bin/env bash
# JBB seven-axis pipeline: theory init, split figure, conditional r24 experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_jbb_pipeline.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== seven_axis JBB pipeline started $(date -Is) ==="

echo "--- download JBB if missing ---"
if [[ ! -f data/jbb_behaviors/harmful_behaviors.csv ]]; then
  "${PY}" scripts/download_jbb_behaviors.py --output data/jbb_behaviors
fi

echo "--- rebuild 70/10/20 split manifest ---"
"${PY}" scripts/build_seven_axis_split.py \
  --config configs/benchmark/router/seven_axis_pair_merge_split_r24.yaml \
  --output data/splits/seven_axis_seed0.json

echo "--- paired theory Nash (n14) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_theory_n14.yaml

echo "--- write co-located fixed_positions for SFT init ---"
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
from infl_ens.utils.agent_init import co_locate_theory_pairs

out_dir = Path("results/seven_axis_theory_n14")
payload = json.loads((out_dir / "positions.json").read_text(encoding="utf-8"))
names = list(payload["positions"].keys())
pos = np.asarray([payload["positions"][n] for n in names], dtype=float)
paired = co_locate_theory_pairs(pos, names)
fixed = {"positions": {names[i]: paired[i].tolist() for i in range(len(names))}}
fixed_path = out_dir / "fixed_positions.json"
fixed_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
print(f"wrote {fixed_path}")
PY

echo "--- theory layout check ---"
LAYOUT="$("${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
from infl_ens.training.pool_dynamics import classify_layout, pairwise_spread

pos_path = Path("results/seven_axis_theory_n14/positions.json")
payload = json.loads(pos_path.read_text())
pos = np.asarray(list(payload["positions"].values()), dtype=float)
spread = pairwise_spread(pos)
layout = classify_layout(pos)
print(layout)
print(f"spread={spread:.4f}")
# Dominant axis per agent (argmax coordinate)
for name, vec in payload["positions"].items():
    dom = int(np.argmax(vec))
    print(f"{name}: axis{dom}={vec[dom]:.3f}")
PY
)"
echo "${LAYOUT}"

echo "--- trait-space figure (decorrelated axes + 70/10/20 slices) ---"
"${PY}" scripts/plot_benchmark_space_heatmaps.py \
  --config configs/benchmark/router/seven_axis_safety.yaml \
  --positions-json results/seven_axis_theory_n14/positions.json \
  --split-manifest data/splits/seven_axis_seed0.json \
  --density-mode empirical \
  --pairwise-bins 48 \
  --dpi 300 \
  --mass-norm per_panel_power \
  --smooth-sigma 2.5 \
  --vmax-percentile 90 \
  --scatter-alpha 0.15 \
  --max-scatter 5000 \
  --output-stem scripts/figures/seven_axis_jbb_resource_slices \
  --title "Seven-axis trait space (JBB jailbreak, 70/10/20 slices)"

if grep -q collapsed <<<"${LAYOUT}"; then
  echo "Theory layout collapsed — skipping r24 split experiment."
  exit 0
fi

echo "--- r24 split closed-loop + pooled baseline + tables ---"
bash scripts/run_seven_axis_split_r24.sh

echo "=== seven_axis JBB pipeline finished $(date -Is) ==="
