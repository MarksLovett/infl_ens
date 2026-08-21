#!/usr/bin/env bash
# Rebuild a detailed empirical resource-density figure for the oracle run.
set -euo pipefail
cd ~/infl_ens

.venv/bin/python - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("results/seven_axis_pair_merge_split/seed0/history.json").read_text())
pos = h[-1]["positions"]
out = Path("results/seven_axis_pair_merge_split/seed0/positions_round5.json")
out.write_text(json.dumps({"positions": pos, "round": h[-1]["round"]}, indent=2))
print(f"wrote {out} ({len(pos)} agents)")
PY

mkdir -p scripts/figures/oracle_run

.venv/bin/python scripts/plot_benchmark_space_heatmaps.py \
  --config configs/benchmark/router/seven_axis_pair_merge_split.yaml \
  --positions-json results/seven_axis_pair_merge_split/seed0/positions_round5.json \
  --density-mode empirical \
  --pairwise-bins 64 \
  --smooth-sigma 1.2 \
  --mass-norm per_panel_power \
  --vmax-percentile 98 \
  --max-scatter 2500 \
  --scatter-alpha 0.08 \
  --dpi 220 \
  --title "Oracle-run resource density (empirical, 64 bins) + round-5 positions" \
  --output-stem scripts/figures/oracle_run/resource_density_detailed

# Also refresh the primary resource_density stem used by the gallery.
cp -f scripts/figures/oracle_run/resource_density_detailed.png scripts/figures/oracle_run/resource_density.png
cp -f scripts/figures/oracle_run/resource_density_detailed.pdf scripts/figures/oracle_run/resource_density.pdf

echo "done"
ls -la scripts/figures/oracle_run/resource_density*
