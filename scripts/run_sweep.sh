#!/usr/bin/env bash
#
# Parameter sweep over the closed-loop SFT trainer.
#
# Usage:
#   scripts/run_sweep.sh seeds   0 1 2 3
#   scripts/run_sweep.sh sigma   0.3 0.5 0.7 0.9
#   scripts/run_sweep.sh kde     0.05 0.08 0.12 0.20
#
# Modes:
#   seeds  → varies `seed` and `closed_loop.sft.seed`, holds sigma_fraction fixed
#   sigma  → varies `sigma_fraction`, holds seed fixed
#   kde    → varies `trait_space.kde_bandwidth`, holds seed and sigma_fraction fixed
#
# Optional environment-variable overrides:
#   BASE_CONFIG      default: configs/benchmark/router/safety_truth_n4_r10.yaml
#   RESULTS_ROOT     default: results/sweep_<mode>
#   SEED             default: 0     (used when mode != seeds)
#   SIGMA_FRACTION   default: 0.5   (used when mode != sigma)
#   KDE_BANDWIDTH    default: null  (used when mode != kde; null → Scott's rule)
#   N_ROUNDS         default: (taken from base config; override to shorten sweeps)
#   SKIP_EXISTING    default: 1     (1 = skip runs whose history.json already exists)
#   POST_PLOT        default: 1     (1 = produce per-run plot after each training)
#   POST_THEORY      default: 1     (1 = also run theory-vs-SFT comparison per run)
#
# Runs are sequential (single-GPU assumed). To launch in tmux, wrap with:
#   tmux new -s sweep "bash scripts/run_sweep.sh seeds 0 1 2 3"
#
# After the sweep finishes, aggregate with:
#   python scripts/plot_sweep.py --root <RESULTS_ROOT> --mode <MODE>

set -euo pipefail

MODE="${1:-}"
shift || true
VALUES=("$@")

if [[ -z "${MODE}" ]] || [[ ${#VALUES[@]} -eq 0 ]]; then
    echo "usage: $0 {seeds|sigma|kde} VAL [VAL ...]" >&2
    exit 1
fi

case "${MODE}" in
    seeds|sigma|kde) ;;
    *) echo "unknown mode: ${MODE}" >&2 ; exit 1 ;;
esac

BASE_CONFIG="${BASE_CONFIG:-configs/benchmark/router/safety_truth_n4_r10.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sweep_${MODE}}"
SEED_DEFAULT="${SEED:-0}"
SIGMA_FRACTION_DEFAULT="${SIGMA_FRACTION:-0.5}"
KDE_BANDWIDTH_DEFAULT="${KDE_BANDWIDTH:-null}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
POST_PLOT="${POST_PLOT:-1}"
POST_THEORY="${POST_THEORY:-1}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "BASE_CONFIG not found: ${BASE_CONFIG}" >&2
    exit 1
fi

mkdir -p "${RESULTS_ROOT}"
echo "sweep mode    : ${MODE}"
echo "values        : ${VALUES[*]}"
echo "base config   : ${BASE_CONFIG}"
echo "results root  : ${RESULTS_ROOT}"
echo "skip existing : ${SKIP_EXISTING}"
echo "post-plot     : ${POST_PLOT}"
echo "post-theory   : ${POST_THEORY}"
echo

for val in "${VALUES[@]}"; do
    overrides=()
    case "${MODE}" in
        seeds)
            seed="${val}"
            sigma_frac="${SIGMA_FRACTION_DEFAULT}"
            kde_bw="${KDE_BANDWIDTH_DEFAULT}"
            slug="seed${seed}"
            ;;
        sigma)
            seed="${SEED_DEFAULT}"
            sigma_frac="${val}"
            kde_bw="${KDE_BANDWIDTH_DEFAULT}"
            slug="sigma$(printf '%g' "${val}")"
            ;;
        kde)
            seed="${SEED_DEFAULT}"
            sigma_frac="${SIGMA_FRACTION_DEFAULT}"
            kde_bw="${val}"
            slug="kde$(printf '%g' "${val}")"
            ;;
    esac

    out_dir="${RESULTS_ROOT}/${slug}"
    history="${out_dir}/history.json"
    log="${RESULTS_ROOT}/${slug}.log"

    if [[ -f "${history}" && "${SKIP_EXISTING}" == "1" ]]; then
        echo "[${slug}] skip (history.json exists)"
    else
        echo "[${slug}] training: seed=${seed} sigma_fraction=${sigma_frac} kde=${kde_bw}"
        overrides=(
            "seed=${seed}"
            "sigma_fraction=${sigma_frac}"
            "output_dir=${out_dir}"
            "closed_loop.sft.output_dir=${out_dir}/agents"
            "closed_loop.sft.seed=${seed}"
        )
        if [[ "${kde_bw}" != "null" ]]; then
            overrides+=("trait_space.kde_bandwidth=${kde_bw}")
        fi
        if [[ -n "${N_ROUNDS:-}" ]]; then
            overrides+=("closed_loop.n_rounds=${N_ROUNDS}")
        fi
        python -m infl_ens.training --config "${BASE_CONFIG}" \
            "${overrides[@]}" \
            2>&1 | tee "${log}"
    fi

    # Optional per-run post-processing.
    if [[ "${POST_PLOT}" == "1" && -f "${history}" ]]; then
        python scripts/plot_closed_loop_history.py \
            --history "${history}" \
            --axis-labels harm hallucination \
            --title "${slug}  (seed=${seed}, sigma=${sigma_frac}, kde=${kde_bw})" \
            --output-stem "scripts/figures/${slug}" \
            > /dev/null
        echo "[${slug}] wrote scripts/figures/${slug}.{pdf,png}"
    fi

    if [[ "${POST_THEORY}" == "1" && -f "${history}" ]]; then
        # We need to point the comparator at a config that captures the
        # exact overrides used for this run. Materialise one alongside the
        # results dir so the comparator can rebuild the trait space the
        # same way.
        run_config="${out_dir}/run_config.yaml"
        python - "${BASE_CONFIG}" "${run_config}" \
                "${seed}" "${sigma_frac}" "${kde_bw}" "${out_dir}" <<'PY'
import sys, json
try:
    import yaml
except ImportError:
    sys.exit(0)
base_path, out_path, seed, sigma_frac, kde_bw, out_dir = sys.argv[1:]
with open(base_path) as fh:
    cfg = yaml.safe_load(fh)
cfg["seed"] = int(seed)
cfg["sigma_fraction"] = float(sigma_frac)
cfg["output_dir"] = out_dir
cfg.setdefault("closed_loop", {}).setdefault("sft", {})["seed"] = int(seed)
cfg["closed_loop"]["sft"]["output_dir"] = f"{out_dir}/agents"
if kde_bw != "null":
    cfg.setdefault("trait_space", {})["kde_bandwidth"] = float(kde_bw)
with open(out_path, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY
        if [[ -f "${run_config}" ]]; then
            python scripts/compare_theory_vs_sft.py \
                --config  "${run_config}" \
                --history "${history}" \
                --axis-labels harm hallucination \
                --title "${slug} theory vs SFT" \
                --output-stem "scripts/figures/${slug}_theory_vs_sft" \
                --summary-json "${out_dir}/theory_vs_sft.json" \
                > /dev/null
            echo "[${slug}] wrote ${out_dir}/theory_vs_sft.json"
        fi
    fi
done

echo
echo "sweep complete."
echo "aggregate with:"
echo "  python scripts/plot_sweep.py --root ${RESULTS_ROOT} --mode ${MODE} \\"
echo "      --output-stem scripts/figures/sweep_${MODE}"
