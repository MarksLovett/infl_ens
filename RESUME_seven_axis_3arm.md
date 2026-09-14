# Resume: seven-axis routing sweep on doob

Handoff doc. Paste this to Claude Code on the new device, in this same repo.
Everything below was verified on 2026-08-27; nothing here is from memory.

---

## 1. What is running RIGHT NOW (do not restart it)

A pipeline is training on the GPU host **doob** in tmux session
`seven_axis_3arm`, started 2026-08-27 09:45:55. It survives your laptop
disconnecting. Local monitors were stopped deliberately; the remote job was
NOT touched.

```
full=12  k3w=0  k3u=12  samp=12  hard=0  gen=0   stage=train (arm: soft)
```

It is retraining 3 of 6 arms and will then re-run analysis. Expected total
~4.3h from 09:45 (soft ~1.7h, hard ~1h, generalist ~0.5h, perround ~16m,
routing ~48m, figures <1m).

**First thing to do:** connect to the VPN, then check it is still alive:

```bash
REMOTE=mlovett@doob.dartmouth.edu MODE=status bash scripts/run_on_doob.sh
# or the compact one-liner probe already installed on the remote:
ssh mlovett@doob.dartmouth.edu 'bash ~/infl_ens/_arm_status.sh'
```

---

## 2. Connection facts (both were wrong at first — these are verified)

- **Account is `mlovett@doob.dartmouth.edu`**, NOT `mslovett@`. `mslovett@`
  fails with `Permission denied (publickey,password)`. `scripts/run_on_doob.sh`
  now defaults to `mlovett@` (this is the one uncommitted local change).
- **Dartmouth VPN is required.** Off-VPN, `doob:22` is filtered while `:80`
  and `:443` answer, so the host looks up but SSH times out. NordVPN does NOT
  work for this — it is a commercial VPN and does not put you on the campus
  network; disconnect it, it can prevent the institutional VPN from taking
  the route. Verify with:
  `Test-NetConnection doob.dartmouth.edu -Port 22` → must be `True`.
- Remote repo is `~/infl_ens`, interpreter `.venv/bin/python`, always with
  `PYTHONPATH=src`. GPU 0 is an A100 80GB; GPU 1 is an A40 usually busy with
  another user's job — stay on GPU 0.

---

## 3. The experiment

`configs/experiments/seven_axis_3arm.yaml` — six arms sweeping routing from
fully distributed to a single sampled winner, all sharing manifest, seed and
theory initialization, plus one pooled generalist.

| arm | selection | k | loss weight | run dir |
|---|---|---|---|---|
| `soft_full` | deterministic (all) | 7 | proportional share | `results/seven_axis_soft_full_pairs/seed0` |
| `soft` | deterministic top-3 | 3 | proportional share | `results/seven_axis_soft_topk3_pairs/seed0` |
| `soft_unit` | deterministic top-3 | 3 | unit | `results/seven_axis_topk3_unit_pairs/seed0` |
| `hard_topk` | **sampled w/o replacement** | 3 | unit | `results/seven_axis_hard_topk3_pairs/seed0` |
| `hard` | sampled | 1 | unit | `results/seven_axis_hard_pairs_matched/seed0` |
| `generalist` | pooled replay of `soft`'s batches | — | — | `results/seven_axis_3arm_generalist/seed0` |

12 rounds each, batch 1654, train 19,848 / val 2,836 / test 5,671. All arms
resolve to trait-space cache fingerprint `3b42c68a8dd334c5`, so the
Qwen3-Embedding-8B encode is reused — never edit the `benchmarks:` or
`trait_space:` blocks or you force a multi-hour re-encode.

What the arms isolate:
- `soft` vs `soft_unit` → loss WEIGHTING at fixed k=3 (the headline)
- `soft_unit` vs `hard_topk` → argmax vs SAMPLED selection at fixed k=3, unit
- `hard_topk` vs `hard` → k at fixed sampling and unit weight
- `soft_full` vs `soft` → k while share-weighted

---

## 4. Why this rerun exists (the bug being fixed)

The first full run completed, but `soft` and `hard` had been trained BEFORE
a large repo refactor, and the refactor changed RNG consumption. Those two
arms therefore saw a **different and slightly smaller training set** than the
three newer arms: 17,510 vs 17,976 unique prompts, Jaccard 0.887. The
generalist was replayed from old-`soft` batches, so the `pooled` column was
data-matched to the old arms only.

That confounded the headline weighting comparison, which spanned two code
versions. So `soft`, `hard` and `generalist` are being retrained under
current code.

**Important subtlety already handled:** retraining the generalist changes
`pooled_nll` in EVERY arm's routing diagnostic, so all five
`routing_ensemble_diagnostics.json` were deleted, not just the retrained
ones. The three kept arms retain their history, adapters and own evals, so
`train` and `perround` skip them while `routing` re-runs for all five.

**Verify the fix landed** once the run finishes and you have pulled results.
Read section 1 of the regenerated cross-analysis:

```bash
sed -n '1,20p' figures/seven_axis_3arm/cross_analysis.md
```

It must report **12/12 rounds identical (min Jaccard 1.0000)** for every arm
pair and conclude that one generalist IS data-matched to all arms. If it
still says `0/12 rounds identical` for any arm, the retrain did not take —
check that arm's run dir was actually deleted before the relaunch.

To confirm the underlying training sets match (the deeper check that caught
the original bug), compare the union of each arm's prompts across rounds;
all five specialists must report the same unique count:

```bash
ssh mlovett@doob.dartmouth.edu 'cd ~/infl_ens && PYTHONPATH=src .venv/bin/python - <<PY
import json
from pathlib import Path
for a in ["seven_axis_soft_full_pairs","seven_axis_soft_topk3_pairs",
          "seven_axis_topk3_unit_pairs","seven_axis_hard_topk3_pairs",
          "seven_axis_hard_pairs_matched"]:
    h = json.loads(Path(f"results/{a}/seed0/history.json").read_text())
    s = set()
    for rec in h:
        b = rec.get("batch_prompts")
        if b: s.update(map(str, b))
        else:
            for lst in (rec.get("agent_prompts") or {}).values():
                s.update(map(str, lst))
    print(f"{a:34s} {len(s)}")
PY'
```
Before the fix this printed 17,976 for the three new arms and 17,510 for the
two old ones. After, all five must agree.

---

## 5. Results from the FIRST (partly confounded) run — for comparison only

Route-then-score, test pool n=5671, round 11. These will be replaced.

| arm | oracle | learned | pooled | learned−pooled | oracle−learned |
|---|---|---|---|---|---|
| Soft k=7 wtd | 1.8857 | 1.9459 | 2.0045 | −0.0586 | −0.0602 |
| Soft k=3 wtd | 1.8900 | 1.9525 | 2.0045 | −0.0520 | −0.0626 |
| Soft k=3 unit | 1.9379 | 1.9837 | 2.0045 | −0.0208 | −0.0457 |
| Sampled k=3 unit | 1.9438 | 1.9870 | 2.0045 | −0.0176 | −0.0432 |
| Hard k=1 | 1.9478 | 2.0187 | 2.0045 | **+0.0142** | −0.0709 |

Findings, and how much to trust them:
- Share-weighted routing produces the best specialists; hard single-winner
  routing is the only arm that LOSES to the generalist. (Direction likely
  robust; the weighting comparison is what the rerun is fixing.)
- Every arm leaves 0.043–0.071 of router headroom, comparable to the
  specialization gain — the ROUTER is the binding constraint, not the
  adapters.
- **The per-pair macro-mean NLL table inverts this ranking.** Unit arms had
  the best average NLL (1.974–1.990) and the WORST oracle (1.938–1.948).
  Averaging a specialist over all seven axes rewards generalists. Do not
  use macro-mean to rank arms; use the routed numbers.
- Pair stability (unaffected by the rerun): all four soft arms held
  within-pair L2 at **exactly 0.000e+00** across all 12 rounds, including
  k=7; `hard` separated to 0.126. Every clone takes its OWN independent
  step — nothing in the code holds partners together. Co-location is a
  prediction the run measures, not a constraint it enforces.

---

## 6. When the run finishes

```bash
REMOTE=mlovett@doob.dartmouth.edu MODE=pull bash scripts/run_on_doob.sh
```

Pulls histories, resolved configs, routing diagnostics, `tables/`, `eval_*`
and all figures into this repo. It deliberately does NOT pull LoRA adapters
or the trait-space cache (tens of GB, they stay on doob).

Expect ~44 files in `figures/seven_axis_3arm/`: five `*_vs_oracle.{tex,pdf}`
bar figures (oracle vs generalist vs specialists, one per arm),
`arm_comparison.{tex,pdf}` (cross-arm overlay), per-arm
`*_final_positions.{pdf,png}` and `*_within_pair.{pdf,png}`,
`*_history.{pdf,png}`, and `cross_analysis.{md,json}`.

Per-arm round-4-vs-11 NLL tables land at
`results/<arm>/seed0/tables/pair_nll_by_round.{csv,md,tex,json}`.

Two convenience scripts are already on the remote (mine, not in git — delete
them whenever): `_arm_status.sh`, `_summarize_tables.py`,
`_summarize_routing.py`. Run the last two with
`cd ~/infl_ens && PYTHONPATH=src .venv/bin/python _summarize_routing.py`.

---

## 7. Local repo state

- Branch `cleanup/lean-pipeline`, HEAD `7175e67`.
- **One uncommitted change:** `scripts/run_on_doob.sh` — the `mlovett@`
  default fix. Worth committing.
- Full test suite passes locally (180 tests).
- **Known failing test on any GPU machine:**
  `tests/test_encoders.py::test_encoder_uses_left_padded_final_token_pooling`
  asserts `model.to_device == "cpu"`, which cannot hold where CUDA exists.
  It passes locally (no GPU) and fails on doob. Not an encoder bug — the
  test bakes in a CPU-only assumption. Worth fixing separately.

---

## 8. Operational gotchas worth keeping

- The tmux window prints `[pipeline finished - press any key]` and WAITS
  after the pipeline exits, success or crash. So **tmux staying alive is not
  evidence the job is healthy** — a crash looks identical to a long arm from
  a session-liveness check. Monitor the log for `pipeline seven_axis_3arm:
  done` plus crash signatures, not just tmux.
- Long CPU-only silences at the start of each arm are normal: the theory
  init is 8000 steps of NumPy grid-Nash ascent, single-core, printing
  nothing, with the GPU idle at ~5 GB.
- `scp -r` MERGES into existing remote dirs, it does not replace. Stale
  deleted files survive and break collection. When syncing after a refactor,
  `rm -rf` the remote `src/ scripts/ tests/ configs/` first.
- Do not pipe a long-running command through `tail -N` if you want to watch
  it — `tail` buffers everything until exit and the log file stays empty.
- GPU memory cycles 20→64 GB across rounds on the k=7 arm. That is the
  PyTorch caching allocator, not a leak; it comes back down.
