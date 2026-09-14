# Handoff: theory-equilibrium initialisation study

Written 2026-09-14. Everything below is committed as of `1785302`; `paper/` is
tracked and clean. Read the "Traps" section before touching the GPU queues.

---

## 1. The one-paragraph version

The study asks whether starting the LoRA pairs at an equilibrium of the
influencer game — solved against the *empirical* prompt cloud — beats the
initialisation every earlier arm used. It does, but only when capacity is
scarce (≤4 pairs). Across competitive reach it does **not** raise the mean; what
it changes is the *smoothness* of the reach response. A second data split
(`dseed1`) is half-scored and is the live piece of work.

---

## 2. What is running right now

Three tmux sessions on `mlovett@doob.dartmouth.edu`, all alive:

| session | what |
|---|---|
| `gpu0_seq` | A100 queue: `…sigma03_d1`, then `…sigma04_d1`, then **ends** |
| `gpu1_seq` | A40 queue, train-only: `…sigma06_d1`, then `…sigma07_d1`, then **ends** |
| `gpu0_rescue` | **armed and waiting** — see below |

`dseed1` equilibrium sweep, 10 arms:

```
0.05 ROUTED   0.10 ROUTED   0.20 ROUTED   0.30 running   0.40 queued
0.45 trained  0.50 trained  0.55 trained  0.60 training  0.70 queued
```

### The rescue job (`/tmp/gpu0_rescue.sh`, session `gpu0_rescue`)

The A40 cannot run seven-pair **routing** (OOMs at 48 GB), so it runs
`--stages manifest,train` only and leaves arms trained-but-unscored. Nothing in
either queue picked them up. This job waits for **both** sequencer sessions to
disappear, then hands the five arms to `/tmp/gpu0_sequence.sh` with
`stages=perround,routing,figures`.

It waits for both — not just `gpu0_seq` — because starting a second sequencer
mid-queue is a known race: the new one sees GPU 0 briefly free between two arms
and launches on top. Costs nothing, since the A40 queue ends first.

Per-arm guards: skips if already routed, never trained, or `history.json` has
< 12 rounds. Reasons logged to `results/gpu0_rescue.log`. **If you kill the A40
mid-arm, that arm is correctly skipped rather than scored half-trained.**

Expect the full ten-arm curve ~9 h from 14:00 on 2026-09-14.

---

## 3. Findings, by how much I trust them

### Solid

- **The equilibrium is an attractor.** Naively-started pairs begin 0.330 from
  the equilibrium (matched L2, 7-D unit box), travel 0.291, and end **0.090**
  away — closing **73 % of the distance to an equilibrium they were never
  given**, in **10 of 10** arms. Equilibrium-started pairs move 0.0153
  (range 0.0108–0.0183), ~19× less. This is the strongest theory confirmation
  in the study. `main.tex` Table 3 (`tab:drift`).
- **Step function in pair count.** Equilibrium start worth +0.026/+0.013/+0.036
  at K = 2/3/4; at K ≥ 5 the five arms scatter around zero (mean −0.000,
  SD 0.0039). At K = 2 it flips routing from a loss (+0.0011) to a gain
  (−0.0247). Threshold moves down one pair.
- **Hard routing reverses sign under the mixture metric**: −0.0156 expected
  → **+0.0341** mixture. The paper previously claimed it was "the only
  configuration that loses to the generalist" without qualification. Fixed in
  abstract, intro and §routing.
- **Reach: no mean effect.** Mean paired Δ over ten reaches = **+0.0007**
  (median +0.0027), equilibrium ahead at 8/10. Every cell shares the same test
  partition (n=5671) and generalist (2.0152), so it is a genuinely paired
  contrast.

### Provisional — the live question

- **Smoothness.** Roughness = mean |2nd difference| along reach. Seed 0:
  naive 0.0164 vs equilibrium 0.0021 (**7.8×**); mixture metric 0.0160 vs
  0.0036 (4.5×). Use roughness, **not SD** — on the mixture metric both curves
  carry a real upward trend in σ, so SD is ~0.0156 either way and reports
  nothing.
- **Registered prediction** for the `dseed1` equilibrium curve, made before the
  data: **roughness < 0.006 and SD < 0.004** (naive dseed1 is 0.00813 / 0.0095).
  At 3 of 10 points: roughness **0.00100**, 7.3× smoother than naive. Tracking,
  but a 3-point roughness is a single second difference.

### Rejected — do not repeat these

- **"Position movement causes the gain zigzag."** Within the naive family,
  larger position steps do **not** predict larger margin changes: Spearman
  ρ = **−0.40** (p = 0.29, n = 9), wrong sign. Equilibrium family +0.67
  (p = 0.050); pooled +0.39 (p = 0.11).
- **"Naive wins where it ends furthest from the equilibrium."** Tempting
  because σ = 0.20 and 0.10 are the two naive wins and the two largest
  residuals — but σ = 0.45 has the third-largest residual and the *most
  negative* naive advantage. ρ = +0.50, p = 0.14, carried by σ = 0.20 alone.
- **"σ = 0.20 is a seed-0 outlier."** It is not. It beats its own split's sweep
  mean in **all four** naive splits (+0.0220, +0.0069, +0.0075, +0.0042). Only
  the *magnitude* is seed-0-specific. Confirmed on dseed1: Δ there is −0.0047
  against seed 0's −0.0215, close to the −0.008 the paper predicts.

### The caveat that matters most

The smoothness result **may be about how the two sweeps were built**, not about
equilibrium. Each naive arm re-solves from a fresh random start at its own σ:
adjacent reaches begin 0.302 apart and span six distinct axis-sets across ten
arms. Equilibrium arms begin 0.014 apart and share one axis-set at 8 of 10.

**The control has not been run:** repeat the naive sweep with positions held
fixed across σ (reuse one naive solve at all ten reaches, change only σ). If the
naive curve smooths, the result is about construction; if it stays rough, it is
about equilibrium. Ten training arms. This is the single most valuable
outstanding experiment.

Also unmeasured: **no run-seed replicates exist anywhere** (checked doob and
local for `seed1`/`seed2`/`_s1`/`_s2`). Every replicate moves run seed and split
seed together, so retraining noise at fixed (σ, init) is unknown. The run seed
moves naive positions but leaves equilibrium positions pinned (`init_mode:
given`), so two arms at σ = 0.50 seed 1 would size the noise *and* test the
mechanism.

---

## 4. Code changes

| file | what |
|---|---|
| `paper/figures_lib.py` | `THEORY_EQ_SIGMA_ARMS` (dseed1 equilibrium arms are *separate run dirs* whose subdir is still `seed0`, so `seeds_for` cannot find them); `roughness()`, `monotone_steps()`, `theory_eq_sigma_contrasts()`, `naive_sigma_contrasts()`, `SIGMA_DEGENERATE` |
| `paper/figures_init_table.py` | new — `fig_sigma_init_table`, the paired naive-vs-equilibrium reach table |
| `paper/figures_tables.py` | registers the above in `TABLE_FIGURES` |
| `paper/make_figures.py` | `fig_replication` left panel rewritten from grouped bars to a **line over σ** (mean of 4 splits + ±1 SD band, individual splits thin behind) |
| `paper/verify_tables.py` | **new — run this before any compile anyone reads** |
| `paper/main.tex` | see below |
| `paper/artifact_template.html` | new subsections in "Result: naive start vs the game equilibrium" |

### `paper/verify_tables.py`

Recomputes every cell of `tab:drift`, `tab:rep`, `tab:routing`, `tab:init` from
`results/` and diffs against the `.tex`. A table whose label it cannot find is a
**failure, not a skip**.

```
cd paper && python verify_tables.py     # exit 0 if the paper matches the data
```

It found **seven wrong digits**: four I introduced (rounding 4 dp to 3 dp by eye
on exact-half values), three inherited from the September draft
(`tab:routing` Sampled-k3 oracle 1.9438→1.9437, Hard-k1 oracle 1.9600→1.9601,
Hard-k1 agreement 0.707→0.706, the last also quoted in prose). All fixed.

**Not yet covered:** numbers appearing only in prose (abstract's 3.4 %/5.4 %,
the drift percentages, roughness ratios) and figure captions. Several were
rewritten this session and are unchecked.

### `paper/main.tex`

New: §`sec:init` ("Where the agents start") with `tab:init` and `tab:drift`;
§`sec:mixture` ("Which ensemble number to report") with `eq:mixture`.
Rewritten: replication 3→4 splits; `tab:routing` gained mixture + uniform
columns; abstract, intro, limitations.

Also fixed a **live rendering bug**: line 486 read `Figure~` + a literal newline
+ `ef{fig:rep}` — a `\r` eaten out of `\ref`, printing broken text in the PDF.

Two results moved against the old text: σ = 0.45 over 0.50 was t(2)=3.1, now
**t(3)=1.78, p=0.17** — no longer significant; σ = 0.70 collapse strengthened to
t(3)=14.9, p=0.001, 4/4 signs.

**Page budget: main text now ends ~p11 against ICLR's 9.** Not addressed —
which results to compress is an editorial call. Obvious candidates: `tab:init`
(a figure carries it) and the uniform-weight paragraph in §routing.

---

## 5. Traps

- **Bash heredocs in this environment mangle backslashes.** `\\n` inside a
  quoted heredoc became a real newline twice, producing broken Python and a
  broken regex. Write files with the Write tool and `scp`, or use the Edit tool.
  This is also how the `\ref` bug got into `main.tex` originally.
- **`nvidia-smi` is broken on doob** — `Failed to initialize NVML: Driver/library
  version mismatch`, NVML 615.71. Cosmetic: new CUDA contexts still initialise
  (verified — arms have started since). But you cannot read GPU memory, so VRAM
  headroom cannot be checked before launching.
- **`ps` on doob looks far busier than it is.** ~40 rows are finished pipelines
  parked on `read -n 1` at `[pipeline finished - press any key]` at 0.0 % CPU.
  Filter on CPU or on `grep -v "bash -c"`.
- **`tmux` prefix matching**: `sigma04` prefixes `sigma045`. Always `-t '=name'`.
- **`DONE` markers can be stale** — `--force` appends, so an old marker survives.
- **`scp` in a `while read` loop eats stdin.** Use `scp -q … < /dev/null`.
- **The local `results/` tree is incomplete and fails silently.** Two arms were
  missing from the drift aggregate until I noticed and pulled `history.json` for
  `theory_eq_sigma005`/`sigma01`. Check counts, do not assume.
- **`amcwhorter` shares the box** (`resume_ca_fixed.py`, `run_ga_shp.py`,
  `run_tx_shp.py`). Not ours — leave alone. A `raddicl` job pinned to `cuda:0`
  was stopped 2026-09-14 with the user's explicit approval.

---

## 6. What to do next

1. **Wait for the sweep** (~9 h from 14:00). When all ten `dseed1` equilibrium
   arms are routed, evaluate the registered prediction:
   ```
   cd paper && python -c "import figures_lib as F; ..."   # roughness over the 9 non-degenerate reaches
   ```
   Report whether roughness < 0.006 and SD < 0.004 held. **Do not quietly
   restate the threshold if it fails.**
2. **Then** update `paper/main.tex` §`sec:init` and the artifact with the
   two-split result, and re-run `verify_tables.py`.
3. **Ask before** running the fixed-position control (§3) or the run-seed
   replicates — both are ~10 and ~2 training arms respectively.
4. **Page budget** needs an editorial decision.

Artifact: <https://claude.ai/code/artifact/8436026f-a563-463d-a60d-8877967eb4ef>
(update with `url=`, and read it first — it is the same page, not a new one).

Unrelated but new in `1785302`: the user added a **behavioral safety evaluation
pipeline** (HarmBench, StrongREJECT, XSTest, TruthfulQA, ConfAIde, BIPIA,
IFEval) under `src/infl_ens/data/behavioral/`. I have not touched or run it.
