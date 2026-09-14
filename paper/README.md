# Paper: routing LoRA specialists in a learned trait space

Incomplete draft targeting ICLR. Named preprint version; switch to the anonymous
submission form by replacing the `\author{...}` block in `main.tex` with `\iclranonymous`.

## Build

```
cd paper
python make_figures.py                 # regenerates every figure from results/
python build_artifact.py               # figures/ -> artifact.html (standalone)
latexmk -pdf main.tex
```

`make_figures.py` is the only entry point for figures; `--list` names them and
`--only sigma_nll radar_sigma` rebuilds a subset. It writes a vector PDF for the
manuscript, a 300-dpi PNG for the artifact, and a JSON sidecar recording which
runs and splits went in, so a figure can always be traced to its inputs.

Figures read `results/` directly. Runs that live only on the training host are
pulled first, and re-running afterwards picks up the new splits with no edits:

```
bash ../scripts/run_on_doob.sh MODE=pull EXPERIMENT=configs/experiments/seven_axis_dseed4.yaml
```
The last clean build was 12 pages (9 of main text, then references and two appendices) with no
errors, no overfull boxes and no undefined references.

Requirements: a Python environment with numpy and matplotlib (`infl_clean` works), and
TeX Live with `natbib`, `booktabs`, `microtype`, `hyperref`.

## Layout

| file | what it is |
| --- | --- |
| `main.tex` | the manuscript |
| `refs.bib` | bibliography; `lovett2026influencer` is a placeholder for the unpublished companion paper |
| `iclr2026_conference.sty` | self-contained reimplementation of the ICLR layout, see below |
| `figures_lib.py` | shared rcParams, palette, split discovery, and the accessors that read `results/` |
| `radar_lib.py` | polar primitives for the pair-position figures |
| `make_figures.py` | the entry point: one function per figure, PDF + PNG + sidecar into `figures/` |
| `figures_extra.py` | the results-walkthrough figures, registered into `make_figures.py` |
| `build_artifact.py` | inlines `figures/*.png` into a standalone `artifact.html` |
| `artifact_template.html` | the walkthrough prose, with `{{fig:name}}` placeholders |
| `figures/` | generated output, not hand-edited |
| `data/prompt_coords.npz` | held-out prompt coordinates, copied from the training host |

## Style file

No official ICLR `.sty` is installed here and TeX Live ships none, so
`iclr2026_conference.sty` reproduces the published layout: US letter, 0.5in side margins,
5.5in by 9in text block, Times, `natbib` author-year, the "Under review as a conference
paper at ICLR 2026" running head, and `\iclrfinalcopy`. To use the official file, drop it
into this directory in place of ours; `main.tex` needs no changes.

One caveat if you edit the author block: `\And` and `\\` inside `\author` conflict with
`amsmath` unless `\@maketitle` is written the way the official NeurIPS/ICLR styles write
it. Ours follows that formulation exactly. `hyperref` also cannot derive PDF metadata from
a title containing `\\`, so `pdftitle` and `pdfauthor` are set explicitly in the preamble.

## Data provenance

Every number in the tables comes from `routing_ensemble_diagnostics.json` and
`eval_test/eval_results.json` under `results/<arm>/<split>/`, at round 11 on the 5,671-prompt
held-out pool. Figures read the same files, so a table and a figure cannot drift apart.
Replicate splits are read live from `results/<arm>/dseed*/` after a `MODE=pull`,
discovered automatically, so `fig_sigma_replicates` and `fig_sigma_table` widen
their `n` as further splits land. (They previously came from a checked-in
`data/replicates.json` snapshot, now retired to `.retired` to avoid a second
source of truth.)

## Open items

- Replication covers four splits of ten. The rest are running; error bars will tighten.
- Seven pairs is not the best pair count at seed 0 -- six beats it on both gain and
  capture. Wants a second split before acting on it.
- The reported router NLL is the expected loss of sampling one expert, not a mixture.
  The evaluator now also computes the sequence-level mixture and a uniform-weight
  control, and `fig_ensemble` is built. Its two mixture series are stamped
  *pending re-score* until the queued routing re-score lands; the expected and
  oracle series in it are already final.
- The Gemma-3-1B generalist had not converged at twelve rounds, so its 17.35% margin is
  inflated. It is greyed out in Table 3 and excluded from the scaling trend.
- The companion influencer's-game paper is unpublished. Replace the `refs.bib` placeholder
  when it has a venue.
- Positions are reported at three reaches only (0.70, 0.40, 0.05); `fig_positions` is where that
  choice lives if you want more.

## Retired figures

Three figures were cut on 2026-09-10 for carrying less information than the
numbers they plotted. Do not re-add them without a reason:

- `fig_gain_envelope` drew a part-to-whole decomposition of differences of
  ~0.005 nats as stacked bars, and its annotations collided with each other. It
  is replaced by the `oracle - pooled` column of `fig_sigma_table`.
- `fig_replication` was dropped from the artifact only: it plots the same sweep
  per replicate on the opposite sign convention, which the table now carries.
  It is still built and still included by `main.tex` as the appendix figure
  `fig:rep`, whose caption says three splits and is stale at four.
- `fig_convergence` had one fact worth keeping -- that the within-pair distance
  is exactly zero under soft routing, because co-location is a fixed point of the
  dynamics rather than an enforced constraint. That moved into the artifact's
  position-update section, where it reads as a check on the theory.

Every figure states its direction of improvement. Use `figures_lib.better(ax,
axis, direction)`, which appends an arrow to the axis label rather than adding a
floating annotation that can collide with the data.
