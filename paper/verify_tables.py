"""Check every hand-transcribed number in main.tex against results/.

The manuscript's tables are typed by hand from analysis output, which is fine
until it isn't: a first pass of this script found four wrong digits in
``tab:drift``, all introduced by rounding a 4-decimal printout to 3 decimals by
eye on exact-half values. None changed a conclusion, and none were findable by
reading.

Run before compiling anything anyone will read:

    python verify_tables.py            # exit 0 if the paper matches the data

Each checker recomputes a table's cells from ``results/`` and diffs them against
what the ``.tex`` actually says. A table that cannot be located is a failure, not
a skip, so renaming a label does not silently disable its check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))

import figures_lib as F  # noqa: E402

TEX = (PAPER / "main.tex").read_text(encoding="utf-8")

SIG = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
NAIVE_SIGMA = {
    0.05: "seven_axis_sigma005_full", 0.10: "seven_axis_sigma01_full",
    0.20: "seven_axis_sigma02_full", 0.30: "seven_axis_sigma03_full",
    0.40: "seven_axis_sigma04_full", 0.45: "seven_axis_sigma045_full",
    0.50: "seven_axis_soft_full_pairs", 0.55: "seven_axis_sigma055_full",
    0.60: "seven_axis_sigma06_full", 0.70: "seven_axis_sigma07_full",
}
EQ_SIGMA = {s: v["seed0"] for s, v in F.THEORY_EQ_SIGMA_ARMS.items()}


# ----------------------------------------------------------------- tex access
def table_body(label: str) -> str:
    """The tabular body of the table carrying ``\\label{label}``."""
    i = TEX.index("label{" + label + "}")
    return TEX[i:TEX.index("end{tabular}", i)]


def rows_of(body: str, label: str, occurrence: int = 0) -> list[str] | None:
    hits = list(re.finditer(re.escape(label) + r"\s*&(.*?)\\\\", body))
    if len(hits) <= occurrence:
        return None
    return [c.strip() for c in hits[occurrence].group(1).split("&")]


def compare(name: str, got: list[str] | None, want: list[str]) -> bool:
    if got is None:
        print("  {:38s} ROW NOT FOUND".format(name))
        return False
    bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
    if len(got) != len(want):
        bad.append(("ncols", len(got), len(want)))
    print("  {:38s} {}".format(name, "ok" if not bad else "MISMATCH " + str(bad)))
    return not bad


# -------------------------------------------------------------- geometry help
def eq_start(run):
    cfg = yaml.safe_load(
        (F.RESULTS / run / "seed0" / "resolved_config.yaml").read_text("utf-8"))
    ip = (cfg.get("closed_loop") or {}).get("init_positions")
    a = (np.array([ip[k] for k in sorted(ip)], float)
         if isinstance(ip, dict) else np.array(ip, float))
    return a.reshape(7, 2, -1).mean(axis=1) if a.shape[0] == 14 else a


def matched(a, b):
    perm, _ = F.match_pairs(a, b)
    return float(np.linalg.norm(a - b[perm], axis=1).mean())


# -------------------------------------------------------------------- checkers
def check_drift() -> bool:
    body = table_body("tab:drift")
    calc = {k: [] for k in ("travel", "s2e", "f2e", "closed", "eqt")}
    for s in SIG:
        ns = np.asarray(F.history_slim(NAIVE_SIGMA[s])["theory_positions"], float)
        nf = np.asarray(F.history_slim(NAIVE_SIGMA[s])["positions"][-1], float)
        es = eq_start(EQ_SIGMA[s])
        ef = np.asarray(F.history_slim(EQ_SIGMA[s])["positions"][-1], float)
        t, a, b = matched(nf, ns), matched(ns, es), matched(nf, es)
        calc["travel"].append(t); calc["s2e"].append(a); calc["f2e"].append(b)
        calc["closed"].append(100 * (1 - b / a))
        calc["eqt"].append(matched(ef, es))
    ok = True
    spec = [("distance travelled", 0, "travel", "{:.3f}"),
            ("start to equilibrium", 0, "s2e", "{:.3f}"),
            ("end to equilibrium", 0, "f2e", "{:.3f}"),
            ("gap closed", 0, "closed", "{:.0f}\\%"),
            ("distance travelled", 1, "eqt", "{:.3f}")]
    for label, occ, key, fmt in spec:
        want = [fmt.format(v) for v in calc[key]] + \
               [fmt.format(float(np.mean(calc[key])))]
        nm = label + (" [equilibrium]" if occ else "")
        ok &= compare(nm, rows_of(body, label, occ), want)
    return ok


def check_rep() -> bool:
    body = table_body("tab:rep")
    nv = F.naive_sigma_contrasts()
    splits = ["seed0", "dseed1", "dseed2", "dseed3"]
    base = np.array([-nv[s][0.50]["gain"] for s in splits])
    ok = True
    for sg in (0.70, 0.50, 0.45, 0.20):
        v = np.array([-nv[s][sg]["gain"] for s in splits])
        cells = ["${:+.4f}$".format(x) for x in v]
        cells += ["${:+.4f}$".format(v.mean()), "{:.4f}".format(v.std(ddof=1))]
        if sg == 0.50:
            cells.append("---")
        else:
            t = stats.ttest_rel(v, base).statistic
            cells.append("${:+.4f}$ ($t={:.1f}$)".format((v - base).mean(), t))
        ok &= compare("rep {:.2f}".format(sg),
                      rows_of(body, "{:.2f}".format(sg)), cells)
    return ok


def check_routing() -> bool:
    body = table_body("tab:routing")
    names = ["Soft $k{=}7$", "Soft $k{=}3$", "Soft $k{=}3$ unit",
             "Sampled $k{=}3$", "Hard $k{=}1$"]
    ok = True
    for name, (_lab, run, sel, k, loss) in zip(names, F.ROUTING_ARMS):
        c = F.contrast(F.diagnostics(run))
        got = rows_of(body, name)
        if got is None:
            print("  {:38s} ROW NOT FOUND".format("routing " + name))
            ok = False
            continue
        # selection / k / loss are prose; check only the numeric tail.
        want = ["{:.4f}".format(c["oracle"]),
                "${:+.4f}$".format(-c["gain"]),
                "$\\mathbf{{{:+.4f}}}$".format(-c["gain_mixture"])
                if name == "Soft $k{=}7$"
                else "${:+.4f}$".format(-c["gain_mixture"]),
                "${:+.4f}$".format(-c["gain_uniform"]),
                "{:.3f}".format(c["agreement"])]
        ok &= compare("routing " + name, got[-5:], want)
    return ok


def check_init() -> bool:
    body = table_body("tab:init")
    eqp = dict(F.THEORY_EQ_PAIR_ARMS)
    nv, eq = [], []
    for k, run in F.PAIR_ARMS:
        nv.append(-F.contrast(F.diagnostics(run))["gain"])
        eq.append(-F.contrast(F.diagnostics(eqp[k]))["gain"])
    nv, eq = np.array(nv), np.array(eq)
    ok = True
    ok &= compare("init naive", rows_of(body, "Naive start"),
                  ["${:+.4f}$".format(v) for v in nv])
    ok &= compare("init equilibrium", rows_of(body, "Equilibrium"),
                  ["${:+.4f}$".format(v) for v in eq])
    ok &= compare("init difference", rows_of(body, "Difference"),
                  ["${:+.4f}$".format(v) for v in (eq - nv)])
    return ok


def main() -> None:
    checks = [("tab:drift", check_drift), ("tab:rep", check_rep),
              ("tab:routing", check_routing), ("tab:init", check_init)]
    ok = True
    for label, fn in checks:
        print(label)
        try:
            ok &= fn()
        except Exception as exc:                       # noqa: BLE001
            print("  FAILED TO CHECK: {}: {}".format(type(exc).__name__, exc))
            ok = False
        print()
    print("every checked number in main.tex matches results/" if ok
          else "MISMATCHES ABOVE -- main.tex disagrees with the data")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
