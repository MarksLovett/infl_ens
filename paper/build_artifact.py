"""Build the standalone results walkthrough from the paper figures.

    python build_artifact.py            # paper/artifact_template.html -> paper/artifact.html

The template is prose with placeholders; this fills them in and inlines every
image as a base64 data URI, so the result is a single self-contained file with
no external requests. Figures come from ``paper/figures/``, which
``make_figures.py`` wrote -- the artifact and the manuscript therefore show the
same renders and cannot drift apart.

Placeholders
    {{fig:name}}    the figure, inlined, with its provenance line
    {{meta:name}}   just the provenance line (splits, runs, seeds)
    {{build:stamp}} when this was built and from what
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"
PLACEHOLDER = re.compile(r"\{\{(fig|meta|build):([a-z_]+)\}\}")


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG header, so the page reserves the right box."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def provenance(name: str) -> str:
    """One line describing what went into a figure, from its sidecar."""
    path = FIGURES / f"fig_{name}.json"
    if not path.is_file():
        return ""
    meta = json.loads(path.read_text(encoding="utf-8"))
    bits: list[str] = []
    if "n_splits" in meta:
        bits.append(f"{meta['n_splits']} data splits ({', '.join(meta['splits'])})")
    elif "splits" in meta:
        bits.append(f"splits: {', '.join(meta['splits'])}")
    elif "seed" in meta:
        bits.append(f"split {meta['seed']}")
    if "n_per_sigma" in meta:
        ns = sorted(set(meta["n_per_sigma"].values()))
        bits.append(f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]} per point")
    if "matched_by" in meta:
        bits.append(f"pairs matched by {meta['matched_by']}")
    if "caveat" in meta:
        bits.append(meta["caveat"])
    if meta.get("generated"):
        bits.append(f"built {meta['generated'][:10]}")
    return " · ".join(bits)


def inline_figure(name: str) -> str:
    """A figure element with the PNG embedded as a data URI."""
    png = FIGURES / f"fig_{name}.png"
    if not png.is_file():
        raise SystemExit(f"missing figure: {png} -- run `python make_figures.py` first")
    data = png.read_bytes()
    w, h = png_size(data)
    b64 = base64.b64encode(data).decode("ascii")
    prov = provenance(name)
    prov_html = f'<p class="prov">{prov}</p>' if prov else ""
    return (
        f'<figure id="fig-{name}">'
        f'<img src="data:image/png;base64,{b64}" alt="{name.replace("_", " ")}" '
        f'width="{w}" height="{h}">'
        f"{prov_html}</figure>"
    )


def build_stamp() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PAPER.parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rev = ""
    py = f"python {sys.version.split()[0]}"
    return " · ".join(x for x in (now, f"repo {rev}" if rev else "", py) if x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default=str(PAPER / "artifact_template.html"))
    ap.add_argument("--out", default=str(PAPER / "artifact.html"))
    ap.add_argument("--max-mb", type=float, default=16.0,
                    help="Refuse to write a page larger than this.")
    args = ap.parse_args()

    template = Path(args.template)
    if not template.is_file():
        raise SystemExit(f"no template at {template}")
    text = template.read_text(encoding="utf-8")

    used: list[str] = []

    def repl(match: re.Match) -> str:
        kind, name = match.group(1), match.group(2)
        if kind == "build":
            return build_stamp()
        if kind == "meta":
            return provenance(name)
        used.append(name)
        return inline_figure(name)

    out_html = PLACEHOLDER.sub(repl, text)

    size_mb = len(out_html.encode("utf-8")) / 1024 / 1024
    if size_mb > args.max_mb:
        raise SystemExit(f"artifact is {size_mb:.1f} MB, over the {args.max_mb} MB limit")

    Path(args.out).write_text(out_html, encoding="utf-8")
    print(f"  wrote {args.out}  ({size_mb:.2f} MB, {len(used)} figures inlined)")
    missing = sorted({p.stem[4:] for p in FIGURES.glob("fig_*.png")} - set(used))
    if missing:
        print(f"  note: built but not referenced by the template: {', '.join(missing)}")


if __name__ == "__main__":
    main()
