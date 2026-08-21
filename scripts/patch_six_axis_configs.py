#!/usr/bin/env python3
"""One-off: strip JBB from seven-axis YAMLs and convert to six-axis / 12 agents."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRS = [
    ROOT / "configs" / "benchmark" / "router",
    ROOT / "configs" / "evaluation",
]

JBB_BLOCK = re.compile(
    r"\n  - kind: jbb_behaviors\n"
    r"    path: data/jbb_behaviors\n"
    r"    include_benign: true\n"
    r"    max_records: null\n",
    re.MULTILINE,
)

JAILBREAK_WEIGHT = re.compile(
    r"    jailbreak: 1\.0\n",
    re.MULTILINE,
)
JAILBREAK_GAMMA = re.compile(
    r"    jailbreak: 2\.5\n",
    re.MULTILINE,
)

OLD_MERGE_GROUPS = re.compile(
    r"  sft_merge_groups:\n"
    r"    - train_as: merge-overrefusal\n"
    r"      names: \[clone-0, clone-13\]\n"
    r"    - train_as: merge-jailbreak\n"
    r"      names: \[clone-1, clone-9\]\n"
    r"    - train_as: merge-policy\n"
    r"      names: \[clone-2, clone-8\]\n"
    r"    - train_as: merge-harm\n"
    r"      names: \[clone-3, clone-6\]\n"
    r"    - train_as: merge-injection\n"
    r"      names: \[clone-4, clone-10\]\n"
    r"    - train_as: merge-hallucination\n"
    r"      names: \[clone-5, clone-11\]\n"
    r"    - train_as: merge-privacy\n"
    r"      names: \[clone-7, clone-12\]\n",
    re.MULTILINE,
)

NEW_MERGE_GROUPS = """  sft_merge_groups:
    - train_as: merge-harm
      names: [clone-0, clone-1]
    - train_as: merge-hallucination
      names: [clone-2, clone-3]
    - train_as: merge-privacy
      names: [clone-4, clone-5]
    - train_as: merge-overrefusal
      names: [clone-6, clone-7]
    - train_as: merge-injection
      names: [clone-8, clone-9]
    - train_as: merge-policy
      names: [clone-10, clone-11]
"""

OLD_VAL_AGENTS = re.compile(
    r"    agents:\n"
    r"      - merge-harm\n"
    r"      - merge-hallucination\n"
    r"      - merge-jailbreak\n"
    r"      - merge-privacy\n"
    r"      - merge-injection\n"
    r"      - merge-overrefusal\n"
    r"      - merge-policy\n",
    re.MULTILINE,
)

NEW_VAL_AGENTS = """    agents:
      - merge-harm
      - merge-hallucination
      - merge-privacy
      - merge-injection
      - merge-overrefusal
      - merge-policy
"""

CLONE_12_13 = re.compile(
    r"\n  - name: clone-12\n  - name: clone-13\n",
    re.MULTILINE,
)


def patch_text(text: str) -> str:
    """Apply six-axis transformations to YAML text."""
    text = JBB_BLOCK.sub("\n", text)
    text = JAILBREAK_WEIGHT.sub("", text)
    text = JAILBREAK_GAMMA.sub("", text)
    text = OLD_MERGE_GROUPS.sub(NEW_MERGE_GROUPS, text)
    text = OLD_VAL_AGENTS.sub(NEW_VAL_AGENTS, text)
    text = CLONE_12_13.sub("\n", text)
    text = text.replace("seven_axis_seed0.json", "six_axis_seed0.json")
    text = text.replace("seven_axis_theory_n14", "six_axis_theory_n12")
    text = text.replace("Seven-axis", "Six-axis")
    text = text.replace("seven-axis", "six-axis")
    text = text.replace("7-axis", "6-axis")
    text = text.replace("14 router agents", "12 router agents")
    text = text.replace("7 pair-merged", "6 pair-merged")
    text = text.replace("14 agents", "12 agents")
    return text


def main() -> None:
    """Patch all router/evaluation YAML configs and rename to six_axis."""
    renamed: list[tuple[Path, Path]] = []
    for directory in CONFIG_DIRS:
        for path in sorted(directory.glob("seven_axis*.yaml")):
            original = path.read_text(encoding="utf-8")
            updated = patch_text(original)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
            new_name = path.name.replace("seven_axis", "six_axis")
            new_name = new_name.replace("theory_n14", "theory_n12")
            new_path = path.with_name(new_name)
            if new_path != path:
                if new_path.exists():
                    new_path.unlink()
                path.rename(new_path)
                renamed.append((path, new_path))
                print(f"renamed -> {new_path.relative_to(ROOT)}")
            else:
                print(f"kept {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
