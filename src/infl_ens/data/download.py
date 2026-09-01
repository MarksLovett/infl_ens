"""Download the safety benchmarks the loaders in :mod:`infl_ens.data.benchmarks` read.

One function per benchmark ``kind``; :func:`download_for_entry` dispatches
on a config ``benchmarks`` entry so the pipeline ``download`` stage can
fetch whatever a config names and is missing on disk.  Every downloader
writes exactly the files the matching offline loader expects.

The ``datasets`` / ``huggingface_hub`` libraries are imported lazily; install
them with ``pip install "infl_ens[ml]"``.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

log = logging.getLogger("infl_ens.download")

BEAVERTAILS_REPO = "PKU-Alignment/BeaverTails"
HALUEVAL_BASE = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
HALUEVAL_TASK_FILES: dict[str, str] = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}
JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "behaviors"
AI4PRIVACY_REPO = "ai4privacy/pii-masking-200k"
ORBENCH_REPO = "orbench-llm/or-bench"
ORBENCH_CONFIGS: tuple[str, ...] = ("or-bench-80k", "or-bench-hard-1k", "or-bench-toxic")
THREAT_MATRIX_REPO = "neuralchemy/prompt-injection-Threat-Matrix"
THREAT_MATRIX_CONFIG = "binary"
DEEPSET_REPO = "deepset/prompt-injections"
DO_NOT_ANSWER_REPO = "LibrAI/do-not-answer"
BENIGN_REPO = "yahma/alpaca-cleaned"


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "downloading benchmarks needs the `datasets` library; "
            "install with `pip install 'infl_ens[ml]'`"
        ) from exc
    return datasets


def download_beavertails(
    dest: Path,
    *,
    split: str = "30k_train",
    max_records: int | None = None,
) -> Path:
    """Write ``<dest>/<split>.jsonl`` from ``PKU-Alignment/BeaverTails``.

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param split: BeaverTails split, e.g. ``30k_train`` or ``30k_test``.
    :type split: str
    :param max_records: Optional cap on the number of records.
    :type max_records: int | None
    :returns: The written file.
    :rtype: pathlib.Path
    """
    ds = _require_datasets().load_dataset(BEAVERTAILS_REPO, split=split)
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / f"{split}.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if max_records is not None and n >= max_records:
                break
            fh.write(json.dumps(dict(row)) + "\n")
            n += 1
    log.info("wrote %d records to %s", n, out_file)
    return out_file


def download_halueval(dest: Path, *, tasks: Sequence[str] = ("qa", "dialogue")) -> list[Path]:
    """Fetch HaluEval task JSON files from the official GitHub repository.

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param tasks: Task identifiers (keys of :data:`HALUEVAL_TASK_FILES`).
    :type tasks: Sequence[str]
    :returns: The written files.
    :rtype: list[pathlib.Path]
    :raises KeyError: For an unknown task.
    """
    from urllib.request import urlopen

    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task in tasks:
        fname = HALUEVAL_TASK_FILES[task]
        url = HALUEVAL_BASE + fname
        path = dest / fname
        with urlopen(url) as resp, path.open("wb") as fh:  # noqa: S310 - fixed https host
            fh.write(resp.read())
        log.info("downloaded %s -> %s", url, path)
        written.append(path)
    return written


def download_jbb_behaviors(dest: Path) -> list[Path]:
    """Write ``harmful_behaviors.csv`` and ``benign_behaviors.csv`` from JBB-Behaviors.

    Tries the ``datasets`` library first and falls back to
    ``huggingface_hub``.

    :param dest: Output directory.
    :type dest: pathlib.Path
    :returns: The written files.
    :rtype: list[pathlib.Path]
    """
    dest.mkdir(parents=True, exist_ok=True)
    harmful, benign = dest / "harmful_behaviors.csv", dest / "benign_behaviors.csv"
    try:
        ds = _require_datasets().load_dataset(JBB_REPO, JBB_CONFIG)
        for split_name, path in (("harmful", harmful), ("benign", benign)):
            rows = ds[split_name]
            fieldnames = list(rows.column_names)
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row[k] for k in fieldnames})
            log.info("wrote %s (%d rows)", path, len(rows))
    except Exception as ds_err:  # noqa: BLE001 - fall back to the hub download
        log.warning("datasets download failed (%s); trying huggingface_hub", ds_err)
        from huggingface_hub import hf_hub_download

        for remote, path in (
            ("data/harmful-behaviors.csv", harmful),
            ("data/benign-behaviors.csv", benign),
        ):
            cached = hf_hub_download(repo_id=JBB_REPO, filename=remote, repo_type="dataset")
            path.write_bytes(Path(cached).read_bytes())
            log.info("wrote %s", path)
    return [harmful, benign]


def _ai4privacy_record(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    src = row.get("source_text") or row.get("unmasked_text")
    tgt = row.get("target_text") or row.get("masked_text")
    if src is not None:
        out["source_text"] = src
    if tgt is not None:
        out["target_text"] = tgt
    for key in ("privacy_mask", "span_labels", "language"):
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def download_ai4privacy(
    dest: Path,
    *,
    language: str = "en",
    max_records: int | None = None,
) -> Path:
    """Write one JSONL of ``ai4privacy/pii-masking-200k`` records.

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param language: Language code to keep (``all`` keeps every language).
    :type language: str
    :param max_records: Optional cap on records written.
    :type max_records: int | None
    :returns: The written file.
    :rtype: pathlib.Path
    """
    ds = _require_datasets().load_dataset(AI4PRIVACY_REPO, split="train")
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / ("english_pii.jsonl" if language == "en" else f"{language}_pii.jsonl")
    accepted = {language, "english"} if language == "en" else {language}
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if language != "all":
                lang = str(row.get("language", "")).lower()
                if lang and lang not in accepted:
                    continue
            rec = _ai4privacy_record(dict(row))
            if "source_text" not in rec:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if max_records is not None and n >= max_records:
                break
    log.info("wrote %d records to %s", n, out_file)
    return out_file


def download_orbench(
    dest: Path,
    *,
    configs: Sequence[str] = ORBENCH_CONFIGS,
    max_records: int | None = None,
) -> list[Path]:
    """Write one CSV per OR-Bench config (``prompt``, ``category`` columns).

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param configs: Dataset configs to fetch.
    :type configs: Sequence[str]
    :param max_records: Optional per-config record cap.
    :type max_records: int | None
    :returns: The written files.
    :rtype: list[pathlib.Path]
    """
    datasets = _require_datasets()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for config in configs:
        ds = datasets.load_dataset(ORBENCH_REPO, config, split="train")
        out_file = dest / f"{config}.csv"
        n = 0
        with out_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["prompt", "category"])
            writer.writeheader()
            for row in ds:
                if max_records is not None and n >= max_records:
                    break
                writer.writerow({"prompt": row.get("prompt", ""), "category": row.get("category", "")})
                n += 1
        log.info("wrote %d records to %s", n, out_file)
        written.append(out_file)
    return written


def _injection_record(row: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    text = row.get("text") or row.get("prompt")
    if not text:
        return None
    label = row.get("label")
    if label is None:
        label = row.get("binary_label")
    if label is None:
        label = row.get("injection")
    if label is None:
        return None
    if isinstance(label, str):
        positive = label.strip().lower() in ("1", "true", "yes", "injection", "malicious", "jailbreak")
        label_int = 1 if positive else 0
    else:
        label_int = 1 if int(label) == 1 else 0
    return {"text": str(text), "label": label_int, "source": source}


def download_prompt_injection(
    dest: Path,
    *,
    source: str = "threat_matrix",
    max_records: int | None = 5000,
    seed: int = 0,
) -> Path:
    """Write ``prompt_injection.jsonl`` (``text``, ``label`` fields).

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param source: ``threat_matrix`` (32k rows, subsampled to
        ``max_records``) or ``deepset`` (the legacy ~662-row set).
    :type source: str
    :param max_records: Record cap (``None`` keeps everything).
    :type max_records: int | None
    :param seed: Shuffle seed when subsampling the threat matrix.
    :type seed: int
    :returns: The written file.
    :rtype: pathlib.Path
    :raises ValueError: For an unknown ``source``.
    """
    datasets = _require_datasets()
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "prompt_injection.jsonl"
    n = 0
    if source == "threat_matrix":
        ds = datasets.load_dataset(THREAT_MATRIX_REPO, THREAT_MATRIX_CONFIG)
        combined = datasets.concatenate_datasets([ds[name] for name in ds])
        indices = list(range(len(combined)))
        random.Random(seed).shuffle(indices)
        if max_records is not None and max_records < len(indices):
            indices = indices[:max_records]
        with out_file.open("w", encoding="utf-8") as fh:
            for idx in indices:
                rec = _injection_record(combined[int(idx)], source=THREAT_MATRIX_REPO)
                if rec is None:
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        manifest: dict[str, Any] = {
            "source": THREAT_MATRIX_REPO, "config": THREAT_MATRIX_CONFIG,
            "n_records": n, "max_records": max_records, "seed": seed,
        }
    elif source == "deepset":
        ds = datasets.load_dataset(DEEPSET_REPO)
        with out_file.open("w", encoding="utf-8") as fh:
            for split_name in ds:
                for row in ds[split_name]:
                    if max_records is not None and n >= max_records:
                        break
                    rec = _injection_record(row, source=DEEPSET_REPO)
                    if rec is None:
                        continue
                    rec["hf_split"] = split_name
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        manifest = {"source": DEEPSET_REPO, "n_records": n, "max_records": max_records}
    else:
        raise ValueError(f"source must be threat_matrix or deepset, got {source!r}")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %d records to %s", n, out_file)
    return out_file


def download_do_not_answer(dest: Path, *, n_benign: int = 5000, seed: int = 0) -> list[Path]:
    """Write ``do_not_answer.jsonl`` plus ``benign_negatives.jsonl`` (Alpaca sample).

    :param dest: Output directory.
    :type dest: pathlib.Path
    :param n_benign: Number of benign Alpaca instructions to sample.
    :type n_benign: int
    :param seed: Shuffle seed for the benign sample.
    :type seed: int
    :returns: The written files.
    :rtype: list[pathlib.Path]
    """
    datasets = _require_datasets()
    dest.mkdir(parents=True, exist_ok=True)
    dna_file = dest / "do_not_answer.jsonl"
    ds = datasets.load_dataset(DO_NOT_ANSWER_REPO, split="train")
    n = 0
    with dna_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            question = row.get("question") or row.get("instruction")
            if not question:
                continue
            fh.write(json.dumps({
                "question": question,
                "risk_area": row.get("risk_area", ""),
                "types_of_harm": row.get("types_of_harm", ""),
            }, ensure_ascii=False) + "\n")
            n += 1
    log.info("wrote %d Do-Not-Answer records to %s", n, dna_file)

    benign_file = dest / "benign_negatives.jsonl"
    ds = datasets.load_dataset(BENIGN_REPO, split="train").shuffle(seed=seed)
    n = 0
    with benign_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if n >= n_benign:
                break
            instruction = row.get("instruction")
            if not instruction:
                continue
            fh.write(json.dumps({"instruction": instruction}, ensure_ascii=False) + "\n")
            n += 1
    log.info("wrote %d benign records to %s", n, benign_file)
    return [dna_file, benign_file]


def _entry_dir(entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    return path.parent if path.suffix else path


def _dl_beavertails(entry: Mapping[str, Any]) -> None:
    path = Path(str(entry["path"]))
    split = path.stem if path.suffix else "30k_train"
    download_beavertails(_entry_dir(entry), split=split)


def _dl_halueval(entry: Mapping[str, Any]) -> None:
    download_halueval(_entry_dir(entry), tasks=tuple(entry.get("tasks") or ("qa", "dialogue")))


def _dl_orbench(entry: Mapping[str, Any]) -> None:
    download_orbench(_entry_dir(entry), configs=tuple(entry.get("configs") or ORBENCH_CONFIGS))


#: ``kind`` -> downloader taking the config ``benchmarks`` entry.
DOWNLOADERS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "beavertails": _dl_beavertails,
    "halueval": _dl_halueval,
    "jbb_behaviors": lambda entry: download_jbb_behaviors(_entry_dir(entry)),
    "ai4privacy": lambda entry: download_ai4privacy(_entry_dir(entry)),
    "orbench": _dl_orbench,
    "prompt_injection": lambda entry: download_prompt_injection(_entry_dir(entry)),
    "do_not_answer": lambda entry: download_do_not_answer(_entry_dir(entry)),
}


def entry_is_present(entry: Mapping[str, Any]) -> bool:
    """Whether the data a benchmark entry points at exists on disk.

    :param entry: Config ``benchmarks`` entry.
    :type entry: Mapping
    :returns: ``True`` when the path exists (a non-empty directory or a file).
    :rtype: bool
    """
    path = Path(str(entry["path"]))
    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def download_for_entry(entry: Mapping[str, Any]) -> None:
    """Download the benchmark a config entry names.

    :param entry: Config ``benchmarks`` entry (``kind`` + ``path`` + loader options).
    :type entry: Mapping
    :raises ValueError: For an unknown ``kind``.
    """
    kind = str(entry.get("kind"))
    if kind not in DOWNLOADERS:
        raise ValueError(f"no downloader for benchmark kind {kind!r}; known: {sorted(DOWNLOADERS)}")
    DOWNLOADERS[kind](entry)


__all__ = [
    "DOWNLOADERS",
    "download_ai4privacy",
    "download_beavertails",
    "download_do_not_answer",
    "download_for_entry",
    "download_halueval",
    "download_jbb_behaviors",
    "download_orbench",
    "download_prompt_injection",
    "entry_is_present",
]
