"""Load and filter the official Popular-but-Wrong compact model-output archives.

The archives live in the paper repository under ``data/model_outputs/`` and are
documented in its ``data/model_outputs/README.md``. ``null`` in an archive means
"unavailable upstream", never zero; this module keeps those values as missing
(``NaN`` / ``<NA>``) and leaves any zero-filling to the feature builder.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ENV_VAR = "POPULAR_BUT_WRONG_REPO"
DEFAULT_REPO_DIR = PROJECT_ROOT / "reference" / "Popular-but-Wrong"
ARCHIVE_SUBDIR = Path("data") / "model_outputs"

DATASET_SIZES = {"movies": 10964, "songs": 2157, "basketball": 13309}
DATASETS = tuple(DATASET_SIZES)
RECORDS_PER_MODEL = sum(DATASET_SIZES.values())  # 26,430

# Public archive slugs (file stems). Paper/analysis-code names map onto them.
MODELS = (
    "llama3-8b",
    "qwen2-7b",
    "gpt-3.5-turbo",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
)
MODEL_ALIASES = {"llama8b": "llama3-8b", "qwen2": "qwen2-7b", "chatgpt": "gpt-3.5-turbo"}

FIELDS = (
    "dataset", "index", "question", "reference", "response", "correct", "confidence",
    "qpop", "gt_pop", "gt_pop_found", "gene_pop", "gene_pop_found",
    "gt_coo", "gene_coo", "q_occ", "gt_occ", "gene_occ",
    "sc_confidence", "vc_confidence", "llm_qpop", "llm_gene_pop", "llm_coo",
)
# Integer fields that may be null; stored as nullable Int64 so missing != 0.
COUNT_FIELDS = (
    "qpop", "gt_pop", "gene_pop", "gt_coo", "gene_coo", "q_occ", "gt_occ", "gene_occ",
    "llm_qpop", "llm_gene_pop", "llm_coo",
)

# Official calibration filter (code/analysis_correlation/calibration_experiment.py):
# skip ambiguous, extremely frequent entity strings.
SINGLE_OCC_THRESHOLD = 6000
EXCLUSION_REASONS = ("empty_response", "noisy_entity", "missing_confidence")


def normalize_model(name: str) -> str:
    key = name.strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODELS} or {tuple(MODEL_ALIASES)}")
    return key


def normalize_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_SIZES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")
    return key


def resolve_archive_dir(repo_dir: str | os.PathLike | None = None) -> Path:
    """Locate ``data/model_outputs``.

    Priority: ``repo_dir`` argument, then ``$POPULAR_BUT_WRONG_REPO``, then
    ``<project>/reference/Popular-but-Wrong``. The path may point at the paper
    repository root or directly at the ``model_outputs`` directory.
    """
    candidate = repo_dir or os.environ.get(REPO_ENV_VAR) or DEFAULT_REPO_DIR
    candidate = Path(candidate).expanduser()
    for path in (candidate / ARCHIVE_SUBDIR, candidate):
        if any(path.glob("*.jsonl.gz")):
            return path
    raise FileNotFoundError(
        f"No *.jsonl.gz archives found under {candidate}. Clone the paper repository to "
        f"{DEFAULT_REPO_DIR} or set {REPO_ENV_VAR} to its location."
    )


def read_archive(path: str | os.PathLike) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_records(rows: list[dict], model: str) -> None:
    """Check counts, identifiers and essential field values of one model archive."""
    problems = []
    if len(rows) != RECORDS_PER_MODEL:
        problems.append(f"{len(rows)} records, expected {RECORDS_PER_MODEL}")
    missing = {field for row in rows for field in FIELDS if field not in row}
    if missing:
        problems.append(f"missing fields {sorted(missing)}")
    else:
        for dataset, size in DATASET_SIZES.items():
            indices = sorted(row["index"] for row in rows if row["dataset"] == dataset)
            if indices != list(range(size)):
                problems.append(f"{dataset}: {len(indices)} records, expected indices 0..{size - 1}")
        unknown = {row["dataset"] for row in rows} - set(DATASET_SIZES)
        if unknown:
            problems.append(f"unknown datasets {sorted(unknown)}")
        if any(row["correct"] not in (0, 1) for row in rows):
            problems.append("correct must be 0 or 1")
        if any(row["confidence"] is not None and not 0 <= row["confidence"] <= 1 for row in rows):
            problems.append("confidence outside [0, 1]")
        if any(not row["reference"] for row in rows):
            problems.append("empty reference list")
        for field in ("qpop", "gene_pop", "gene_coo"):
            if any(row[field] is not None and (not isinstance(row[field], int) or row[field] < 0) for row in rows):
                problems.append(f"{field} must be a non-negative integer or null")
        if any(not row["gene_pop_found"] and row["gene_pop"] != 0 for row in rows):
            problems.append("gene_pop must be 0 when gene_pop_found is false")
    if problems:
        raise ValueError(f"Invalid archive for {model}: " + "; ".join(problems))


def load_model_outputs(
    models: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    repo_dir: str | os.PathLike | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load archives into one DataFrame keyed by ``(model, dataset, index)``.

    Validation always runs on the complete archive, before dataset filtering.
    Nothing is filtered here except by the requested models/datasets.
    """
    archive_dir = resolve_archive_dir(repo_dir)
    models = [normalize_model(m) for m in models] if models else list(MODELS)
    datasets = [normalize_dataset(d) for d in datasets] if datasets else list(DATASETS)

    frames = []
    for model in models:
        path = archive_dir / f"{model}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing archive {path}")
        rows = read_archive(path)
        if validate:
            validate_records(rows, model)
        frame = pd.DataFrame.from_records([row for row in rows if row["dataset"] in datasets], columns=FIELDS)
        frame.insert(0, "model", model)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    for field in COUNT_FIELDS:
        df[field] = df[field].astype("Int64")
    for field in ("confidence", "sc_confidence", "vc_confidence"):
        df[field] = df[field].astype(float)
    df["correct"] = df["correct"].astype(int)
    return df


def exclusion_reasons(df: pd.DataFrame) -> pd.Series:
    """First applicable official exclusion reason per record (``NA`` if retained).

    Mirrors ``collect_samples`` in the official calibration code, in order:
    empty response; single-entity Wikipedia occurrence > 6000 for the question
    entity (and, for movies/songs, the generated entity), with unavailable
    occurrence counts treated as 0 as upstream does; missing token probabilities.
    Popularity / co-occurrence lookups that fail do *not* exclude a record.
    """
    empty = df["response"].fillna("") == ""
    q_occ = df["q_occ"].fillna(0)
    gene_occ = df["gene_occ"].fillna(0)
    noisy = (q_occ > SINGLE_OCC_THRESHOLD) | (
        (df["dataset"] != "basketball") & (gene_occ > SINGLE_OCC_THRESHOLD)
    )
    masks = {"empty_response": empty, "noisy_entity": noisy, "missing_confidence": df["confidence"].isna()}

    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    for reason in EXCLUSION_REASONS:
        reasons = reasons.mask(reasons.isna() & masks[reason].astype(bool), reason)
    return reasons


def apply_paper_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(retained_records, report)``; the report has one row per (model, dataset)."""
    reasons = exclusion_reasons(df)
    retained = df[reasons.isna()].reset_index(drop=True)

    keys = ["model", "dataset"]
    report = df.groupby(keys, sort=False).size().rename("n_total").to_frame()
    excluded = pd.crosstab([df["model"], df["dataset"]], reasons.fillna("retained"))
    for reason in EXCLUSION_REASONS:
        report[reason] = excluded[reason] if reason in excluded else 0
    grouped = retained.groupby(keys, sort=False)["correct"]
    report["n_retained"] = grouped.size()
    report["n_correct"] = grouped.sum()
    report = report.fillna(0).astype(int)
    report["accuracy"] = (report["n_correct"] / report["n_retained"]).round(4)
    return retained, report.reset_index()
