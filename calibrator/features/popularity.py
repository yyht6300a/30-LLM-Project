"""Offline popularity lookup: PopQ (qpop), PopGe (gene_pop) and RPopGe (gene_coo).

Reads the paper repository's precomputed resources -- no Wikipedia, Wikidata or
running model needed:

* ``data/{movies,songs,basketball}.jsonl``: question-entity Wikidata sitelinks.
* ``res/gt_gene_entity_popularity_*.jsonl``: answer-entity Wikidata sitelinks.
* ``res/cooccurrence_*.json``: Wikipedia document co-occurrence of question and
  answer entities.
* ``res/single_occurrence_*.json``: Wikipedia document counts per entity, used by
  the paper to exclude ambiguous, extremely frequent names.

``res/`` is created by the paper's ``scripts/materialize_compact_data.py``.

Entity normalization and matching copy the code that produced the archived
features (``calibration_experiment.py`` / ``build_compact_release_data.py``), not
the newer ``build_external_popularity.py``, which normalizes differently.

A value that is not in the tables is returned as ``None`` with ``*_found=False``;
a found value of 0 is a genuine zero. Only (question, answer) pairs produced by
the paper's six models are covered: an arbitrary user question or a new answer
string will usually be reported as not found.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from ..data_prep.loader import DATASETS, DEFAULT_REPO_DIR, REPO_ENV_VAR, SINGLE_OCC_THRESHOLD
from ..data_prep.splits import FEATURE_COLUMNS

QUESTION_PREFIXES = {
    "movies": "Who is the director of the movie ",
    "songs": "Who is the performer of the song ",
    "basketball": "Where is the birthplace of the basketball player ",
}
RESOURCE_SUFFIX = "qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b"
POP_FILE = Path("res") / f"gt_gene_entity_popularity_{RESOURCE_SUFFIX}.jsonl"
COO_FILE = Path("res") / f"cooccurrence_{RESOURCE_SUFFIX}.json"
SINGLE_FILE = Path("res") / f"single_occurrence_{RESOURCE_SUFFIX}.json"
QA_FILES = {dataset: Path("data") / f"{dataset}.jsonl" for dataset in DATASETS}
LOOKUP_FEATURES = ("qpop", "gene_pop", "gene_coo")


def infer_dataset(question: str) -> str | None:
    """Dataset whose question template ``question`` starts with, else ``None``."""
    for dataset, prefix in QUESTION_PREFIXES.items():
        if question.startswith(prefix):
            return dataset
    return None


def question_entity(question: str, dataset: str) -> str:
    """Question entity with original casing; a bare entity passes through unchanged."""
    return question.replace(QUESTION_PREFIXES[dataset], "")


def clean_entity(text: str | None, dataset: str) -> str:
    """Official ``remove_punctuation_edges``: generated answer -> answer entity.

    Drops newlines and anything from the first "("; keeps only the part before
    the first comma for basketball, or for movies/songs answers of at most 20
    characters; strips leading/trailing non-word characters.
    """
    value = (text or "").replace("\n", "").split("(")[0].strip()
    if dataset == "basketball" or len(value) <= 20:
        value = value.split(",")[0].strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", value).strip()


def resolve_repo_dir(repo_dir: str | os.PathLike | None = None) -> Path:
    """Paper repository root: argument, then ``$POPULAR_BUT_WRONG_REPO``, then ``reference/Popular-but-Wrong``."""
    root = Path(repo_dir or os.environ.get(REPO_ENV_VAR) or DEFAULT_REPO_DIR).expanduser()
    missing = [str(p) for p in (POP_FILE, COO_FILE, SINGLE_FILE, *QA_FILES.values()) if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Popularity resources missing under {root}: {missing}. Point {REPO_ENV_VAR} at the paper "
            "repository and run `python scripts/materialize_compact_data.py` there to create res/."
        )
    return root


def _read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sitelinks(value) -> int | None:
    """Official ``pop_value``: 'No', None or non-integers mean not found."""
    value = value.get("popularity") if isinstance(value, dict) else value
    if value in ("No", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class PopularityLookup:
    """In-memory offline tables; build once and reuse (loading takes about a second)."""

    def __init__(self, repo_dir: str | os.PathLike | None = None):
        root = resolve_repo_dir(repo_dir)
        # PopQ is attached per question in the QA files; key by the exact question
        # entity (Movies "Coma"/"COMA" and Songs "Change"/"chAngE" differ only by case).
        self.question_pop = {
            dataset: {question_entity(row["question"], dataset): _sitelinks(row["popularity"])
                      for row in _read_jsonl(root / path)}
            for dataset, path in QA_FILES.items()
        }
        self.entity_pop = {}
        for row in _read_jsonl(root / POP_FILE):
            for entity, value in row.items():
                self.entity_pop[entity] = _sitelinks(value)
        self.cooccurrence = json.loads((root / COO_FILE).read_text(encoding="utf-8"))
        self.occurrence = json.loads((root / SINGLE_FILE).read_text(encoding="utf-8"))

    def lookup(self, question: str, response: str | None, dataset: str | None = None) -> dict:
        """Popularity features of one (question, generated answer) pair.

        ``dataset`` is inferred from the question template when omitted; if it
        cannot be determined nothing is found. With an explicit ``dataset`` the
        question may also be the bare question entity.
        """
        dataset = dataset.strip().lower() if dataset else infer_dataset(question)
        if dataset is not None and dataset not in QUESTION_PREFIXES:
            raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")

        q_entity = question_entity(question, dataset) if dataset else question
        gene_entity = clean_entity(response, dataset) if dataset else ""
        q_key, gene_key = q_entity.lower(), gene_entity.lower()

        qpop = self.question_pop[dataset].get(q_entity) if dataset else None
        gene_pop = self.entity_pop.get(gene_entity) if gene_entity else None
        gene_coo = self.cooccurrence.get(q_key, {}).get(gene_key) if dataset and gene_entity else None
        q_occ = self.occurrence.get(q_key) if dataset else None
        gene_occ = self.occurrence.get(gene_key) if gene_entity else None

        noisy = (q_occ or 0) > SINGLE_OCC_THRESHOLD or (
            dataset != "basketball" and (gene_occ or 0) > SINGLE_OCC_THRESHOLD)
        return {
            "dataset": dataset,
            "question_entity": q_entity,
            "gene_entity": gene_entity,
            "qpop": qpop,
            "gene_pop": gene_pop,
            "gene_coo": gene_coo,
            "qpop_found": qpop is not None,
            "gene_pop_found": gene_pop is not None,
            "gene_coo_found": gene_coo is not None,
            "q_occ": q_occ,
            "gene_occ": gene_occ,
            # The paper excludes such samples from training; calibrated scores are unreliable here.
            "noisy_entity": bool(dataset) and noisy,
        }

    def feature_vector(
        self,
        question: str,
        response: str | None,
        confidence: float,
        dataset: str | None = None,
        features=FEATURE_COLUMNS,
    ) -> np.ndarray:
        """Calibrator input in ``features`` order, unavailable popularity filled with 0 as in the paper."""
        unknown = [f for f in features if f != "confidence" and f not in LOOKUP_FEATURES]
        if unknown:
            raise ValueError(f"Cannot look up {unknown}; supported: confidence, {', '.join(LOOKUP_FEATURES)}")
        result = self.lookup(question, response, dataset)
        values = {f: result[f] or 0 for f in LOOKUP_FEATURES}
        values["confidence"] = confidence
        return np.array([values[f] for f in features], dtype=float)
