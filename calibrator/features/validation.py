"""Check the offline lookup against the features stored in the official archives."""

from __future__ import annotations

import pandas as pd

from .popularity import PopularityLookup

COMPARED = ("qpop", "gene_pop", "gene_coo", "q_occ", "gene_occ")
STATUSES = ("match", "zero_filled", "missing_both", "mismatch")


def _status(ours, archived) -> str:
    """``zero_filled``: lookup reports unavailable, archive stores the paper's 0 placeholder."""
    archived = None if pd.isna(archived) else int(archived)
    if ours is None:
        return "missing_both" if archived is None else "zero_filled" if archived == 0 else "mismatch"
    return "match" if ours == archived else "mismatch"


def compare_with_archives(lookup: PopularityLookup, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute features from each record's raw question/response and compare.

    Returns ``(summary, mismatches)``: per-(model, dataset) status counts, and
    one row per record with any mismatch (values or ``gene_pop_found`` flag).
    """
    rows = []
    for rec in records[["model", "dataset", "index", "question", "response", *COMPARED, "gene_pop_found"]].itertuples(index=False):
        found = lookup.lookup(rec.question, rec.response, rec.dataset)
        row = {"model": rec.model, "dataset": rec.dataset, "index": rec.index}
        for field in COMPARED:
            row[field] = _status(found[field], getattr(rec, field))
        row["gene_pop_found"] = "match" if found["gene_pop_found"] == bool(rec.gene_pop_found) else "mismatch"
        rows.append(row)
    status = pd.DataFrame(rows)

    checked = [*COMPARED, "gene_pop_found"]
    inconsistent = (status[checked] == "mismatch").any(axis=1)
    summary = status.groupby(["model", "dataset"], sort=False).size().rename("n_records").to_frame()
    summary["n_consistent"] = (~inconsistent).groupby([status["model"], status["dataset"]], sort=False).sum()
    for field in ("qpop", "gene_pop", "gene_coo"):
        counts = pd.crosstab([status["model"], status["dataset"]], status[field])
        for name in STATUSES:
            summary[f"{field}_{name}"] = counts[name] if name in counts else 0
    for field in ("gene_pop_found", "q_occ", "gene_occ"):
        summary[f"{field}_mismatch"] = (status[field] == "mismatch").groupby(
            [status["model"], status["dataset"]], sort=False).sum()
    summary = summary.fillna(0).astype(int).reset_index()

    mismatches = records.loc[inconsistent.to_numpy()].reset_index(drop=True)
    return summary, mismatches
