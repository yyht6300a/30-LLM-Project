"""Validate the offline popularity lookup against the official archives.

    python -m calibrator.features [--model qwen2.5-7b] [--dataset movies]
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..data_prep.loader import load_model_outputs
from .popularity import PopularityLookup
from .validation import compare_with_archives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", help="paper repository (default: $POPULAR_BUT_WRONG_REPO or reference/Popular-but-Wrong)")
    parser.add_argument("--model", action="append")
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()

    lookup = PopularityLookup(args.repo_dir)
    records = load_model_outputs(args.model, args.dataset, repo_dir=args.repo_dir)
    summary, mismatches = compare_with_archives(lookup, records)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
        total = summary[["n_records", "n_consistent"]].sum()
        print(f"\n{total.n_consistent:,} / {total.n_records:,} records consistent; {len(mismatches):,} with mismatches")
        if len(mismatches):
            print(mismatches[["model", "dataset", "index", "question", "response", "qpop", "gene_pop",
                              "gene_pop_found", "gene_coo", "q_occ", "gene_occ"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
