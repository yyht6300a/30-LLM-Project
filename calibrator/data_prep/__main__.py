"""Print the data inventory and split summaries.

    python -m calibrator.data_prep                       # six-model inventory
    python -m calibrator.data_prep --model qwen2.5-7b --dataset movies
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .loader import DATASETS, MODELS, apply_paper_filters, load_model_outputs
from .splits import FEATURE_COLUMNS, PROTOCOLS, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", help="paper repository (default: $POPULAR_BUT_WRONG_REPO or reference/Popular-but-Wrong)")
    parser.add_argument("--model", action="append", help=f"repeatable; default all of {MODELS}")
    parser.add_argument("--dataset", action="append", help=f"repeatable; default all of {DATASETS}")
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS, help="print split summaries for these protocols")
    parser.add_argument("--features", nargs="+", default=FEATURE_COLUMNS)
    args = parser.parse_args()

    records = load_model_outputs(args.model, args.dataset, repo_dir=args.repo_dir)
    _, report = apply_paper_filters(records)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Loaded {len(records):,} records")
        print(report.to_string(index=False))

    protocols = args.protocol or (list(PROTOCOLS) if args.model and args.dataset else [])
    for model, dataset in report[["model", "dataset"]].itertuples(index=False):
        for protocol in protocols:
            split = load_split(model, dataset, protocol, features=args.features, records=records)
            print(f"\n[{model} / {dataset} / {protocol}]")
            print(json.dumps(split.stats, indent=2))


if __name__ == "__main__":
    main()
