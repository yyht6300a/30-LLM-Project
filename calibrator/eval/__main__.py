"""CLI: run the P2 calibration evaluation and write JSON + Markdown reports."""

from __future__ import annotations

import argparse

from calibrator.eval.evaluate import run_evaluation, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P2 calibration evaluation.")
    parser.add_argument("--models", nargs="*", default=None, help="Model slugs to evaluate.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset slugs to evaluate.")
    parser.add_argument("--protocol", default="balanced", choices=["balanced", "balanced_train", "natural"])
    parser.add_argument("--output-dir", default="calibrator/eval/output", help="Where to write reports.")
    parser.add_argument("--repo-dir", default=None, help="Paper repo root or model_outputs dir.")
    args = parser.parse_args()

    results = run_evaluation(
        models=args.models,
        datasets=args.datasets,
        protocol=args.protocol,
        repo_dir=args.repo_dir,
    )
    paths = write_reports(results, args.output_dir)
    print("wrote", paths["json"])
    print("wrote", paths["markdown"])


if __name__ == "__main__":
    main()
