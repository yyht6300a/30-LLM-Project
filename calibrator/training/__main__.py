"""CLI: train and export calibrator MLPs for the ablation feature subsets.

For every (model, dataset) pair this exports three deployable variants:

- ``full`` (no filename suffix): c + PopQ + PopGe + RPopGe  (main calibrator)
- ``c``:                           c only                     (ablation)
- ``c_rpopge``:                    c + RPopGe                 (ablation)

Each variant is saved as ``<stem>.pt`` (PyTorch state_dict), ``<stem>.npz``
(batch-norm-folded NumPy weights, loadable without torch via ``load_folded``)
and ``<stem>.json`` (feature order, scaler, threshold, metrics), where ``stem``
is ``<model>__<dataset>`` for ``full`` and ``<model>__<dataset>__<variant>``
otherwise.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from calibrator.data_prep import DATASETS, MODELS, load_model_outputs, load_split
from calibrator.eval.metrics import auroc, brier, conf_wrong, ece, select_threshold
from calibrator.training.mlp import MLPCalibrator

FEATURE_VARIANTS = {
    "full": ["confidence", "qpop", "gene_pop", "gene_coo"],
    "c": ["confidence"],
    "c_rpopge": ["confidence", "gene_coo"],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "calibrator" / "training" / "artifacts"


def _stem(model: str, dataset: str, variant: str) -> str:
    return f"{model}__{dataset}" if variant == "full" else f"{model}__{dataset}__{variant}"


def train_and_save(model: str, dataset: str, variant: str, features: list[str], records, artifacts_dir: Path) -> dict:
    split = load_split(model, dataset, protocol="balanced", features=features, records=records)
    X_tr, y_tr = split.X_trainval, split.y_trainval
    X_te, y_te = split.X_test, split.y_test

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    mlp = MLPCalibrator().fit(X_tr_s, y_tr)
    p_tr = mlp.predict_proba(X_tr_s)
    p_te = mlp.predict_proba(X_te_s)
    thr = select_threshold(y_tr, p_tr)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(model, dataset, variant)

    folded = mlp.export()
    npz = {f"W{i}": w for i, (w, _) in enumerate(folded["hidden"])}
    npz.update({f"b{i}": b for i, (_, b) in enumerate(folded["hidden"])})
    npz["W_out"] = folded["out"][0]
    npz["b_out"] = folded["out"][1]
    npz["n_hidden"] = len(folded["hidden"])
    np.savez_compressed(artifacts_dir / f"{stem}.npz", **npz)

    torch.save(
        {
            "state_dict": mlp.model.state_dict(),
            "in_dim": mlp.in_dim,
            "hidden": list(mlp.hidden),
            "dropout": mlp.dropout,
        },
        artifacts_dir / f"{stem}.pt",
    )

    metadata = {
        "model": model,
        "dataset": dataset,
        "variant": variant,
        "features": list(features),
        "protocol": "balanced",
        "mlp": {
            "hidden": list(mlp.hidden),
            "lr": mlp.lr,
            "weight_decay": mlp.weight_decay,
            "dropout": mlp.dropout,
            "seed": mlp.seed,
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": thr,
        "n_trainval": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "files": {"pt": f"{stem}.pt", "npz": f"{stem}.npz"},
        "test_metrics": {
            "alignment": float(((p_te >= thr) == y_te).mean()),
            "auroc": auroc(y_te, p_te),
            "ece": ece(y_te, p_te),
            "brier": brier(y_te, p_te),
            "conf_wrong": conf_wrong(y_te, p_te),
        },
    }
    (artifacts_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export P2 calibrator weights.")
    parser.add_argument("--models", nargs="*", default=None, help="Model slugs to train.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset slugs to train.")
    parser.add_argument("--variants", nargs="*", default=None, choices=list(FEATURE_VARIANTS), help="Feature variants to export.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--repo-dir", default=None, help="Paper repo root or model_outputs dir.")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    models = args.models or list(MODELS)
    datasets = args.datasets or list(DATASETS)
    variants = args.variants or list(FEATURE_VARIANTS)
    records = load_model_outputs(repo_dir=args.repo_dir)

    manifest = {}
    for model, dataset, variant in itertools.product(models, datasets, variants):
        features = FEATURE_VARIANTS[variant]
        metadata = train_and_save(model, dataset, variant, features, records, artifacts_dir)
        stem = _stem(model, dataset, variant)
        manifest[f"{model}/{dataset}/{variant}"] = {
            "stem": stem,
            "features": list(features),
            "json": f"{stem}.json",
            "pt": f"{stem}.pt",
            "npz": f"{stem}.npz",
            "threshold": metadata["threshold"],
            "alignment": metadata["test_metrics"]["alignment"],
        }
        print(f"trained {model}/{dataset}/{variant}: alignment={metadata['test_metrics']['alignment']:.3f}")

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", artifacts_dir / "manifest.json")


if __name__ == "__main__":
    main()
