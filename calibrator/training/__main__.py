"""CLI: train the deployable MLP(full) calibrator per (model, dataset) and export weights.

Exports three artifacts per (model, dataset) under ``artifacts/`` (gitignored):
- ``<model>__<dataset>.pt``  : full PyTorch state_dict (retraining / audit)
- ``<model>__<dataset>.npz`` : batch-norm-folded NumPy inference weights, loadable
  without torch via ``calibrator.training.mlp.load_folded``
- ``<model>__<dataset>.json``: feature order, scaler, threshold and test metrics
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

FEATURES = ["confidence", "qpop", "gene_pop", "gene_coo"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "calibrator" / "training" / "artifacts"


def train_and_save(model: str, dataset: str, records, artifacts_dir: Path) -> dict:
    split = load_split(model, dataset, protocol="balanced", features=FEATURES, records=records)
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
    stem = f"{model}__{dataset}"

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
        "features": FEATURES,
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
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--repo-dir", default=None, help="Paper repo root or model_outputs dir.")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    models = args.models or list(MODELS)
    datasets = args.datasets or list(DATASETS)
    records = load_model_outputs(repo_dir=args.repo_dir)

    manifest = {}
    for model, dataset in itertools.product(models, datasets):
        metadata = train_and_save(model, dataset, records, artifacts_dir)
        manifest[f"{model}/{dataset}"] = {
            "json": f"{model}__{dataset}.json",
            "pt": f"{model}__{dataset}.pt",
            "npz": f"{model}__{dataset}.npz",
            "threshold": metadata["threshold"],
            "alignment": metadata["test_metrics"]["alignment"],
        }
        print(f"trained {model}/{dataset}: alignment={metadata['test_metrics']['alignment']:.3f}")

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", artifacts_dir / "manifest.json")


if __name__ == "__main__":
    main()
