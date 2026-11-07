"""Run the full calibration comparison and produce the P2 metrics report.

For every (model, dataset) pair this fits, on the 50% training half, each
baseline and MLP variant from the paper, selects the decision threshold on the
training half, and reports Alignment / AUROC / ECE / Brier / Confw on the
held-out test half. Threshold selection and feature columns follow P1's
``load_split`` interface exactly.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from calibrator.data_prep import DATASETS, MODELS, load_model_outputs, load_split
from calibrator.eval.metrics import auroc, brier, conf_wrong, ece, select_threshold
from calibrator.training.baselines import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    ThresholdModel,
)
from calibrator.training.mlp import MLPCalibrator

# name -> (kind, features, probabilistic). "probabilistic" means the score is a
# genuine probability in [0, 1], so ECE/Brier/Confw are meaningful.
METHODS: dict[str, tuple[str, list[str], bool]] = {
    "Prob": ("threshold", ["confidence"], True),
    "Consis": ("threshold", ["sc_confidence"], True),
    "Verbal": ("threshold", ["vc_confidence"], True),
    "PopQ": ("threshold", ["qpop"], False),
    "PopGe": ("threshold", ["gene_pop"], False),
    "RPopGe": ("threshold", ["gene_coo"], False),
    "Temp": ("temp", ["confidence"], True),
    "Platt": ("platt", ["confidence"], True),
    "Isotonic": ("isotonic", ["confidence"], True),
    "MLP(c)": ("mlp", ["confidence"], True),
    "MLP(c+RPopGe)": ("mlp", ["confidence", "gene_coo"], True),
    "MLP(full)": ("mlp", ["confidence", "qpop", "gene_pop", "gene_coo"], True),
}

MODEL_LABELS = {
    "llama3-8b": "Llama-3-8B",
    "qwen2-7b": "Qwen2-7B",
    "gpt-3.5-turbo": "ChatGPT",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-14b": "Qwen2.5-14B",
    "qwen2.5-32b": "Qwen2.5-32B",
}
DATASET_LABELS = {"movies": "Movies", "songs": "Songs", "basketball": "Basketball"}


def evaluate_combination(
    model: str,
    dataset: str,
    records,
    protocol: str = "balanced",
) -> dict[str, dict]:
    """Fit every method for one (model, dataset) and return its test metrics."""
    out: dict[str, dict] = {}
    for name, (kind, features, probabilistic) in METHODS.items():
        split = load_split(model, dataset, protocol=protocol, features=features, records=records)
        X_tr, y_tr = split.X_trainval, split.y_trainval
        X_te, y_te = split.X_test, split.y_test
        entry: dict = {"n_test": int(len(y_te))}

        if kind == "threshold":
            m = ThresholdModel().fit(X_tr[:, 0], y_tr)
            pred = m.predict(X_te[:, 0])
            entry["threshold"] = m.threshold
            entry["flip"] = m.flip
            entry["alignment"] = float(np.mean(pred == y_te))
            entry["auroc"] = auroc(y_te, X_te[:, 0])
            if probabilistic:
                entry.update(_probabilistic_metrics(y_te, X_te[:, 0]))

        elif kind in ("temp", "platt", "isotonic"):
            cls = {"temp": TemperatureScaler, "platt": PlattScaler, "isotonic": IsotonicCalibrator}[kind]
            m = cls().fit(X_tr, y_tr)
            p_tr, p_te = m.predict_proba(X_tr), m.predict_proba(X_te)
            thr = select_threshold(y_tr, p_tr)
            entry["threshold"] = thr
            entry["alignment"] = float(np.mean((p_te >= thr) == y_te))
            entry.update(_full_metrics(y_te, p_te))

        elif kind == "mlp":
            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
            m = MLPCalibrator().fit(X_tr_s, y_tr)
            p_tr, p_te = m.predict_proba(X_tr_s), m.predict_proba(X_te_s)
            thr = select_threshold(y_tr, p_tr)
            entry["threshold"] = thr
            entry["alignment"] = float(np.mean((p_te >= thr) == y_te))
            entry.update(_full_metrics(y_te, p_te))

        out[name] = entry
    return out


def _full_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": auroc(y, p),
        "ece": ece(y, p),
        "brier": brier(y, p),
        "conf_wrong": conf_wrong(y, p),
    }


def _probabilistic_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "ece": ece(y, p),
        "brier": brier(y, p),
        "conf_wrong": conf_wrong(y, p),
    }


def open_source_models(dataset: str) -> list[str]:
    """Models included in the paper's averaging; Qwen2.5 is excluded on Basketball."""
    if dataset == "basketball":
        return ["llama3-8b", "qwen2-7b"]
    return ["llama3-8b", "qwen2-7b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]


def run_evaluation(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    protocol: str = "balanced",
    repo_dir=None,
) -> dict:
    models = models or list(MODELS)
    datasets = datasets or list(DATASETS)
    records = load_model_outputs(repo_dir=repo_dir)
    results: dict[str, dict[str, dict]] = {}
    for model, dataset in itertools.product(models, datasets):
        results[f"{model}/{dataset}"] = evaluate_combination(
            model, dataset, records=records, protocol=protocol
        )
    return results


def alignment_summary(results: dict) -> dict:
    """Average Alignment per method using the paper's open-source averaging rule."""
    methods = list(METHODS)
    summary = {m: [] for m in methods}
    for dataset in DATASETS:
        for model in open_source_models(dataset):
            combo = results.get(f"{model}/{dataset}")
            if not combo:
                continue
            for m in methods:
                if "alignment" in combo[m]:
                    summary[m].append(combo[m]["alignment"])
    return {m: float(np.mean(v)) if v else float("nan") for m, v in summary.items()}


def render_markdown(results: dict) -> str:
    lines: list[str] = [
        "# P2 校准器训练与评测报告",
        "",
        "协议：balanced（论文主表）。Alignment 为在训练半区上选定阈值后、在留出测试集上的二值化准确率。",
        "",
    ]
    methods = list(METHODS)
    for dataset in DATASETS:
        lines.append(f"## {DATASET_LABELS[dataset]}")
        lines.append("")
        header = ["Method"] + [MODEL_LABELS[m] for m in MODELS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for name in methods:
            row = [name]
            for model in MODELS:
                entry = results.get(f"{model}/{dataset}", {}).get(name, {})
                row.append(f"{entry['alignment']:.3f}" if "alignment" in entry else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    summary = alignment_summary(results)
    lines.append("## Alignment 汇总（开放模型平均，Basketball 不含 Qwen2.5）")
    lines.append("")
    lines.append("| Method | Avg Alignment |")
    lines.append("|---|---|")
    for name in methods:
        lines.append(f"| {name} | {summary[name]:.3f} |")
    lines.append("")
    return "\n".join(lines)


def write_reports(results: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration_report.md").write_text(render_markdown(results), encoding="utf-8")
    return {
        "json": str(output_dir / "calibration_results.json"),
        "markdown": str(output_dir / "calibration_report.md"),
    }
