"""Coverage-accuracy curves: raw confidence c (calibrator off) vs. MLP(full) (calibrator on).

Sort test questions by score, highest first. Answering the top k directly gives
coverage = k / n and accuracy = fraction correct among those k (risk-coverage
curve, Kamath et al. 2020, Sec. 3.1, with accuracy instead of risk). At equal
coverage, the higher curve is the score that better pushes wrong answers to
retrieval.

Data: the P2 balanced test split for the deployment model (docs/01), which the
exported weights never saw. Scores come from the archived paper snapshot, not
from fresh vLLM generations.

    python -m evaluation.coverage_accuracy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrator.data_prep import load_model_outputs, load_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "calibrator" / "training" / "artifacts"
OUTPUT = Path(__file__).resolve().parent / "output"
MODEL = "qwen2.5-7b"
DATASETS = ["movies", "songs", "basketball"]
REPORT_COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]


def coverage_accuracy(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and accuracy of the top-k questions by score, for k = 1..n."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    hits = np.cumsum(np.asarray(y)[order])
    k = np.arange(1, len(order) + 1)
    return k / len(order), hits / k


def load_calibrator(model: str, dataset: str):
    """Exported MLP(full) as (predict_fn, metadata), evaluated in NumPy.

    Same forward pass as calibrator.training.mlp.folded_predict, re-implemented
    because importing calibrator.training pulls in torch.
    """
    meta = json.loads((ARTIFACTS / f"{model}__{dataset}.json").read_text(encoding="utf-8"))
    z = np.load(ARTIFACTS / meta["files"]["npz"])
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_hidden"]))]
    mean, scale = np.asarray(meta["scaler_mean"]), np.asarray(meta["scaler_scale"])

    def predict(X: np.ndarray) -> np.ndarray:
        a = (np.asarray(X, dtype=float) - mean) / scale
        for W, b in layers:
            a = np.maximum(a @ W.T + b, 0.0)
        logit = (a @ z["W_out"].T + z["b_out"]).ravel()
        return 1.0 / (1.0 + np.exp(-logit))

    return predict, meta


def accuracy_at(cov: np.ndarray, acc: np.ndarray, target: float) -> float:
    return float(acc[np.searchsorted(cov, target - 1e-9)])


def main() -> None:
    records = load_model_outputs(models=[MODEL])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4), sharey=True)
    lines = [f"# Coverage-accuracy, {MODEL}, balanced test split", ""]

    for ax, dataset in zip(axes, DATASETS):
        predict, meta = load_calibrator(MODEL, dataset)
        split = load_split(MODEL, dataset, protocol=meta["protocol"], features=meta["features"], records=records)
        y = split.y_test
        scores = {
            "off: raw c": split.X_test[:, meta["features"].index("confidence")],
            "on: MLP(full)": predict(split.X_test),
        }
        lines += [f"## {dataset} (n_test = {len(y)})", "",
                  "| Score | " + " | ".join(f"cov {c:.1f}" for c in REPORT_COVERAGES) + " |",
                  "|---|" + "---|" * len(REPORT_COVERAGES)]
        for name, s in scores.items():
            cov, acc = coverage_accuracy(y, s)
            ax.plot(cov, acc, label=name)
            lines.append(f"| {name} | " + " | ".join(f"{accuracy_at(cov, acc, c):.3f}" for c in REPORT_COVERAGES) + " |")
        lines.append("")
        ax.set_title(f"{dataset} (n={len(y)})")
        ax.set_xlabel("coverage (fraction answered directly)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("accuracy on answered questions")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Coverage-accuracy, {MODEL}, balanced test split")
    fig.tight_layout()
    png = OUTPUT / f"coverage_accuracy_{MODEL}.png"
    fig.savefig(png, dpi=150)
    md = OUTPUT / f"coverage_accuracy_{MODEL}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {png} and {md}")


if __name__ == "__main__":
    main()
