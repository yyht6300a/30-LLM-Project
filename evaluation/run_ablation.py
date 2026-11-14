"""End-to-end ablation: calibrator on/off x retrieval on/off (evaluation/README.md).

One vLLM generation per test question gives the answer and raw confidence c;
P3's CalibrationService (the code behind the MCP ``calibrate`` tool) gives the
calibrated probability. All four configurations are then derived from the same
records, because neither switch changes the generated answer:

    decision source  raw        -> answer iff c >= P2's Prob threshold
                     calibrated -> answer iff P(correct) >= artifact threshold
    retrieval        off        -> non-answer is a refusal (counts as wrong)
                     on         -> non-answer goes to search_qa (local KB)

Questions: the P2 balanced test split of the deployment model, which the
calibrator never saw during training. Correctness: the paper's normalized
containment rule (``has_answer``).

Run on a GPU node with vLLM up (model_runtime/vllm/start_qwen25_7b.sh):

    export PYTHONPATH="$PWD" OPENHANDS_SUPPRESS_BANNER=1
    python -m evaluation.run_ablation                 # generate + metrics
    python -m evaluation.run_ablation --metrics-only  # recompute from saved records

Environment: the agent_integration env (openhands-sdk) plus pandas,
scikit-learn, matplotlib. M3_VLLM_BASE_URL / M3_MODEL as in agent_integration.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_integration"))

from calibrated_qa import DecisionPolicy, generate_answer  # noqa: E402
from search_tool import KnowledgeBase  # noqa: E402

from calibrator.data_prep import load_model_outputs, load_split  # noqa: E402
from evaluation.coverage_accuracy import accuracy_at, coverage_accuracy  # noqa: E402
from mcp_server.service import CalibrationService  # noqa: E402

BASE_URL = os.getenv("M3_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("M3_MODEL", "qwen2.5-7b")
DATASETS = ["movies", "songs", "basketball"]
OUTPUT = Path(__file__).resolve().parent / "output"
RECORDS = OUTPUT / "ablation_records.jsonl"
P2_RESULTS = ROOT / "calibrator" / "eval" / "output" / "calibration_results.json"
ARTIFACTS = ROOT / "calibrator" / "training" / "artifacts"
CONFIGS = [(mode, retrieval) for retrieval in ("off", "on") for mode in ("raw", "calibrated")]
REPORT_COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]

# --- correctness: the paper's has_answer (qa_generation/utils/utils.py) --------
_TOKEN = regex.compile(r"([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])", flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE)
_ARTICLES = regex.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = " ".join(_ARTICLES.sub(" ", text).split())
    return [m.group().lower() for m in _TOKEN.finditer(text)]


def has_answer(references: list[str], text: str) -> int:
    """1 if the normalized tokens of any reference occur contiguously in text."""
    words = _tokens(text)
    for ref in references:
        r = _tokens(ref)
        if any(words[i:i + len(r)] == r for i in range(len(words) - len(r) + 1)):
            return 1
    return 0


# --- generation pass (GPU) -------------------------------------------------------
def test_questions(records=None) -> list[dict]:
    rows = []
    records = records if records is not None else load_model_outputs(models=[MODEL])
    for dataset in DATASETS:
        meta = json.loads((ARTIFACTS / f"{MODEL}__{dataset}.json").read_text(encoding="utf-8"))
        split = load_split(MODEL, dataset, protocol=meta["protocol"], records=records)
        for _, r in split.meta_test.iterrows():
            rows.append({"dataset": dataset, "index": int(r["index"]), "question": r["question"], "reference": list(r["reference"])})
    return rows


def run_question(q: dict, service: CalibrationService, kb: KnowledgeBase, generate=generate_answer) -> dict:
    rec = dict(q, answer=None, raw_confidence=None, calibrated_probability=None, threshold=None, error=None)
    try:
        gen = generate(base_url=BASE_URL, model=MODEL, question=q["question"])
        rec["answer"], rec["raw_confidence"] = gen.answer, gen.raw_confidence
        cal = service.calibrate(question=q["question"], answer=gen.answer, confidence=gen.raw_confidence, model=MODEL, domain=q["dataset"])
        rec.update(calibrated_probability=cal["calibrated_probability"], threshold=cal["threshold"], status=cal["status"], reason=cal["reason"])
    except Exception as exc:  # noqa: BLE001 - record and keep going; counted as a non-answer
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["correct_closed"] = has_answer(q["reference"], rec["answer"] or "")
    hits = kb.search(q["question"])
    # The agent answers from the retrieved reference aliases; score them as its answer.
    rec["kb_found"] = bool(hits)
    rec["correct_retrieved"] = has_answer(q["reference"], ", ".join(a for h in hits for a in h["references"]))
    return rec


def generate_records(questions: list[dict], generate=generate_answer) -> list[dict]:
    service, kb = CalibrationService(), KnowledgeBase()
    out = []
    for i, q in enumerate(questions, 1):
        out.append(run_question(q, service, kb, generate))
        if i % 100 == 0:
            print(f"{i}/{len(questions)}", flush=True)
    return out


# --- metrics (CPU) -----------------------------------------------------------------
def raw_thresholds() -> dict[str, float]:
    """Raw-c threshold per dataset: P2's Prob baseline, tuned like the calibrator's."""
    p2 = json.loads(P2_RESULTS.read_text(encoding="utf-8"))
    return {d: p2[f"{MODEL}/{d}"]["Prob"]["threshold"] for d in DATASETS}


def decide(rec: dict, mode: str, raw_threshold: float) -> str:
    if rec["raw_confidence"] is None:  # generation failed
        return "unknown"
    decision, _ = DecisionPolicy.decide(
        mode=mode, calibrated_probability=rec["calibrated_probability"], threshold=rec["threshold"],
        raw_confidence=rec["raw_confidence"], raw_threshold=raw_threshold,
    )
    return decision


def config_metrics(recs: list[dict], mode: str, retrieval: str, raw_threshold: float) -> dict:
    n = len(recs)
    answered = [decide(r, mode, raw_threshold) == "answer" for r in recs]
    correct = sum(
        r["correct_closed"] if a else (r["correct_retrieved"] if retrieval == "on" else 0)
        for r, a in zip(recs, answered)
    )
    non_answer = n - sum(answered)
    return {
        "n": n,
        "accuracy": correct / n,
        "refusal_rate": non_answer / n if retrieval == "off" else None,
        "retrieval_rate": non_answer / n if retrieval == "on" else None,
    }


def curve_scores(recs: list[dict]) -> dict[str, list[float]]:
    # Questions the calibrator cannot score (unknown) are ranked last.
    return {
        "raw c": [r["raw_confidence"] if r["raw_confidence"] is not None else -1.0 for r in recs],
        "calibrated": [r["calibrated_probability"] if r["calibrated_probability"] is not None else -1.0 for r in recs],
    }


def write_report(records: list[dict]) -> None:
    thresholds = raw_thresholds()
    results, lines = {}, [f"# End-to-end ablation, {MODEL}, P2 balanced test split", ""]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        recs = [r for r in records if r["dataset"] == dataset]
        n_err = sum(r["error"] is not None for r in recs)
        lines += [f"## {dataset} (n = {len(recs)}, generation errors = {n_err}, raw threshold = {thresholds[dataset]:.4f})", "",
                  "| Decision | Retrieval | Accuracy | Refusal rate | Retrieval rate |", "|---|---|---|---|---|"]
        results[dataset] = {}
        for mode, retrieval in CONFIGS:
            m = config_metrics(recs, mode, retrieval, thresholds[dataset])
            results[dataset][f"{mode}/retrieval_{retrieval}"] = m
            fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(f"| {mode} | {retrieval} | {fmt(m['accuracy'])} | {fmt(m['refusal_rate'])} | {fmt(m['retrieval_rate'])} |")

        y = [r["correct_closed"] for r in recs]
        lines += ["", "Coverage-accuracy (accuracy of directly answered questions):", "",
                  "| Score | " + " | ".join(f"cov {c:.1f}" for c in REPORT_COVERAGES) + " |", "|---|" + "---|" * len(REPORT_COVERAGES)]
        for name, s in curve_scores(recs).items():
            cov, acc = coverage_accuracy(y, s)
            ax.plot(cov, acc, label=name)
            lines.append(f"| {name} | " + " | ".join(f"{accuracy_at(cov, acc, c):.3f}" for c in REPORT_COVERAGES) + " |")
            results[dataset][f"curve/{name}"] = {f"{c:.1f}": accuracy_at(cov, acc, c) for c in REPORT_COVERAGES}
        lines.append("")
        ax.set_title(f"{dataset} (n={len(recs)})")
        ax.set_xlabel("coverage (fraction answered directly)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("accuracy on answered questions")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"End-to-end coverage-accuracy, {MODEL} via vLLM")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ablation_curves.png", dpi=150)
    (OUTPUT / "ablation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUTPUT / "ablation_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {OUTPUT}/ablation_metrics.md, ablation_metrics.json, ablation_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics-only", action="store_true", help="skip generation, reuse saved records")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        records = generate_records(test_questions())
        RECORDS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    records = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines()]
    write_report(records)


if __name__ == "__main__":
    main()
