from types import SimpleNamespace

import pytest

from evaluation.run_ablation import config_metrics, generate_records, has_answer


def _snapshot_or_skip():
    from calibrator.data_prep import load_model_outputs

    try:
        return load_model_outputs(models=["qwen2.5-7b"])
    except FileNotFoundError:
        pytest.skip("paper snapshot not available")


def test_has_answer_matches_archived_correct_labels():
    records = _snapshot_or_skip()
    rows = records.to_dict("records") if hasattr(records, "to_dict") else records
    mismatches = [r for r in rows if has_answer(list(r["reference"]), r["response"] or "") != r["correct"]]
    assert len(rows) == 26430
    assert mismatches == []


def test_config_metrics_hand_example():
    # raw threshold 0.5: q1 answered (right), q2 answered (wrong), q3 non-answer
    recs = [
        {"raw_confidence": 0.9, "calibrated_probability": 0.9, "threshold": 0.5, "correct_closed": 1, "correct_retrieved": 1},
        {"raw_confidence": 0.8, "calibrated_probability": 0.2, "threshold": 0.5, "correct_closed": 0, "correct_retrieved": 1},
        {"raw_confidence": 0.1, "calibrated_probability": 0.1, "threshold": 0.5, "correct_closed": 0, "correct_retrieved": 1},
    ]
    assert config_metrics(recs, "raw", "off", 0.5) == {"n": 3, "accuracy": 1 / 3, "refusal_rate": 1 / 3, "retrieval_rate": None}
    assert config_metrics(recs, "raw", "on", 0.5) == {"n": 3, "accuracy": 2 / 3, "refusal_rate": None, "retrieval_rate": 1 / 3}
    # calibrated sends the overconfident wrong q2 to retrieval
    assert config_metrics(recs, "calibrated", "on", 0.5) == {"n": 3, "accuracy": 1.0, "refusal_rate": None, "retrieval_rate": 2 / 3}


def test_pipeline_on_snapshot_answers_reproduces_p2_alignment():
    """Feed archived answers instead of vLLM; the calibrated decisions must match P2's test Alignment."""
    import json

    from calibrator.data_prep import load_split
    from evaluation.run_ablation import ARTIFACTS, decide

    records = _snapshot_or_skip()
    split = load_split("qwen2.5-7b", "movies", protocol="balanced", records=records)
    snap = {r["question"]: r for _, r in split.meta_test.iterrows()}
    questions = [{"dataset": "movies", "index": int(r["index"]), "question": q, "reference": list(r["reference"])} for q, r in snap.items()]

    def fake_generate(*, base_url, model, question):
        return SimpleNamespace(answer=snap[question]["response"], raw_confidence=float(snap[question]["confidence"]))

    recs = generate_records(questions, generate=fake_generate)
    assert all(r["error"] is None for r in recs)
    assert [r["correct_closed"] for r in recs] == [int(snap[r["question"]]["correct"]) for r in recs]
    assert all(r["kb_found"] and r["correct_retrieved"] == 1 for r in recs)

    meta = json.loads((ARTIFACTS / "qwen2.5-7b__movies.json").read_text(encoding="utf-8"))
    alignment = sum((decide(r, "calibrated", 0.0) == "answer") == bool(r["correct_closed"]) for r in recs) / len(recs)
    # one known feature difference (P1 README: qwen2.5-7b movies #7717) may flip at most one decision
    assert abs(alignment - meta["test_metrics"]["alignment"]) <= 1 / len(recs) + 1e-12
