"""Tests for calibrator.data_prep.

Integration tests need the paper repository (see calibrator/README.md) and are
skipped when its archives cannot be found.
"""

import gzip
import json
import math

import numpy as np
import pandas as pd
import pytest

from calibrator.data_prep import (
    DATASET_SIZES,
    FEATURE_COLUMNS,
    MODELS,
    PROTOCOLS,
    apply_paper_filters,
    balance_indices,
    build_features,
    exclusion_reasons,
    feature_available,
    load_model_outputs,
    load_split,
    resolve_archive_dir,
    split_indices,
)
from calibrator.data_prep.loader import FIELDS, validate_records

try:
    ARCHIVE_DIR = resolve_archive_dir()
except FileNotFoundError:
    ARCHIVE_DIR = None
needs_data = pytest.mark.skipif(ARCHIVE_DIR is None, reason="paper archives not found")

# Retained sample counts from the official code/analysis_correlation/calibration_results.json
# ("conf+all_ext_mlp": natural_n_used if balanced else n_used) and the balanced pool sizes.
OFFICIAL_RETAINED = {
    "movies": {"llama3-8b": 9529, "qwen2-7b": 9582, "gpt-3.5-turbo": 9597,
               "qwen2.5-7b": 9578, "qwen2.5-14b": 9557, "qwen2.5-32b": 9565},
    "songs": {"llama3-8b": 1284, "qwen2-7b": 1460, "gpt-3.5-turbo": 1536,
              "qwen2.5-7b": 1518, "qwen2.5-14b": 1498, "qwen2.5-32b": 1557},
    "basketball": {m: 13300 for m in MODELS},
}
OFFICIAL_BALANCED = {
    ("movies", "gpt-3.5-turbo"): 1656, ("movies", "qwen2.5-7b"): 3498,
    ("songs", "qwen2-7b"): 648, ("songs", "qwen2.5-7b"): 296,
    ("songs", "qwen2.5-14b"): 602, ("songs", "qwen2.5-32b"): 484,
    ("basketball", "llama3-8b"): 3542, ("basketball", "qwen2-7b"): 2610,
    ("basketball", "qwen2.5-7b"): 1126, ("basketball", "qwen2.5-14b"): 1954,
    ("basketball", "qwen2.5-32b"): 2066,
}
SETTING3_NAMES = {"llama3-8b": "llama8b", "qwen2-7b": "qwen2", "gpt-3.5-turbo": "chatgpt",
                  "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-32b": "Qwen2.5-32B"}


@pytest.fixture(scope="module")
def records():
    return load_model_outputs()


@pytest.fixture(scope="module")
def qwen_movies(records):
    return {p: load_split("Qwen2.5-7B", "Movies", p, records=records) for p in PROTOCOLS}


# ── Synthetic unit tests (no data needed) ────────────────────────────────────

def make_row(dataset="movies", index=0, **overrides):
    row = dict.fromkeys(FIELDS)
    row.update(dataset=dataset, index=index, question="Q", reference=["A"], response="A",
               correct=1, confidence=0.9, qpop=5, gt_pop=3, gt_pop_found=True,
               gene_pop=3, gene_pop_found=True, gene_coo=2, q_occ=10, gene_occ=10)
    row.update(overrides)
    return row


def test_validation_rejects_truncated_archive(tmp_path):
    rows = [make_row(index=i) for i in range(3)]
    with pytest.raises(ValueError, match="expected 26430"):
        validate_records(rows, "toy")
    path = tmp_path / "data" / "model_outputs" / "qwen2.5-7b.jsonl.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)
    assert resolve_archive_dir(tmp_path) == path.parent
    with pytest.raises(ValueError, match="Invalid archive"):
        load_model_outputs(["qwen2.5-7b"], repo_dir=tmp_path)
    assert len(load_model_outputs(["qwen2.5-7b"], repo_dir=tmp_path, validate=False)) == 3


def test_validation_rejects_missing_field_and_bad_values():
    rows = [make_row()]
    del rows[0]["gene_coo"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_records(rows, "toy")
    with pytest.raises(ValueError, match="correct must be 0 or 1"):
        validate_records([make_row(correct=2)], "toy")
    with pytest.raises(ValueError, match="gene_pop must be 0"):
        validate_records([make_row(gene_pop=4, gene_pop_found=False)], "toy")


def test_exclusion_rules_follow_official_order():
    df = pd.DataFrame([
        make_row(response=""),                                   # empty
        make_row(response="", q_occ=9999),                       # empty wins over noisy
        make_row(q_occ=6001),                                    # noisy question entity
        make_row(gene_occ=6001),                                 # noisy generated entity
        make_row(dataset="basketball", gene_occ=6001),           # basketball ignores gene_occ
        make_row(q_occ=None, gene_occ=None),                     # unavailable occ -> 0, kept
        make_row(q_occ=6000),                                    # threshold is strict
        make_row(confidence=None),                               # missing token probs
    ])
    assert exclusion_reasons(df).tolist() == [
        "empty_response", "empty_response", "noisy_entity", "noisy_entity",
        pd.NA, pd.NA, pd.NA, "missing_confidence"]


def test_missing_popularity_is_zero_filled_but_flagged():
    df = pd.DataFrame([
        make_row(gene_pop=0, gene_pop_found=False, gene_coo=None, llm_coo=None),  # unavailable
        make_row(gene_pop=0, gene_pop_found=True, gene_coo=0, llm_coo=7),         # genuine zeros
    ])
    df["gene_coo"] = df["gene_coo"].astype("Int64")
    assert feature_available(df, "gene_pop").tolist() == [False, True]
    assert feature_available(df, "gene_coo").tolist() == [False, True]
    X, y = build_features(df, ["gene_pop", "gene_coo", "llm_coo"])
    np.testing.assert_array_equal(X, [[0, 0, 0], [0, 0, 7]])
    np.testing.assert_array_equal(y, [1, 1])
    with pytest.raises(ValueError, match="missing values"):
        build_features(pd.DataFrame([make_row(vc_confidence=None)]), ["vc_confidence"])
    with pytest.raises(ValueError, match="Invalid features"):
        build_features(df, ["gt_coo"])  # ground-truth fields leak the label


def official_balance_samples(samples, seed=42):
    """Verbatim copy of balance_samples in the official calibration_experiment.py."""
    rng = np.random.default_rng(seed)
    pos = [s for s in samples if s["acc"] == 1]
    neg = [s for s in samples if s["acc"] == 0]
    n_min = min(len(pos), len(neg))
    pos_idx = rng.choice(len(pos), n_min, replace=False)
    neg_idx = rng.choice(len(neg), n_min, replace=False)
    balanced = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    rng.shuffle(balanced)
    return balanced


def test_balance_indices_matches_official_implementation():
    y = (np.random.default_rng(0).random(997) < 0.15).astype(int)
    expected = [s["id"] for s in official_balance_samples([{"id": i, "acc": v} for i, v in enumerate(y)])]
    chosen = balance_indices(y)
    assert chosen.tolist() == expected
    assert y[chosen].sum() * 2 == len(chosen)


def test_split_indices_protocols_on_synthetic_labels():
    y = (np.random.default_rng(1).random(2000) < 0.1).astype(int)
    for protocol in PROTOCOLS:
        parts = split_indices(y, protocol)
        sets = [set(parts[k]) for k in ("train", "val", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert parts["balanced"] == (protocol != "natural")
    assert y[split_indices(y, "natural")["test"]].mean() == pytest.approx(0.1, abs=0.02)
    keep = np.ones_like(y, dtype=bool)
    keep[:50] = False
    parts = split_indices(y, "natural", keep=keep)
    assert min(np.concatenate([parts[k] for k in ("train", "val", "test")])) >= 50
    with pytest.raises(ValueError):
        split_indices(y, "unknown")


# ── Integration tests on the official archives ───────────────────────────────

@needs_data
def test_all_six_archives_load_with_expected_counts(records):
    assert len(records) == 158_580
    assert set(records["model"]) == set(MODELS)
    counts = records.groupby(["model", "dataset"]).size()
    for model in MODELS:
        for dataset, size in DATASET_SIZES.items():
            assert counts[(model, dataset)] == size
    assert not records.duplicated(["model", "dataset", "index"]).any()
    assert list(records.columns[:3]) == ["model", "dataset", "index"]


@needs_data
def test_model_and_dataset_filtering_accepts_paper_names():
    df = load_model_outputs(["Qwen2.5-7B"], ["Movies"])
    assert len(df) == 10964
    assert set(df["model"]) == {"qwen2.5-7b"} and set(df["dataset"]) == {"movies"}
    assert set(load_model_outputs(["chatgpt"], ["songs"])["model"]) == {"gpt-3.5-turbo"}
    with pytest.raises(ValueError):
        load_model_outputs(["gpt-4"])


@needs_data
def test_missing_values_are_preserved_not_zeroed(records):
    assert records["gene_occ"].isna().sum() == 45            # qwen2.5-32b, null upstream
    assert records["confidence"].isna().sum() == 214         # qwen2-7b empty responses
    genuine_zero = records["gene_pop_found"] & (records["gene_pop"] == 0)
    assert genuine_zero.sum() > 0                            # found entity with 0 sitelinks
    assert (records.loc[~records["gene_pop_found"], "gene_pop"] == 0).all()
    assert str(records["gene_coo"].dtype) == "Int64"


@needs_data
def test_filter_report_accounts_for_every_record_and_matches_official(records):
    retained, report = apply_paper_filters(records)
    excluded = report[["empty_response", "noisy_entity", "missing_confidence"]].sum(axis=1)
    assert (report["n_total"] == excluded + report["n_retained"]).all()
    assert report["n_retained"].sum() == len(retained)
    for row in report.itertuples():
        assert row.n_retained == OFFICIAL_RETAINED[row.dataset][row.model], (row.model, row.dataset)


@needs_data
def test_balanced_pool_sizes_match_official(records):
    for dataset in DATASET_SIZES:
        for model in MODELS:
            split = load_split(model, dataset, "balanced", records=records)
            n = split.stats["retained"]["n"]
            pool = len(split.y_train) + len(split.y_val) + len(split.y_test)
            assert pool == OFFICIAL_BALANCED.get((dataset, model), n), (model, dataset)
            assert split.stats["balanced"] == ((dataset, model) in OFFICIAL_BALANCED)


@needs_data
@pytest.mark.skipif(ARCHIVE_DIR is None or not (ARCHIVE_DIR.parents[1] / "code/analysis_correlation/setting3_results.json").is_file(),
                    reason="official setting3_results.json not available")
def test_natural_split_matches_official_setting3(records):
    per_pair = json.loads((ARCHIVE_DIR.parents[1] / "code/analysis_correlation/setting3_results.json").read_text())["per_pair"]
    for dataset in DATASET_SIZES:
        for model in MODELS:
            ref = per_pair[f"{dataset}__{SETTING3_NAMES[model]}__raw"]
            split = load_split(model, dataset, "natural", records=records)
            assert len(split.y_trainval) == ref["n_train"] and len(split.y_test) == ref["n_test"]
            assert split.y_trainval.mean() == pytest.approx(ref["mean_acc_train"], abs=1e-12)
            assert split.y_test.mean() == pytest.approx(ref["mean_acc_test"], abs=1e-12)


@needs_data
def test_features_and_labels_align_with_metadata(qwen_movies):
    split = qwen_movies["balanced"]
    assert split.features == tuple(FEATURE_COLUMNS)
    for X, y, meta in [(split.X_train, split.y_train, split.meta_train),
                       (split.X_val, split.y_val, split.meta_val),
                       (split.X_test, split.y_test, split.meta_test)]:
        assert X.shape == (len(meta), 4) and X.dtype == float and y.dtype.kind == "i"
        np.testing.assert_array_equal(y, meta["correct"])
        np.testing.assert_array_equal(X[:, 0], meta["confidence"])
        np.testing.assert_array_equal(X[:, 1], meta["qpop"].astype(float))
        np.testing.assert_array_equal(X[:, 2], meta["gene_pop"].astype(float))  # raw, unscaled
        np.testing.assert_array_equal(X[:, 3], meta["gene_coo"].fillna(0).astype(float))
        assert set(np.unique(y)) <= {0, 1}
    assert (split.meta_train["gene_pop_available"] == split.meta_train["gene_pop_found"]).all()


@needs_data
def test_feature_subsets_keep_requested_order(records, qwen_movies):
    subset = load_split("qwen2.5-7b", "movies", features=["gene_coo", "confidence"], records=records)
    full = qwen_movies["balanced"]
    np.testing.assert_array_equal(subset.X_train, full.X_train[:, [3, 0]])
    np.testing.assert_array_equal(subset.y_test, full.y_test)


@needs_data
def test_splits_are_deterministic(records, qwen_movies):
    for protocol in PROTOCOLS:
        again = load_split("qwen2.5-7b", "movies", protocol, records=records)
        first = qwen_movies[protocol]
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
    other = load_split("qwen2.5-7b", "movies", seed=7, records=records)
    assert not other.meta_test["index"].equals(qwen_movies["balanced"].meta_test["index"])


@needs_data
def test_train_val_test_are_disjoint(qwen_movies):
    for protocol, split in qwen_movies.items():
        ids = [set(m["index"]) for m in (split.meta_train, split.meta_val, split.meta_test)]
        assert sum(map(len, ids)) == len(set.union(*ids)), protocol
        assert len(ids[1]) == math.ceil(0.1 * (len(ids[0]) + len(ids[1])))  # sklearn rounds up


@needs_data
def test_balancing_behaviour(records, qwen_movies):
    balanced, balanced_train, natural = (qwen_movies[p] for p in PROTOCOLS)
    # Qwen2.5-7B Movies accuracy 0.183 < 0.25 -> balanced to 1749 + 1749.
    assert balanced.stats["retained"] == {"n": 9578, "n_correct": 1749, "n_wrong": 7829, "accuracy": 0.1826}
    assert len(balanced.y_trainval) == len(balanced.y_test) == 1749
    assert balanced.y_trainval.sum() + balanced.y_test.sum() == 1749
    # Setting 2: same natural test half as Setting 3, balanced training half.
    np.testing.assert_array_equal(balanced_train.meta_test["index"], natural.meta_test["index"])
    assert balanced_train.y_trainval.mean() == pytest.approx(0.5, abs=0.001)
    assert natural.y_test.mean() == pytest.approx(0.1826, abs=0.001)
    assert not natural.stats["balanced"]
    # Accuracy inside [0.25, 0.75] -> no balancing, identical to natural.
    a = load_split("llama3-8b", "movies", "balanced", records=records)
    b = load_split("llama3-8b", "movies", "natural", records=records)
    assert not a.stats["balanced"]
    np.testing.assert_array_equal(a.meta_test["index"], b.meta_test["index"])


@needs_data
def test_official_drop_policy_for_verbalized_confidence(records):
    # 31 retained qwen2.5-14b movies rows lack vc_confidence; upstream drops them after balancing.
    split = load_split("qwen2.5-14b", "movies", "natural", features=["vc_confidence"], records=records)
    assert split.stats["n_dropped_missing_after_balancing"] == 31
    assert len(split.y_trainval) + len(split.y_test) == 9557 - 31
    assert not np.isnan(split.X_train).any()


@needs_data
def test_strict_missing_mode_drops_unavailable_popularity(records, qwen_movies):
    strict = load_split("qwen2.5-7b", "movies", "natural", missing="drop", records=records)
    assert strict.stats["excluded"]["unavailable_feature"] == 1337
    assert strict.stats["retained"]["n"] == 9578 - 1337
    assert strict.meta_train["gene_pop_found"].all()


# sha256 of the exact train/val/test indices; verified identical on Python 3.10 (numpy 2.2)
# and 3.13 (numpy 2.3). A change means numpy/sklearn RNG streams drifted -> splits differ.
GOLDEN_SPLIT_DIGESTS = {
    "balanced": "f1e52e8ff471cad8",
    "balanced_train": "079629fc584f2c09",
    "natural": "9a3733ed7a98c356",
}


@needs_data
def test_split_indices_match_golden_digest(qwen_movies):
    import hashlib

    for protocol, split in qwen_movies.items():
        key = ",".join(f"{name}:" + " ".join(map(str, meta["index"])) for name, meta in
                       [("train", split.meta_train), ("val", split.meta_val), ("test", split.meta_test)])
        assert hashlib.sha256(key.encode()).hexdigest()[:16] == GOLDEN_SPLIT_DIGESTS[protocol], protocol
