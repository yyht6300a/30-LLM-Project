"""Tests for calibrator.features (offline popularity lookup).

Integration tests need the paper repository with a materialized ``res/`` (see
calibrator/README.md) and are skipped otherwise.
"""

import json

import numpy as np
import pytest

from calibrator.data_prep import FEATURE_COLUMNS, exclusion_reasons, load_model_outputs, load_split
from calibrator.features import (
    PopularityLookup,
    clean_entity,
    compare_with_archives,
    infer_dataset,
    question_entity,
    resolve_repo_dir,
)

try:
    resolve_repo_dir()
    HAVE_DATA = True
except FileNotFoundError:
    HAVE_DATA = False
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="paper repository with res/ not found")

MOVIE_Q = "Who is the director of the movie "
SONG_Q = "Who is the performer of the song "
PLAYER_Q = "Where is the birthplace of the basketball player "


# ── Synthetic resources (no data needed) ─────────────────────────────────────

@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    (root / "data").mkdir()
    (root / "res").mkdir()
    qa = {
        "movies": [("Coma", 20), ("COMA", 16), ("Alien", 50), ("Moulin Rouge!", 60)],
        "songs": [("Hello", 30)],
        "basketball": [("Yao Ming", 80)],
    }
    prefixes = {"movies": MOVIE_Q, "songs": SONG_Q, "basketball": PLAYER_Q}
    for dataset, rows in qa.items():
        with open(root / "data" / f"{dataset}.jsonl", "w", encoding="utf-8") as f:
            for entity, pop in rows:
                f.write(json.dumps({"question": prefixes[dataset] + entity, "reference": ["x"],
                                    "popularity": pop, "question_entity": entity}) + "\n")
    suffix = "qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b"
    with open(root / "res" / f"gt_gene_entity_popularity_{suffix}.jsonl", "w", encoding="utf-8") as f:
        for entity, pop in [("Ridley Scott", {"popularity": 70}), ("Nobody", {"popularity": 0}),
                            ("Ghost", {"popularity": "No"}), ("Shanghai", 120), ("", {"popularity": 6})]:
            f.write(json.dumps({entity: pop}) + "\n")
    (root / "res" / f"cooccurrence_{suffix}.json").write_text(json.dumps(
        {"alien": {"ridley scott": 40, "nobody": 0}, "yao ming": {"shanghai": 9}, "coma": {}}))
    (root / "res" / f"single_occurrence_{suffix}.json").write_text(json.dumps(
        {"alien": 900, "ridley scott": 7000, "yao ming": 50, "shanghai": 9000, "hello": 6001}))
    return PopularityLookup(root)


def test_clean_entity_follows_official_rules():
    assert clean_entity("Ridley Scott (director)", "movies") == "Ridley Scott"
    assert clean_entity('"Ridley Scott."\n', "movies") == "Ridley Scott"
    assert clean_entity("Scott, Ridley", "movies") == "Scott"                     # <= 20 chars: comma split
    long = "Ridley Scott, Tony Scott and others"
    assert clean_entity(long, "movies") == long                                    # > 20 chars: kept
    assert clean_entity(long, "basketball") == "Ridley Scott"                     # basketball: always split
    assert clean_entity("Shanghai, China", "basketball") == "Shanghai"
    assert clean_entity("/>.", "movies") == "" and clean_entity(None, "songs") == ""


def test_question_parsing():
    assert infer_dataset(MOVIE_Q + "Alien") == "movies"
    assert infer_dataset(PLAYER_Q + "Yao Ming") == "basketball"
    assert infer_dataset("What is the capital of France?") is None
    assert question_entity(SONG_Q + "Hello", "songs") == "Hello"
    assert question_entity("Hello", "songs") == "Hello"


def test_lookup_found_values(toy):
    r = toy.lookup(MOVIE_Q + "Alien", "Ridley Scott")
    assert r["dataset"] == "movies" and r["question_entity"] == "Alien" and r["gene_entity"] == "Ridley Scott"
    assert (r["qpop"], r["gene_pop"], r["gene_coo"]) == (50, 70, 40)
    assert r["qpop_found"] and r["gene_pop_found"] and r["gene_coo_found"]
    assert (r["q_occ"], r["gene_occ"]) == (900, 7000)
    assert r["noisy_entity"]                                   # gene_occ > 6000 on movies
    b = toy.lookup(PLAYER_Q + "Yao Ming", "Shanghai, China")
    assert (b["qpop"], b["gene_pop"], b["gene_coo"]) == (80, 120, 9)
    assert not b["noisy_entity"]                               # basketball ignores gene_occ
    assert toy.lookup("Hello", "Adele", dataset="Songs")["noisy_entity"]   # q_occ > 6000


def test_lookup_trailing_punctuation(toy):
    # Agent-appended '?' must be resolved, but a title's own trailing '!'/'?'
    # must be preserved (e.g. "Moulin Rouge!").
    q = toy.lookup(MOVIE_Q + "Alien?", "Ridley Scott")
    assert q["question_entity"] == "Alien" and q["qpop_found"] and q["qpop"] == 50

    m = toy.lookup(MOVIE_Q + "Moulin Rouge!", "Baz Luhrmann")
    assert m["question_entity"] == "Moulin Rouge!" and m["qpop_found"] and m["qpop"] == 60

    m2 = toy.lookup(MOVIE_Q + "Moulin Rouge!?", "Baz Luhrmann")
    assert m2["question_entity"] == "Moulin Rouge!" and m2["qpop_found"] and m2["qpop"] == 60


def test_missing_values_differ_from_genuine_zeros(toy):
    zero = toy.lookup(MOVIE_Q + "Alien", "Nobody")
    assert zero["gene_pop"] == 0 and zero["gene_pop_found"]
    assert zero["gene_coo"] == 0 and zero["gene_coo_found"]
    absent = toy.lookup(MOVIE_Q + "Alien", "Somebody Else")
    assert absent["gene_pop"] is None and not absent["gene_pop_found"]
    assert absent["gene_coo"] is None and not absent["gene_coo_found"]
    assert absent["gene_occ"] is None
    no_label = toy.lookup(MOVIE_Q + "Alien", "Ghost")          # 'No' in the table = not found
    assert no_label["gene_pop"] is None and not no_label["gene_pop_found"]
    unknown_q = toy.lookup(MOVIE_Q + "Unknown Film", "Ridley Scott")
    assert unknown_q["qpop"] is None and not unknown_q["qpop_found"]
    assert unknown_q["gene_pop"] == 70 and not unknown_q["gene_coo_found"]


def test_empty_answer_has_no_entity(toy):
    # The archived table maps "" to 6 sitelinks (upstream artifact); an empty answer has no entity.
    for response in ("", None, "/>."):
        r = toy.lookup(MOVIE_Q + "Alien", response)
        assert r["gene_entity"] == "" and not r["gene_pop_found"] and not r["gene_coo_found"]


def test_matching_case_rules(toy):
    assert toy.lookup(MOVIE_Q + "Coma", "x")["qpop"] == 20      # qpop: exact question entity
    assert toy.lookup(MOVIE_Q + "COMA", "x")["qpop"] == 16
    assert not toy.lookup(MOVIE_Q + "coma", "x")["qpop_found"]
    assert not toy.lookup(MOVIE_Q + "Alien", "ridley scott")["gene_pop_found"]  # sitelinks: case-sensitive
    assert toy.lookup(MOVIE_Q + "ALIEN", "Ridley Scott")["gene_coo"] == 40      # coo: lowercased keys


def test_uncovered_questions_are_reported_not_guessed(toy):
    r = toy.lookup("What is the capital of France?", "Paris")
    assert r["dataset"] is None and not (r["qpop_found"] or r["gene_pop_found"] or r["gene_coo_found"])
    assert not r["noisy_entity"]
    with pytest.raises(ValueError):
        toy.lookup("Alien", "x", dataset="books")


def test_feature_vector_zero_fills_like_the_paper(toy):
    v = toy.feature_vector(MOVIE_Q + "Alien", "Somebody Else", confidence=0.8)
    np.testing.assert_array_equal(v, [0.8, 50, 0, 0])                # FEATURE_COLUMNS order
    v = toy.feature_vector(MOVIE_Q + "Alien", "Ridley Scott", 0.8, features=["gene_coo", "confidence"])
    np.testing.assert_array_equal(v, [40, 0.8])
    with pytest.raises(ValueError, match="Cannot look up"):
        toy.feature_vector(MOVIE_Q + "Alien", "x", 0.8, features=["llm_coo"])


def test_missing_resources_raise_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="materialize_compact_data"):
        resolve_repo_dir(tmp_path)


# ── Integration with the official archives ───────────────────────────────────

@pytest.fixture(scope="module")
def lookup():
    return PopularityLookup()


@pytest.fixture(scope="module")
def records():
    return load_model_outputs()


@needs_data
def test_official_readme_example(lookup):
    r = lookup.lookup("Who is the director of the movie The Intouchables", "Eric Toledano")
    assert (r["qpop"], r["gene_coo"], r["q_occ"], r["gene_occ"]) == (60, 2, 63, 2)
    assert r["gene_pop"] is None and not r["gene_pop_found"]   # archived as gene_pop=0, found=false


@needs_data
def test_lookup_matches_all_archived_features(lookup, records):
    summary, mismatches = compare_with_archives(lookup, records)
    assert summary["n_records"].sum() == 158_580 and len(summary) == 18
    for field in ("qpop", "gene_coo"):
        assert summary[f"{field}_match"].sum() == 158_580 - summary[f"{field}_zero_filled"].sum()
    assert summary["qpop_zero_filled"].sum() == 0
    assert summary[["qpop_mismatch", "gene_coo_mismatch", "q_occ_mismatch", "gene_occ_mismatch"]].sum().sum() == 0
    # Every remaining difference is the empty-entity artifact ("" -> 6 sitelinks upstream).
    empty = [clean_entity(r, d) == "" for r, d in zip(records["response"], records["dataset"])]
    assert len(mismatches) == sum(empty) == 215
    assert (mismatches["gene_pop"] == 6).all() and mismatches["gene_pop_found"].all()
    assert summary["gene_pop_mismatch"].sum() == summary["gene_pop_found_mismatch"].sum() == 215


@needs_data
def test_noisy_flag_agrees_with_data_prep_filter(lookup, records):
    subset = records[records["response"] != ""]
    noisy = [lookup.lookup(q, r, d)["noisy_entity"]
             for q, r, d in zip(subset["question"], subset["response"], subset["dataset"])]
    np.testing.assert_array_equal(noisy, (exclusion_reasons(subset) == "noisy_entity").to_numpy())


@needs_data
@pytest.mark.parametrize("dataset", ["movies", "songs", "basketball"])
def test_inference_features_equal_training_features(lookup, records, dataset):
    split = load_split("qwen2.5-7b", dataset, "natural", records=records)
    meta = split.meta_test
    X = np.array([lookup.feature_vector(q, r, c, d) for q, r, c, d in
                  zip(meta["question"], meta["response"], meta["confidence"], meta["dataset"])])
    differs = ~np.all(X == split.X_test, axis=1)
    # Only Qwen2.5-7B Movies #7717 (response "/>.", empty entity) may differ, and only in gene_pop.
    assert set(meta.loc[differs, "index"]) <= {7717}
    np.testing.assert_array_equal(X[:, [0, 1, 3]], split.X_test[:, [0, 1, 3]])
    assert list(FEATURE_COLUMNS) == ["confidence", "qpop", "gene_pop", "gene_coo"]
