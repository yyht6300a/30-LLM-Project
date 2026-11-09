from __future__ import annotations

import math

import pytest

from mcp_server.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
)


@pytest.fixture
def store() -> ArtifactStore:
    return ArtifactStore()


def test_qwen_movies_golden_prediction(
    store: ArtifactStore,
) -> None:
    result = store.predict(
        model="qwen2.5-7b",
        domain="movies",
        features={
            "confidence": 0.8954265,
            "qpop": 60,
            "gene_pop": 15,
            "gene_coo": 9,
        },
    )

    assert result["artifact"] == "qwen2.5-7b__movies"
    assert result["feature_order"] == [
        "confidence",
        "qpop",
        "gene_pop",
        "gene_coo",
    ]
    assert result["calibrated_probability"] == pytest.approx(
        0.9668865976,
        abs=1e-6,
    )
    assert result["threshold"] == pytest.approx(
        0.3449274898,
        abs=1e-7,
    )
    assert result["decision"] == "answer"


def test_unknown_model_is_rejected(
    store: ArtifactStore,
) -> None:
    with pytest.raises(ArtifactNotFoundError):
        store.predict(
            model="not-a-real-model",
            domain="movies",
            features={
                "confidence": 0.8,
                "qpop": 1,
                "gene_pop": 1,
                "gene_coo": 1,
            },
        )


def test_missing_feature_is_rejected(
    store: ArtifactStore,
) -> None:
    with pytest.raises(ValueError, match="Missing input features"):
        store.predict(
            model="qwen2.5-7b",
            domain="movies",
            features={
                "confidence": 0.8,
                "qpop": 1,
                "gene_pop": 1,
            },
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_confidence_is_rejected(
    store: ArtifactStore,
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        store.predict(
            model="qwen2.5-7b",
            domain="movies",
            features={
                "confidence": confidence,
                "qpop": 1,
                "gene_pop": 1,
                "gene_coo": 1,
            },
        )


def test_result_is_finite(
    store: ArtifactStore,
) -> None:
    result = store.predict(
        model="qwen2.5-7b",
        domain="songs",
        features={
            "confidence": 0.8,
            "qpop": 10,
            "gene_pop": 20,
            "gene_coo": 3,
        },
    )

    probability = result["calibrated_probability"]
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0