from __future__ import annotations

import math

import pytest

from mcp_server.service import CalibrationService


@pytest.fixture(scope="module")
def service() -> CalibrationService:
    return CalibrationService()


def test_full_calibration_path(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie The Intouchables"
        ),
        answer="Olivier Nakache",
        confidence=0.8954265,
        model="qwen2.5-7b",
        domain="movies",
    )

    assert result["status"] == "ok"
    assert result["decision"] == "answer"
    assert result["reason"] is None
    assert result["feature_coverage"] == {
        "qpop": True,
        "gene_pop": True,
        "gene_coo": True,
    }
    assert result["feature_values"] == {
        "confidence": 0.8954265,
        "qpop": 60.0,
        "gene_pop": 15.0,
        "gene_coo": 9.0,
    }
    assert result["calibrated_probability"] == pytest.approx(
        0.9668865976,
        abs=1e-6,
    )


def test_domain_can_be_inferred(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie The Intouchables"
        ),
        answer="Olivier Nakache",
        confidence=0.8954265,
        model="qwen2.5-7b",
    )

    assert result["domain"] == "movies"
    assert result["status"] == "ok"


def test_partial_coverage_is_zero_filled(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie The Intouchables"
        ),
        answer="Eric Toledano",
        confidence=0.9888770878,
        model="qwen2.5-7b",
        domain="movies",
    )

    assert result["status"] == "degraded"
    assert result["reason"] == "partial_feature_coverage"
    assert result["feature_coverage"]["gene_pop"] is False
    assert result["feature_values"]["gene_pop"] == 0.0
    assert result["calibrated_probability"] is not None
    assert math.isfinite(result["calibrated_probability"])
    assert result["decision"] in {"answer", "retrieve"}


def test_all_features_missing_returns_unknown(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie "
            "Definitely Not A Real Movie 12345"
        ),
        answer="Definitely Not A Real Director 67890",
        confidence=0.8,
        model="qwen2.5-7b",
        domain="movies",
    )

    assert result["status"] == "degraded"
    assert result["reason"] == "no_popularity_features"
    assert result["decision"] == "unknown"
    assert result["calibrated_probability"] is None
    assert result["fallback_score"] == 0.8


def test_unknown_model_does_not_crash(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie The Intouchables"
        ),
        answer="Olivier Nakache",
        confidence=0.8954265,
        model="unknown-model",
        domain="movies",
    )

    assert result["status"] == "degraded"
    assert result["reason"] == "artifact_not_found"
    assert result["decision"] == "unknown"


def test_unsupported_domain_returns_unknown(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question="What is the capital of France?",
        answer="Paris",
        confidence=0.9,
        model="qwen2.5-7b",
        domain="geography",
    )

    assert result["status"] == "degraded"
    assert result["reason"] == "domain_not_supported"
    assert result["decision"] == "unknown"


def test_empty_answer_returns_unknown(
    service: CalibrationService,
) -> None:
    result = service.calibrate(
        question=(
            "Who is the director of the movie The Intouchables"
        ),
        answer="",
        confidence=0.7,
        model="qwen2.5-7b",
        domain="movies",
    )

    assert result["reason"] == "empty_answer"
    assert result["decision"] == "unknown"


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_invalid_confidence_is_rejected(
    service: CalibrationService,
    confidence,
) -> None:
    with pytest.raises(ValueError):
        service.calibrate(
            question=(
                "Who is the director of the movie The Intouchables"
            ),
            answer="Olivier Nakache",
            confidence=confidence,
            model="qwen2.5-7b",
            domain="movies",
        )