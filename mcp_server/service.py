from __future__ import annotations

import math
from pathlib import Path

from calibrator.features.popularity import (
    PopularityLookup,
    infer_dataset,
)

from mcp_server.artifact_store import (
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactValidationError,
)


class CalibrationService:
    """Popularity lookup plus deployable calibrator inference."""

    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None = None,
        repo_dir: str | Path | None = None,
    ) -> None:
        self.artifacts = (
            ArtifactStore()
            if artifacts_dir is None
            else ArtifactStore(artifacts_dir)
        )
        self.lookup = PopularityLookup(repo_dir)

    @staticmethod
    def _validate_confidence(confidence: float) -> float:
        if isinstance(confidence, bool):
            raise ValueError("confidence must be a number, not bool")

        try:
            value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc

        if not math.isfinite(value):
            raise ValueError("confidence must be finite")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {value}"
            )

        return value

    @staticmethod
    def _unknown_result(
        *,
        model: str,
        domain: str | None,
        confidence: float,
        reason: str,
        warnings: list[str] | None = None,
        **extra,
    ) -> dict:
        result = {
            "status": "degraded",
            "model": model,
            "domain": domain,
            "raw_confidence": confidence,
            "fallback_score": confidence,
            "calibrated_probability": None,
            "threshold": None,
            "decision": "unknown",
            "artifact": None,
            "reason": reason,
            "warnings": warnings or [],
        }
        result.update(extra)
        return result

    def calibrate(
        self,
        *,
        question: str,
        answer: str,
        confidence: float,
        model: str,
        domain: str | None = None,
    ) -> dict:
        confidence = self._validate_confidence(confidence)

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        model = model.strip().lower()
        question = question.strip()
        answer = answer.strip() if isinstance(answer, str) else ""

        resolved_domain = (
            domain.strip().lower()
            if isinstance(domain, str) and domain.strip()
            else infer_dataset(question)
        )

        if resolved_domain is None:
            return self._unknown_result(
                model=model,
                domain=None,
                confidence=confidence,
                reason="domain_not_inferred",
            )

        if not answer:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="empty_answer",
            )

        try:
            lookup = self.lookup.lookup(
                question,
                answer,
                dataset=resolved_domain,
            )
        except ValueError as exc:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="domain_not_supported",
                warnings=[str(exc)],
            )

        coverage = {
            "qpop": bool(lookup["qpop_found"]),
            "gene_pop": bool(lookup["gene_pop_found"]),
            "gene_coo": bool(lookup["gene_coo_found"]),
        }

        feature_values = {
            "confidence": confidence,
            "qpop": (
                float(lookup["qpop"])
                if lookup["qpop"] is not None
                else 0.0
            ),
            "gene_pop": (
                float(lookup["gene_pop"])
                if lookup["gene_pop"] is not None
                else 0.0
            ),
            "gene_coo": (
                float(lookup["gene_coo"])
                if lookup["gene_coo"] is not None
                else 0.0
            ),
        }

        context = {
            "question_entity": lookup["question_entity"],
            "answer_entity": lookup["gene_entity"],
            "feature_values": feature_values,
            "feature_coverage": coverage,
            "noisy_entity": bool(lookup["noisy_entity"]),
        }

        if lookup["noisy_entity"]:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="noisy_entity",
                warnings=[
                    "Entity occurrence exceeds the paper filtering threshold"
                ],
                **context,
            )

        missing_features = [
            name for name, found in coverage.items() if not found
        ]

        if len(missing_features) == len(coverage):
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="no_popularity_features",
                warnings=[
                    "All popularity features are unavailable"
                ],
                **context,
            )

        try:
            prediction = self.artifacts.predict(
                model=model,
                domain=resolved_domain,
                features=feature_values,
            )
        except ArtifactNotFoundError as exc:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="artifact_not_found",
                warnings=[str(exc)],
                **context,
            )
        except ArtifactValidationError as exc:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="artifact_invalid",
                warnings=[str(exc)],
                **context,
            )
        except ArtifactError as exc:
            return self._unknown_result(
                model=model,
                domain=resolved_domain,
                confidence=confidence,
                reason="artifact_error",
                warnings=[str(exc)],
                **context,
            )

        status = "ok" if not missing_features else "degraded"
        warnings = (
            []
            if not missing_features
            else [
                "Missing popularity features were zero-filled "
                "according to the paper protocol: "
                + ", ".join(missing_features)
            ]
        )

        return {
            "status": status,
            "model": model,
            "domain": resolved_domain,
            "raw_confidence": confidence,
            "fallback_score": confidence,
            "calibrated_probability": prediction[
                "calibrated_probability"
            ],
            "threshold": prediction["threshold"],
            "decision": prediction["decision"],
            "artifact": prediction["artifact"],
            "reason": (
                None
                if status == "ok"
                else "partial_feature_coverage"
            ),
            "warnings": warnings,
            **context,
        }