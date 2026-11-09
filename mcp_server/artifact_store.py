from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = (
    PROJECT_ROOT / "calibrator" / "training" / "artifacts"
)

SAFE_NAME = re.compile(r"^[a-zA-Z0-9._-]+$")


class ArtifactError(RuntimeError):
    """Base exception for deployable calibrator artifacts."""


class ArtifactNotFoundError(ArtifactError):
    """Requested model/domain artifact does not exist."""


class ArtifactValidationError(ArtifactError):
    """Artifact exists but is malformed or inconsistent."""


@dataclass(frozen=True)
class LoadedArtifact:
    model: str
    domain: str
    features: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    threshold: float
    predict_proba: Callable[[np.ndarray], np.ndarray]


def _load_numpy_predictor(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load batch-normalization-folded MLP weights without importing torch."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            n_hidden = int(np.asarray(archive["n_hidden"]).item())

            hidden = [
                (
                    np.asarray(archive[f"W{i}"], dtype=np.float64),
                    np.asarray(archive[f"b{i}"], dtype=np.float64),
                )
                for i in range(n_hidden)
            ]

            output_weight = np.asarray(
                archive["W_out"], dtype=np.float64
            )
            output_bias = np.asarray(
                archive["b_out"], dtype=np.float64
            )
    except Exception as exc:
        raise ArtifactValidationError(
            f"Unable to load NumPy weights from {path}: {exc}"
        ) from exc

    if not hidden:
        raise ArtifactValidationError(f"No hidden layers found in {path}")

    def predict_proba(values: np.ndarray) -> np.ndarray:
        activation = np.asarray(values, dtype=np.float64)

        if activation.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got {activation.shape}"
            )

        for weight, bias in hidden:
            activation = activation @ weight.T + bias
            activation = np.maximum(activation, 0.0)

        logits = (
            activation @ output_weight.T + output_bias
        ).reshape(-1)

        # Numerically stable enough for these small calibration networks.
        logits = np.clip(logits, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-logits))

    return predict_proba


class ArtifactStore:
    def __init__(
        self,
        artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._cache: dict[tuple[str, str], LoadedArtifact] = {}

    @staticmethod
    def _validate_name(name: str, field: str) -> str:
        normalized = name.strip().lower()

        if not normalized or not SAFE_NAME.fullmatch(normalized):
            raise ValueError(f"Invalid {field}: {name!r}")

        return normalized

    def _load(self, model: str, domain: str) -> LoadedArtifact:
        model = self._validate_name(model, "model")
        domain = self._validate_name(domain, "domain")

        cache_key = (model, domain)
        if cache_key in self._cache:
            return self._cache[cache_key]

        stem = f"{model}__{domain}"
        metadata_path = self.artifacts_dir / f"{stem}.json"
        weights_path = self.artifacts_dir / f"{stem}.npz"

        missing = [
            str(path)
            for path in (metadata_path, weights_path)
            if not path.is_file()
        ]
        if missing:
            raise ArtifactNotFoundError(
                f"Missing artifact files for {model}/{domain}: {missing}"
            )

        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ArtifactValidationError(
                f"Unable to read {metadata_path}: {exc}"
            ) from exc

        if metadata.get("model") != model:
            raise ArtifactValidationError(
                f"Model mismatch in {metadata_path}: "
                f"{metadata.get('model')!r} != {model!r}"
            )

        if metadata.get("dataset") != domain:
            raise ArtifactValidationError(
                f"Domain mismatch in {metadata_path}: "
                f"{metadata.get('dataset')!r} != {domain!r}"
            )

        features = tuple(metadata.get("features", []))
        if not features or len(set(features)) != len(features):
            raise ArtifactValidationError(
                f"Invalid feature list in {metadata_path}: {features}"
            )

        scaler_mean = np.asarray(
            metadata.get("scaler_mean"), dtype=np.float64
        )
        scaler_scale = np.asarray(
            metadata.get("scaler_scale"), dtype=np.float64
        )

        expected_shape = (len(features),)
        if scaler_mean.shape != expected_shape:
            raise ArtifactValidationError(
                f"Invalid scaler_mean shape: {scaler_mean.shape}"
            )

        if scaler_scale.shape != expected_shape:
            raise ArtifactValidationError(
                f"Invalid scaler_scale shape: {scaler_scale.shape}"
            )

        if not np.all(np.isfinite(scaler_mean)):
            raise ArtifactValidationError("scaler_mean is not finite")

        if not np.all(np.isfinite(scaler_scale)):
            raise ArtifactValidationError("scaler_scale is not finite")

        if np.any(scaler_scale <= 0):
            raise ArtifactValidationError(
                "scaler_scale must contain positive values"
            )

        threshold = float(metadata.get("threshold"))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ArtifactValidationError(
                f"Invalid threshold: {threshold}"
            )

        artifact = LoadedArtifact(
            model=model,
            domain=domain,
            features=features,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            threshold=threshold,
            predict_proba=_load_numpy_predictor(weights_path),
        )

        # Fail during startup/load rather than on the first real request.
        probe = artifact.predict_proba(
            np.zeros((1, len(features)), dtype=np.float64)
        )
        if probe.shape != (1,) or not np.all(np.isfinite(probe)):
            raise ArtifactValidationError(
                f"Invalid predictor output for {stem}: {probe}"
            )

        self._cache[cache_key] = artifact
        return artifact

    def predict(
        self,
        *,
        model: str,
        domain: str,
        features: Mapping[str, float],
    ) -> dict:
        artifact = self._load(model, domain)

        missing = [
            name for name in artifact.features if name not in features
        ]
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        raw = np.asarray(
            [features[name] for name in artifact.features],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(raw)):
            raise ValueError("Input features must all be finite")

        confidence_index = (
            artifact.features.index("confidence")
            if "confidence" in artifact.features
            else None
        )
        if confidence_index is not None:
            confidence = float(raw[confidence_index])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"confidence must be in [0, 1], got {confidence}"
                )

        scaled = (
            raw - artifact.scaler_mean
        ) / artifact.scaler_scale

        probability = float(
            artifact.predict_proba(scaled.reshape(1, -1))[0]
        )

        if not math.isfinite(probability):
            raise ArtifactValidationError(
                "Calibrator returned a non-finite probability"
            )

        if not 0.0 <= probability <= 1.0:
            raise ArtifactValidationError(
                f"Calibrator returned invalid probability {probability}"
            )

        decision = (
            "answer"
            if probability >= artifact.threshold
            else "retrieve"
        )

        return {
            "model": artifact.model,
            "domain": artifact.domain,
            "artifact": f"{artifact.model}__{artifact.domain}",
            "feature_order": list(artifact.features),
            "raw_features": raw.tolist(),
            "scaled_features": scaled.tolist(),
            "calibrated_probability": probability,
            "threshold": artifact.threshold,
            "decision": decision,
        }