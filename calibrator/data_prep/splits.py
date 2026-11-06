"""Feature matrices and reproducible train/val/test splits for calibrator training.

Protocols (all per (model, dataset), ``random_state=42``):

* ``balanced`` -- paper main table (``calibration_experiment.py``): undersample
  the retained pool to 1:1 if its accuracy is outside [0.25, 0.75], then a 50/50
  stratified split.
* ``balanced_train`` -- Setting 2 (``setting2_natural_test.py``): 50/50 split of
  the natural pool, then balance only the training half; test stays natural.
* ``natural`` -- Setting 3 / appendix (``setting3_natural_train_test.py``):
  50/50 split, no balancing.

In every protocol 10% of the training half is held out (stratified) as the
validation set, exactly as the official ``run_mlp`` does for early stopping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .loader import apply_paper_filters, load_model_outputs, normalize_dataset, normalize_model

FEATURE_COLUMNS = ["confidence", "qpop", "gene_pop", "gene_coo"]
TARGET_COLUMN = "correct"

SEED = 42
TEST_SIZE = 0.5
VAL_SIZE = 0.1
BALANCE_RANGE = (0.25, 0.75)
PROTOCOLS = ("balanced", "balanced_train", "natural")

# How the official calibration code treats an unavailable value of each usable
# feature: "zero" = filled with 0; "drop" = record removed from that experiment
# (after balancing, right before the 50/50 split). Ground-truth fields
# (gt_pop, gt_coo, ...) are deliberately absent: they leak the label.
OFFICIAL_MISSING = {
    "confidence": "drop",
    "qpop": "zero",
    "gene_pop": "zero",
    "gene_coo": "zero",
    "llm_qpop": "zero",
    "llm_gene_pop": "zero",
    "llm_coo": "zero",
    "sc_confidence": "drop",
    "vc_confidence": "drop",
}


def feature_available(df: pd.DataFrame, feature: str) -> pd.Series:
    """True where the upstream value exists. A found entity with 0 sitelinks is available."""
    if feature == "gene_pop":
        return df["gene_pop_found"].astype(bool)
    return df[feature].notna()


def check_features(features) -> list[str]:
    features = list(features)
    unknown = [f for f in features if f not in OFFICIAL_MISSING]
    if unknown or not features or len(set(features)) != len(features):
        raise ValueError(f"Invalid features {features}; choose distinct names from {list(OFFICIAL_MISSING)}")
    return features


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Zero-fill zero-policy features; drop-policy gaps stay NaN."""
    columns = []
    for feature in features:
        values = df[feature].astype("Float64")
        if OFFICIAL_MISSING[feature] == "zero":
            values = values.fillna(0)
        columns.append(values.to_numpy(dtype=float, na_value=np.nan))
    return np.column_stack(columns) if len(df) else np.empty((0, len(features)))


def build_features(df: pd.DataFrame, features=FEATURE_COLUMNS) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unscaled) ``X`` in the given column order and binary ``y``.

    Zero-policy features are filled with 0; drop-policy features must already
    be complete (``load_split`` removes those records).
    """
    features = check_features(features)
    X = _feature_matrix(df, features)
    if np.isnan(X).any():
        bad = [f for f, col in zip(features, X.T) if np.isnan(col).any()]
        raise ValueError(f"{bad} have missing values; drop those records first")
    return X, df[TARGET_COLUMN].to_numpy(dtype=int)


def needs_balancing(y: np.ndarray) -> bool:
    ratio = float(np.mean(y))
    return ratio < BALANCE_RANGE[0] or ratio > BALANCE_RANGE[1]


def balance_indices(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Positions of a 1:1 undersample, drawn exactly like the official ``balance_samples``."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_min = min(len(pos), len(neg))
    chosen = np.concatenate([
        pos[rng.choice(len(pos), n_min, replace=False)],
        neg[rng.choice(len(neg), n_min, replace=False)],
    ])
    rng.shuffle(chosen)
    return chosen


def split_indices(
    y: np.ndarray,
    protocol: str = "balanced",
    keep: np.ndarray | None = None,
    seed: int = SEED,
) -> dict:
    """Return disjoint ``train``/``val``/``test`` positions into ``y`` plus balancing info.

    ``keep`` marks records usable for the requested features (official
    drop-policy); it is applied after whole-pool balancing, as upstream does.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    y = np.asarray(y)
    keep = np.ones(len(y), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)

    pool = np.arange(len(y))
    balanced = protocol == "balanced" and needs_balancing(y)
    if balanced:
        pool = pool[balance_indices(y, seed)]
    n_pool = len(pool)
    pool = pool[keep[pool]]
    n_dropped = n_pool - len(pool)

    train_half, test = train_test_split(pool, test_size=TEST_SIZE, random_state=seed, stratify=y[pool])
    if protocol == "balanced_train" and needs_balancing(y[train_half]):
        train_half = train_half[balance_indices(y[train_half], seed)]
        balanced = True
    train, val = train_test_split(train_half, test_size=VAL_SIZE, random_state=seed, stratify=y[train_half])
    return {"train": train, "val": val, "test": test, "balanced": balanced, "n_dropped_missing": n_dropped}


@dataclass(frozen=True)
class CalibrationSplit:
    """Aligned arrays and metadata for one (model, dataset, protocol)."""

    model: str
    dataset: str
    protocol: str
    features: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    stats: dict = field(default_factory=dict)

    @property
    def X_trainval(self) -> np.ndarray:
        """Full 50% training half. Upstream fits StandardScaler and searches thresholds on it."""
        return np.concatenate([self.X_train, self.X_val])

    @property
    def y_trainval(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val])


def _class_counts(y: np.ndarray) -> dict:
    return {"n": int(len(y)), "n_correct": int(y.sum()), "n_wrong": int(len(y) - y.sum()),
            "accuracy": round(float(y.mean()), 4) if len(y) else None}


def load_split(
    model: str,
    dataset: str,
    protocol: str = "balanced",
    features=FEATURE_COLUMNS,
    missing: str = "official",
    seed: int = SEED,
    records: pd.DataFrame | None = None,
    repo_dir: str | os.PathLike | None = None,
) -> CalibrationSplit:
    """Build the train/val/test split P2 trains calibrators on.

    ``missing="official"`` reproduces the paper (popularity lookups that failed
    count as 0). ``missing="drop"`` instead excludes, before balancing, every
    record whose selected features are unavailable (e.g. ``gene_pop_found`` is
    false) -- a sensitivity setting, not the paper protocol.

    Pass ``records`` (from ``load_model_outputs``) to avoid re-reading archives.
    """
    model, dataset = normalize_model(model), normalize_dataset(dataset)
    features = check_features(features)
    if missing not in ("official", "drop"):
        raise ValueError("missing must be 'official' or 'drop'")

    if records is None:
        records = load_model_outputs([model], [dataset], repo_dir=repo_dir)
    subset = records[(records["model"] == model) & (records["dataset"] == dataset)]
    if subset.empty:
        raise ValueError(f"No records for {model}/{dataset}")
    retained, report = apply_paper_filters(subset)
    excluded = {k: int(v) for k, v in report.iloc[0].items()
                if k not in ("model", "dataset", "n_total", "n_retained", "n_correct", "accuracy")}

    available = pd.DataFrame({f: feature_available(retained, f) for f in features})
    if missing == "drop":
        complete = available.all(axis=1)
        excluded["unavailable_feature"] = int((~complete).sum())
        retained, available = retained[complete].reset_index(drop=True), available[complete].reset_index(drop=True)
    zero_filled = {f: int((~available[f]).sum()) for f in features if OFFICIAL_MISSING[f] == "zero"}

    keep = np.ones(len(retained), dtype=bool)
    for f in features:
        if OFFICIAL_MISSING[f] == "drop":
            keep &= available[f].to_numpy()

    X, y = _feature_matrix(retained, features), retained[TARGET_COLUMN].to_numpy(dtype=int)
    parts = split_indices(y, protocol, keep=keep, seed=seed)

    meta = retained.join(available.add_suffix("_available"))
    take = {name: parts[name] for name in ("train", "val", "test")}
    stats = {
        "model": model, "dataset": dataset, "protocol": protocol, "features": list(features),
        "missing": missing, "seed": seed,
        "n_total": int(len(subset)), "excluded": excluded,
        "retained": _class_counts(y),
        "balanced": parts["balanced"],
        "n_dropped_missing_after_balancing": parts["n_dropped_missing"],
        "zero_filled_unavailable": zero_filled,
        **{name: _class_counts(y[idx]) for name, idx in take.items()},
    }
    return CalibrationSplit(
        model=model, dataset=dataset, protocol=protocol, features=tuple(features),
        X_train=X[take["train"]], y_train=y[take["train"]],
        X_val=X[take["val"]], y_val=y[take["val"]],
        X_test=X[take["test"]], y_test=y[take["test"]],
        meta_train=meta.iloc[take["train"]].reset_index(drop=True),
        meta_val=meta.iloc[take["val"]].reset_index(drop=True),
        meta_test=meta.iloc[take["test"]].reset_index(drop=True),
        stats=stats,
    )
