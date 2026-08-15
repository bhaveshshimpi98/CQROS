"""CQROS Factor Orthogonalization combination redundancy filtering.

Purpose:
    Apply greedy, rank-preserving absolute-Pearson redundancy filtering to
    Factor Combination candidates using validation-window combination signals
    built from equal-weight member factor observations.

Responsibilities:
    - Validate orthogonalization correlation configuration
    - Build equal-weight combination signals from factor observation panels
    - Compute pairwise absolute Pearson correlations with overlap counts
    - Apply greedy acceptance/rejection in combination-rank order
    - Emit auditable redundancy decision rows
    - Remain free of persistence, CLI, and storage logic

Dependencies:
    ``numpy``, ``polars``, ``cqros.factor_orthogonalization.exceptions``, and
    ``cqros.factor_selection.redundancy`` for observation-source Protocol and
    Phase 3B default thresholds.

Public API:
    ``DEFAULT_MAX_COMBINATION_CORRELATION``, ``DEFAULT_MIN_CORRELATION_OVERLAP``,
    ``ORTHOGONALIZATION_METHOD``, ``ORTHOGONALIZATION_VERSION``,
    ``REASON_ACCEPTED``, ``REASON_REDUNDANT``, ``OrthogonalizationConfig``,
    ``require_max_combination_correlation``, ``require_min_correlation_overlap``,
    ``require_orthogonalization_config``, ``apply_greedy_combination_filter``
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import polars as pl

from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_selection.redundancy import DEFAULT_MAX_FACTOR_CORRELATION
from cqros.factor_selection.redundancy import (
    DEFAULT_MIN_CORRELATION_OVERLAP as _PHASE3B_MIN_CORRELATION_OVERLAP,
)

__all__ = [
    "DEFAULT_MAX_COMBINATION_CORRELATION",
    "DEFAULT_MIN_CORRELATION_OVERLAP",
    "ORTHOGONALIZATION_METHOD",
    "ORTHOGONALIZATION_VERSION",
    "REASON_ACCEPTED",
    "REASON_REDUNDANT",
    "OrthogonalizationConfig",
    "apply_greedy_combination_filter",
    "require_max_combination_correlation",
    "require_min_correlation_overlap",
    "require_orthogonalization_config",
]

_ERROR_MAX_CORR: Final[str] = "FORTH_MAX_COMBINATION_CORRELATION_INVALID"
_ERROR_MIN_OVERLAP: Final[str] = "FORTH_MIN_OVERLAP_INVALID"

# Mirror Phase 3B defaults explicitly for combination-unit orthogonalization.
DEFAULT_MAX_COMBINATION_CORRELATION: Final[float] = DEFAULT_MAX_FACTOR_CORRELATION
DEFAULT_MIN_CORRELATION_OVERLAP: Final[int] = _PHASE3B_MIN_CORRELATION_OVERLAP

ORTHOGONALIZATION_METHOD: Final[str] = "correlation_filter"
ORTHOGONALIZATION_VERSION: Final[str] = "1.0.0"

REASON_ACCEPTED: Final[str] = "accepted"
REASON_REDUNDANT: Final[str] = "redundant"

_KEY_SEP: Final[str] = "\x1f"
_EQUAL_WEIGHT: Final[str] = "equal_weight"


@dataclass(frozen=True, slots=True)
class OrthogonalizationConfig:
    """Immutable combination orthogonalization configuration.

    Attributes:
        max_combination_correlation: Absolute Pearson threshold for redundancy.
        min_overlap: Minimum pairwise complete observations required.
    """

    max_combination_correlation: float
    min_overlap: int


def require_max_combination_correlation(value: object) -> float:
    """Validate and return a correlation threshold in ``(0, 1)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorOrthogonalizationError(
            "max_combination_correlation must be a float in (0, 1)",
            error_code=_ERROR_MAX_CORR,
            details={
                "parameter": "max_combination_correlation",
                "value": value,
                "actual_type": type(value).__name__,
            },
        )
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise FactorOrthogonalizationError(
            "max_combination_correlation must be a float in (0, 1)",
            error_code=_ERROR_MAX_CORR,
            details={"parameter": "max_combination_correlation", "value": value},
        )
    return threshold


def require_min_correlation_overlap(value: object) -> int:
    """Validate and return a positive integer minimum overlap."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorOrthogonalizationError(
            "min_overlap must be a positive integer",
            error_code=_ERROR_MIN_OVERLAP,
            details={
                "parameter": "min_overlap",
                "value": value,
                "actual_type": type(value).__name__,
            },
        )
    if value <= 0:
        raise FactorOrthogonalizationError(
            "min_overlap must be a positive integer",
            error_code=_ERROR_MIN_OVERLAP,
            details={"parameter": "min_overlap", "value": value},
        )
    return value


def require_orthogonalization_config(
    *,
    max_combination_correlation: object = DEFAULT_MAX_COMBINATION_CORRELATION,
    min_overlap: object = DEFAULT_MIN_CORRELATION_OVERLAP,
) -> OrthogonalizationConfig:
    """Validate and assemble an ``OrthogonalizationConfig``."""
    return OrthogonalizationConfig(
        max_combination_correlation=require_max_combination_correlation(
            max_combination_correlation
        ),
        min_overlap=require_min_correlation_overlap(min_overlap),
    )


def apply_greedy_combination_filter(
    combinations: pl.DataFrame,
    observations: pl.DataFrame,
    config: OrthogonalizationConfig,
) -> list[dict[str, object]]:
    """Apply greedy combination redundancy filtering in rank order.

    ``combinations`` must already be sorted by ascending
    ``combination_rank`` then ``combination_id``. Observations may be empty;
    insufficient overlap never forces rejection.

    Returns:
        One decision dictionary per input combination row.
    """
    rows = combinations.to_dicts()
    if len(rows) == 0:
        return []

    wide = _pivot_observations(observations)
    series_cache = _build_combination_series_cache(wide, rows)

    accepted: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for candidate in rows:
        decision = _evaluate_candidate(
            candidate=candidate,
            accepted=accepted,
            series_cache=series_cache,
            config=config,
        )
        decisions.append(decision)
        if not bool(decision["redundancy_rejected"]):
            accepted.append(candidate)

    return decisions


def _factor_key(name: str, version: str) -> str:
    """Compose a deterministic wide-column key for name+version identity."""
    return f"{name}{_KEY_SEP}{version}"


def _pivot_observations(observations: pl.DataFrame) -> pl.DataFrame:
    """Pivot long observations to wide columns keyed by factor identity."""
    required = ("symbol", "open_time", "factor_name", "factor_version", "factor_value")
    if observations.height == 0:
        return pl.DataFrame(schema={"symbol": pl.String, "open_time": pl.Int64})
    missing = [column for column in required if column not in observations.columns]
    if missing:
        return pl.DataFrame(schema={"symbol": pl.String, "open_time": pl.Int64})

    prepared = (
        observations.select(
            pl.col("symbol"),
            pl.col("open_time"),
            pl.col("factor_name"),
            pl.col("factor_version"),
            pl.col("factor_value"),
        )
        .with_columns(
            (pl.col("factor_name") + pl.lit(_KEY_SEP) + pl.col("factor_version")).alias(
                "_factor_key"
            )
        )
        .filter(pl.col("factor_value").is_not_null())
    )
    if prepared.height == 0:
        return pl.DataFrame(schema={"symbol": pl.String, "open_time": pl.Int64})
    return prepared.pivot(
        values="factor_value",
        index=["symbol", "open_time"],
        on="_factor_key",
        aggregate_function="first",
    )


def _build_combination_series_cache(
    wide: pl.DataFrame,
    combinations: Sequence[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Materialize equal-weight combination signal arrays keyed by combination_id."""
    cache: dict[str, np.ndarray] = {}
    if wide.height == 0:
        return cache

    for row in combinations:
        combination_id = str(row["combination_id"])
        names = _as_string_list(row["factor_names"])
        versions = _as_string_list(row["factor_versions"])
        method = str(row.get("combination_method", _EQUAL_WEIGHT))
        if method != _EQUAL_WEIGHT:
            continue
        if len(names) == 0 or len(names) != len(versions):
            continue
        member_arrays: list[np.ndarray] = []
        skip = False
        for name, version in zip(names, versions, strict=True):
            key = _factor_key(name, version)
            if key not in wide.columns:
                skip = True
                break
            member_arrays.append(wide[key].to_numpy())
        if skip or len(member_arrays) == 0:
            continue
        stacked = np.vstack(member_arrays)
        # Equal-weight mean; NaN where any member is non-finite.
        with np.errstate(invalid="ignore"):
            finite_mask = np.isfinite(stacked).all(axis=0)
            signal = np.full(stacked.shape[1], np.nan, dtype=np.float64)
            signal[finite_mask] = stacked[:, finite_mask].mean(axis=0)
        cache[combination_id] = signal
    return cache


def _evaluate_candidate(
    *,
    candidate: dict[str, object],
    accepted: Sequence[dict[str, object]],
    series_cache: dict[str, np.ndarray],
    config: OrthogonalizationConfig,
) -> dict[str, object]:
    """Evaluate one combination against already accepted combinations."""
    combination_id = str(candidate["combination_id"])
    base: dict[str, object] = {
        "combination_id": combination_id,
        "redundancy_checked": True,
        "redundancy_rejected": False,
        "redundancy_reference_combination_id": None,
        "correlation_score": None,
        "correlation_overlap": None,
        "orthogonalization_reason": REASON_ACCEPTED,
    }
    if len(accepted) == 0:
        return base

    candidate_series = series_cache.get(combination_id)
    for reference in accepted:
        reference_id = str(reference["combination_id"])
        reference_series = series_cache.get(reference_id)
        abs_corr, overlap = _pairwise_abs_pearson(candidate_series, reference_series)
        if overlap < config.min_overlap:
            continue
        if abs_corr is None:
            continue
        if abs_corr >= config.max_combination_correlation:
            return {
                **base,
                "redundancy_rejected": True,
                "redundancy_reference_combination_id": reference_id,
                "correlation_score": abs_corr,
                "correlation_overlap": overlap,
                "orthogonalization_reason": REASON_REDUNDANT,
            }
    return base


def _pairwise_abs_pearson(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> tuple[float | None, int]:
    """Return ``(|pearson|, overlap)`` using pairwise complete observations."""
    if left is None or right is None:
        return None, 0
    if left.shape != right.shape:
        return None, 0
    mask = np.isfinite(left) & np.isfinite(right)
    overlap = int(mask.sum())
    if overlap == 0:
        return None, 0
    left_vals = left[mask]
    right_vals = right[mask]
    if left_vals.size < 2:
        return None, overlap
    if float(np.std(left_vals)) == 0.0 or float(np.std(right_vals)) == 0.0:
        return None, overlap
    corr = float(np.corrcoef(left_vals, right_vals)[0, 1])
    if not math.isfinite(corr):
        return None, overlap
    return abs(corr), overlap


def _as_string_list(value: object) -> list[str]:
    """Normalize list-like member identity columns to ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in cast(Sequence[object], value)]
    return [str(value)]
