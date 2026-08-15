"""CQROS Factor Selection correlation redundancy filtering.

Purpose:
    Apply greedy, rank-preserving absolute-Pearson redundancy filtering to a
    ranked Factor Selection candidate pool using validation-window factor
    observations.

Responsibilities:
    - Validate redundancy configuration parameters
    - Define the factor-observation panel loading contract
    - Load and pivot observation panels once per timeframe
    - Compute pairwise absolute Pearson correlations with overlap counts
    - Apply greedy acceptance/rejection in selection-rank order
    - Emit auditable redundancy decision columns
    - Remain free of scoring, ranking, persistence, and CLI logic

Dependencies:
    ``polars``, ``numpy``, and ``cqros.factor_selection.exceptions``.

Public API:
    ``DEFAULT_CANDIDATE_N``, ``DEFAULT_MAX_FACTOR_CORRELATION``,
    ``DEFAULT_MIN_CORRELATION_OVERLAP``, ``REASON_OUTSIDE_CANDIDATE_N``,
    ``REASON_OUTSIDE_TOP_N``, ``REASON_REDUNDANT``, ``REASON_TOP_N``,
    ``FactorObservationSource``, ``RedundancyConfig``,
    ``require_candidate_n``, ``require_max_factor_correlation``,
    ``require_min_correlation_overlap``, ``require_redundancy_config``,
    ``apply_greedy_redundancy_filter``, ``pairwise_abs_pearson``
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
import polars as pl

from cqros.factor_selection.exceptions import FactorSelectionError

__all__ = [
    "DEFAULT_CANDIDATE_N",
    "DEFAULT_MAX_FACTOR_CORRELATION",
    "DEFAULT_MIN_CORRELATION_OVERLAP",
    "REASON_OUTSIDE_CANDIDATE_N",
    "REASON_OUTSIDE_TOP_N",
    "REASON_REDUNDANT",
    "REASON_TOP_N",
    "FactorObservationSource",
    "RedundancyConfig",
    "apply_greedy_redundancy_filter",
    "pairwise_abs_pearson",
    "require_candidate_n",
    "require_max_factor_correlation",
    "require_min_correlation_overlap",
    "require_redundancy_config",
]

_ERROR_CANDIDATE_N: Final[str] = "FSEL_CANDIDATE_N_INVALID"
_ERROR_MAX_CORR: Final[str] = "FSEL_MAX_FACTOR_CORRELATION_INVALID"
_ERROR_MIN_OVERLAP: Final[str] = "FSEL_MIN_OVERLAP_INVALID"
_ERROR_CANDIDATE_LT_TOP: Final[str] = "FSEL_CANDIDATE_N_LT_TOP_N"

DEFAULT_CANDIDATE_N: Final[int] = 40
DEFAULT_MAX_FACTOR_CORRELATION: Final[float] = 0.90
DEFAULT_MIN_CORRELATION_OVERLAP: Final[int] = 500

REASON_TOP_N: Final[str] = "top_n"
REASON_OUTSIDE_TOP_N: Final[str] = "outside_top_n"
REASON_REDUNDANT: Final[str] = "redundant"
REASON_OUTSIDE_CANDIDATE_N: Final[str] = "outside_candidate_n"

_KEY_SEP: Final[str] = "\x1f"


@dataclass(frozen=True, slots=True)
class RedundancyConfig:
    """Immutable redundancy-filter configuration.

    Attributes:
        top_n: Final number of accepted factors retained per timeframe.
        candidate_n: Maximum ranked candidates considered by the filter.
        max_factor_correlation: Absolute Pearson threshold for redundancy.
        min_overlap: Minimum pairwise complete observations required.
    """

    top_n: int
    candidate_n: int
    max_factor_correlation: float
    min_overlap: int


@runtime_checkable
class FactorObservationSource(Protocol):
    """Load validation-window factor observations for redundancy filtering."""

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        """Return long-format observations for the requested factors.

        Required columns:
            ``symbol``, ``open_time``, ``factor_name``, ``factor_version``,
            ``factor_value``.

        Args:
            timeframe: Bar interval for the panel.
            factor_names: Factor names to load.
            factor_versions: Matching factor versions (parallel to names).
            start_time: Inclusive validation-window start (UTC ms).
            end_time: Inclusive validation-window end (UTC ms).

        Returns:
            Observation rows restricted to the validation window. May be empty.
        """
        ...


def require_candidate_n(candidate_n: object) -> int:
    """Validate and return a positive integer candidate pool size."""
    if isinstance(candidate_n, bool) or not isinstance(candidate_n, int):
        raise FactorSelectionError(
            "candidate_n must be a positive integer",
            error_code=_ERROR_CANDIDATE_N,
            details={
                "parameter": "candidate_n",
                "value": candidate_n,
                "actual_type": type(candidate_n).__name__,
            },
        )
    if candidate_n <= 0:
        raise FactorSelectionError(
            "candidate_n must be a positive integer",
            error_code=_ERROR_CANDIDATE_N,
            details={"parameter": "candidate_n", "value": candidate_n},
        )
    return candidate_n


def require_max_factor_correlation(value: object) -> float:
    """Validate and return a correlation threshold in ``(0, 1)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorSelectionError(
            "max_factor_correlation must be a float in (0, 1)",
            error_code=_ERROR_MAX_CORR,
            details={
                "parameter": "max_factor_correlation",
                "value": value,
                "actual_type": type(value).__name__,
            },
        )
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise FactorSelectionError(
            "max_factor_correlation must be a float in (0, 1)",
            error_code=_ERROR_MAX_CORR,
            details={"parameter": "max_factor_correlation", "value": value},
        )
    return threshold


def require_min_correlation_overlap(value: object) -> int:
    """Validate and return a positive integer minimum overlap."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorSelectionError(
            "min_overlap must be a positive integer",
            error_code=_ERROR_MIN_OVERLAP,
            details={
                "parameter": "min_overlap",
                "value": value,
                "actual_type": type(value).__name__,
            },
        )
    if value <= 0:
        raise FactorSelectionError(
            "min_overlap must be a positive integer",
            error_code=_ERROR_MIN_OVERLAP,
            details={"parameter": "min_overlap", "value": value},
        )
    return value


def require_redundancy_config(
    *,
    top_n: int,
    candidate_n: object = DEFAULT_CANDIDATE_N,
    max_factor_correlation: object = DEFAULT_MAX_FACTOR_CORRELATION,
    min_overlap: object = DEFAULT_MIN_CORRELATION_OVERLAP,
) -> RedundancyConfig:
    """Validate and assemble a ``RedundancyConfig``.

    Raises:
        FactorSelectionError: If any parameter is invalid or
            ``candidate_n < top_n``.
    """
    validated_candidate = require_candidate_n(candidate_n)
    if validated_candidate < top_n:
        raise FactorSelectionError(
            "candidate_n must be greater than or equal to top_n",
            error_code=_ERROR_CANDIDATE_LT_TOP,
            details={"candidate_n": validated_candidate, "top_n": top_n},
        )
    return RedundancyConfig(
        top_n=top_n,
        candidate_n=validated_candidate,
        max_factor_correlation=require_max_factor_correlation(max_factor_correlation),
        min_overlap=require_min_correlation_overlap(min_overlap),
    )


def apply_greedy_redundancy_filter(
    ranked: pl.DataFrame,
    observations: pl.DataFrame,
    config: RedundancyConfig,
) -> pl.DataFrame:
    """Apply greedy redundancy filtering to one timeframe's ranked rows.

    ``ranked`` must contain exactly one timeframe and already-assigned
    ``selection_rank`` values. Observations may be empty.

    Returns:
        A DataFrame with the original ranked columns plus redundancy audit
        fields and finalized ``selected`` / ``selection_reason``.
    """
    ordered = ranked.sort(
        "selection_rank",
        "factor_name",
        "factor_version",
        maintain_order=True,
    )
    rows = ordered.to_dicts()
    if len(rows) == 0:
        return ordered

    candidate_limit = min(config.candidate_n, len(rows))
    candidates = rows[:candidate_limit]
    outside = rows[candidate_limit:]

    wide = _pivot_observations(observations, candidates)
    series_cache = _build_series_cache(wide, candidates)

    accepted: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for candidate in candidates:
        decision = _evaluate_candidate(
            candidate=candidate,
            accepted=accepted,
            series_cache=series_cache,
            config=config,
        )
        decisions.append(decision)
        if not bool(decision["redundancy_rejected"]):
            accepted.append(candidate)

    selected_keys = {
        (row["factor_name"], row["factor_version"]) for row in accepted[: config.top_n]
    }

    finalized: list[dict[str, object]] = []
    for decision in decisions:
        key = (decision["factor_name"], decision["factor_version"])
        if bool(decision["redundancy_rejected"]):
            reason = REASON_REDUNDANT
            selected = False
        elif key in selected_keys:
            reason = REASON_TOP_N
            selected = True
        else:
            reason = REASON_OUTSIDE_TOP_N
            selected = False
        finalized.append(
            {
                **decision,
                "selected": selected,
                "selection_reason": reason,
            }
        )

    for row in outside:
        finalized.append(
            {
                "factor_name": row["factor_name"],
                "factor_version": row["factor_version"],
                "candidate_rank": None,
                "redundancy_checked": False,
                "redundancy_rejected": False,
                "redundancy_reference_factor": None,
                "redundancy_reference_factor_version": None,
                "redundancy_correlation": None,
                "redundancy_overlap": None,
                "selected": False,
                "selection_reason": REASON_OUTSIDE_CANDIDATE_N,
            }
        )

    audit = pl.DataFrame(finalized)
    return ordered.join(
        audit,
        on=["factor_name", "factor_version"],
        how="left",
    )


def _factor_key(name: str, version: str) -> str:
    """Compose a deterministic wide-column key for name+version identity."""
    return f"{name}{_KEY_SEP}{version}"


def _pivot_observations(
    observations: pl.DataFrame,
    candidates: Sequence[dict[str, object]],
) -> pl.DataFrame:
    """Pivot long observations to wide columns keyed by factor identity."""
    required = ("symbol", "open_time", "factor_name", "factor_version", "factor_value")
    if observations.height == 0:
        return pl.DataFrame(schema={"symbol": pl.String, "open_time": pl.Int64})
    missing = [column for column in required if column not in observations.columns]
    if missing:
        return pl.DataFrame(schema={"symbol": pl.String, "open_time": pl.Int64})

    keys = {_factor_key(str(row["factor_name"]), str(row["factor_version"])) for row in candidates}
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
        .filter(pl.col("_factor_key").is_in(list(keys)))
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


def _build_series_cache(
    wide: pl.DataFrame,
    candidates: Sequence[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Materialize numpy arrays for correlation without repeated frame scans."""
    cache: dict[str, np.ndarray] = {}
    if wide.height == 0:
        return cache
    for row in candidates:
        key = _factor_key(str(row["factor_name"]), str(row["factor_version"]))
        if key in wide.columns:
            cache[key] = wide[key].to_numpy()
    return cache


def _evaluate_candidate(
    *,
    candidate: dict[str, object],
    accepted: Sequence[dict[str, object]],
    series_cache: dict[str, np.ndarray],
    config: RedundancyConfig,
) -> dict[str, object]:
    """Evaluate one candidate against already accepted factors."""
    selection_rank = candidate["selection_rank"]
    if not isinstance(selection_rank, int):
        raise FactorSelectionError(
            "selection_rank must be an integer",
            error_code="FSEL_SELECTION_RANK_INVALID",
            details={"selection_rank": selection_rank},
        )
    base = {
        "factor_name": candidate["factor_name"],
        "factor_version": candidate["factor_version"],
        "candidate_rank": selection_rank,
        "redundancy_checked": True,
        "redundancy_rejected": False,
        "redundancy_reference_factor": None,
        "redundancy_reference_factor_version": None,
        "redundancy_correlation": None,
        "redundancy_overlap": None,
    }
    if len(accepted) == 0:
        return base

    candidate_key = _factor_key(str(candidate["factor_name"]), str(candidate["factor_version"]))
    candidate_series = series_cache.get(candidate_key)

    for reference in accepted:
        reference_key = _factor_key(str(reference["factor_name"]), str(reference["factor_version"]))
        reference_series = series_cache.get(reference_key)
        abs_corr, overlap = pairwise_abs_pearson(candidate_series, reference_series)
        if overlap < config.min_overlap:
            continue
        if abs_corr is None:
            continue
        if abs_corr >= config.max_factor_correlation:
            return {
                **base,
                "redundancy_rejected": True,
                "redundancy_reference_factor": reference["factor_name"],
                "redundancy_reference_factor_version": reference["factor_version"],
                "redundancy_correlation": abs_corr,
                "redundancy_overlap": overlap,
            }
    return base


def pairwise_abs_pearson(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> tuple[float | None, int]:
    """Return ``(|pearson|, overlap)`` using pairwise complete observations.

    Shared by the legacy wide-pivot path and the memory-efficient inner-join
    path so Pearson / overlap decisions use the same numerical implementation.
    """
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
    # Constant series → undefined Pearson; treat as non-redundant.
    if float(np.std(left_vals)) == 0.0 or float(np.std(right_vals)) == 0.0:
        return None, overlap
    corr = float(np.corrcoef(left_vals, right_vals)[0, 1])
    if not math.isfinite(corr):
        return None, overlap
    return abs(corr), overlap
