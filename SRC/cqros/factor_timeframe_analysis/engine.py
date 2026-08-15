"""CQROS Factor Timeframe Analysis Engine contracts and implementation.

Purpose:
    Convert a canonical Factor Selection dataset into a deterministic
    timeframe-analysis DataFrame conforming to ``TIMEFRAME_ANALYSIS_SCHEMA``.

Responsibilities:
    - Define ``FactorTimeframeAnalysisEngine`` as the shared analysis contract
    - Provide ``SimpleFactorTimeframeAnalysisEngine`` for best-timeframe
      selection across available timeframes
    - Validate Factor Selection DataFrame structure
    - Restrict participation to rows where ``selected`` is ``True``
    - Rank timeframes within each factor by selection score
    - Emit ``selected`` for combination eligibility and record Factor
      Selection lineage via ``source_selection_version``
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.factor_timeframe_analysis.exceptions``, and
    ``cqros.factor_timeframe_analysis.schema``.

Public API:
    ``FactorTimeframeAnalysisEngine``, ``SimpleFactorTimeframeAnalysisEngine``,
    ``FACTOR_SELECTION_INPUT_COLUMNS``, ``DEFAULT_SOURCE_SELECTION_VERSION``,
    ``validate_factor_selection_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.factor_timeframe_analysis.schema import (
    CANONICAL_COLUMN_ORDER,
    TIMEFRAME_ANALYSIS_SCHEMA,
    TimeframeAnalysisStatus,
)

__all__ = [
    "DEFAULT_SOURCE_SELECTION_VERSION",
    "FACTOR_SELECTION_INPUT_COLUMNS",
    "FactorTimeframeAnalysisEngine",
    "SimpleFactorTimeframeAnalysisEngine",
    "validate_factor_selection_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "FTA_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "FTA_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FTA_MISSING_COLUMNS"
_ERROR_NO_SELECTED: Final[str] = "FTA_NO_SELECTED"
_ERROR_SOURCE_VERSION: Final[str] = "FTA_SOURCE_SELECTION_VERSION"

_GROUP_COLUMNS: Final[tuple[str, ...]] = ("factor_name", "factor_version")

DEFAULT_SOURCE_SELECTION_VERSION: Final[str] = "unspecified"

# Factor Selection columns required to assemble a timeframe-analysis row.
FACTOR_SELECTION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "timeframe",
    "selection_score",
    "selection_rank",
    "selected",
    "selection_time",
    "status",
)


@runtime_checkable
class FactorTimeframeAnalysisEngine(Protocol):
    """Structural contract for converting selection rows into timeframe analysis.

    Implementations own timeframe-analysis semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, factor_selection: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Selection dataset into a timeframe-analysis DataFrame.

        Args:
            factor_selection: Canonical Factor Selection dataset.
                Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``TIMEFRAME_ANALYSIS_SCHEMA``.
        """
        ...


class SimpleFactorTimeframeAnalysisEngine:
    """Generate deterministic best-timeframe rows from Factor Selection output.

    Only rows with ``selected == True`` participate. Within each
    ``(factor_name, factor_version)`` group, timeframes are ordered by
    descending ``selection_score`` (ties broken by ascending ``timeframe``).

    Metrics:

    - ``best_timeframe`` / ``best_selection_score``: winning timeframe score
    - ``timeframe_rank``: always ``1`` for the winner
    - ``winner_margin``: best minus second-best score (null when ``n < 2``)
    - ``score_gap``: best minus worst score (null when ``n < 2``)
    - ``timeframe_stability``: ``1 - clamp(std / score_gap, 0, 1)``;
      ``1.0`` when only one timeframe or zero gap
    - ``timeframe_confidence``: mean of clipped absolute ``winner_margin``
      and ``timeframe_stability``; ``1.0`` when only one timeframe
    - ``selected``: ``True`` when status is ``PASS``; otherwise ``False``
    - ``source_selection_version``: Factor Selection lineage version string

    Status is ``PASS`` when ``best_timeframe`` is non-null/non-empty and
    ``best_selection_score`` is non-null; otherwise ``FAIL``.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical inputs.
    """

    __slots__ = ("_source_selection_version",)

    def __init__(
        self,
        *,
        source_selection_version: str = DEFAULT_SOURCE_SELECTION_VERSION,
    ) -> None:
        """Initialize the engine with Factor Selection lineage version.

        Args:
            source_selection_version: Non-blank lineage string identifying the
                Factor Selection artifact generation that produced the input
                panel (for example the calendar year ``\"2026\"``).

        Raises:
            FactorTimeframeAnalysisError: If ``source_selection_version`` is
                blank.
        """
        self._source_selection_version = _require_source_selection_version(source_selection_version)

    def build(self, factor_selection: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Selection dataset into finalized analysis rows.

        Args:
            factor_selection: Canonical Factor Selection dataset.
                Must not be mutated.

        Returns:
            A new DataFrame matching ``TIMEFRAME_ANALYSIS_SCHEMA``.

        Raises:
            FactorTimeframeAnalysisError: If the input fails structural
                validation, required columns are missing, or no selected
                rows remain for analysis.
        """
        frame = validate_factor_selection_frame(factor_selection)
        _require_columns(frame, FACTOR_SELECTION_INPUT_COLUMNS, "factor_selection")
        return _build_timeframe_analysis_rows(
            frame,
            source_selection_version=self._source_selection_version,
        )


def validate_factor_selection_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Selection dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        FactorTimeframeAnalysisError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorTimeframeAnalysisError(
            "factor_selection frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_selection",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise FactorTimeframeAnalysisError(
            "factor_selection frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_selection", "rows": frame.height},
        )
    return frame


def _build_timeframe_analysis_rows(
    factor_selection: pl.DataFrame,
    *,
    source_selection_version: str,
) -> pl.DataFrame:
    """Assemble canonical timeframe-analysis rows from Factor Selection."""
    selected = factor_selection.filter(pl.col("selected") == True)  # noqa: E712
    if selected.height == 0:
        raise FactorTimeframeAnalysisError(
            "factor_selection frame contains no selected factors",
            error_code=_ERROR_NO_SELECTED,
            details={
                "dataset": "factor_selection",
                "rows": factor_selection.height,
                "selected_rows": 0,
            },
        )

    # One row per (factor, version, timeframe): keep highest score, then latest time.
    deduplicated = selected.sort(
        "factor_name",
        "factor_version",
        "timeframe",
        "selection_score",
        "selection_time",
        descending=[False, False, False, True, True],
        nulls_last=True,
        maintain_order=True,
    ).unique(subset=["factor_name", "factor_version", "timeframe"], keep="first")

    # Winner first within each factor group.
    ordered = deduplicated.sort(
        "factor_name",
        "factor_version",
        "selection_score",
        "timeframe",
        descending=[False, False, True, False],
        nulls_last=True,
        maintain_order=True,
    )

    aggregated = ordered.group_by(list(_GROUP_COLUMNS), maintain_order=True).agg(
        pl.col("factor_category").first().alias("factor_category"),
        pl.col("selection_time").first().alias("analysis_time"),
        pl.col("timeframe").first().alias("best_timeframe"),
        pl.col("selection_score").first().alias("best_selection_score"),
        pl.col("selection_score").len().alias("_timeframe_count"),
        pl.col("selection_score").min().alias("_min_score"),
        pl.col("selection_score").drop_nulls().std(ddof=0).alias("_std_score"),
        pl.col("selection_score")
        .sort(descending=True, nulls_last=True)
        .slice(1, 1)
        .first()
        .alias("_second_score"),
    )

    pass_status = TimeframeAnalysisStatus.PASS.value
    fail_status = TimeframeAnalysisStatus.FAIL.value

    scored = aggregated.with_columns(
        pl.lit(1).cast(pl.Int32).alias("timeframe_rank"),
        pl.when(pl.col("_timeframe_count") < 2)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("best_selection_score") - pl.col("_second_score"))
        .alias("winner_margin"),
        pl.when(pl.col("_timeframe_count") < 2)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("best_selection_score") - pl.col("_min_score"))
        .alias("score_gap"),
    )

    with_stability = scored.with_columns(
        pl.when(pl.col("_timeframe_count") < 2)
        .then(pl.lit(1.0))
        .when(pl.col("score_gap").is_null() | (pl.col("score_gap") == 0.0))
        .then(pl.lit(1.0))
        .when(pl.col("_std_score").is_null())
        .then(pl.lit(1.0))
        .otherwise(
            (1.0 - (pl.col("_std_score") / pl.col("score_gap")).clip(0.0, 1.0)).clip(0.0, 1.0)
        )
        .alias("timeframe_stability")
    )

    with_confidence = with_stability.with_columns(
        pl.when(pl.col("_timeframe_count") < 2)
        .then(pl.lit(1.0))
        .when(pl.col("winner_margin").is_null())
        .then(0.5 * pl.col("timeframe_stability"))
        .otherwise(
            (
                0.5 * pl.col("winner_margin").clip(0.0, 1.0) + 0.5 * pl.col("timeframe_stability")
            ).clip(0.0, 1.0)
        )
        .alias("timeframe_confidence")
    )

    finalized = (
        with_confidence.with_columns(
            pl.when(
                pl.col("best_timeframe").is_not_null()
                & (pl.col("best_timeframe") != "")
                & pl.col("best_selection_score").is_not_null()
            )
            .then(pl.lit(pass_status))
            .otherwise(pl.lit(fail_status))
            .alias("status")
        )
        .with_columns(
            (pl.col("status") == pass_status).alias("selected"),
            pl.lit(source_selection_version).alias("source_selection_version"),
        )
        .select(list(CANONICAL_COLUMN_ORDER))
    )

    return finalized.sort("factor_name", "factor_version").cast(TIMEFRAME_ANALYSIS_SCHEMA)


def _require_source_selection_version(value: object) -> str:
    """Validate and return a non-blank Factor Selection lineage version."""
    if not isinstance(value, str) or value.strip() == "":
        raise FactorTimeframeAnalysisError(
            "source_selection_version must be a non-blank string",
            error_code=_ERROR_SOURCE_VERSION,
            details={"source_selection_version": value},
        )
    return value.strip()


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorTimeframeAnalysisError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )
