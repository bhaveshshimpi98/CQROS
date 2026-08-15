"""CQROS Factor Combination Engine contracts and implementation.

Purpose:
    Convert a Factor Timeframe Analysis dataset into a deterministic
    pairwise factor-combination DataFrame conforming to
    ``FACTOR_COMBINATION_SCHEMA``.

Responsibilities:
    - Define ``FactorCombinationEngine`` as the shared combination contract
    - Provide ``SimpleFactorCombinationEngine`` for equal-weight pair generation
    - Validate Factor Timeframe Analysis DataFrame structure
    - Restrict participation to rows where ``selected`` is ``True``
    - Emit deterministic ``N`` choose ``2`` combination rows
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.factor_combination.exceptions``, and
    ``cqros.factor_combination.schema``.

Public API:
    ``FactorCombinationEngine``, ``SimpleFactorCombinationEngine``,
    ``FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS``,
    ``validate_factor_timeframe_analysis_frame``
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.factor_combination.exceptions import FactorCombinationError
from cqros.factor_combination.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_COMBINATION_SCHEMA,
    FactorCombinationStatus,
)

__all__ = [
    "FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS",
    "FactorCombinationEngine",
    "SimpleFactorCombinationEngine",
    "validate_factor_timeframe_analysis_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "FCOMB_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "FCOMB_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FCOMB_MISSING_COLUMNS"
_ERROR_NO_SELECTED: Final[str] = "FCOMB_NO_SELECTED"
_ERROR_INSUFFICIENT_FACTORS: Final[str] = "FCOMB_INSUFFICIENT_FACTORS"

_COMBINATION_METHOD: Final[str] = "equal_weight"
_COMBINATION_SIZE: Final[int] = 2
_ID_SEPARATOR: Final[str] = "|"

# Factor Timeframe Analysis columns required to assemble a combination row.
FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "best_timeframe",
    "best_selection_score",
    "timeframe_confidence",
    "selected",
)


@runtime_checkable
class FactorCombinationEngine(Protocol):
    """Structural contract for converting timeframe analysis into combinations.

    Implementations own combination-generation semantics. Pipeline
    orchestration delegates exclusively through this contract.
    Implementations must return a new DataFrame and must not mutate the
    input frame.
    """

    def build(self, factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Timeframe Analysis dataset into combinations.

        Args:
            factor_timeframe_analysis: Canonical Factor Timeframe Analysis
                dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``FACTOR_COMBINATION_SCHEMA``.
        """
        ...


class SimpleFactorCombinationEngine:
    """Generate deterministic equal-weight pairwise factor combinations.

    Only rows with ``selected == True`` participate. For ``N`` selected
    factors the engine emits ``N`` choose ``2`` pairs. Each pair uses
    ``combination_method = equal_weight`` and ``combination_size = 2``.

    Metrics:

    - ``combination_id``: alphabetically sorted factor names joined by ``|``
    - ``timeframe``: shared timeframe when equal; otherwise the timeframe of
      the higher ``best_selection_score`` (ties break by lexicographic
      timeframe)
    - ``stability_score``: mean of member ``timeframe_confidence`` values
    - ``combination_score``: mean of member ``best_selection_score`` values
    - ``confidence_score``: mean of member ``timeframe_confidence`` and
      min-max-normalized ``best_selection_score`` values
    - Statistical metric columns remain null in this initial version
    - ``combination_rank``: dense descending rank by ``combination_score``
      with ``combination_id`` ascending as the deterministic tie-break
    - ``status``: always ``PASS``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Pair generation is fully deterministic for identical inputs.
        ``analysis_time`` is current UTC epoch milliseconds.
    """

    __slots__ = ()

    def build(self, factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Timeframe Analysis dataset into combination rows.

        Args:
            factor_timeframe_analysis: Canonical Factor Timeframe Analysis
                dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``FACTOR_COMBINATION_SCHEMA``.

        Raises:
            FactorCombinationError: If the input fails structural validation,
                required columns are missing, no selected rows remain, or
                fewer than two selected factors are available.
        """
        frame = validate_factor_timeframe_analysis_frame(factor_timeframe_analysis)
        _require_columns(
            frame,
            FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS,
            "factor_timeframe_analysis",
        )
        return _build_factor_combination_rows(frame)


def validate_factor_timeframe_analysis_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Timeframe Analysis dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        FactorCombinationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorCombinationError(
            "factor_timeframe_analysis frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_timeframe_analysis",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise FactorCombinationError(
            "factor_timeframe_analysis frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_timeframe_analysis", "rows": frame.height},
        )
    return frame


def _build_factor_combination_rows(factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical pairwise combination rows from timeframe analysis."""
    selected = factor_timeframe_analysis.filter(pl.col("selected") == True)  # noqa: E712
    if selected.height == 0:
        raise FactorCombinationError(
            "factor_timeframe_analysis frame contains no selected factors",
            error_code=_ERROR_NO_SELECTED,
            details={
                "dataset": "factor_timeframe_analysis",
                "rows": factor_timeframe_analysis.height,
                "selected_rows": 0,
            },
        )

    # One row per (factor, version): keep highest selection score.
    deduplicated = selected.sort(
        "factor_name",
        "factor_version",
        "best_selection_score",
        descending=[False, False, True],
        nulls_last=True,
        maintain_order=True,
    ).unique(subset=["factor_name", "factor_version"], keep="first")

    if deduplicated.height < _COMBINATION_SIZE:
        raise FactorCombinationError(
            "factor_timeframe_analysis frame requires at least two selected factors",
            error_code=_ERROR_INSUFFICIENT_FACTORS,
            details={
                "dataset": "factor_timeframe_analysis",
                "selected_rows": deduplicated.height,
                "minimum_required": _COMBINATION_SIZE,
            },
        )

    normalized = deduplicated.with_columns(
        _minmax_normalize_global(pl.col("best_selection_score")).alias(
            "_normalized_selection_score"
        )
    ).sort("factor_name", "factor_version", maintain_order=True)

    left = normalized.select(
        pl.col("factor_name").alias("factor_name_a"),
        pl.col("factor_version").alias("factor_version_a"),
        pl.col("factor_category").alias("factor_category_a"),
        pl.col("best_timeframe").alias("best_timeframe_a"),
        pl.col("best_selection_score").alias("best_selection_score_a"),
        pl.col("timeframe_confidence").alias("timeframe_confidence_a"),
        pl.col("_normalized_selection_score").alias("_normalized_selection_score_a"),
    )
    right = normalized.select(
        pl.col("factor_name").alias("factor_name_b"),
        pl.col("factor_version").alias("factor_version_b"),
        pl.col("factor_category").alias("factor_category_b"),
        pl.col("best_timeframe").alias("best_timeframe_b"),
        pl.col("best_selection_score").alias("best_selection_score_b"),
        pl.col("timeframe_confidence").alias("timeframe_confidence_b"),
        pl.col("_normalized_selection_score").alias("_normalized_selection_score_b"),
    )

    pairs = left.join(right, how="cross").filter(
        (pl.col("factor_name_a") < pl.col("factor_name_b"))
        | (
            (pl.col("factor_name_a") == pl.col("factor_name_b"))
            & (pl.col("factor_version_a") < pl.col("factor_version_b"))
        )
    )

    analysis_time_ms = int(datetime.now(UTC).timestamp() * 1000)
    pass_status = FactorCombinationStatus.PASS.value

    scored = pairs.with_columns(
        (pl.col("factor_name_a") + pl.lit(_ID_SEPARATOR) + pl.col("factor_name_b")).alias(
            "combination_id"
        ),
        pl.concat_list("factor_name_a", "factor_name_b").alias("factor_names"),
        pl.concat_list("factor_version_a", "factor_version_b").alias("factor_versions"),
        pl.concat_list("factor_category_a", "factor_category_b").alias("factor_categories"),
        _resolve_combination_timeframe().alias("timeframe"),
        pl.lit(_COMBINATION_SIZE).cast(pl.Int32).alias("combination_size"),
        pl.lit(_COMBINATION_METHOD).alias("combination_method"),
        pl.lit(analysis_time_ms).cast(pl.Int64).alias("analysis_time"),
        pl.lit(None, dtype=pl.Float64).alias("information_coefficient"),
        pl.lit(None, dtype=pl.Float64).alias("rank_information_coefficient"),
        pl.lit(None, dtype=pl.Float64).alias("ic_information_ratio"),
        pl.lit(None, dtype=pl.Float64).alias("quantile_spread"),
        pl.lit(None, dtype=pl.Float64).alias("hit_rate"),
        pl.lit(None, dtype=pl.Float64).alias("turnover"),
        pl.lit(None, dtype=pl.Float64).alias("correlation_penalty"),
        pl.lit(None, dtype=pl.Float64).alias("diversification_score"),
        ((pl.col("timeframe_confidence_a") + pl.col("timeframe_confidence_b")) / 2.0).alias(
            "stability_score"
        ),
        (
            (
                pl.col("timeframe_confidence_a")
                + pl.col("timeframe_confidence_b")
                + pl.col("_normalized_selection_score_a")
                + pl.col("_normalized_selection_score_b")
            )
            / 4.0
        ).alias("confidence_score"),
        ((pl.col("best_selection_score_a") + pl.col("best_selection_score_b")) / 2.0).alias(
            "combination_score"
        ),
        pl.lit(pass_status).alias("status"),
    )

    ranked = scored.sort(
        "combination_score",
        "combination_id",
        descending=[True, False],
        nulls_last=True,
        maintain_order=True,
    ).with_columns((pl.int_range(pl.len()) + 1).cast(pl.Int32).alias("combination_rank"))

    return (
        ranked.sort("combination_rank", maintain_order=True)
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(FACTOR_COMBINATION_SCHEMA)
    )


def _resolve_combination_timeframe() -> pl.Expr:
    """Choose the pair timeframe from shared or higher-scoring member values."""
    return (
        pl.when(pl.col("best_timeframe_a") == pl.col("best_timeframe_b"))
        .then(pl.col("best_timeframe_a"))
        .when(pl.col("best_selection_score_a") > pl.col("best_selection_score_b"))
        .then(pl.col("best_timeframe_a"))
        .when(pl.col("best_selection_score_b") > pl.col("best_selection_score_a"))
        .then(pl.col("best_timeframe_b"))
        .otherwise(pl.min_horizontal("best_timeframe_a", "best_timeframe_b"))
    )


def _minmax_normalize_global(expression: pl.Expr) -> pl.Expr:
    """Min-max normalize ``expression`` across the frame to ``[0, 1]``.

    Null inputs become ``0.0``. When the frame has zero range, finite values
    normalize to ``1.0`` so a lone score retains full component credit.
    """
    minimum = expression.min()
    maximum = expression.max()
    return (
        pl.when(expression.is_null())
        .then(pl.lit(0.0))
        .when(maximum == minimum)
        .then(pl.lit(1.0))
        .otherwise((expression - minimum) / (maximum - minimum))
    )


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorCombinationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )
