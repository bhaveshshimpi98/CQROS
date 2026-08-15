"""CQROS Factor Selection detailed CSV audit export.

Purpose:
    Produce a research-auditable CSV representation of Factor Selection
    decisions without replacing the canonical Parquet dataset.

Responsibilities:
    - Assemble one detailed audit row per factor per timeframe
    - Preserve raw Factor Validation metrics and selection decisions
    - Expose normalized components, weights, and contributions using the
      locked engine scoring implementation
    - Write per-timeframe and optional combined detailed CSV files
    - Remain free of ranking changes, statistical gates, and Phase 3
      redundancy filtering

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``,
    ``cqros.factor_selection.engine``, and
    ``cqros.factor_selection.exceptions``.

Public API:
    ``DETAILED_AUDIT_COLUMNS``, ``COMBINED_DETAILED_CSV_NAME``,
    ``build_detailed_audit_frame``, ``detailed_csv_path``,
    ``combined_detailed_csv_path``, ``write_detailed_csv``,
    ``write_combined_detailed_csv``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import FILE_EXTENSION_CSV, STORAGE_DIR_FACTOR_SELECTION
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_selection.engine import (
    NORMALIZATION_METHOD,
    SCORING_METHOD,
    attach_selection_score_components,
    validate_factor_validation_frame,
)
from cqros.factor_selection.exceptions import FactorSelectionError

__all__ = [
    "COMBINED_DETAILED_CSV_NAME",
    "DETAILED_AUDIT_COLUMNS",
    "build_detailed_audit_frame",
    "combined_detailed_csv_path",
    "contribution_sum_expression",
    "detailed_csv_path",
    "write_combined_detailed_csv",
    "write_detailed_csv",
]

_logger = logging.getLogger(__name__)

_ERROR_SELECTION_TYPE: Final[str] = "FSEL_DETAILED_SELECTION_TYPE"
_ERROR_SELECTION_EMPTY: Final[str] = "FSEL_DETAILED_SELECTION_EMPTY"
_ERROR_JOIN_MISMATCH: Final[str] = "FSEL_DETAILED_JOIN_MISMATCH"
_ERROR_FRAMES_EMPTY: Final[str] = "FSEL_DETAILED_FRAMES_EMPTY"

_JOIN_KEYS: Final[tuple[str, ...]] = ("factor_name", "factor_version", "timeframe")

_OPTIONAL_VALIDATION_CONTEXT: Final[tuple[str, ...]] = (
    "dataset_version",
    "label_version",
    "validation_start_time",
    "validation_end_time",
)

_CONTRIBUTION_COLUMNS: Final[tuple[str, ...]] = (
    "information_coefficient_contribution",
    "rank_information_coefficient_contribution",
    "ic_information_ratio_contribution",
    "quantile_spread_contribution",
    "monotonicity_contribution",
    "ic_decay_contribution",
    "turnover_contribution",
)

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_rank",
    "factor_name",
    "factor_version",
)

COMBINED_DETAILED_CSV_NAME: Final[str] = f"factor_selection_detailed{FILE_EXTENSION_CSV}"

DETAILED_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    # Identity
    "factor_name",
    "factor_version",
    "factor_category",
    "timeframe",
    # Validation context
    "validation_time",
    "validation_status",
    "validation_start_time",
    "validation_end_time",
    "dataset_version",
    "label_version",
    "observations",
    # Raw factor metrics
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "ic_p_value",
    "ic_decay",
    "quantile_spread",
    "monotonicity_score",
    "turnover",
    # Ranking components
    "abs_information_coefficient",
    "abs_rank_information_coefficient",
    "inverse_turnover",
    # Normalized components
    "information_coefficient_normalized",
    "rank_information_coefficient_normalized",
    "ic_information_ratio_normalized",
    "quantile_spread_normalized",
    "monotonicity_normalized",
    "ic_decay_normalized",
    "turnover_normalized",
    # Weights
    "information_coefficient_weight",
    "rank_information_coefficient_weight",
    "ic_information_ratio_weight",
    "quantile_spread_weight",
    "monotonicity_weight",
    "ic_decay_weight",
    "turnover_weight",
    # Weighted contributions
    "information_coefficient_contribution",
    "rank_information_coefficient_contribution",
    "ic_information_ratio_contribution",
    "quantile_spread_contribution",
    "monotonicity_contribution",
    "ic_decay_contribution",
    "turnover_contribution",
    # Final decision (from canonical selection output)
    "selection_score",
    "selection_rank",
    "selected",
    "status",
    "selection_reason",
    # Configuration audit
    "top_n",
    "candidate_n",
    "scoring_method",
    "normalization_method",
    "max_factor_correlation",
    "min_correlation_overlap",
    # Redundancy audit
    "candidate_rank",
    "redundancy_checked",
    "redundancy_rejected",
    "redundancy_reference_factor",
    "redundancy_reference_factor_version",
    "redundancy_correlation",
    "redundancy_overlap",
    # Partition identity
    "manager",
    "exchange",
    "market",
)


def build_detailed_audit_frame(
    factor_validation: pl.DataFrame,
    factor_selection: pl.DataFrame,
    *,
    top_n: int,
    manager: str,
    exchange: Exchange,
    market: Market,
    redundancy_audit: pl.DataFrame | None = None,
    candidate_n: int | None = None,
    max_factor_correlation: float | None = None,
    min_correlation_overlap: int | None = None,
) -> pl.DataFrame:
    """Build a detailed audit DataFrame for Factor Selection decisions.

    Selection decision fields (``selection_score``, ``selection_rank``,
    ``selected``, ``status``, ``selection_reason``) are taken from the
    canonical ``factor_selection`` frame and are not recomputed. Normalized
    components and contributions are produced by
    ``attach_selection_score_components`` so they match the locked engine.
    Redundancy audit columns are joined from ``redundancy_audit`` when
    provided.
    """
    from cqros.factor_selection.redundancy import (
        DEFAULT_CANDIDATE_N,
        DEFAULT_MAX_FACTOR_CORRELATION,
        DEFAULT_MIN_CORRELATION_OVERLAP,
    )

    validation = validate_factor_validation_frame(factor_validation)
    selection = _require_selection_frame(factor_selection)

    validation_prepared = _prepare_validation_side(validation)
    scored = attach_selection_score_components(validation_prepared)

    selection_side = selection.select(
        pl.col("factor_name"),
        pl.col("factor_version"),
        pl.col("timeframe"),
        pl.col("selection_score"),
        pl.col("selection_rank"),
        pl.col("selected"),
        pl.col("status"),
        pl.col("selection_reason"),
    )

    scored_side = scored.drop("selection_score")
    joined = scored_side.join(selection_side, on=list(_JOIN_KEYS), how="inner")
    if joined.height != selection.height:
        raise FactorSelectionError(
            "detailed audit join did not cover every factor selection row",
            error_code=_ERROR_JOIN_MISMATCH,
            details={
                "factor_validation_rows": validation.height,
                "factor_selection_rows": selection.height,
                "joined_rows": joined.height,
                "join_keys": _JOIN_KEYS,
            },
        )

    resolved_candidate_n = DEFAULT_CANDIDATE_N if candidate_n is None else int(candidate_n)
    resolved_max_corr = (
        DEFAULT_MAX_FACTOR_CORRELATION
        if max_factor_correlation is None
        else float(max_factor_correlation)
    )
    resolved_min_overlap = (
        DEFAULT_MIN_CORRELATION_OVERLAP
        if min_correlation_overlap is None
        else int(min_correlation_overlap)
    )

    if redundancy_audit is not None and redundancy_audit.height > 0:
        audit_cols = [
            "factor_name",
            "factor_version",
            "timeframe",
            "candidate_rank",
            "redundancy_checked",
            "redundancy_rejected",
            "redundancy_reference_factor",
            "redundancy_reference_factor_version",
            "redundancy_correlation",
            "redundancy_overlap",
            "candidate_n",
            "max_factor_correlation",
            "min_correlation_overlap",
        ]
        present = [column for column in audit_cols if column in redundancy_audit.columns]
        joined = joined.join(
            redundancy_audit.select(present),
            on=list(_JOIN_KEYS),
            how="left",
        )
        if "candidate_n" not in joined.columns:
            joined = joined.with_columns(pl.lit(resolved_candidate_n).alias("candidate_n"))
        if "max_factor_correlation" not in joined.columns:
            joined = joined.with_columns(pl.lit(resolved_max_corr).alias("max_factor_correlation"))
        if "min_correlation_overlap" not in joined.columns:
            joined = joined.with_columns(
                pl.lit(resolved_min_overlap).alias("min_correlation_overlap")
            )
    else:
        joined = joined.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("candidate_rank"),
            pl.lit(False).alias("redundancy_checked"),
            pl.lit(False).alias("redundancy_rejected"),
            pl.lit(None, dtype=pl.String).alias("redundancy_reference_factor"),
            pl.lit(None, dtype=pl.String).alias("redundancy_reference_factor_version"),
            pl.lit(None, dtype=pl.Float64).alias("redundancy_correlation"),
            pl.lit(None, dtype=pl.Int64).alias("redundancy_overlap"),
            pl.lit(resolved_candidate_n).alias("candidate_n"),
            pl.lit(resolved_max_corr).alias("max_factor_correlation"),
            pl.lit(resolved_min_overlap).alias("min_correlation_overlap"),
        )

    assembled = joined.with_columns(
        pl.lit(int(top_n)).alias("top_n"),
        pl.lit(SCORING_METHOD).alias("scoring_method"),
        pl.lit(NORMALIZATION_METHOD).alias("normalization_method"),
        pl.lit(manager).alias("manager"),
        pl.lit(exchange).alias("exchange"),
        pl.lit(market).alias("market"),
    ).select(list(DETAILED_AUDIT_COLUMNS))

    return assembled.sort(list(_SORT_COLUMNS))


def detailed_csv_path(
    storage_root: Path,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    timeframe: Timeframe,
    year: int,
) -> Path:
    """Return the per-timeframe detailed audit CSV path.

    Layout::

        factor_selection/{manager}/{exchange}/{market}/{timeframe}/{year}_detailed.csv
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_SELECTION
        / manager
        / exchange
        / market
        / timeframe
        / f"{year}_detailed{FILE_EXTENSION_CSV}"
    )


def combined_detailed_csv_path(
    storage_root: Path,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
) -> Path:
    """Return the combined detailed audit CSV path across timeframes.

    Layout::

        factor_selection/{manager}/{exchange}/{market}/factor_selection_detailed.csv
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_SELECTION
        / manager
        / exchange
        / market
        / COMBINED_DETAILED_CSV_NAME
    )


def write_detailed_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Write a detailed audit DataFrame to ``path`` as CSV.

    Args:
        frame: Detailed audit frame to persist.
        path: Destination CSV path.

    Returns:
        ``path`` after a successful write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.select(list(DETAILED_AUDIT_COLUMNS)).sort(list(_SORT_COLUMNS))
    ordered.write_csv(path)
    _logger.info(
        "Wrote factor selection detailed CSV",
        extra={"path": str(path), "rows": ordered.height, "columns": ordered.width},
    )
    return path


def write_combined_detailed_csv(frames: list[pl.DataFrame], path: Path) -> Path:
    """Concatenate detailed audit frames and write a combined CSV.

    Args:
        frames: Per-partition detailed audit frames.
        path: Destination combined CSV path.

    Returns:
        ``path`` after a successful write.

    Raises:
        FactorSelectionError: If ``frames`` is empty.
    """
    if len(frames) == 0:
        raise FactorSelectionError(
            "combined detailed CSV requires at least one audit frame",
            error_code=_ERROR_FRAMES_EMPTY,
            details={"frame_count": 0},
        )
    combined = pl.concat(frames, how="vertical_relaxed").select(list(DETAILED_AUDIT_COLUMNS))
    combined = combined.sort(list(_SORT_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(path)
    _logger.info(
        "Wrote factor selection combined detailed CSV",
        extra={"path": str(path), "rows": combined.height, "columns": combined.width},
    )
    return path


def contribution_sum_expression() -> pl.Expr:
    """Return an expression summing all weighted contribution columns."""
    total: pl.Expr = pl.lit(0.0)
    for column in _CONTRIBUTION_COLUMNS:
        total = total + pl.col(column)
    return total.alias("contribution_sum")


def _require_selection_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Factor Selection DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorSelectionError(
            "factor_selection frame must be a polars DataFrame",
            error_code=_ERROR_SELECTION_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorSelectionError(
            "factor_selection frame must contain at least one row",
            error_code=_ERROR_SELECTION_EMPTY,
            details={"rows": frame.height},
        )
    return frame


def _prepare_validation_side(frame: pl.DataFrame) -> pl.DataFrame:
    """Select validation metrics and ensure lineage context columns exist."""
    columns: list[pl.Expr] = [
        pl.col("factor_name"),
        pl.col("factor_version"),
        pl.col("factor_category"),
        pl.col("timeframe"),
        pl.col("validation_time"),
        pl.col("status").alias("validation_status"),
        pl.col("observations"),
        pl.col("information_coefficient"),
        pl.col("rank_information_coefficient"),
        pl.col("ic_information_ratio"),
        pl.col("ic_p_value"),
        pl.col("ic_decay"),
        pl.col("quantile_spread"),
        pl.col("monotonicity_score"),
        pl.col("turnover"),
    ]
    for column in _OPTIONAL_VALIDATION_CONTEXT:
        if column in frame.columns:
            columns.append(pl.col(column))
        else:
            dtype = pl.Int64 if column.endswith("_time") else pl.String
            columns.append(pl.lit(None, dtype=dtype).alias(column))
    return frame.select(columns)
