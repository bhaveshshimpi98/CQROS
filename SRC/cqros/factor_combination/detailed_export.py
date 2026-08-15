"""CQROS Factor Combination detailed CSV audit export.

Purpose:
    Produce a research-auditable CSV representation of Factor Combination
    decisions without replacing the canonical Parquet dataset.

Responsibilities:
    - Assemble one detailed audit row per combination per timeframe with
      lineage provenance columns attached
    - Write per-timeframe and optional combined detailed CSV files
    - Remain free of combination algorithm changes and Phase 3 logic

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``, and
    ``cqros.factor_combination.exceptions``.

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

from cqros.core.constants import FILE_EXTENSION_CSV, STORAGE_DIR_FACTOR_COMBINATION
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_combination.exceptions import FactorCombinationError

__all__ = [
    "COMBINED_DETAILED_CSV_NAME",
    "DETAILED_AUDIT_COLUMNS",
    "build_detailed_audit_frame",
    "combined_detailed_csv_path",
    "detailed_csv_path",
    "write_combined_detailed_csv",
    "write_detailed_csv",
]

_logger = logging.getLogger(__name__)

_ERROR_COMBINATION_TYPE: Final[str] = "FCOMB_DETAILED_COMBINATION_TYPE"
_ERROR_COMBINATION_EMPTY: Final[str] = "FCOMB_DETAILED_COMBINATION_EMPTY"
_ERROR_FRAMES_EMPTY: Final[str] = "FCOMB_DETAILED_FRAMES_EMPTY"

_LIST_COLUMNS: Final[tuple[str, ...]] = (
    "factor_names",
    "factor_versions",
    "factor_categories",
)

_LIST_SEPARATOR: Final[str] = "|"

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "combination_rank",
    "combination_id",
)

COMBINED_DETAILED_CSV_NAME: Final[str] = f"factor_combination_detailed{FILE_EXTENSION_CSV}"

# Combination schema fields + lineage provenance + partition identity.
DETAILED_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    # Combination identity
    "combination_id",
    "factor_names",
    "factor_versions",
    "factor_categories",
    "timeframe",
    "combination_size",
    "combination_method",
    "analysis_time",
    # Combination metrics
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "quantile_spread",
    "hit_rate",
    "turnover",
    "correlation_penalty",
    "diversification_score",
    "stability_score",
    "confidence_score",
    "combination_score",
    "combination_rank",
    "status",
    # Lineage provenance
    "source_fta_version",
    "source_selection_version",
    # Partition identity
    "manager",
    "exchange",
    "market",
    "year",
)


def build_detailed_audit_frame(
    combination_frame: pl.DataFrame,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    year: int,
    source_fta_version: str,
    source_selection_version: str,
) -> pl.DataFrame:
    """Build a detailed audit DataFrame for Factor Combination decisions.

    Attaches lineage provenance columns and partition identity to the
    canonical combination frame. The canonical combination columns are
    preserved unchanged.

    Args:
        combination_frame: Canonical Factor Combination DataFrame.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year of the partition.
        source_fta_version: Version string of the Factor Timeframe Analysis
            dataset that generated this combination partition.
        source_selection_version: Version string of the Factor Selection
            dataset consumed by the FTA pipeline.

    Returns:
        Augmented DataFrame with ``DETAILED_AUDIT_COLUMNS`` in canonical
        order, sorted by timeframe, combination_rank, and combination_id.

    Raises:
        FactorCombinationError: If ``combination_frame`` is not a DataFrame
            or is empty.
    """
    validated = _require_combination_frame(combination_frame)
    assembled = validated.with_columns(
        pl.lit(source_fta_version).alias("source_fta_version"),
        pl.lit(source_selection_version).alias("source_selection_version"),
        pl.lit(manager).alias("manager"),
        pl.lit(exchange).alias("exchange"),
        pl.lit(market).alias("market"),
        pl.lit(int(year)).alias("year"),
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

        factor_combination/{manager}/{exchange}/{market}/{timeframe}/{year}_detailed.csv

    Args:
        storage_root: Data lake storage root directory.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        timeframe: Trade bar interval.
        year: Calendar year of the partition.

    Returns:
        Path to ``{year}_detailed.csv`` under the per-timeframe hierarchy.
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_COMBINATION
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

        factor_combination/{manager}/{exchange}/{market}/factor_combination_detailed.csv

    Args:
        storage_root: Data lake storage root directory.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.

    Returns:
        Path to ``factor_combination_detailed.csv`` under the
        manager/exchange/market directory.
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_COMBINATION
        / manager
        / exchange
        / market
        / COMBINED_DETAILED_CSV_NAME
    )


def write_detailed_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Write a detailed audit DataFrame to ``path`` as CSV.

    List-typed columns (``factor_names``, ``factor_versions``,
    ``factor_categories``) are serialized to pipe-delimited strings before
    writing because CSV does not support nested data.

    Args:
        frame: Detailed audit frame to persist. Must contain all
            ``DETAILED_AUDIT_COLUMNS``.
        path: Destination CSV path.

    Returns:
        ``path`` after a successful write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.select(list(DETAILED_AUDIT_COLUMNS)).sort(list(_SORT_COLUMNS))
    ordered = _serialize_list_columns(ordered)
    ordered.write_csv(path)
    _logger.info(
        "Wrote factor combination detailed CSV",
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
        FactorCombinationError: If ``frames`` is empty.
    """
    if len(frames) == 0:
        raise FactorCombinationError(
            "combined detailed CSV requires at least one audit frame",
            error_code=_ERROR_FRAMES_EMPTY,
            details={"frame_count": 0},
        )
    combined = pl.concat(frames, how="vertical_relaxed").select(list(DETAILED_AUDIT_COLUMNS))
    combined = combined.sort(list(_SORT_COLUMNS))
    combined = _serialize_list_columns(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(path)
    _logger.info(
        "Wrote factor combination combined detailed CSV",
        extra={"path": str(path), "rows": combined.height, "columns": combined.width},
    )
    return path


def _serialize_list_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert known List[String] columns to pipe-delimited String columns.

    Required before writing CSV because CSV format does not support nested
    (list) data types.

    Args:
        frame: DataFrame potentially containing list columns.

    Returns:
        DataFrame with list columns converted to pipe-delimited strings.
    """
    exprs: list[pl.Expr] = []
    for col in frame.columns:
        if col in _LIST_COLUMNS and frame.schema[col] == pl.List(pl.String):
            exprs.append(pl.col(col).list.join(_LIST_SEPARATOR).alias(col))
        else:
            exprs.append(pl.col(col))
    return frame.select(exprs)


def _require_combination_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Factor Combination DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorCombinationError(
            "combination_frame must be a polars DataFrame",
            error_code=_ERROR_COMBINATION_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorCombinationError(
            "combination_frame must contain at least one row",
            error_code=_ERROR_COMBINATION_EMPTY,
            details={"rows": frame.height},
        )
    return frame
