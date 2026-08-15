"""CQROS Factor Timeframe Analysis detailed CSV audit export.

Purpose:
    Produce a research-auditable CSV representation of Factor Timeframe
    Analysis decisions without replacing the canonical Parquet dataset.

Responsibilities:
    - Assemble one detailed audit row per FTA factor, preserving all FTA
      schema columns verbatim
    - Append partition identity and source timeframe lineage fields
    - Write per-year detailed CSV files alongside the Parquet partitions
    - Remain free of engine internals, ranking changes, and Phase 3 logic

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``,
    ``cqros.factor_timeframe_analysis.exceptions``, and
    ``cqros.factor_timeframe_analysis.schema``.

Public API:
    ``DETAILED_AUDIT_COLUMNS``, ``build_detailed_audit_frame``,
    ``detailed_csv_path``, ``write_detailed_csv``

Notes:
    ``source_timeframes`` is a caller-supplied comma-joined string of the
    Factor Selection timeframes whose partitions were loaded when producing
    the FTA frame (e.g. ``"1h,4h,1d"``). It is stored verbatim for lineage
    and is not parsed by this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import FILE_EXTENSION_CSV, STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS
from cqros.core.types import Exchange, Market
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.factor_timeframe_analysis.schema import FACTOR_TIMEFRAME_ANALYSIS_COLUMNS

__all__ = [
    "DETAILED_AUDIT_COLUMNS",
    "build_detailed_audit_frame",
    "detailed_csv_path",
    "write_detailed_csv",
]

_logger = logging.getLogger(__name__)

_ERROR_FTA_TYPE: Final[str] = "FTA_DETAILED_FRAME_TYPE"
_ERROR_FTA_EMPTY: Final[str] = "FTA_DETAILED_FRAME_EMPTY"

_SORT_COLUMNS: Final[tuple[str, ...]] = ("factor_name", "factor_version")

DETAILED_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    # All FTA schema columns (verbatim from FACTOR_TIMEFRAME_ANALYSIS_COLUMNS)
    *FACTOR_TIMEFRAME_ANALYSIS_COLUMNS,
    # Partition identity and source lineage
    "manager",
    "exchange",
    "market",
    "year",
    "source_timeframes",
)


def build_detailed_audit_frame(
    fta_frame: pl.DataFrame,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    year: int,
    source_timeframes: str,
) -> pl.DataFrame:
    """Build a detailed audit DataFrame for Factor Timeframe Analysis decisions.

    All FTA schema columns are preserved verbatim from ``fta_frame``. Partition
    identity (manager, exchange, market, year) and the joined source timeframes
    string are appended as lineage columns.

    Args:
        fta_frame: Canonical FTA DataFrame conforming to
            ``TIMEFRAME_ANALYSIS_SCHEMA``. Must not be empty.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year of the partition.
        source_timeframes: Comma-joined string of Factor Selection timeframes
            that were loaded when producing the FTA frame (e.g. ``"1h,4h,1d"``).

    Returns:
        Detailed audit DataFrame with ``DETAILED_AUDIT_COLUMNS`` column order,
        sorted by factor_name, factor_version.

    Raises:
        FactorTimeframeAnalysisError: If ``fta_frame`` is not a polars
            DataFrame or contains no rows.
    """
    _require_fta_frame(fta_frame)
    assembled = fta_frame.with_columns(
        pl.lit(manager).alias("manager"),
        pl.lit(exchange).alias("exchange"),
        pl.lit(market).alias("market"),
        pl.lit(int(year)).alias("year"),
        pl.lit(source_timeframes).alias("source_timeframes"),
    ).select(list(DETAILED_AUDIT_COLUMNS))
    return assembled.sort(list(_SORT_COLUMNS))


def detailed_csv_path(
    storage_root: Path,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    year: int,
) -> Path:
    """Return the per-year detailed audit CSV path.

    Layout::

        factor_timeframe_analysis/{manager}/{exchange}/{market}/{year}_detailed.csv

    Args:
        storage_root: Data lake storage root directory.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year of the partition.

    Returns:
        Absolute path to the ``{year}_detailed.csv`` file.
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS
        / manager
        / exchange
        / market
        / f"{year}_detailed{FILE_EXTENSION_CSV}"
    )


def write_detailed_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Write a detailed audit DataFrame to ``path`` as CSV.

    Creates parent directories if they do not exist. The frame is re-ordered
    to ``DETAILED_AUDIT_COLUMNS`` and sorted by factor_name, factor_version
    before writing.

    Args:
        frame: Detailed audit frame to persist. Must contain all
            ``DETAILED_AUDIT_COLUMNS`` columns.
        path: Destination CSV path.

    Returns:
        ``path`` after a successful write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.select(list(DETAILED_AUDIT_COLUMNS)).sort(list(_SORT_COLUMNS))
    ordered.write_csv(path)
    _logger.info(
        "Wrote factor timeframe analysis detailed CSV",
        extra={"path": str(path), "rows": ordered.height, "columns": ordered.width},
    )
    return path


def _require_fta_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty FTA polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorTimeframeAnalysisError(
            "fta_frame must be a polars DataFrame",
            error_code=_ERROR_FTA_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorTimeframeAnalysisError(
            "fta_frame must contain at least one row",
            error_code=_ERROR_FTA_EMPTY,
            details={"rows": frame.height},
        )
    return frame
