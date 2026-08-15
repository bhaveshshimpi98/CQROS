"""CQROS Factor Orthogonalization detailed CSV audit export.

Purpose:
    Produce a research-auditable CSV representation of Factor Orthogonalization
    decisions without replacing the canonical Parquet dataset.

Responsibilities:
    - Assemble one detailed audit row per combination with lineage and
      partition identity attached
    - Write per-timeframe and optional combined detailed CSV files
    - Remain free of orthogonalization algorithm changes

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``, and
    ``cqros.factor_orthogonalization.exceptions``.

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

from cqros.core.constants import FILE_EXTENSION_CSV, STORAGE_DIR_FACTOR_ORTHOGONALIZATION
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.schema import CANONICAL_COLUMN_ORDER

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

_ERROR_ORTHO_TYPE: Final[str] = "FORTH_DETAILED_ORTHO_TYPE"
_ERROR_ORTHO_EMPTY: Final[str] = "FORTH_DETAILED_ORTHO_EMPTY"
_ERROR_FRAMES_EMPTY: Final[str] = "FORTH_DETAILED_FRAMES_EMPTY"

_LIST_COLUMNS: Final[tuple[str, ...]] = (
    "factor_names",
    "factor_versions",
    "factor_categories",
)

_LIST_SEPARATOR: Final[str] = "|"

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "orthogonalization_rank",
    "source_combination_rank",
    "combination_id",
)

COMBINED_DETAILED_CSV_NAME: Final[str] = f"factor_orthogonalization_detailed{FILE_EXTENSION_CSV}"

DETAILED_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    *CANONICAL_COLUMN_ORDER,
    "manager",
    "exchange",
    "market",
    "year",
)


def build_detailed_audit_frame(
    orthogonalization_frame: pl.DataFrame,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    year: int,
) -> pl.DataFrame:
    """Build a detailed audit DataFrame for Factor Orthogonalization decisions."""
    validated = _require_orthogonalization_frame(orthogonalization_frame)
    assembled = validated.with_columns(
        pl.lit(manager).alias("manager"),
        pl.lit(exchange).alias("exchange"),
        pl.lit(market).alias("market"),
        pl.lit(int(year)).alias("year"),
    ).select(list(DETAILED_AUDIT_COLUMNS))
    return assembled.sort(list(_SORT_COLUMNS), nulls_last=True)


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

        factor_orthogonalization/{manager}/{exchange}/{market}/{timeframe}/{year}_detailed.csv
    """
    return (
        storage_root
        / STORAGE_DIR_FACTOR_ORTHOGONALIZATION
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
    """Return the combined detailed audit CSV path across timeframes."""
    return (
        storage_root
        / STORAGE_DIR_FACTOR_ORTHOGONALIZATION
        / manager
        / exchange
        / market
        / COMBINED_DETAILED_CSV_NAME
    )


def write_detailed_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Write a detailed audit DataFrame to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.select(list(DETAILED_AUDIT_COLUMNS)).sort(
        list(_SORT_COLUMNS),
        nulls_last=True,
    )
    ordered = _serialize_list_columns(ordered)
    ordered.write_csv(path)
    _logger.info(
        "Wrote factor orthogonalization detailed CSV",
        extra={"path": str(path), "rows": ordered.height, "columns": ordered.width},
    )
    return path


def write_combined_detailed_csv(frames: list[pl.DataFrame], path: Path) -> Path:
    """Concatenate detailed audit frames and write a combined CSV."""
    if len(frames) == 0:
        raise FactorOrthogonalizationError(
            "combined detailed CSV requires at least one audit frame",
            error_code=_ERROR_FRAMES_EMPTY,
            details={"frame_count": 0},
        )
    combined = pl.concat(frames, how="vertical_relaxed").select(list(DETAILED_AUDIT_COLUMNS))
    combined = combined.sort(list(_SORT_COLUMNS), nulls_last=True)
    combined = _serialize_list_columns(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(path)
    _logger.info(
        "Wrote factor orthogonalization combined detailed CSV",
        extra={"path": str(path), "rows": combined.height, "columns": combined.width},
    )
    return path


def _serialize_list_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert known List[String] columns to pipe-delimited String columns."""
    exprs: list[pl.Expr] = []
    for col in frame.columns:
        if col in _LIST_COLUMNS and frame.schema[col] == pl.List(pl.String):
            exprs.append(pl.col(col).list.join(_LIST_SEPARATOR).alias(col))
        else:
            exprs.append(pl.col(col))
    return frame.select(exprs)


def _require_orthogonalization_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty orthogonalization DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorOrthogonalizationError(
            "orthogonalization_frame must be a polars DataFrame",
            error_code=_ERROR_ORTHO_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorOrthogonalizationError(
            "orthogonalization_frame must contain at least one row",
            error_code=_ERROR_ORTHO_EMPTY,
            details={"rows": frame.height},
        )
    return frame
