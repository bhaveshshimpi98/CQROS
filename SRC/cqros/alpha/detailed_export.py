"""CQROS Alpha detailed CSV audit export.

Purpose:
    Produce a research-auditable CSV representation of Alpha partitions
    without replacing the canonical Parquet dataset.

Responsibilities:
    - Write per-partition detailed CSV files from Alpha frames
    - Write optional combined detailed CSV files across partitions
    - Export only ``ALPHA_SCHEMA`` columns (no lineage fields)
    - Remain free of alpha algorithms and repository filesystem walks

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``,
    ``cqros.alpha.exceptions``, and ``cqros.alpha.schema``.

Public API:
    ``COMBINED_DETAILED_CSV_NAME``, ``detailed_csv_path``,
    ``combined_detailed_csv_path``, ``write_detailed_csv``,
    ``write_combined_detailed_csv``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

from cqros.alpha.exceptions import AlphaError
from cqros.alpha.schema import ALPHA_COLUMNS, ALPHA_SCHEMA
from cqros.core.constants import FILE_EXTENSION_CSV, STORAGE_DIR_ALPHA
from cqros.core.types import Exchange, Market, Symbol, Timeframe

__all__ = [
    "COMBINED_DETAILED_CSV_NAME",
    "combined_detailed_csv_path",
    "detailed_csv_path",
    "write_combined_detailed_csv",
    "write_detailed_csv",
]

_logger = logging.getLogger(__name__)

_ERROR_FRAME_TYPE: Final[str] = "ALPHA_DETAILED_FRAME_TYPE"
_ERROR_FRAMES_EMPTY: Final[str] = "ALPHA_DETAILED_FRAMES_EMPTY"

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_set_id",
    "prediction_time",
)

COMBINED_DETAILED_CSV_NAME: Final[str] = f"alpha_detailed{FILE_EXTENSION_CSV}"


def detailed_csv_path(
    storage_root: Path,
    *,
    manager: str,
    exchange: Exchange,
    market: Market,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
) -> Path:
    """Return the per-partition detailed Alpha CSV path.

    Layout::

        alpha/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}_detailed.csv
    """
    return (
        storage_root
        / STORAGE_DIR_ALPHA
        / manager
        / exchange
        / market
        / symbol
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
    """Return the combined detailed Alpha CSV path across partitions."""
    return (
        storage_root / STORAGE_DIR_ALPHA / manager / exchange / market / COMBINED_DETAILED_CSV_NAME
    )


def write_detailed_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Write an Alpha DataFrame to ``path`` as CSV using ``ALPHA_SCHEMA`` columns."""
    ordered = _require_alpha_columns(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = ordered.sort(list(_SORT_COLUMNS), nulls_last=True)
    ordered.write_csv(path)
    _logger.info(
        "Wrote alpha detailed CSV",
        extra={"path": str(path), "rows": ordered.height, "columns": ordered.width},
    )
    return path


def write_combined_detailed_csv(frames: list[pl.DataFrame], path: Path) -> Path:
    """Concatenate Alpha frames and write a combined detailed CSV."""
    if len(frames) == 0:
        raise AlphaError(
            "combined detailed CSV requires at least one alpha frame",
            error_code=_ERROR_FRAMES_EMPTY,
            details={"frame_count": 0},
        )
    combined = pl.concat(
        [_require_alpha_columns(frame) for frame in frames],
        how="vertical_relaxed",
    ).select(list(ALPHA_COLUMNS))
    combined = combined.sort(list(_SORT_COLUMNS), nulls_last=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(path)
    _logger.info(
        "Wrote alpha combined detailed CSV",
        extra={"path": str(path), "rows": combined.height, "columns": combined.width},
    )
    return path


def _require_alpha_columns(frame: object) -> pl.DataFrame:
    """Validate ``frame`` and select ``ALPHA_SCHEMA`` columns."""
    if not isinstance(frame, pl.DataFrame):
        raise AlphaError(
            "alpha detailed export frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame.select(list(ALPHA_COLUMNS)).cast(ALPHA_SCHEMA)
