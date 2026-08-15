"""CQROS Reporting Engine contracts and simple implementation.

Purpose:
    Convert a canonical analytics-metrics ledger into a deterministic
    reporting DataFrame conforming to ``REPORTING_SCHEMA``.

Responsibilities:
    - Define ``ReportingEngine`` as the shared reporting contract
    - Provide ``SimpleReportingEngine`` for metadata-row generation
    - Validate analytics DataFrame structure
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.reporting.exceptions``, and ``cqros.reporting.schema``.

Public API:
    ``ReportingEngine``, ``SimpleReportingEngine``,
    ``ANALYTICS_INPUT_COLUMNS``, ``validate_analytics_frame``
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.schema import (
    CANONICAL_COLUMN_ORDER,
    REPORTING_SCHEMA,
    ReportingStatus,
)

__all__ = [
    "ANALYTICS_INPUT_COLUMNS",
    "ReportingEngine",
    "SimpleReportingEngine",
    "validate_analytics_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "REP_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "REP_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "REP_MISSING_COLUMNS"
_ERROR_OPEN_TIME_ORDER: Final[str] = "REP_OPEN_TIME_ORDER"

_REPORT_NAME: Final[str] = "performance_report"
_REPORT_TYPE: Final[str] = "analytics"
_REPORT_FORMAT: Final[str] = "parquet"
_REPORT_VERSION: Final[str] = "v1"
_REPORT_PATH: Final[str] = ""

# Analytics columns required to assemble a reporting row.
ANALYTICS_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
)


@runtime_checkable
class ReportingEngine(Protocol):
    """Structural contract for converting analytics ledgers into reporting.

    Implementations own reporting-metadata semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, analytics: pl.DataFrame) -> pl.DataFrame:
        """Convert an analytics ledger into a reporting DataFrame.

        Args:
            analytics: Canonical analytics dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``REPORTING_SCHEMA``.
        """
        ...


class SimpleReportingEngine:
    """Generate deterministic reporting metadata rows from analytics rows.

    Rules:
        - One output row per input analytics row
        - Identity columns (``symbol``, ``timeframe``, ``open_time``,
          ``manager``) are preserved from the input row
        - ``open_time`` is emitted as epoch milliseconds (``Int64``)
        - Report metadata fields are deterministic v1 constants
        - ``generated_at`` equals the row ``open_time``
        - ``status`` is always ``GENERATED``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        This engine emits reporting metadata only; it does not create
        files or render PDF/HTML/Markdown artifacts.
    """

    __slots__ = ()

    def build(self, analytics: pl.DataFrame) -> pl.DataFrame:
        """Convert an analytics ledger into finalized reporting metadata.

        Args:
            analytics: Canonical analytics dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``REPORTING_SCHEMA``.

        Raises:
            ReportingValidationError: If the input fails structural
                validation, required columns are missing, or timestamps
                are unsorted.
        """
        frame = validate_analytics_frame(analytics)
        _require_columns(frame, ANALYTICS_INPUT_COLUMNS, "analytics")
        ordered = frame.sort("open_time", maintain_order=True)
        _require_sorted_open_times(ordered)
        return _build_reporting_rows(ordered)


def validate_analytics_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate analytics dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        ReportingValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise ReportingValidationError(
            "analytics frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": "analytics", "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise ReportingValidationError(
            "analytics frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "analytics", "rows": frame.height},
        )
    return frame


def _build_reporting_rows(analytics: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical reporting rows from a sorted analytics ledger."""
    open_times = [_to_epoch_ms(value) for value in analytics["open_time"].to_list()]
    row_count = len(open_times)

    assembled = pl.DataFrame(
        {
            "symbol": analytics["symbol"].to_list(),
            "timeframe": analytics["timeframe"].to_list(),
            "open_time": open_times,
            "manager": analytics["manager"].to_list(),
            "report_name": [_REPORT_NAME] * row_count,
            "report_type": [_REPORT_TYPE] * row_count,
            "report_format": [_REPORT_FORMAT] * row_count,
            "report_version": [_REPORT_VERSION] * row_count,
            "report_path": [_REPORT_PATH] * row_count,
            "generated_at": open_times,
            "status": [ReportingStatus.GENERATED.value] * row_count,
        }
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(REPORTING_SCHEMA)


def _to_epoch_ms(value: object) -> int:
    """Convert an analytics ``open_time`` value to epoch milliseconds."""
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000.0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise ReportingValidationError(
        "open_time must be datetime or integer epoch milliseconds",
        error_code=_ERROR_FRAME_TYPE,
        details={"actual_type": type(value).__name__, "value": repr(value)},
    )


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ReportingValidationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_sorted_open_times(frame: pl.DataFrame) -> None:
    """Raise when ``open_time`` is not non-decreasing after sorting."""
    open_times = frame["open_time"].to_list()
    for index in range(1, len(open_times)):
        if open_times[index] < open_times[index - 1]:
            raise ReportingValidationError(
                "open_time must be sorted in non-decreasing order",
                error_code=_ERROR_OPEN_TIME_ORDER,
                details={
                    "index": index,
                    "open_time": open_times[index],
                    "previous_open_time": open_times[index - 1],
                },
            )
