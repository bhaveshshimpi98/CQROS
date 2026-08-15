"""CQROS Monitoring Engine contracts and simple implementation.

Purpose:
    Convert a canonical reporting ledger into a deterministic
    monitoring DataFrame conforming to ``MONITORING_SCHEMA``.

Responsibilities:
    - Define ``MonitoringEngine`` as the shared monitoring contract
    - Provide ``SimpleMonitoringEngine`` for monitor-row generation
    - Validate reporting DataFrame structure
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.monitoring.exceptions``, and ``cqros.monitoring.schema``.

Public API:
    ``MonitoringEngine``, ``SimpleMonitoringEngine``,
    ``REPORTING_INPUT_COLUMNS``, ``validate_reporting_frame``
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.monitoring.exceptions import MonitoringValidationError
from cqros.monitoring.schema import (
    CANONICAL_COLUMN_ORDER,
    MONITORING_SCHEMA,
    MonitoringStatus,
)

__all__ = [
    "REPORTING_INPUT_COLUMNS",
    "MonitoringEngine",
    "SimpleMonitoringEngine",
    "validate_reporting_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "MON_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "MON_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "MON_MISSING_COLUMNS"
_ERROR_OPEN_TIME_ORDER: Final[str] = "MON_OPEN_TIME_ORDER"

_MONITOR_TYPE: Final[str] = "system"
_MONITOR_NAME: Final[str] = "report_monitor"
_SEVERITY: Final[str] = "NORMAL"
_METRIC_NAME: Final[str] = "report_generation"
_METRIC_VALUE: Final[float] = 1.0
_THRESHOLD: Final[float] = 1.0
_ALERT: Final[bool] = False

# Reporting columns required to assemble a monitoring row.
REPORTING_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "manager",
)


@runtime_checkable
class MonitoringEngine(Protocol):
    """Structural contract for converting reporting ledgers into monitoring.

    Implementations own monitoring-row semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, reporting: pl.DataFrame) -> pl.DataFrame:
        """Convert a reporting ledger into a monitoring DataFrame.

        Args:
            reporting: Canonical reporting dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``MONITORING_SCHEMA``.
        """
        ...


class SimpleMonitoringEngine:
    """Generate deterministic monitoring rows from reporting rows.

    Rules:
        - One output row per input reporting row
        - Identity columns (``symbol``, ``timeframe``, ``open_time``,
          ``manager``) are preserved from the input row
        - ``open_time`` is emitted as epoch milliseconds (``Int64``)
        - Monitor metadata fields are deterministic v1 constants
        - ``alert`` is always ``False``
        - ``status`` is always ``NORMAL``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        This engine emits monitoring metadata only; it does not evaluate
        thresholds or raise alerts.
    """

    __slots__ = ()

    def build(self, reporting: pl.DataFrame) -> pl.DataFrame:
        """Convert a reporting ledger into finalized monitoring rows.

        Args:
            reporting: Canonical reporting dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``MONITORING_SCHEMA``.

        Raises:
            MonitoringValidationError: If the input fails structural
                validation, required columns are missing, or timestamps
                are unsorted.
        """
        frame = validate_reporting_frame(reporting)
        _require_columns(frame, REPORTING_INPUT_COLUMNS, "reporting")
        ordered = frame.sort("open_time", maintain_order=True)
        _require_sorted_open_times(ordered)
        return _build_monitoring_rows(ordered)


def validate_reporting_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate reporting dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        MonitoringValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise MonitoringValidationError(
            "reporting frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": "reporting", "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise MonitoringValidationError(
            "reporting frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "reporting", "rows": frame.height},
        )
    return frame


def _build_monitoring_rows(reporting: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical monitoring rows from a sorted reporting ledger."""
    open_times = [_to_epoch_ms(value) for value in reporting["open_time"].to_list()]
    row_count = len(open_times)

    assembled = pl.DataFrame(
        {
            "symbol": reporting["symbol"].to_list(),
            "timeframe": reporting["timeframe"].to_list(),
            "open_time": open_times,
            "manager": reporting["manager"].to_list(),
            "monitor_type": [_MONITOR_TYPE] * row_count,
            "monitor_name": [_MONITOR_NAME] * row_count,
            "severity": [_SEVERITY] * row_count,
            "metric_name": [_METRIC_NAME] * row_count,
            "metric_value": [_METRIC_VALUE] * row_count,
            "threshold": [_THRESHOLD] * row_count,
            "alert": [_ALERT] * row_count,
            "status": [MonitoringStatus.NORMAL.value] * row_count,
        }
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MONITORING_SCHEMA)


def _to_epoch_ms(value: object) -> int:
    """Convert a reporting ``open_time`` value to epoch milliseconds."""
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000.0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise MonitoringValidationError(
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
        raise MonitoringValidationError(
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
            raise MonitoringValidationError(
                "open_time must be sorted in non-decreasing order",
                error_code=_ERROR_OPEN_TIME_ORDER,
                details={
                    "index": index,
                    "open_time": open_times[index],
                    "previous_open_time": open_times[index - 1],
                },
            )
