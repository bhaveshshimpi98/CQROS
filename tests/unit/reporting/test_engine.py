"""Unit tests for CQROS ``SimpleReportingEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.reporting import (
    REPORTING_SCHEMA,
    ReportingStatus,
    ReportingValidationError,
    SimpleReportingEngine,
)
from cqros.reporting.engine import ANALYTICS_INPUT_COLUMNS, validate_analytics_frame
from cqros.reporting.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_MANAGER = "simple"
_SYMBOL = "BTCUSDT"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _epoch_ms(value: datetime) -> int:
    """Convert a UTC datetime to epoch milliseconds."""
    return int(value.timestamp() * 1000.0)


def _analytics_frame(
    *,
    open_times: list[datetime] | list[int] | None = None,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
) -> pl.DataFrame:
    """Build a minimal analytics frame for reporting engine tests."""
    open_times = open_times if open_times is not None else [_open_time(0)]
    row_count = len(open_times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": open_times,
            "manager": [manager] * row_count,
        }
    )


def _build(
    engine: SimpleReportingEngine,
    *,
    analytics: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build reporting metadata with a default analytics frame."""
    return engine.build(analytics if analytics is not None else _analytics_frame())


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """ANALYTICS_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
    ):
        assert column in ANALYTICS_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_analytics_frame_rejects_non_dataframe() -> None:
    """validate_analytics_frame rejects non-DataFrame inputs with REP_FRAME_TYPE."""
    with pytest.raises(ReportingValidationError) as exc_info:
        validate_analytics_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "REP_FRAME_TYPE"


def test_validate_analytics_frame_rejects_empty_dataframe() -> None:
    """validate_analytics_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"symbol": []})
    with pytest.raises(ReportingValidationError) as exc_info:
        validate_analytics_frame(empty)
    assert exc_info.value.error_code == "REP_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty analytics frames."""
    empty = pl.DataFrame(schema={column: pl.Utf8 for column in ("symbol",)}).clear()
    with pytest.raises(ReportingValidationError) as exc_info:
        SimpleReportingEngine().build(empty)
    assert exc_info.value.error_code == "REP_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_analytics_columns() -> None:
    """Missing required analytics columns raise REP_MISSING_COLUMNS."""
    engine = SimpleReportingEngine()
    with pytest.raises(ReportingValidationError) as exc_info:
        _build(engine, analytics=_analytics_frame().drop("manager"))
    assert exc_info.value.error_code == "REP_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Report metadata generation
# ---------------------------------------------------------------------------


def test_report_metadata_fields_are_deterministic_constants() -> None:
    """Report metadata fields use deterministic v1 constants."""
    result = _build(SimpleReportingEngine())
    assert result["report_name"].to_list() == ["performance_report"]
    assert result["report_type"].to_list() == ["analytics"]
    assert result["report_format"].to_list() == ["parquet"]
    assert result["report_version"].to_list() == ["v1"]
    assert result["report_path"].to_list() == [""]


def test_one_reporting_row_per_analytics_row() -> None:
    """Engine emits exactly one reporting row for each analytics row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(open_times=[t0, t1]),
    )
    assert result.height == 2


def test_generated_at_equals_open_time() -> None:
    """generated_at is copied from the row open_time epoch milliseconds."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(open_times=[t0, t1]),
    )
    open_times = result["open_time"].to_list()
    assert result["generated_at"].to_list() == open_times
    assert open_times == [_epoch_ms(t0), _epoch_ms(t1)]


# ---------------------------------------------------------------------------
# Status generation
# ---------------------------------------------------------------------------


def test_status_is_generated_for_every_row() -> None:
    """status is GENERATED for every reporting row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(open_times=[t0, t1]),
    )
    assert result["status"].to_list() == [
        ReportingStatus.GENERATED.value,
        ReportingStatus.GENERATED.value,
    ]


def test_single_row_preserves_generated_status() -> None:
    """A single-row analytics ledger produces GENERATED status."""
    result = _build(SimpleReportingEngine())
    assert result["status"].to_list() == [ReportingStatus.GENERATED.value]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and REPORTING_SCHEMA dtypes."""
    result = _build(SimpleReportingEngine())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == REPORTING_SCHEMA
    assert result.schema["open_time"] == pl.Int64
    assert result.schema["generated_at"] == pl.Int64
    assert result.schema["report_name"] == pl.Utf8
    assert result.schema["status"] == pl.Utf8


def test_open_time_converted_to_epoch_milliseconds() -> None:
    """open_time is emitted as Int64 epoch milliseconds."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(open_times=[t0, t1]),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1)]


def test_manager_is_preserved_on_every_row() -> None:
    """manager column preserves upstream lineage on every row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(
            manager="custom-manager",
            open_times=[t0, t1],
        ),
    )
    assert result["manager"].to_list() == ["custom-manager", "custom-manager"]


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied analytics frame."""
    analytics = _analytics_frame()
    before = analytics.clone()
    SimpleReportingEngine().build(analytics)
    assert_frame_equal(analytics, before)


def test_output_is_deterministic() -> None:
    """Identical analytics inputs produce identical reporting outputs."""
    analytics = _analytics_frame(open_times=[_open_time(0), _open_time(1)])
    engine = SimpleReportingEngine()
    first = engine.build(analytics)
    second = engine.build(analytics)
    assert_frame_equal(first, second)


def test_multiple_timestamps_sorted_by_open_time() -> None:
    """Output rows are sorted by open_time ascending."""
    t0, t2, t1 = _open_time(0), _open_time(2), _open_time(1)
    result = _build(
        SimpleReportingEngine(),
        analytics=_analytics_frame(open_times=[t2, t0, t1]),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1), _epoch_ms(t2)]
    assert result["generated_at"].to_list() == [_epoch_ms(t0), _epoch_ms(t1), _epoch_ms(t2)]
