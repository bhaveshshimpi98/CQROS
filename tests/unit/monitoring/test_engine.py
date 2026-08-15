"""Unit tests for CQROS ``SimpleMonitoringEngine``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.monitoring import (
    MONITORING_SCHEMA,
    MonitoringStatus,
    MonitoringValidationError,
    SimpleMonitoringEngine,
)
from cqros.monitoring.engine import REPORTING_INPUT_COLUMNS, validate_reporting_frame
from cqros.monitoring.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_MANAGER = "simple"
_SYMBOL = "BTCUSDT"


def _open_time(index: int = 0) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)


def _epoch_ms(value: datetime) -> int:
    """Convert a UTC datetime to epoch milliseconds."""
    return int(value.timestamp() * 1000.0)


def _reporting_frame(
    *,
    open_times: list[datetime] | list[int] | None = None,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
) -> pl.DataFrame:
    """Build a minimal reporting frame for monitoring engine tests."""
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
    engine: SimpleMonitoringEngine,
    *,
    reporting: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build monitoring rows with a default reporting frame."""
    return engine.build(reporting if reporting is not None else _reporting_frame())


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """REPORTING_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "symbol",
        "timeframe",
        "open_time",
        "manager",
    ):
        assert column in REPORTING_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_reporting_frame_rejects_non_dataframe() -> None:
    """validate_reporting_frame rejects non-DataFrame inputs with MON_FRAME_TYPE."""
    with pytest.raises(MonitoringValidationError) as exc_info:
        validate_reporting_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "MON_FRAME_TYPE"


def test_validate_reporting_frame_rejects_empty_dataframe() -> None:
    """validate_reporting_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"symbol": []})
    with pytest.raises(MonitoringValidationError) as exc_info:
        validate_reporting_frame(empty)
    assert exc_info.value.error_code == "MON_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty reporting frames."""
    empty = pl.DataFrame(schema={column: pl.Utf8 for column in ("symbol",)}).clear()
    with pytest.raises(MonitoringValidationError) as exc_info:
        SimpleMonitoringEngine().build(empty)
    assert exc_info.value.error_code == "MON_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_reporting_columns() -> None:
    """Missing required reporting columns raise MON_MISSING_COLUMNS."""
    engine = SimpleMonitoringEngine()
    with pytest.raises(MonitoringValidationError) as exc_info:
        _build(engine, reporting=_reporting_frame().drop("manager"))
    assert exc_info.value.error_code == "MON_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Monitor metadata generation
# ---------------------------------------------------------------------------


def test_monitor_metadata_fields_are_deterministic_constants() -> None:
    """Monitor metadata fields use deterministic v1 constants."""
    result = _build(SimpleMonitoringEngine())
    assert result["monitor_type"].to_list() == ["system"]
    assert result["monitor_name"].to_list() == ["report_monitor"]
    assert result["severity"].to_list() == ["NORMAL"]
    assert result["metric_name"].to_list() == ["report_generation"]
    assert result["metric_value"].to_list() == [1.0]
    assert result["threshold"].to_list() == [1.0]
    assert result["alert"].to_list() == [False]


def test_one_monitoring_row_per_reporting_row() -> None:
    """Engine emits exactly one monitoring row for each reporting row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleMonitoringEngine(),
        reporting=_reporting_frame(open_times=[t0, t1]),
    )
    assert result.height == 2


# ---------------------------------------------------------------------------
# Status generation
# ---------------------------------------------------------------------------


def test_status_is_normal_for_every_row() -> None:
    """status is NORMAL for every monitoring row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleMonitoringEngine(),
        reporting=_reporting_frame(open_times=[t0, t1]),
    )
    assert result["status"].to_list() == [
        MonitoringStatus.NORMAL.value,
        MonitoringStatus.NORMAL.value,
    ]


def test_single_row_preserves_normal_status() -> None:
    """A single-row reporting ledger produces NORMAL status."""
    result = _build(SimpleMonitoringEngine())
    assert result["status"].to_list() == [MonitoringStatus.NORMAL.value]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and MONITORING_SCHEMA dtypes."""
    result = _build(SimpleMonitoringEngine())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == MONITORING_SCHEMA
    assert result.schema["open_time"] == pl.Int64
    assert result.schema["metric_value"] == pl.Float64
    assert result.schema["threshold"] == pl.Float64
    assert result.schema["alert"] == pl.Boolean
    assert result.schema["monitor_name"] == pl.Utf8
    assert result.schema["status"] == pl.Utf8


def test_open_time_converted_to_epoch_milliseconds() -> None:
    """open_time is emitted as Int64 epoch milliseconds."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleMonitoringEngine(),
        reporting=_reporting_frame(open_times=[t0, t1]),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1)]


def test_manager_is_preserved_on_every_row() -> None:
    """manager column preserves upstream lineage on every row."""
    t0, t1 = _open_time(0), _open_time(1)
    result = _build(
        SimpleMonitoringEngine(),
        reporting=_reporting_frame(
            manager="custom-manager",
            open_times=[t0, t1],
        ),
    )
    assert result["manager"].to_list() == ["custom-manager", "custom-manager"]


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied reporting frame."""
    reporting = _reporting_frame()
    before = reporting.clone()
    SimpleMonitoringEngine().build(reporting)
    assert_frame_equal(reporting, before)


def test_output_is_deterministic() -> None:
    """Identical reporting inputs produce identical monitoring outputs."""
    reporting = _reporting_frame(open_times=[_open_time(0), _open_time(1)])
    engine = SimpleMonitoringEngine()
    first = engine.build(reporting)
    second = engine.build(reporting)
    assert_frame_equal(first, second)


def test_multiple_timestamps_sorted_by_open_time() -> None:
    """Output rows are sorted by open_time ascending."""
    t0, t2, t1 = _open_time(0), _open_time(2), _open_time(1)
    result = _build(
        SimpleMonitoringEngine(),
        reporting=_reporting_frame(open_times=[t2, t0, t1]),
    )
    assert result["open_time"].to_list() == [_epoch_ms(t0), _epoch_ms(t1), _epoch_ms(t2)]
