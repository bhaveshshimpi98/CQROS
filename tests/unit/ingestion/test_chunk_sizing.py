"""Unit tests for CQROS adaptive download chunk sizing."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from cqros.core.constants import (
    DAYS_PER_WEEK,
    MILLISECONDS_PER_DAY,
    MILLISECONDS_PER_HOUR,
    MILLISECONDS_PER_MINUTE,
)
from cqros.core.exceptions import ValidationError
from cqros.ingestion.chunk_sizing import (
    DAYS_PER_MONTH_APPROXIMATION,
    DEFAULT_CHUNK_SAFETY_FACTOR,
    DOWNLOAD_TIMEFRAMES,
    AdaptiveChunkSizingStrategy,
    FixedChunkSizingStrategy,
    effective_kline_count,
    timeframe_duration_ms,
)
from cqros.ingestion.downloader import (
    DEFAULT_KLINE_REQUEST_LIMIT,
    DownloadPlanner,
    DownloadTask,
)

_SYMBOL = "BTCUSDT"
_START = 1_700_000_000_000
_DAY = MILLISECONDS_PER_DAY
_YEAR = 365 * _DAY

_EXPECTED_DURATION_MS: dict[str, int] = {
    "1m": MILLISECONDS_PER_MINUTE,
    "3m": 3 * MILLISECONDS_PER_MINUTE,
    "5m": 5 * MILLISECONDS_PER_MINUTE,
    "15m": 15 * MILLISECONDS_PER_MINUTE,
    "30m": 30 * MILLISECONDS_PER_MINUTE,
    "1h": MILLISECONDS_PER_HOUR,
    "2h": 2 * MILLISECONDS_PER_HOUR,
    "4h": 4 * MILLISECONDS_PER_HOUR,
    "6h": 6 * MILLISECONDS_PER_HOUR,
    "8h": 8 * MILLISECONDS_PER_HOUR,
    "12h": 12 * MILLISECONDS_PER_HOUR,
    "1d": MILLISECONDS_PER_DAY,
    "3d": 3 * MILLISECONDS_PER_DAY,
    "1w": DAYS_PER_WEEK * MILLISECONDS_PER_DAY,
    "1M": DAYS_PER_MONTH_APPROXIMATION * MILLISECONDS_PER_DAY,
}


def _assert_contiguous_coverage(
    tasks: tuple[DownloadTask, ...],
    *,
    start_time: int,
    end_time: int,
) -> None:
    """Assert tasks cover ``[start_time, end_time]`` without gaps or overlaps."""
    assert tasks
    assert tasks[0].start_time == start_time
    assert tasks[-1].end_time == end_time
    for index in range(len(tasks) - 1):
        assert tasks[index].end_time + 1 == tasks[index + 1].start_time
        assert tasks[index].end_time >= tasks[index].start_time


def _expected_task_count(
    *,
    start_time: int,
    end_time: int,
    chunk_size_ms: int,
) -> int:
    """Compute expected task count for inclusive window splitting."""
    span = end_time - start_time + 1
    return math.ceil(span / chunk_size_ms)


@pytest.mark.parametrize("timeframe", sorted(DOWNLOAD_TIMEFRAMES))
def test_timeframe_duration_ms_matches_expected_values(timeframe: str) -> None:
    """Every allowlisted download timeframe converts to a known duration."""
    assert timeframe_duration_ms(timeframe) == _EXPECTED_DURATION_MS[timeframe]


def test_download_timeframes_match_required_set() -> None:
    """Allowlist matches the Binance intervals required for adaptive planning."""
    assert DOWNLOAD_TIMEFRAMES == frozenset(_EXPECTED_DURATION_MS)


def test_timeframe_duration_ms_rejects_invalid_values() -> None:
    """Unsupported and malformed timeframes fail with validation errors."""
    with pytest.raises(ValidationError) as unsupported:
        timeframe_duration_ms("2x")
    assert unsupported.value.error_code == "INGESTION-CHUNK-SIZING-002"

    with pytest.raises(ValidationError) as wrong_type:
        timeframe_duration_ms(60)  # type: ignore[arg-type]
    assert wrong_type.value.error_code == "INGESTION-CHUNK-SIZING-001"

    with pytest.raises(ValidationError) as missing:
        timeframe_duration_ms("1s")
    assert missing.value.error_code == "INGESTION-CHUNK-SIZING-002"


def test_effective_kline_count_uses_floor_of_limit_times_safety() -> None:
    """Effective candle budget is ``floor(kline_limit * safety_factor)``."""
    assert effective_kline_count(kline_limit=1_500, safety_factor=0.90) == 1_350
    assert effective_kline_count(kline_limit=100, safety_factor=0.99) == 99


def test_effective_kline_count_rejects_invalid_configuration() -> None:
    """Invalid limits and safety factors fail fast."""
    with pytest.raises(ValidationError) as limit_error:
        effective_kline_count(kline_limit=0, safety_factor=0.9)
    assert limit_error.value.error_code == "INGESTION-CHUNK-SIZING-006"

    with pytest.raises(ValidationError) as factor_error:
        effective_kline_count(kline_limit=100, safety_factor=0.0)
    assert factor_error.value.error_code == "INGESTION-CHUNK-SIZING-007"

    with pytest.raises(ValidationError) as tiny:
        effective_kline_count(kline_limit=1, safety_factor=0.4)
    assert tiny.value.error_code == "INGESTION-CHUNK-SIZING-005"


def test_fixed_and_adaptive_strategies_are_immutable() -> None:
    """Chunk sizing strategies are frozen dataclasses."""
    fixed = FixedChunkSizingStrategy(size_ms=_DAY)
    adaptive = AdaptiveChunkSizingStrategy(kline_limit=DEFAULT_KLINE_REQUEST_LIMIT)
    assert is_dataclass(fixed)
    assert is_dataclass(adaptive)
    with pytest.raises(FrozenInstanceError):
        fixed.size_ms = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        adaptive.kline_limit = 1  # type: ignore[misc]


def test_adaptive_chunk_size_formula() -> None:
    """Adaptive chunk size equals effective candles times timeframe duration."""
    strategy = AdaptiveChunkSizingStrategy(
        kline_limit=DEFAULT_KLINE_REQUEST_LIMIT,
        safety_factor=DEFAULT_CHUNK_SAFETY_FACTOR,
    )
    candles = effective_kline_count(
        kline_limit=DEFAULT_KLINE_REQUEST_LIMIT,
        safety_factor=DEFAULT_CHUNK_SAFETY_FACTOR,
    )
    assert strategy.chunk_size_ms("1h") == candles * timeframe_duration_ms("1h")
    assert strategy.chunk_size_ms("1m") == candles * timeframe_duration_ms("1m")


@pytest.mark.parametrize(
    ("timeframe", "min_days", "max_days"),
    [
        ("1m", 0.9, 1.0),
        ("5m", 4.0, 5.0),
        ("15m", 13.0, 15.0),
        ("1h", 50.0, 60.0),
        ("4h", 200.0, 230.0),
    ],
)
def test_adaptive_chunk_duration_matches_expected_scale(
    timeframe: str,
    min_days: float,
    max_days: float,
) -> None:
    """Derived chunk durations match the institutional-scale expectations."""
    strategy = AdaptiveChunkSizingStrategy(kline_limit=DEFAULT_KLINE_REQUEST_LIMIT)
    days = strategy.chunk_size_ms(timeframe) / _DAY
    assert min_days <= days <= max_days


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
def test_adaptive_planner_covers_year_without_gaps_or_overlaps(timeframe: str) -> None:
    """Adaptive planning covers a full year contiguously for key timeframes."""
    planner = DownloadPlanner()
    end_time = _START + _YEAR - 1
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe=timeframe,
        start_time=_START,
        end_time=end_time,
    )
    chunk_size = planner.chunk_strategy.chunk_size_ms(timeframe)
    assert len(tasks) == _expected_task_count(
        start_time=_START,
        end_time=end_time,
        chunk_size_ms=chunk_size,
    )
    _assert_contiguous_coverage(tasks, start_time=_START, end_time=end_time)
    assert all(task.symbol == _SYMBOL for task in tasks)
    assert all(task.timeframe == timeframe for task in tasks)


def test_adaptive_planner_one_hour_uses_far_fewer_tasks_than_daily_chunking() -> None:
    """One year of 1h data is planned in single-digit tasks, not 365."""
    adaptive = DownloadPlanner()
    fixed_day = DownloadPlanner(chunk_size_ms=_DAY)
    end_time = _START + _YEAR - 1

    adaptive_tasks = adaptive.plan(
        symbol=_SYMBOL,
        timeframe="1h",
        start_time=_START,
        end_time=end_time,
    )
    fixed_tasks = fixed_day.plan(
        symbol=_SYMBOL,
        timeframe="1h",
        start_time=_START,
        end_time=end_time,
    )

    assert len(fixed_tasks) == 365
    assert 1 <= len(adaptive_tasks) <= 10
    assert len(adaptive_tasks) < len(fixed_tasks)


def test_adaptive_planner_one_day_fits_year_in_single_task() -> None:
    """One year of 1d data fits inside one adaptive download task."""
    planner = DownloadPlanner()
    end_time = _START + _YEAR - 1
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="1d",
        start_time=_START,
        end_time=end_time,
    )
    assert len(tasks) == 1
    assert tasks[0] == DownloadTask(
        symbol=_SYMBOL,
        timeframe="1d",
        start_time=_START,
        end_time=end_time,
    )


def test_adaptive_planner_very_short_range_is_single_task() -> None:
    """Ranges shorter than one chunk produce exactly one task."""
    planner = DownloadPlanner()
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="1h",
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert tasks == (
        DownloadTask(
            symbol=_SYMBOL,
            timeframe="1h",
            start_time=_START,
            end_time=_START + 1_000,
        ),
    )


def test_adaptive_planner_boundary_exact_chunk_size() -> None:
    """A range exactly equal to one chunk yields a single full-size task."""
    planner = DownloadPlanner()
    chunk_size = planner.chunk_strategy.chunk_size_ms("15m")
    end_time = _START + chunk_size - 1
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="15m",
        start_time=_START,
        end_time=end_time,
    )
    assert len(tasks) == 1
    assert tasks[0].start_time == _START
    assert tasks[0].end_time == end_time


def test_adaptive_planner_boundary_one_millisecond_over_chunk() -> None:
    """Crossing the chunk boundary by one millisecond creates a second task."""
    planner = DownloadPlanner()
    chunk_size = planner.chunk_strategy.chunk_size_ms("5m")
    end_time = _START + chunk_size
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="5m",
        start_time=_START,
        end_time=end_time,
    )
    assert len(tasks) == 2
    _assert_contiguous_coverage(tasks, start_time=_START, end_time=end_time)
    assert tasks[0].end_time - tasks[0].start_time + 1 == chunk_size
    assert tasks[1].start_time == tasks[0].end_time + 1


def test_adaptive_planner_rejects_invalid_timeframe() -> None:
    """Adaptive planning rejects unsupported timeframes."""
    planner = DownloadPlanner()
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            timeframe="2x",
            start_time=_START,
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-CHUNK-SIZING-002"


def test_fixed_chunk_planner_ignores_invalid_timeframe() -> None:
    """Fixed chunk overrides keep opaque timeframe stamping behavior."""
    planner = DownloadPlanner(chunk_size_ms=_DAY)
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="2x",
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert tasks == (
        DownloadTask(
            symbol=_SYMBOL,
            timeframe="2x",
            start_time=_START,
            end_time=_START + 1_000,
        ),
    )


def test_planner_rejects_mutually_exclusive_configuration() -> None:
    """Fixed chunk size and injected strategy cannot both be provided."""
    with pytest.raises(ValidationError) as exc_info:
        DownloadPlanner(
            chunk_size_ms=_DAY,
            chunk_strategy=FixedChunkSizingStrategy(size_ms=_DAY),
        )
    assert exc_info.value.error_code == "INGESTION-DOWNLOADER-009"


def test_planner_accepts_injected_strategy() -> None:
    """Callers may inject a custom chunk sizing strategy."""
    strategy = FixedChunkSizingStrategy(size_ms=10_000)
    planner = DownloadPlanner(chunk_strategy=strategy)
    assert planner.chunk_strategy is strategy
    assert planner.chunk_size_ms is None
    tasks = planner.plan(
        symbol=_SYMBOL,
        timeframe="1h",
        start_time=_START,
        end_time=_START + 25_000,
    )
    assert len(tasks) == 3
    _assert_contiguous_coverage(
        tasks,
        start_time=_START,
        end_time=_START + 25_000,
    )


def test_planner_invalid_timestamps_unchanged() -> None:
    """Invalid timestamp types still raise the original downloader error."""
    planner = DownloadPlanner()
    with pytest.raises(ValidationError) as start_error:
        planner.plan(
            symbol=_SYMBOL,
            timeframe="1h",
            start_time="1700000000000",  # type: ignore[arg-type]
            end_time=_START,
        )
    assert start_error.value.error_code == "INGESTION-DOWNLOADER-003"

    with pytest.raises(ValidationError) as end_error:
        planner.plan(
            symbol=_SYMBOL,
            timeframe="1h",
            start_time=_START,
            end_time=True,  # type: ignore[arg-type]
        )
    assert end_error.value.error_code == "INGESTION-DOWNLOADER-003"


def test_default_safety_factor_is_ninety_percent() -> None:
    """Default adaptive safety factor retains 10% exchange headroom."""
    assert DEFAULT_CHUNK_SAFETY_FACTOR == 0.90
