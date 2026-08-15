"""Unit tests for CQROS historical taker buy/sell volume downloader."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_MINUTE,
)
from cqros.core.exceptions import ValidationError
from cqros.ingestion import (
    DEFAULT_TAKER_VOLUME_CHUNK_SAFETY_FACTOR,
    DEFAULT_TAKER_VOLUME_PERIOD,
    DEFAULT_TAKER_VOLUME_REQUEST_LIMIT,
    TAKER_VOLUME_PERIODS,
    TakerVolumeDownloader,
    TakerVolumeDownloadPlanner,
    TakerVolumeDownloadTask,
)
from cqros.ingestion.chunk_sizing import DEFAULT_CHUNK_SAFETY_FACTOR
from cqros.ingestion.taker_volume import (
    TakerVolumeDownloader as TakerVolumeDownloaderDirect,
)
from cqros.ingestion.taker_volume import (
    TakerVolumeDownloadPlanner as TakerVolumeDownloadPlannerDirect,
)
from cqros.ingestion.taker_volume import (
    TakerVolumeDownloadTask as TakerVolumeDownloadTaskDirect,
)

_SYMBOL = "BTCUSDT"
_PERIOD = "5m"
_START = 1_700_000_000_000
_FIVE_MINUTES = 5 * MILLISECONDS_PER_MINUTE
_CHUNK = 30 * _FIVE_MINUTES


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _taker_volume(
    timestamp: int,
    *,
    buy_vol: str = "387.3300",
    sell_vol: str = "248.5030",
    buy_sell_ratio: str | None = None,
) -> dict[str, Any]:
    """Build a Binance-shaped taker buy/sell volume object."""
    payload: dict[str, Any] = {
        "buyVol": buy_vol,
        "sellVol": sell_vol,
        "timestamp": str(timestamp),
    }
    if buy_sell_ratio is not None:
        payload["buySellRatio"] = buy_sell_ratio
    else:
        sell = float(sell_vol)
        if sell != 0.0:
            payload["buySellRatio"] = str(float(buy_vol) / sell)
    return payload


def test_exports_match_module_symbols() -> None:
    """Package exports match the taker-volume module classes and defaults."""
    assert TakerVolumeDownloadTask is TakerVolumeDownloadTaskDirect
    assert TakerVolumeDownloadPlanner is TakerVolumeDownloadPlannerDirect
    assert TakerVolumeDownloader is TakerVolumeDownloaderDirect
    assert DEFAULT_TAKER_VOLUME_PERIOD == "5m"
    assert DEFAULT_TAKER_VOLUME_REQUEST_LIMIT == 500
    assert DEFAULT_TAKER_VOLUME_CHUNK_SAFETY_FACTOR == DEFAULT_CHUNK_SAFETY_FACTOR
    assert "5m" in TAKER_VOLUME_PERIODS
    assert "1d" in TAKER_VOLUME_PERIODS


def test_taker_volume_download_task_is_immutable() -> None:
    """TakerVolumeDownloadTask is a frozen slotted dataclass."""
    task = TakerVolumeDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + 1_000,
    )
    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.symbol = "ETHUSDT"  # type: ignore[misc]


def test_planner_rejects_non_positive_chunk_size() -> None:
    """Planner construction fails fast on invalid chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        TakerVolumeDownloadPlanner(chunk_size_ms=0)
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-001"


def test_planner_rejects_invalid_safety_factor() -> None:
    """Planner construction fails fast on invalid safety factors."""
    with pytest.raises(ValidationError) as exc_info:
        TakerVolumeDownloadPlanner(safety_factor=0.0)
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-003"


def test_planner_rejects_unsupported_period() -> None:
    """Planner rejects periods outside the Binance taker-volume allowlist."""
    planner = TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            period="8h",
            start_time=_START,
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-008"


def test_planner_rejects_non_integer_timestamps() -> None:
    """Planner rejects non-integer Unix millisecond timestamps."""
    planner = TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK)
    with pytest.raises(ValidationError) as exc_info:
        planner.plan(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time="1700000000000",  # type: ignore[arg-type]
            end_time=_START + 1_000,
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-006"


def test_planner_splits_range_into_contiguous_tasks() -> None:
    """Long ranges are split into contiguous inclusive chunks."""
    planner = TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + (2 * _CHUNK) + 100,
    )

    assert len(tasks) == 3
    assert all(isinstance(task, TakerVolumeDownloadTask) for task in tasks)
    assert tasks[0] == TakerVolumeDownloadTask(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + _CHUNK - 1,
    )
    assert tasks[1].start_time == tasks[0].end_time + 1
    assert tasks[2].end_time == _START + (2 * _CHUNK) + 100
    assert tasks[0].end_time + 1 == tasks[1].start_time
    assert tasks[1].end_time + 1 == tasks[2].start_time


def test_planner_uses_period_aware_chunk_sizing_by_default() -> None:
    """Without a fixed chunk size, period duration drives task windows."""
    planner = TakerVolumeDownloadPlanner(request_limit=10, safety_factor=1.0)
    tasks = planner.plan(
        symbol=_SYMBOL,
        period=_PERIOD,
        start_time=_START,
        end_time=_START + (15 * _FIVE_MINUTES),
    )
    assert len(tasks) == 2
    assert tasks[0].end_time - tasks[0].start_time + 1 == 10 * _FIVE_MINUTES


def test_planner_returns_empty_tuple_when_start_after_end() -> None:
    """Inverted ranges produce no tasks."""
    planner = TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK)
    assert (
        planner.plan(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START + 1,
            end_time=_START,
        )
        == ()
    )


def test_downloader_rejects_non_positive_request_limit() -> None:
    """Downloader construction fails fast on invalid request limits."""
    with pytest.raises(ValidationError) as exc_info:
        TakerVolumeDownloader(
            client=MagicMock(),
            repository=MagicMock(),
            planner=TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
            request_limit=0,
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-005"


def test_download_symbol_fetches_persists_and_avoids_paths() -> None:
    """Symbol download paginates taker volume and persists year partitions."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        side_effect=[
            [
                _taker_volume(_START),
                _taker_volume(
                    _START + _FIVE_MINUTES,
                    buy_vol="100.0",
                    sell_vol="50.0",
                ),
            ],
            [],
        ]
    )
    repository = MagicMock()
    planner = TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK)
    downloader = TakerVolumeDownloader(
        client,
        repository,
        planner,
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _FIVE_MINUTES),
        )
    )

    assert client.get_taker_buy_sell_volume.await_count >= 1
    repository.save_taker_volume.assert_called_once()
    args, kwargs = repository.save_taker_volume.call_args
    frame = args[0]
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert kwargs["exchange"] == EXCHANGE_BINANCE
    assert kwargs["market"] == MARKET_USDT_PERPETUAL
    assert kwargs["symbol"] == _SYMBOL
    assert kwargs["timeframe"] == _PERIOD
    assert kwargs["year"] == 2023
    assert "path" not in kwargs


def test_download_symbol_paginates_until_page_under_limit() -> None:
    """Full pages continue pagination; a short page ends the task fetch."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        side_effect=[
            [
                _taker_volume(_START),
                _taker_volume(_START + _FIVE_MINUTES),
            ],
            [_taker_volume(_START + (2 * _FIVE_MINUTES))],
        ]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (3 * _FIVE_MINUTES),
        )
    )

    assert client.get_taker_buy_sell_volume.await_count == 2
    first_call = client.get_taker_buy_sell_volume.await_args_list[0]
    second_call = client.get_taker_buy_sell_volume.await_args_list[1]
    assert first_call.args == (_SYMBOL, _PERIOD)
    assert first_call.kwargs["start_time"] == _START
    assert second_call.kwargs["start_time"] == _START + _FIVE_MINUTES + 1
    frame = repository.save_taker_volume.call_args.args[0]
    assert frame.height == 3


def test_download_symbol_skips_persist_when_empty() -> None:
    """Empty exchange responses do not write partitions."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(return_value=[])
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    repository.save_taker_volume.assert_not_called()


def test_fetch_symbol_returns_empty_frame_for_inverted_range() -> None:
    """Inverted ranges return an empty canonical taker-volume frame."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock()
    downloader = TakerVolumeDownloader(
        client,
        MagicMock(),
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START + 1,
            end_time=_START,
        )
    )

    assert frame.height == 0
    assert frame.columns == [
        "symbol",
        "timestamp",
        "buy_volume",
        "sell_volume",
        "buy_sell_ratio",
    ]
    client.get_taker_buy_sell_volume.assert_not_called()


def test_download_universe_processes_symbols_sequentially() -> None:
    """Universe download invokes symbol download for each symbol in order."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        side_effect=[
            [_taker_volume(_START)],
            [_taker_volume(_START)],
        ]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_universe(
            ["BTCUSDT", "ETHUSDT"],
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert repository.save_taker_volume.call_count == 2
    symbols = [call.kwargs["symbol"] for call in repository.save_taker_volume.call_args_list]
    assert symbols == ["BTCUSDT", "ETHUSDT"]
    frames = [call.args[0] for call in repository.save_taker_volume.call_args_list]
    assert frames[0]["symbol"].to_list() == ["BTCUSDT"]
    assert frames[1]["symbol"].to_list() == ["ETHUSDT"]


def test_invalid_taker_volume_payload_raises_validation_error() -> None:
    """Malformed taker-volume payloads fail with a validation error."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(return_value={"not": "a list"})
    downloader = TakerVolumeDownloader(
        client,
        MagicMock(),
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-009"


def test_non_mapping_taker_volume_row_raises_validation_error() -> None:
    """Non-object taker-volume rows fail validation."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(return_value=[["not", "an", "object"]])
    downloader = TakerVolumeDownloader(
        client,
        MagicMock(),
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-010"


def test_missing_required_taker_volume_fields_raise_validation_error() -> None:
    """Rows missing required vendor fields fail validation."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[{"timestamp": _START, "buyVol": "1.0"}]
    )
    downloader = TakerVolumeDownloader(
        client,
        MagicMock(),
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-011"
    assert "sellVol" in exc_info.value.details["missing"]


def test_persisted_frame_schema_and_mapped_ratio() -> None:
    """Persisted frames map vendor fields into canonical columns."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[_taker_volume(_START, buy_vol="100.0", sell_vol="40.0")]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_taker_volume.call_args.args[0]
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL],
            "timestamp": [_START],
            "buy_volume": [100.0],
            "sell_volume": [40.0],
            "buy_sell_ratio": [2.5],
        }
    )
    assert_frame_equal(frame, expected)


def test_buy_sell_ratio_is_null_when_sell_volume_is_zero() -> None:
    """Zero sell volume produces a null buy_sell_ratio when ratio is omitted."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[
            {
                "buyVol": "25.0",
                "sellVol": "0.0",
                "timestamp": str(_START),
            }
        ]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_taker_volume.call_args.args[0]
    assert frame["buy_sell_ratio"].null_count() == 1
    assert frame["buy_volume"].to_list() == [25.0]
    assert frame["sell_volume"].to_list() == [0.0]


def test_vendor_buy_sell_ratio_is_mapped() -> None:
    """Vendor buySellRatio maps into buy_sell_ratio when present."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[
            {
                "buyVol": "10.0",
                "sellVol": "5.0",
                "buySellRatio": "1.5586",
                "timestamp": str(_START),
            }
        ]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    frame = repository.save_taker_volume.call_args.args[0]
    assert frame["buy_volume"].to_list() == [10.0]
    assert frame["sell_volume"].to_list() == [5.0]
    assert frame["buy_sell_ratio"].to_list()[0] == pytest.approx(1.5586)
    assert frame["timestamp"].to_list() == [_START]


def test_fetch_symbol_does_not_persist() -> None:
    """fetch_symbol returns data without writing repository partitions."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(return_value=[_taker_volume(_START)])
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    frame = _run(
        downloader.fetch_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + 1_000,
        )
    )

    assert frame.height == 1
    repository.save_taker_volume.assert_not_called()


def test_pagination_stops_when_cursor_does_not_advance() -> None:
    """Stuck pagination cursors terminate instead of looping forever."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[
            _taker_volume(_START),
            _taker_volume(_START),
        ]
    )
    repository = MagicMock()
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
        request_limit=2,
    )

    _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + _FIVE_MINUTES,
        )
    )

    assert client.get_taker_buy_sell_volume.await_count == 2
    frame = repository.save_taker_volume.call_args.args[0]
    assert frame.height == 4


def test_invalid_buy_volume_value_raises_validation_error() -> None:
    """Non-numeric buyVol values fail validation."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(
        return_value=[
            {
                "buyVol": "not-a-number",
                "sellVol": "1.0",
                "timestamp": str(_START),
            }
        ]
    )
    downloader = TakerVolumeDownloader(
        client,
        MagicMock(),
        TakerVolumeDownloadPlanner(chunk_size_ms=_CHUNK),
    )

    with pytest.raises(ValidationError) as exc_info:
        _run(
            downloader.download_symbol(
                symbol=_SYMBOL,
                period=_PERIOD,
                start_time=_START,
                end_time=_START + 1_000,
            )
        )
    assert exc_info.value.error_code == "INGESTION-TAKER-VOLUME-013"
