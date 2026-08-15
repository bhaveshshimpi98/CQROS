"""Unit tests for resumable downloaders across all ingestion datasets."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from cqros.ingestion.chunk_sizing import timeframe_duration_ms
from cqros.ingestion.downloader import DownloadPlanner, HistoricalDownloader
from cqros.ingestion.long_short_ratio import (
    LongShortDownloader,
    LongShortDownloadPlanner,
    LongShortRatioKind,
)
from cqros.ingestion.open_interest import (
    OpenInterestDownloader,
    OpenInterestDownloadPlanner,
)
from cqros.ingestion.resume import DownloadStatus
from cqros.ingestion.taker_volume import (
    TakerVolumeDownloader,
    TakerVolumeDownloadPlanner,
)

_SYMBOL = "BTCUSDT"
_START = 1_700_000_000_000
_PERIOD = "1h"
_INTERVAL = timeframe_duration_ms(_PERIOD)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _kline(open_time: int) -> list[Any]:
    """Build a minimal Binance-shaped kline array."""
    return [
        open_time,
        "100.0",
        "101.0",
        "99.0",
        "100.5",
        "10.0",
        open_time + _INTERVAL - 1,
        "1000.0",
        42,
        "5.0",
        "500.0",
        "0",
    ]


def _open_interest(timestamp: int) -> dict[str, Any]:
    """Build a Binance-shaped open-interest history object."""
    return {
        "symbol": _SYMBOL,
        "sumOpenInterest": "1.0",
        "sumOpenInterestValue": "2.0",
        "timestamp": str(timestamp),
    }


def _taker_volume(timestamp: int) -> dict[str, Any]:
    """Build a Binance-shaped taker-volume history object."""
    return {
        "buySellRatio": "1.2",
        "buyVol": "10.0",
        "sellVol": "8.0",
        "timestamp": str(timestamp),
    }


def _long_short(timestamp: int) -> dict[str, Any]:
    """Build a Binance-shaped long/short ratio object."""
    return {
        "symbol": _SYMBOL,
        "longAccount": "0.6",
        "shortAccount": "0.4",
        "longShortRatio": "1.5",
        "timestamp": str(timestamp),
    }


def test_ohlcv_fresh_full_download() -> None:
    """Empty OHLCV storage performs a full historical download."""
    client = MagicMock()
    client.get_klines = AsyncMock(return_value=[_kline(_START)])
    repository = MagicMock()
    repository.get_latest_ohlcv_timestamp.return_value = None
    downloader = HistoricalDownloader(client, repository, DownloadPlanner())
    result = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_PERIOD,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )
    assert result.status is DownloadStatus.FULL
    assert result.rows_downloaded == 1
    repository.save_ohlcv.assert_called_once()
    client.get_klines.assert_awaited()


def test_ohlcv_resume_and_partial_update() -> None:
    """OHLCV resume starts at latest + bar duration and reports UPDATED."""
    latest = _START
    client = MagicMock()
    client.get_klines = AsyncMock(return_value=[_kline(latest + _INTERVAL)])
    repository = MagicMock()
    repository.get_latest_ohlcv_timestamp.return_value = latest
    downloader = HistoricalDownloader(client, repository, DownloadPlanner())
    result = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_PERIOD,
            start_time=_START - _INTERVAL,
            end_time=latest + (2 * _INTERVAL),
        )
    )
    assert result.status is DownloadStatus.UPDATED
    assert result.rows_downloaded == 1
    assert client.get_klines.await_args.kwargs["start_time"] == latest + _INTERVAL
    repository.save_ohlcv.assert_called_once()


def test_ohlcv_skipped_when_up_to_date() -> None:
    """Already-current OHLCV storage skips exchange calls."""
    latest = _START
    client = MagicMock()
    client.get_klines = AsyncMock()
    repository = MagicMock()
    repository.get_latest_ohlcv_timestamp.return_value = latest
    downloader = HistoricalDownloader(client, repository, DownloadPlanner())
    result = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            timeframe=_PERIOD,
            start_time=_START - _INTERVAL,
            end_time=latest + _INTERVAL,
        )
    )
    assert result.status is DownloadStatus.SKIPPED
    assert result.rows_downloaded == 0
    client.get_klines.assert_not_awaited()
    repository.save_ohlcv.assert_not_called()


def test_open_interest_resume_skips_and_updates() -> None:
    """Open-interest downloader resumes and skips using period interval."""
    client = MagicMock()
    client.get_open_interest_history = AsyncMock(return_value=[_open_interest(_START + _INTERVAL)])
    repository = MagicMock()
    repository.get_latest_open_interest_timestamp.return_value = _START
    downloader = OpenInterestDownloader(
        client,
        repository,
        OpenInterestDownloadPlanner(),
    )

    skipped = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )
    assert skipped.status is DownloadStatus.SKIPPED
    client.get_open_interest_history.assert_not_awaited()

    updated = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _INTERVAL),
        )
    )
    assert updated.status is DownloadStatus.UPDATED
    assert updated.rows_downloaded == 1
    assert client.get_open_interest_history.await_args.kwargs["start_time"] == _START + _INTERVAL


def test_taker_volume_fresh_and_resume() -> None:
    """Taker-volume downloader supports full and resumed windows."""
    client = MagicMock()
    client.get_taker_buy_sell_volume = AsyncMock(return_value=[_taker_volume(_START + _INTERVAL)])
    repository = MagicMock()
    repository.get_latest_taker_volume_timestamp.return_value = None
    downloader = TakerVolumeDownloader(
        client,
        repository,
        TakerVolumeDownloadPlanner(),
    )
    full = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + _INTERVAL,
        )
    )
    assert full.status is DownloadStatus.FULL
    repository.save_taker_volume.assert_called_once()

    repository.get_latest_taker_volume_timestamp.return_value = _START
    client.get_taker_buy_sell_volume.reset_mock()
    repository.save_taker_volume.reset_mock()
    client.get_taker_buy_sell_volume.return_value = [_taker_volume(_START + _INTERVAL)]
    updated = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _INTERVAL),
        )
    )
    assert updated.status is DownloadStatus.UPDATED
    assert client.get_taker_buy_sell_volume.await_args.kwargs["start_time"] == _START + _INTERVAL


def test_long_short_resume_routes_dataset_kind() -> None:
    """Long/short resume asks repository with the dataset namespace string."""
    client = MagicMock()
    fetch = AsyncMock(return_value=[_long_short(_START + _INTERVAL)])
    client.get_global_long_short_account_ratio = fetch
    client.get_top_long_short_account_ratio = AsyncMock()
    client.get_top_long_short_position_ratio = AsyncMock()
    repository = MagicMock()
    repository.get_latest_long_short_timestamp.return_value = _START
    downloader = LongShortDownloader(
        client,
        repository,
        LongShortDownloadPlanner(),
    )
    result = _run(
        downloader.download_symbol(
            symbol=_SYMBOL,
            kind=LongShortRatioKind.GLOBAL_ACCOUNT,
            period=_PERIOD,
            start_time=_START,
            end_time=_START + (2 * _INTERVAL),
        )
    )
    assert result.status is DownloadStatus.UPDATED
    assert (
        repository.get_latest_long_short_timestamp.call_args.kwargs["dataset"]
        == LongShortRatioKind.GLOBAL_ACCOUNT.value
    )
    assert fetch.await_args.kwargs["start_time"] == _START + _INTERVAL
