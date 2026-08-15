"""Unit tests for resumable FundingDownloader behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock

import polars as pl

from cqros.ingestion.funding import (
    DEFAULT_FUNDING_INTERVAL_MS,
    DEFAULT_FUNDING_TIMEFRAME,
    FundingDownloader,
    FundingDownloadPlanner,
)
from cqros.ingestion.resume import DownloadStatus


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _downloader(
    *,
    latest: int | None,
) -> tuple[FundingDownloader, MagicMock, AsyncMock]:
    """Compose a FundingDownloader with mocked collaborators."""
    client = MagicMock()
    fetch = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "fundingTime": 1_700_000_000_000 + DEFAULT_FUNDING_INTERVAL_MS,
                "fundingRate": "0.0001",
                "markPrice": "100.0",
            }
        ]
    )
    client.get_funding_rates = fetch
    repository = MagicMock()
    repository.get_latest_funding_timestamp.return_value = latest
    repository.save_funding = MagicMock()
    downloader = FundingDownloader(client, repository, FundingDownloadPlanner())
    return downloader, repository, fetch


def test_funding_download_full_when_empty() -> None:
    """Empty storage performs a full historical funding download."""
    downloader, repository, fetch = _downloader(latest=None)
    result = _run(
        downloader.download_symbol(
            symbol="BTCUSDT",
            start_time=1_700_000_000_000,
            end_time=1_700_000_000_000 + DEFAULT_FUNDING_INTERVAL_MS,
        )
    )
    assert result.status is DownloadStatus.FULL
    assert result.rows_downloaded == 1
    repository.get_latest_funding_timestamp.assert_called_once()
    fetch.assert_awaited()
    repository.save_funding.assert_called_once()


def test_funding_download_skipped_when_up_to_date() -> None:
    """Already-current funding storage skips exchange calls."""
    latest = 1_700_000_000_000
    downloader, repository, fetch = _downloader(latest=latest)
    result = _run(
        downloader.download_symbol(
            symbol="BTCUSDT",
            start_time=1_600_000_000_000,
            end_time=latest + DEFAULT_FUNDING_INTERVAL_MS,
        )
    )
    assert result.status is DownloadStatus.SKIPPED
    assert result.rows_downloaded == 0
    fetch.assert_not_awaited()
    repository.save_funding.assert_not_called()


def test_funding_download_resumes_after_latest() -> None:
    """Resume funding downloads from latest + 8h interval."""
    latest = 1_700_000_000_000
    downloader, repository, fetch = _downloader(latest=latest)
    end_time = latest + (2 * DEFAULT_FUNDING_INTERVAL_MS)
    result = _run(
        downloader.download_symbol(
            symbol="BTCUSDT",
            start_time=1_600_000_000_000,
            end_time=end_time,
        )
    )
    assert result.status is DownloadStatus.UPDATED
    assert result.rows_downloaded == 1
    called_start = fetch.await_args.kwargs["start_time"]
    assert called_start == latest + DEFAULT_FUNDING_INTERVAL_MS
    repository.save_funding.assert_called_once()
    assert (
        repository.get_latest_funding_timestamp.call_args.kwargs["timeframe"]
        == DEFAULT_FUNDING_TIMEFRAME
    )
