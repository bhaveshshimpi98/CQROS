"""Unit tests for CQROS resumable download window helpers."""

from __future__ import annotations

import pytest

from cqros.ingestion.resume import (
    DownloadResult,
    DownloadStatus,
    resolve_resume_window,
)


def test_resolve_resume_window_full_when_empty() -> None:
    """Missing storage yields a full download from the requested start."""
    window = resolve_resume_window(
        latest_timestamp=None,
        requested_start=100,
        requested_end=500,
        interval_ms=10,
    )
    assert window.status is DownloadStatus.FULL
    assert window.start_time == 100
    assert window.end_time == 500


def test_resolve_resume_window_updated_from_latest_plus_interval() -> None:
    """Existing storage advances start to latest + interval."""
    window = resolve_resume_window(
        latest_timestamp=200,
        requested_start=100,
        requested_end=500,
        interval_ms=10,
    )
    assert window.status is DownloadStatus.UPDATED
    assert window.start_time == 210


def test_resolve_resume_window_respects_requested_start() -> None:
    """Adjusted start never moves earlier than the caller request."""
    window = resolve_resume_window(
        latest_timestamp=50,
        requested_start=100,
        requested_end=500,
        interval_ms=10,
    )
    assert window.status is DownloadStatus.UPDATED
    assert window.start_time == 100


def test_resolve_resume_window_skipped_when_up_to_date() -> None:
    """adjusted_start >= end skips the download."""
    window = resolve_resume_window(
        latest_timestamp=490,
        requested_start=100,
        requested_end=500,
        interval_ms=10,
    )
    assert window.status is DownloadStatus.SKIPPED
    assert window.start_time == 500


def test_resolve_resume_window_rejects_non_positive_interval() -> None:
    """Non-positive intervals fail fast."""
    with pytest.raises(ValueError, match="interval_ms must be greater than 0"):
        resolve_resume_window(
            latest_timestamp=None,
            requested_start=0,
            requested_end=1,
            interval_ms=0,
        )


def test_download_result_format_progress() -> None:
    """Progress labels match the CLI contract."""
    assert DownloadResult(status=DownloadStatus.SKIPPED).format_progress() == "SKIPPED (up to date)"
    assert (
        DownloadResult(status=DownloadStatus.UPDATED, rows_downloaded=36).format_progress()
        == "UPDATED (+36 rows)"
    )
    assert (
        DownloadResult(status=DownloadStatus.FULL, rows_downloaded=10).format_progress()
        == "FULL DOWNLOAD"
    )
