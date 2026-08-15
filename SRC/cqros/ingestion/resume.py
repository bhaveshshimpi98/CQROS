"""CQROS resumable download range resolution.

Purpose:
    Provide shared, deterministic helpers that adjust historical download
    windows from the latest persisted timestamp so downloaders remain
    idempotent without duplicating resume arithmetic.

Responsibilities:
    - Represent immutable ``DownloadResult`` outcomes for CLI progress
    - Resolve adjusted start times from latest timestamp + interval
    - Classify downloads as full, updated, or skipped
    - Remain free of repository I/O, planner chunking, and exchange calls

Dependencies:
    ``enum``, ``dataclasses``, and ``cqros.core.types``.

Public API:
    ``DownloadStatus``, ``DownloadResult``, and ``resolve_resume_window``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from cqros.core.types import UnixTimestampMs

__all__ = [
    "DownloadResult",
    "DownloadStatus",
    "coerce_latest_timestamp",
    "resolve_resume_window",
]

_ERROR_INTERVAL: Final[str] = "INGESTION-RESUME-001"


class DownloadStatus(StrEnum):
    """Resumable download outcome for progress reporting."""

    FULL = "full"
    UPDATED = "updated"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Immutable outcome of a resumable ``download_symbol`` call.

    Attributes:
        status: Whether the call performed a full download, an incremental
            update, or skipped because storage was already current.
        rows_downloaded: Count of rows fetched from the exchange for this
            call. ``0`` when skipped or when the exchange returned no rows.
    """

    status: DownloadStatus
    rows_downloaded: int = 0

    def format_progress(self) -> str:
        """Return a single-line progress label for CLI output."""
        if self.status is DownloadStatus.SKIPPED:
            return "SKIPPED (up to date)"
        if self.status is DownloadStatus.UPDATED:
            return f"UPDATED (+{self.rows_downloaded} rows)"
        return "FULL DOWNLOAD"


@dataclass(frozen=True, slots=True)
class ResumeWindow:
    """Resolved download window after applying resume rules.

    Attributes:
        status: Provisional status before fetch (``SKIPPED`` means no fetch).
        start_time: Inclusive start to pass to the planner when not skipped.
        end_time: Inclusive end unchanged from the caller request.
    """

    status: DownloadStatus
    start_time: UnixTimestampMs
    end_time: UnixTimestampMs


def coerce_latest_timestamp(value: object) -> UnixTimestampMs | None:
    """Normalize repository latest-timestamp responses to ``int | None``.

    Args:
        value: Raw repository return value.

    Returns:
        ``value`` when it is a non-bool ``int``; otherwise ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def resolve_resume_window(
    *,
    latest_timestamp: UnixTimestampMs | None,
    requested_start: UnixTimestampMs,
    requested_end: UnixTimestampMs,
    interval_ms: int,
) -> ResumeWindow:
    """Resolve the effective download window for a resumable downloader.

    Args:
        latest_timestamp: Latest persisted event timestamp, or ``None`` when
            no readable partitions exist.
        requested_start: Caller-requested inclusive range start (Unix ms).
        requested_end: Caller-requested inclusive range end (Unix ms).
        interval_ms: Dataset-native step after the latest timestamp (for
            example bar duration or the 8h funding interval).

    Returns:
        Immutable ``ResumeWindow``. When ``status`` is ``SKIPPED``, no
        exchange fetch should be performed.

    Raises:
        ValueError: If ``interval_ms`` is not positive.
    """
    if interval_ms <= 0:
        raise ValueError(
            f"interval_ms must be greater than 0 (got {interval_ms})",
        )

    if latest_timestamp is None:
        return ResumeWindow(
            status=DownloadStatus.FULL,
            start_time=requested_start,
            end_time=requested_end,
        )

    adjusted_start = latest_timestamp + interval_ms
    effective_start = max(requested_start, adjusted_start)
    if effective_start >= requested_end:
        return ResumeWindow(
            status=DownloadStatus.SKIPPED,
            start_time=effective_start,
            end_time=requested_end,
        )

    return ResumeWindow(
        status=DownloadStatus.UPDATED,
        start_time=effective_start,
        end_time=requested_end,
    )
