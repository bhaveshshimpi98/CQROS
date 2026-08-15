"""Unit tests for CQROS factor orthogonalization verification CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import pytest

from cqros.cli.verify_factor_orthogonalization import (
    VerifyFactorOrthogonalizationSummary,
    build_options,
    build_parser,
    format_summary,
    main,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT
from cqros.core.exceptions import ValidationError


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def test_build_parser_defaults() -> None:
    """Parser defaults match existing verify CLI conventions."""
    parser = build_parser()
    args = parser.parse_args([])
    options = build_options(args)
    assert options.manager is None
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.workers == ResearchConfig().worker_count


def test_build_options_rejects_invalid_workers() -> None:
    """Non-positive workers raise ValidationError."""
    parser = build_parser()
    args = parser.parse_args(["--workers", "0"])
    with pytest.raises(ValidationError):
        build_options(args)


def test_format_summary_pass() -> None:
    """Summary formatting includes PASS repository status."""
    summary = VerifyFactorOrthogonalizationSummary(
        panels_verified=0,
        datasets_verified=0,
        timeframes_verified=0,
        successful_tasks=0,
        failed_tasks=0,
        rows_checked=0,
        duration_seconds=0.5,
        repository_passed=True,
    )
    text = format_summary(summary)
    assert "CQROS Factor Orthogonalization Verification Summary" in text
    assert "PASS" in text


def test_main_with_empty_storage(tmp_path: Path) -> None:
    """Main succeeds when no orthogonalization partitions exist."""
    code = _run(main(["--storage-root", str(tmp_path)]))
    assert code == 0
