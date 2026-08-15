"""Unit tests for CQROS factor orthogonalization generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path

import pytest

from cqros.cli.generate_factor_orthogonalization import (
    FactorOrthogonalizationGenerationSummary,
    build_options,
    build_parser,
    format_summary,
    main,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import DEFAULT_STORAGE_ROOT
from cqros.core.exceptions import ValidationError
from cqros.factor_orthogonalization import (
    DEFAULT_MAX_COMBINATION_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
)


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def test_build_parser_defaults() -> None:
    """Parser exposes required manager and Phase 3B-aligned defaults."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple"])
    options = build_options(args)
    assert options.manager == "simple"
    assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
    assert options.max_combination_correlation == DEFAULT_MAX_COMBINATION_CORRELATION
    assert options.min_correlation_overlap == DEFAULT_MIN_CORRELATION_OVERLAP
    assert options.workers == ResearchConfig().worker_count
    assert options.export_detailed_csv is False


def test_build_options_rejects_invalid_workers() -> None:
    """Non-positive workers raise ValidationError."""
    parser = build_parser()
    args = parser.parse_args(["--manager", "simple", "--workers", "0"])
    with pytest.raises(ValidationError):
        build_options(args)


def test_format_summary_deterministic() -> None:
    """Summary formatting is stable for empty runs."""
    summary = FactorOrthogonalizationGenerationSummary(
        manager="simple",
        panels=0,
        rows=0,
        successful_tasks=0,
        failed_tasks=0,
        skipped_tasks=0,
        duration_seconds=1.25,
        output_directory=Path("data/factor_orthogonalization"),
        failed_task_labels=(),
    )
    text = format_summary(summary)
    assert "CQROS Factor Orthogonalization Generation Summary" in text
    assert "Manager: simple" in text
    assert "Successful: 0" in text


def test_main_with_empty_storage(tmp_path: Path) -> None:
    """Main succeeds with zero discovered combination partitions."""
    code = _run(
        main(
            [
                "--manager",
                "simple",
                "--storage-root",
                str(tmp_path),
            ]
        )
    )
    assert code == 0
