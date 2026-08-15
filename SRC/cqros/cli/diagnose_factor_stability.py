"""CQROS Factor Stability + OOS IC diagnostic CLI.

Purpose:
    Discover existing Purged-CV ledger partitions and run the read-only
    Factor Stability diagnostic without mutating production lake artifacts.

Responsibilities:
    - Parse CLI arguments for factor-stability diagnosis
    - Discover Purged-CV ledger partitions (never invent timeframes)
    - Wire repositories into ``FactorStabilityDiagnostic``
    - Write CSV reports under ``reports/purged_cv`` by default
    - Print critical diagnostic answers and immutability status
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``argparse``, ``asyncio``, ``logging``, ``cqros.config``, ``cqros.core``,
    ``cqros.factor_selection``, ``cqros.factor_validation``, ``cqros.factors``,
    ``cqros.purged_cv``, ``cqros.reporting.factor_stability_diagnostic``,
    ``cqros.storage``, and ``cqros.walk_forward``.

Public API:
    ``build_parser``, ``build_options``, ``discover_work``, ``run_diagnostic``,
    ``format_summary``, and ``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_validation import FactorValidationRepository
from cqros.factors import FactorsRepository
from cqros.purged_cv import PurgedCVPartitionRef, PurgedCVRepository
from cqros.reporting.factor_stability_diagnostic import (
    DEFAULT_OUTPUT_ROOT,
    FactorStabilityDiagnostic,
    FactorStabilityDiagnosticResult,
)
from cqros.storage import LabelRepository, ParquetStore, StorageLayout
from cqros.walk_forward import WalkForwardRepository
from cqros.walk_forward.evaluation_input import WalkForwardInputBuilder

__all__ = [
    "DiscoveredWorkItem",
    "FactorStabilityOptions",
    "FactorStabilitySummary",
    "build_options",
    "build_parser",
    "discover_work",
    "format_summary",
    "main",
    "run_diagnostic",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_DEFAULT_ENGINE: Final[str] = "simple"

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-DIAGNOSE-FACTOR-STABILITY-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-DIAGNOSE-FACTOR-STABILITY-002"
_ERROR_YEAR: Final[str] = "CLI-DIAGNOSE-FACTOR-STABILITY-003"
_ERROR_MANAGER: Final[str] = "CLI-DIAGNOSE-FACTOR-STABILITY-004"


@dataclass(frozen=True, slots=True)
class FactorStabilityOptions:
    """Immutable CLI options for factor-stability diagnosis."""

    storage_root: Path
    manager: str
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    workers: int
    verbose: bool
    debug: bool
    report_output: Path


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered Purged-CV ledger panel ready for diagnosis."""

    manager: str
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactorStabilitySummary:
    """Immutable aggregate summary for a factor-stability diagnostic run."""

    manager: str
    panels: int
    folds: int
    timeframes: tuple[str, ...]
    duration_seconds: float
    report_directory: Path
    verdict: str
    primary_conclusion: str
    ledger_hashes_unchanged: bool
    critical_answers: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-stability diagnostic argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-diagnose-factor-stability",
        description=(
            "Diagnose Factor Stability and negative OOS IC using existing "
            "Purged-CV / Factor Selection / Labels / Factors artifacts. "
            "Read-only: never mutates production lake parquet."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for discovery and diagnosis lineage.",
    )
    parser.add_argument(
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist. Omit to discover all ledger timeframes.",
    )
    parser.add_argument(
        "--years",
        dest="years",
        nargs="*",
        default=None,
        metavar="YEAR",
        help="Optional calendar-year allowlist. Omit to discover all ledger years.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=(
            f"Reserved concurrency hint (default: {_DEFAULT_WORKER_COUNT}). "
            "Panels are diagnosed sequentially for deterministic CSV output."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and full failure tracebacks.",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Storage root for dataset tiers (default: {DEFAULT_STORAGE_ROOT}).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=f"CSV report directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
    )
    return parser


def build_options(args: argparse.Namespace) -> FactorStabilityOptions:
    """Map parsed CLI arguments onto immutable options."""
    workers = int(args.workers)
    if workers <= 0:
        raise ValidationError(
            "workers must be greater than 0",
            error_code=_ERROR_WORKERS,
            details={"parameter": "workers", "value": workers},
        )
    manager = str(args.manager).strip()
    if not manager:
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
            details={"parameter": "manager", "value": args.manager},
        )
    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )
    return FactorStabilityOptions(
        storage_root=storage_root,
        manager=manager,
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        report_output=Path(args.output),
    )


def discover_work(
    purged_cv_repository: PurgedCVRepository,
    options: FactorStabilityOptions,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover Purged-CV ledger panels matching CLI filters."""
    partitions = purged_cv_repository.discover_partitions(
        managers=(options.manager,),
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: FactorStabilitySummary) -> str:
    """Render a deterministic factor-stability diagnostic summary."""
    lines = [
        "=======================================",
        "CQROS Factor Stability Diagnostic",
        "=======================================",
        "",
        f"Manager: {summary.manager}",
        f"Verdict: {summary.verdict}",
        f"Primary Conclusion: {summary.primary_conclusion}",
        "",
        f"Timeframes: {', '.join(summary.timeframes) if summary.timeframes else '(none)'}",
        f"Panels: {summary.panels}",
        f"Folds Accounted: {summary.folds}",
        f"Ledger Unchanged: {summary.ledger_hashes_unchanged}",
        "",
        f"Duration: {_format_duration(summary.duration_seconds)}",
        f"Report directory: {summary.report_directory.as_posix()}",
        "",
        "Critical Answers",
        "---------------",
    ]
    for key in sorted(summary.critical_answers):
        lines.append(f"{key}: {summary.critical_answers[key]}")
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the factor-stability diagnostic CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        summary = await run_diagnostic(options)
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        if args.debug:
            _logger.exception("Factor stability diagnostic failed")
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS if summary.verdict != "FAIL" else _EXIT_FAILURE


async def run_diagnostic(options: FactorStabilityOptions) -> FactorStabilitySummary:
    """Execute the diagnostic for discovered work and return a summary."""
    started = time.perf_counter()
    layout = StorageLayout(options.storage_root)
    datastore = ParquetStore()
    purged_cv_repository = PurgedCVRepository(layout, datastore)
    work = discover_work(purged_cv_repository, options)
    diagnostic = FactorStabilityDiagnostic(
        purged_cv_repository=purged_cv_repository,
        walk_forward_repository=WalkForwardRepository(layout, datastore),
        factor_selection_repository=FactorSelectionRepository(layout, datastore),
        factors_repository=FactorsRepository(layout, datastore),
        label_repository=LabelRepository(layout, datastore),
        walk_forward_input_builder=WalkForwardInputBuilder(
            FactorsRepository(layout, datastore),
            LabelRepository(layout, datastore),
        ),
        factor_validation_repository=FactorValidationRepository(layout, datastore),
        output_root=options.report_output,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    # Sequential diagnosis keeps CSV ordering deterministic regardless of workers.
    _ = options.workers
    result = await asyncio.to_thread(
        diagnostic.run,
        manager=options.manager,
        engine=_DEFAULT_ENGINE,
        timeframes=options.timeframes,
        years=options.years,
        storage_root=options.storage_root,
    )
    return _build_summary(
        options=options,
        work=work,
        result=result,
        duration_seconds=time.perf_counter() - started,
    )


def _build_summary(
    *,
    options: FactorStabilityOptions,
    work: Sequence[DiscoveredWorkItem],
    result: FactorStabilityDiagnosticResult,
    duration_seconds: float,
) -> FactorStabilitySummary:
    critical: dict[str, str] = {}
    for row in result.global_summary.iter_rows(named=True):
        metric = str(row["metric"])
        if metric.startswith("q") or metric in {"verdict", "selection_intensity"}:
            critical[metric] = str(row["value"])
    hashes_unchanged = result.parquet_hashes_before == result.parquet_hashes_after
    return FactorStabilitySummary(
        manager=options.manager,
        panels=len(work),
        folds=result.folds_accounted,
        timeframes=result.timeframes_analyzed,
        duration_seconds=duration_seconds,
        report_directory=options.report_output,
        verdict=result.verdict,
        primary_conclusion=result.primary_conclusion,
        ledger_hashes_unchanged=hashes_unchanged,
        critical_answers=critical,
    )


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("cqros").setLevel(level)


def _normalize_timeframes(
    values: Sequence[str] | None,
) -> tuple[Timeframe, ...] | None:
    if values is None:
        return None
    normalized: list[Timeframe] = []
    for timeframe in values:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValidationError(
                f"unsupported timeframe: {timeframe}",
                error_code=_ERROR_TIMEFRAME,
                details={"parameter": "timeframes", "value": timeframe},
            )
        if timeframe not in normalized:
            normalized.append(timeframe)
    return tuple(normalized) if normalized else None


def _normalize_years(values: Sequence[str] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    normalized: list[int] = []
    for raw in values:
        try:
            year = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            ) from exc
        if year < 1:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            )
        if year not in normalized:
            normalized.append(year)
    return tuple(sorted(normalized)) if normalized else None


def _group_partitions(
    partitions: Sequence[PurgedCVPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.manager, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)
    items: list[DiscoveredWorkItem] = []
    for (manager, timeframe), years in sorted(grouped.items()):
        items.append(
            DiscoveredWorkItem(
                manager=manager,
                timeframe=timeframe,
                years=tuple(sorted(set(years))),
            )
        )
    return tuple(items)


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60.0)
    rem = seconds - (minutes * 60.0)
    return f"{minutes}m {rem:.2f}s"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
