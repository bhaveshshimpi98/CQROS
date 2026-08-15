"""CQROS factor validation researcher report CLI.

Purpose:
    Provide an argparse-based research entry point that loads existing factor
    validation parquet ledgers and writes read-only researcher reports.

Responsibilities:
    - Parse CLI arguments for report generation
    - Wire ``StorageLayout``, ``ParquetStore``, and ``FactorValidationRepository``
    - Delegate report construction to ``FactorValidationReporter``
    - Remain free of factor generation, validation math, selection, and
      parquet mutation

Dependencies:
    ``argparse``, ``asyncio``, ``cqros.core``, ``cqros.factor_validation``,
    ``cqros.reporting.factor_validation_report``, and ``cqros.storage``.

Public API:
    ``ReportFactorValidationOptions``, ``build_parser``, ``build_options``,
    ``run_report``, ``format_summary``, and ``main``.

Notes:
    This module is a thin composition root. It never writes factor validation
    parquet files and never modifies validation business logic.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.factor_validation import FactorValidationRepository
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_validation_report import (
    DEFAULT_OUTPUT_ROOT,
    REPORT_FORMATS,
    FactorValidationReporter,
    ReportFormat,
)
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "ReportFactorValidationOptions",
    "build_options",
    "build_parser",
    "format_summary",
    "main",
    "run_report",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_MANAGER: Final[str] = "CLI-REPORT-FACTOR-VALIDATION-001"
_ERROR_FORMAT: Final[str] = "CLI-REPORT-FACTOR-VALIDATION-002"
_ERROR_OUTPUT: Final[str] = "CLI-REPORT-FACTOR-VALIDATION-003"


@dataclass(frozen=True, slots=True)
class ReportFactorValidationOptions:
    """Immutable CLI options for factor validation reporting.

    Attributes:
        storage_root: Storage root containing ``factor_validation``.
        manager: Order manager whose ledgers are reported.
        output: Report root directory. The manager subdirectory is appended.
        formats: Ordered unique report formats to emit.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and full failure
            tracebacks.
    """

    storage_root: Path
    manager: str
    output: Path
    formats: tuple[ReportFormat, ...]
    verbose: bool
    debug: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the factor validation report argument parser.

    Returns:
        Configured ``ArgumentParser`` for report flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-report-factor-validation",
        description=(
            "Generate read-only CQROS factor validation researcher reports "
            "from existing factor validation parquet ledgers."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order-manager identity whose validation ledgers are reported.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=(
            "Report root directory. Reports are written under "
            f"<output>/<manager>/ (default: {DEFAULT_OUTPUT_ROOT.as_posix()})."
        ),
    )
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="*",
        default=None,
        metavar="FORMAT",
        help=("Report formats to emit: html, csv, xlsx. " "Omit to emit all three formats."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and log complete failure tracebacks.",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Storage root for dataset tiers (default: {DEFAULT_STORAGE_ROOT}).",
    )
    return parser


def build_options(args: argparse.Namespace) -> ReportFactorValidationOptions:
    """Map parsed CLI arguments onto ``ReportFactorValidationOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable reporting options.

    Raises:
        ValidationError: If manager, output, or formats are invalid.
    """
    manager = str(args.manager).strip()
    if not manager:
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
        )

    output = Path(DEFAULT_OUTPUT_ROOT if args.output is None else args.output)
    if str(output).strip() == "":
        raise ValidationError(
            "output must be a non-empty path",
            error_code=_ERROR_OUTPUT,
        )

    raw_formats = args.formats
    if raw_formats is None or len(raw_formats) == 0:
        formats: tuple[ReportFormat, ...] = REPORT_FORMATS
    else:
        normalized: list[ReportFormat] = []
        seen: set[ReportFormat] = set()
        for item in raw_formats:
            value = str(item).strip().lower()
            if value not in REPORT_FORMATS:
                raise ValidationError(
                    f"Unsupported report format '{item}'",
                    error_code=_ERROR_FORMAT,
                    details={"format": item, "allowed": list(REPORT_FORMATS)},
                )
            if value == "html":
                format_name: ReportFormat = "html"
            elif value == "csv":
                format_name = "csv"
            else:
                format_name = "xlsx"
            if format_name not in seen:
                normalized.append(format_name)
                seen.add(format_name)
        formats = tuple(normalized)

    storage_root = (
        Path(DEFAULT_STORAGE_ROOT) if args.storage_root is None else Path(args.storage_root)
    )
    return ReportFactorValidationOptions(
        storage_root=storage_root,
        manager=manager,
        output=output,
        formats=formats,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def format_summary(*, manager: str, output_dir: Path, formats: tuple[ReportFormat, ...]) -> str:
    """Format a concise completion summary for stdout.

    Args:
        manager: Reported manager identity.
        output_dir: Directory containing written report files.
        formats: Formats that were requested.

    Returns:
        Multi-line human-readable summary text.
    """
    lines = [
        "Factor Validation Report",
        f"manager: {manager}",
        f"output: {output_dir.as_posix()}",
        f"formats: {', '.join(formats)}",
        "status: COMPLETE",
    ]
    return "\n".join(lines)


def run_report(
    *,
    repository: FactorValidationRepository,
    options: ReportFactorValidationOptions,
) -> Path:
    """Load validation ledgers and write the selected researcher reports.

    Args:
        repository: Read-only factor validation repository.
        options: Parsed CLI options.

    Returns:
        Destination directory containing written report artifacts.
    """
    reporter = FactorValidationReporter(
        repository,
        manager=options.manager,
        output_dir=options.output,
        exchange=_EXCHANGE,
        market=_MARKET,
        logger=_logger,
    )
    reporter.load()
    return reporter.write(formats=options.formats)


async def main(argv: list[str] | None = None) -> int:
    """CLI entry point for factor validation researcher reports.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` success, ``1`` failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILURE

    _configure_logging(verbose=options.verbose, debug=options.debug)
    started = time.perf_counter()
    repository = FactorValidationRepository(
        StorageLayout(options.storage_root),
        ParquetStore(),
        logger=_logger,
    )
    try:
        output_dir = await asyncio.to_thread(
            run_report,
            repository=repository,
            options=options,
        )
    except (ReportingValidationError, CQROSError) as exc:
        if options.debug:
            _logger.exception("Factor validation report failed")
        else:
            _logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        if options.debug:
            _logger.exception("Unexpected factor validation report failure")
        else:
            _logger.error("%s: %s", type(exc).__name__, exc)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_FAILURE

    elapsed = time.perf_counter() - started
    print(
        format_summary(
            manager=options.manager,
            output_dir=output_dir,
            formats=options.formats,
        )
    )
    print(f"duration_seconds: {elapsed:.3f}")
    return _EXIT_SUCCESS


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure process logging for the report CLI."""
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
