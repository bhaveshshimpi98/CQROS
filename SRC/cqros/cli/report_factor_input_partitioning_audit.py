"""CQROS factor-input partitioning implementation audit CLI.

Purpose:
    Provide an argparse entry point for the post-implementation dependency /
    partition audit without regenerating production research ledgers.

Public API:
    ``build_parser``, ``build_options``, ``run_report``, ``main``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import DEFAULT_STORAGE_ROOT
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.reporting.factor_input_partitioning_audit import (
    DEFAULT_OUTPUT_ROOT,
    FactorInputPartitioningAuditReporter,
    FactorInputPartitioningAuditResult,
)

__all__ = [
    "ReportFactorInputPartitioningAuditOptions",
    "build_options",
    "build_parser",
    "format_summary",
    "main",
    "run_report",
]

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1
_ERROR_YEAR: Final[str] = "CLI-REPORT-INPUT-PART-AUDIT-001"


@dataclass(frozen=True, slots=True)
class ReportFactorInputPartitioningAuditOptions:
    """Immutable CLI options for the partitioning implementation audit."""

    storage_root: Path
    output: Path
    year: int
    symbol: str
    timeframe: str
    verbose: bool
    debug: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the factor-input partitioning audit argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-report-factor-input-partitioning-audit",
        description=(
            "Generate a read-only factor-input partitioning implementation "
            "audit from processed market data."
        ),
    )
    parser.add_argument("--year", type=int, required=True, metavar="YEAR")
    parser.add_argument("--symbol", default="BTCUSDT", metavar="SYMBOL")
    parser.add_argument("--timeframe", default="1d", metavar="TIMEFRAME")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path(DEFAULT_STORAGE_ROOT),
        metavar="PATH",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def build_options(args: argparse.Namespace) -> ReportFactorInputPartitioningAuditOptions:
    """Map parsed CLI arguments onto immutable options."""
    year = int(args.year)
    if year <= 0:
        raise ValidationError(
            "year must be a positive integer",
            error_code=_ERROR_YEAR,
            details={"parameter": "year", "value": year},
        )
    return ReportFactorInputPartitioningAuditOptions(
        storage_root=Path(args.storage_root),
        output=Path(args.output),
        year=year,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def run_report(
    options: ReportFactorInputPartitioningAuditOptions,
) -> FactorInputPartitioningAuditResult:
    """Execute the factor-input partitioning audit reporter."""
    level = logging.DEBUG if options.debug else logging.INFO if options.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    reporter = FactorInputPartitioningAuditReporter(
        storage_root=options.storage_root,
        output_root=options.output,
    )
    return reporter.run(year=options.year, symbol=options.symbol, timeframe=options.timeframe)


def format_summary(result: FactorInputPartitioningAuditResult) -> str:
    """Render a concise human-readable audit summary."""
    return (
        result.summary_path.read_text(encoding="utf-8").rstrip()
        + "\n\nReport paths\n"
        + f"- summary: {result.summary_path.as_posix()}\n"
        + f"- audit_csv: {result.audit_csv_path.as_posix()}\n"
        + f"- hashes_before: {result.hashes_before_path.as_posix()}\n"
        + f"- hashes_after: {result.hashes_after_path.as_posix()}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the factor-input partitioning audit CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        result = run_report(options)
    except (CQROSError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    print(format_summary(result), end="")
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
