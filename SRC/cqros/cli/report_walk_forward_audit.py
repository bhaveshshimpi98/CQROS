"""CQROS Walk-Forward consolidated audit report CLI.

Purpose:
    Provide an argparse-based entry point that discovers production
    Walk-Forward parquet partitions and writes read-only consolidated CSV
    audit reports.

Responsibilities:
    - Parse CLI arguments for Walk-Forward audit reporting
    - Wire ``StorageLayout`` into ``WalkForwardAuditReporter``
    - Print the discovery table before writing reports
    - Print final report paths and global totals
    - Remain free of Walk-Forward engine mutation and parquet writes

Dependencies:
    ``argparse``, ``logging``, ``cqros.core``, ``cqros.reporting``, and
    ``cqros.storage``.

Public API:
    ``ReportWalkForwardAuditOptions``, ``build_parser``, ``build_options``,
    ``run_report``, ``format_summary``, and ``main``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import DEFAULT_STORAGE_ROOT
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.walk_forward_audit_report import (
    DEFAULT_OUTPUT_ROOT,
    WalkForwardAuditReporter,
    WalkForwardAuditResult,
    format_discovery_table,
)
from cqros.storage import StorageLayout

__all__ = [
    "ReportWalkForwardAuditOptions",
    "build_options",
    "build_parser",
    "format_summary",
    "main",
    "run_report",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_ENGINE: Final[str] = "simple"

_ERROR_ENGINE: Final[str] = "CLI-REPORT-WALK-FORWARD-AUDIT-001"
_ERROR_OUTPUT: Final[str] = "CLI-REPORT-WALK-FORWARD-AUDIT-002"


@dataclass(frozen=True, slots=True)
class ReportWalkForwardAuditOptions:
    """Immutable CLI options for Walk-Forward audit reporting.

    Attributes:
        storage_root: Storage root containing ``walk_forward``.
        output: Directory that receives the three CSV reports.
        engine: Engine label recorded in the detail CSV.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and full failure
            tracebacks.
    """

    storage_root: Path
    output: Path
    engine: str
    verbose: bool
    debug: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the Walk-Forward audit report argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-report-walk-forward-audit",
        description=(
            "Generate read-only consolidated Walk-Forward CSV audit reports "
            "from existing walk_forward parquet partitions."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=f"Report output directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        default=_DEFAULT_ENGINE,
        metavar="NAME",
        help=(
            "Engine label recorded in the detail CSV. Walk-Forward partitions "
            f"do not persist engine identity (default: {_DEFAULT_ENGINE})."
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


def build_options(args: argparse.Namespace) -> ReportWalkForwardAuditOptions:
    """Map parsed CLI arguments onto ``ReportWalkForwardAuditOptions``."""
    engine = str(args.engine).strip()
    if not engine:
        raise ValidationError(
            "engine must be a non-empty string",
            error_code=_ERROR_ENGINE,
            details={"engine": args.engine},
        )
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        raise ValidationError(
            "output path must be a directory",
            error_code=_ERROR_OUTPUT,
            details={"output": str(output)},
        )
    storage_root = (
        Path(args.storage_root) if args.storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
    )
    return ReportWalkForwardAuditOptions(
        storage_root=storage_root,
        output=output,
        engine=engine,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def run_report(options: ReportWalkForwardAuditOptions) -> WalkForwardAuditResult:
    """Execute the Walk-Forward audit reporter and print discovery output."""
    _configure_logging(verbose=options.verbose, debug=options.debug)
    reporter = WalkForwardAuditReporter(
        StorageLayout(options.storage_root),
        output_dir=options.output,
        engine=options.engine,
    )
    detail, parquet_paths, hashes_before = reporter.collect()
    print(format_discovery_table(detail))
    print()
    result = reporter.emit(detail, parquet_paths, hashes_before)
    print(format_summary(result))
    return result


def format_summary(result: WalkForwardAuditResult) -> str:
    """Format final report paths and global totals."""
    totals = {
        str(metric): value
        for metric, value in result.global_summary.select(["metric", "value"]).iter_rows()
    }
    lines = [
        "Report paths:",
        f"  {result.paths.detail}",
        f"  {result.paths.timeframe_summary}",
        f"  {result.paths.global_summary}",
        "Global totals:",
        f"  timeframes={totals.get('total_timeframes')}",
        f"  years={totals.get('total_years')}",
        f"  symbols={totals.get('total_symbols')}",
        f"  rows={totals.get('total_rows')}",
        f"  selected_rows={totals.get('total_selected_rows')}",
        f"  pass_rows={totals.get('total_pass_rows')}",
        f"  fail_rows={totals.get('total_fail_rows')}",
        f"  unique_folds={totals.get('total_folds')}",
        f"  parquet_unmodified=" f"{result.parquet_hashes_before == result.parquet_hashes_after}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Walk-Forward consolidated audit reporting."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        run_report(options)
    except (CQROSError, ReportingValidationError, ValidationError, OSError, ValueError) as exc:
        _logger.error("Walk-Forward audit report failed: %s", exc)
        if bool(getattr(args, "debug", False)):
            _logger.exception("Walk-Forward audit report traceback")
        return _EXIT_FAILURE
    return _EXIT_SUCCESS


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure module logging from CLI verbosity flags."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level)


if __name__ == "__main__":
    sys.exit(main())
