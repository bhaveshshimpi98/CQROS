"""CQROS 1d dense-factor root-cause investigation CLI.

Purpose:
    Provide an argparse entry point that runs the read-only 1d dense-factor
    (PVT/OBV/OI) root-cause investigation without mutating production lake
    ledgers.

Responsibilities:
    - Parse CLI arguments for the dense-factor root-cause reporter
    - Wire ``FactorStability1dDenseRootCauseReporter``
    - Print the human-readable summary and report paths
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``argparse``, ``logging``, ``cqros.core``, and
    ``cqros.reporting.factor_stability_1d_dense_root_cause``.

Public API:
    ``ReportFactorStability1dDenseRootCauseOptions``, ``build_parser``,
    ``build_options``, ``run_report``, ``format_summary``, and ``main``.
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
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_1d_dense_root_cause import (
    DEFAULT_OUTPUT_ROOT,
    FactorStability1dDenseRootCauseReporter,
    FactorStability1dDenseRootCauseResult,
)

__all__ = [
    "ReportFactorStability1dDenseRootCauseOptions",
    "build_options",
    "build_parser",
    "format_summary",
    "main",
    "run_report",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_ERROR_MANAGER: Final[str] = "CLI-REPORT-1D-DENSE-RC-001"
_ERROR_OUTPUT: Final[str] = "CLI-REPORT-1D-DENSE-RC-002"
_ERROR_YEAR: Final[str] = "CLI-REPORT-1D-DENSE-RC-003"


@dataclass(frozen=True, slots=True)
class ReportFactorStability1dDenseRootCauseOptions:
    """Immutable CLI options for the 1d dense-factor root-cause investigation."""

    storage_root: Path
    output: Path
    manager: str
    year: int | None
    verbose: bool
    debug: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the 1d dense-factor root-cause investigation argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-report-factor-stability-1d-dense-root-cause",
        description=(
            "Generate read-only 1d dense-factor (PVT/OBV/OI) root-cause reports "
            "from existing selection / purged_cv_evaluation / processed artifacts."
        ),
    )
    parser.add_argument(
        "--manager",
        dest="manager",
        required=True,
        metavar="NAME",
        help="Order manager identity used for partition discovery.",
    )
    parser.add_argument(
        "--year",
        dest="year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Optional year filter. Defaults to the latest discovered 1d year.",
    )
    parser.add_argument(
        "--storage-root",
        dest="storage_root",
        type=Path,
        default=Path(DEFAULT_STORAGE_ROOT),
        metavar="PATH",
        help=f"Storage root for dataset tiers (default: {DEFAULT_STORAGE_ROOT}).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=f"Report directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
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
    return parser


def build_options(args: argparse.Namespace) -> ReportFactorStability1dDenseRootCauseOptions:
    """Map parsed CLI arguments onto immutable options."""
    manager = str(args.manager).strip()
    if not manager:
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
            details={"parameter": "manager", "value": args.manager},
        )
    year = args.year
    if year is not None and int(year) <= 0:
        raise ValidationError(
            "year must be a positive integer",
            error_code=_ERROR_YEAR,
            details={"parameter": "year", "value": year},
        )
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        raise ValidationError(
            "output path must be a directory",
            error_code=_ERROR_OUTPUT,
            details={"output": str(output)},
        )
    return ReportFactorStability1dDenseRootCauseOptions(
        storage_root=Path(args.storage_root),
        output=output,
        manager=manager,
        year=int(year) if year is not None else None,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def run_report(
    options: ReportFactorStability1dDenseRootCauseOptions,
) -> FactorStability1dDenseRootCauseResult:
    """Execute the 1d dense-factor root-cause reporter."""
    _configure_logging(verbose=options.verbose, debug=options.debug)
    reporter = FactorStability1dDenseRootCauseReporter(
        storage_root=options.storage_root,
        output_root=options.output,
        manager=options.manager,
    )
    return reporter.run(year=options.year)


def format_summary(result: FactorStability1dDenseRootCauseResult) -> str:
    """Render a concise human-readable investigation summary."""
    lines = [
        result.summary_text.rstrip(),
        "",
        "Report paths",
        "------------",
    ]
    for key in sorted(result.paths):
        lines.append(f"{key}: {result.paths[key].as_posix()}")
    lines.append("")
    lines.append(f"VERDICT: {result.verdict}")
    lines.append(f"PRIMARY_CAUSE: {result.primary_cause}")
    lines.append(f"SECONDARY_CAUSES: {result.secondary_causes}")
    lines.append(f"CONFIDENCE: {result.confidence}")
    lines.append(f"PRODUCTION_ARTIFACTS_UNCHANGED: {result.production_artifacts_unchanged}")
    lines.append(f"DETERMINISTIC: {result.deterministic}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the 1d dense-factor root-cause investigation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        result = run_report(options)
    except (CQROSError, ReportingValidationError) as exc:
        print(str(exc), file=sys.stderr)
        if bool(getattr(args, "debug", False)):
            _logger.exception("1d dense-factor root-cause investigation failed")
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        if bool(getattr(args, "debug", False)):
            _logger.exception("1d dense-factor root-cause investigation failed")
        return _EXIT_FAILURE

    print(format_summary(result), end="")
    return _EXIT_SUCCESS


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


if __name__ == "__main__":
    raise SystemExit(main())
