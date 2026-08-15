"""CQROS Factor Selection Stability review CLI.

Purpose:
    Provide an argparse entry point that discovers Factor Selection partitions
    and runs the read-only selection-stability reporter without mutating
    production lake ledgers.

Responsibilities:
    - Parse CLI arguments for selection-stability reporting
    - Wire ``FactorStabilitySelectionReporter``
    - Print concise human verdicts per timeframe and report paths
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``argparse``, ``logging``, ``cqros.core``, and
    ``cqros.reporting.factor_stability_selection_report``.

Public API:
    ``ReportFactorStabilityOptions``, ``build_parser``, ``build_options``,
    ``run_report``, ``format_summary``, and ``main``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import DEFAULT_STORAGE_ROOT, SUPPORTED_TIMEFRAMES
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Timeframe
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_selection_report import (
    DEFAULT_OUTPUT_ROOT,
    FactorStabilitySelectionReporter,
    FactorStabilitySelectionResult,
)

__all__ = [
    "ReportFactorStabilityOptions",
    "build_options",
    "build_parser",
    "format_summary",
    "main",
    "run_report",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_ERROR_MANAGER: Final[str] = "CLI-REPORT-FACTOR-STABILITY-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-REPORT-FACTOR-STABILITY-002"
_ERROR_OUTPUT: Final[str] = "CLI-REPORT-FACTOR-STABILITY-003"


@dataclass(frozen=True, slots=True)
class ReportFactorStabilityOptions:
    """Immutable CLI options for Factor Selection stability reporting."""

    storage_root: Path
    output: Path
    manager: str
    timeframes: tuple[Timeframe, ...] | None
    verbose: bool
    debug: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the Factor Selection stability report argument parser."""
    parser = argparse.ArgumentParser(
        prog="cqros-report-factor-stability",
        description=(
            "Generate read-only Factor Selection stability CSV reports from "
            "existing factor_selection / walk_forward_evaluation / "
            "purged_cv_evaluation artifacts."
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
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist. Omit to discover all selection timeframes.",
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
        help=f"CSV report directory (default: {DEFAULT_OUTPUT_ROOT.as_posix()}).",
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


def build_options(args: argparse.Namespace) -> ReportFactorStabilityOptions:
    """Map parsed CLI arguments onto immutable options."""
    manager = str(args.manager).strip()
    if not manager:
        raise ValidationError(
            "manager must be a non-empty string",
            error_code=_ERROR_MANAGER,
            details={"parameter": "manager", "value": args.manager},
        )
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        raise ValidationError(
            "output path must be a directory",
            error_code=_ERROR_OUTPUT,
            details={"output": str(output)},
        )
    return ReportFactorStabilityOptions(
        storage_root=Path(args.storage_root),
        output=output,
        manager=manager,
        timeframes=_normalize_timeframes(args.timeframes),
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def run_report(options: ReportFactorStabilityOptions) -> FactorStabilitySelectionResult:
    """Execute the Factor Selection stability reporter."""
    _configure_logging(verbose=options.verbose, debug=options.debug)
    reporter = FactorStabilitySelectionReporter(
        storage_root=options.storage_root,
        output_root=options.output,
        manager=options.manager,
    )
    return reporter.run(timeframes=options.timeframes)


def format_summary(result: FactorStabilitySelectionResult) -> str:
    """Render a concise human-readable selection-stability summary."""
    lines = [
        "=======================================",
        "CQROS Factor Selection Stability Review",
        "=======================================",
        "",
        f"production_ledgers_unchanged: {result.production_ledgers_unchanged}",
        "",
        "Verdicts",
        "--------",
    ]
    if result.verdicts:
        for timeframe in sorted(result.verdicts):
            lines.append(f"{timeframe}: {result.verdicts[timeframe]}")
    else:
        lines.append("(none)")

    if result.global_summary.height > 0:
        lines.extend(["", "Key metrics", "-----------"])
        for row in result.global_summary.sort("timeframe").iter_rows(named=True):
            lines.append(
                f"{row['timeframe']}: selected={row['selected_factors']} "
                f"tested={row['tested_factors']} "
                f"wf_oriented={_fmt(row['wf_oriented_oos_ic'])} "
                f"pcv_oriented={_fmt(row['pcv_oriented_oos_ic'])} "
                f"status={row['status']}"
            )

    lines.extend(["", "Report paths", "------------"])
    for key in sorted(result.paths):
        lines.append(f"{key}: {result.paths[key].as_posix()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Factor Selection stability reporting."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options(args)
        result = run_report(options)
    except (CQROSError, ReportingValidationError) as exc:
        print(str(exc), file=sys.stderr)
        if bool(getattr(args, "debug", False)):
            _logger.exception("Factor selection stability report failed")
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        if bool(getattr(args, "debug", False)):
            _logger.exception("Factor selection stability report failed")
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


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
