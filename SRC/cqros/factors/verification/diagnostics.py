"""CQROS factor verification diagnostics.

Purpose:
    Provide immutable structured diagnostics that distinguish expected
    rolling-window warmup NULLs, mathematically defined domain NULLs, and
    genuine factor-value corruption.

Responsibilities:
    - Define null, invalid-numeric, and warning diagnostic value objects
    - Define ``FactorVerificationDiagnostics`` and
      ``FactorVerificationReport`` aggregates
    - Format human-readable ``--debug`` diagnostic text
    - Format a global FAIL report locating Unexpected NULL and +Inf
      findings by partition

Dependencies:
    Python standard library only.

Public API:
    ``FactorInvalidNumericDiagnostic``, ``FactorNullDiagnostic``,
    ``FactorVerificationDiagnostics``, ``FactorVerificationReport``,
    ``FactorWarningDiagnostic``, ``GlobalFailureFinding``,
    ``InvalidNumericKind``, ``NullClassification``,
    ``collect_global_failure_findings``, ``format_factor_diagnostics``,
    ``format_global_failure_report``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

__all__ = [
    "FactorInvalidNumericDiagnostic",
    "FactorNullDiagnostic",
    "FactorVerificationDiagnostics",
    "FactorVerificationReport",
    "FactorWarningDiagnostic",
    "GlobalFailureFinding",
    "InvalidNumericKind",
    "NullClassification",
    "collect_global_failure_findings",
    "format_factor_diagnostics",
    "format_global_failure_report",
]

_SEPARATOR: Final[str] = "--------------------------------"
_ISSUE_UNEXPECTED_NULL: Final[str] = "Unexpected NULL"
_ISSUE_DOMAIN_NULL: Final[str] = "Domain NULL"
_ISSUE_POSITIVE_INF: Final[str] = "+Inf"


class NullClassification(StrEnum):
    """Classification of factor-value NULL runs."""

    WARMUP_NULLS = "WARMUP_NULLS"
    DOMAIN_NULLS = "DOMAIN_NULLS"
    UNEXPECTED_NULLS = "UNEXPECTED_NULLS"


class InvalidNumericKind(StrEnum):
    """Classification of invalid factor_value numerics."""

    POSITIVE_INFINITY = "+inf"
    NEGATIVE_INFINITY = "-inf"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    NON_FINITE = "non-finite"


@dataclass(frozen=True, slots=True)
class FactorNullDiagnostic:
    """Structured NULL finding for one factor in a partition.

    Attributes:
        factor_name: Factor identity.
        count: Number of NULL ``factor_value`` rows for the factor.
        first_open_time: Earliest NULL ``open_time`` (epoch milliseconds).
        last_open_time: Latest NULL ``open_time`` (epoch milliseconds).
        only_at_beginning: Whether NULLs occupy only the leading run.
        appears_after_valid: Whether any NULL appears after a valid value.
        classification: Warmup, domain, or unexpected classification.
    """

    factor_name: str
    count: int
    first_open_time: int
    last_open_time: int
    only_at_beginning: bool
    appears_after_valid: bool
    classification: NullClassification


@dataclass(frozen=True, slots=True)
class FactorInvalidNumericDiagnostic:
    """Structured invalid-numeric finding for one factor observation.

    Attributes:
        factor_name: Factor identity.
        open_time: Observation ``open_time`` (epoch milliseconds).
        kind: Invalid numeric classification.
        count: Number of matching observations at this timestamp.
    """

    factor_name: str
    open_time: int
    kind: InvalidNumericKind
    count: int


@dataclass(frozen=True, slots=True)
class FactorWarningDiagnostic:
    """Structured warning with type, optional factor, and count.

    Attributes:
        warning_type: Machine-readable warning type.
        factor_name: Factor identity when applicable; ``None`` for
            frame-level warnings.
        count: Number of affected rows or occurrences.
    """

    warning_type: str
    factor_name: str | None
    count: int


@dataclass(frozen=True, slots=True)
class FactorVerificationDiagnostics:
    """Immutable collection of structured factor verification findings.

    Attributes:
        null_diagnostics: Per-factor NULL classifications.
        invalid_numeric_diagnostics: Per-observation invalid numerics.
        warning_diagnostics: Structured warnings for every emitted warning.
    """

    null_diagnostics: tuple[FactorNullDiagnostic, ...]
    invalid_numeric_diagnostics: tuple[FactorInvalidNumericDiagnostic, ...]
    warning_diagnostics: tuple[FactorWarningDiagnostic, ...]

    @property
    def has_content(self) -> bool:
        """Return whether any diagnostic category is non-empty."""
        return (
            len(self.null_diagnostics) > 0
            or len(self.invalid_numeric_diagnostics) > 0
            or len(self.warning_diagnostics) > 0
        )


@dataclass(frozen=True, slots=True)
class FactorVerificationReport:
    """Factor verification report with counters and structured diagnostics.

    Attributes:
        rows_checked: Total rows inspected.
        duplicate_timestamp_rows: Duplicate long-format primary-key rows.
        null_rows: Rows containing NULL in any required column.
        nan_rows: Rows containing NaN ``factor_value``.
        invalid_timestamp_rows: Rows with invalid ``open_time``.
        invalid_numeric_rows: Rows with infinite or overflow/underflow values.
        warmup_null_rows: ``factor_value`` NULLs classified as warmup.
        domain_null_rows: ``factor_value`` NULLs classified as domain NULLs.
        unexpected_null_rows: NULL rows that are neither warmup nor domain.
        positive_inf_rows: Rows with ``+inf`` ``factor_value``.
        negative_inf_rows: Rows with ``-inf`` ``factor_value``.
        non_finite_rows: Rows with non-finite non-infinite ``factor_value``.
        warnings: Deterministic human-readable warning strings.
        diagnostics: Structured diagnostics for debugging.
        passed: Overall pass/fail; warmup and domain NULLs do not fail.
    """

    rows_checked: int
    duplicate_timestamp_rows: int
    null_rows: int
    nan_rows: int
    invalid_timestamp_rows: int
    invalid_numeric_rows: int
    warmup_null_rows: int
    domain_null_rows: int
    unexpected_null_rows: int
    positive_inf_rows: int
    negative_inf_rows: int
    non_finite_rows: int
    warnings: tuple[str, ...]
    diagnostics: FactorVerificationDiagnostics
    passed: bool


@dataclass(frozen=True, slots=True)
class GlobalFailureFinding:
    """One Unexpected NULL or +Inf finding located to a partition.

    Attributes:
        symbol: Factor symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        factor_name: Factor identity.
        issue: Human-readable issue label (``Unexpected NULL`` or ``+Inf``).
        count: Number of affected observations.
        first_open_time: Earliest affected ``open_time`` (epoch milliseconds).
        last_open_time: Latest affected ``open_time`` (epoch milliseconds).
    """

    symbol: str
    timeframe: str
    year: int
    factor_name: str
    issue: str
    count: int
    first_open_time: int
    last_open_time: int


def collect_global_failure_findings(
    *,
    symbol: str,
    timeframe: str,
    year: int,
    diagnostics: FactorVerificationDiagnostics,
) -> tuple[GlobalFailureFinding, ...]:
    """Extract Unexpected NULL and +Inf findings for one partition.

    Warmup NULLs, domain NULLs, and non-+Inf invalid numerics are excluded.

    Args:
        symbol: Factor symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        diagnostics: Structured verification diagnostics for the partition.

    Returns:
        Deterministically ordered findings for the partition.
    """
    findings: list[GlobalFailureFinding] = []

    for item in diagnostics.null_diagnostics:
        if item.classification != NullClassification.UNEXPECTED_NULLS:
            continue
        findings.append(
            GlobalFailureFinding(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                factor_name=item.factor_name,
                issue=_ISSUE_UNEXPECTED_NULL,
                count=item.count,
                first_open_time=item.first_open_time,
                last_open_time=item.last_open_time,
            )
        )

    positive_inf_by_factor: dict[str, list[FactorInvalidNumericDiagnostic]] = {}
    for item in diagnostics.invalid_numeric_diagnostics:
        if item.kind != InvalidNumericKind.POSITIVE_INFINITY:
            continue
        positive_inf_by_factor.setdefault(item.factor_name, []).append(item)

    for factor_name in sorted(positive_inf_by_factor):
        items = positive_inf_by_factor[factor_name]
        open_times = sorted(entry.open_time for entry in items)
        findings.append(
            GlobalFailureFinding(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                factor_name=factor_name,
                issue=_ISSUE_POSITIVE_INF,
                count=sum(entry.count for entry in items),
                first_open_time=open_times[0],
                last_open_time=open_times[-1],
            )
        )

    findings.sort(
        key=lambda finding: (
            finding.symbol,
            finding.timeframe,
            finding.year,
            finding.factor_name,
            finding.issue,
            finding.first_open_time,
        )
    )
    return tuple(findings)


def format_global_failure_report(
    findings: Sequence[GlobalFailureFinding],
) -> str:
    """Render a global FAIL report locating Unexpected NULL and +Inf findings.

    Args:
        findings: Partition-scoped failure findings across the repository.

    Returns:
        Multi-line failure report. Empty string when ``findings`` is empty.
    """
    if not findings:
        return ""

    ordered = sorted(
        findings,
        key=lambda finding: (
            finding.symbol,
            finding.timeframe,
            finding.year,
            finding.factor_name,
            finding.issue,
            finding.first_open_time,
        ),
    )

    blocks: list[str] = []
    for finding in ordered:
        blocks.append(_format_global_failure_block(finding))

    partitions = {(finding.symbol, finding.timeframe, finding.year) for finding in ordered}
    symbols = {finding.symbol for finding in ordered}
    factors = {finding.factor_name for finding in ordered}

    footer = "\n".join(
        [
            f"Affected partitions: {len(partitions)}",
            f"Affected symbols: {len(symbols)}",
            f"Affected factors: {len(factors)}",
        ]
    )
    body = f"\n{_SEPARATOR}\n".join(blocks)
    return (
        "=====================================\n"
        "CQROS Factor Failure Report\n"
        "=====================================\n"
        "\n"
        f"{body}\n"
        f"{_SEPARATOR}\n"
        "\n"
        f"{footer}\n"
    )


def format_factor_diagnostics(diagnostics: FactorVerificationDiagnostics) -> str:
    """Render detailed diagnostics for ``--debug`` CLI output.

    Args:
        diagnostics: Structured verification diagnostics.

    Returns:
        Multi-line diagnostic text. Empty string when there is no content.
    """
    if not diagnostics.has_content:
        return ""

    blocks: list[str] = []
    for item in diagnostics.null_diagnostics:
        blocks.append(
            "\n".join(
                [
                    f"Factor: {item.factor_name}",
                    f"Issue: {_null_issue_label(item.classification)}",
                    f"Count: {item.count}",
                    "Range:",
                    _format_open_time(item.first_open_time),
                    "->",
                    _format_open_time(item.last_open_time),
                ]
            )
        )

    for item in diagnostics.invalid_numeric_diagnostics:
        blocks.append(
            "\n".join(
                [
                    f"Factor: {item.factor_name}",
                    _invalid_numeric_title(item.kind),
                    "Timestamp",
                    _format_open_time(item.open_time),
                    f"Count: {item.count}",
                ]
            )
        )

    for item in diagnostics.warning_diagnostics:
        # NULL and invalid-numeric warnings are already rendered above.
        if item.warning_type in {
            NullClassification.WARMUP_NULLS.value,
            NullClassification.DOMAIN_NULLS.value,
            NullClassification.UNEXPECTED_NULLS.value,
            "POSITIVE_INFINITY",
            "NEGATIVE_INFINITY",
            "OVERFLOW",
            "UNDERFLOW",
            "NON_FINITE",
        }:
            continue
        factor_label = item.factor_name if item.factor_name is not None else "*"
        blocks.append(
            "\n".join(
                [
                    f"Factor: {factor_label}",
                    item.warning_type,
                    f"Count: {item.count}",
                ]
            )
        )

    if not blocks:
        return ""
    return f"\n{_SEPARATOR}\n".join(blocks) + f"\n{_SEPARATOR}\n"


def _null_issue_label(classification: NullClassification) -> str:
    """Return the human-readable Issue label for a NULL classification."""
    mapping: dict[NullClassification, str] = {
        NullClassification.WARMUP_NULLS: "Warmup NULLs",
        NullClassification.DOMAIN_NULLS: _ISSUE_DOMAIN_NULL,
        NullClassification.UNEXPECTED_NULLS: _ISSUE_UNEXPECTED_NULL,
    }
    return mapping[classification]


def _format_global_failure_block(finding: GlobalFailureFinding) -> str:
    """Render one partition-scoped failure finding block."""
    lines = [
        f"Symbol: {finding.symbol}",
        f"Timeframe: {finding.timeframe}",
        f"Year: {finding.year}",
        f"Factor: {finding.factor_name}",
        f"Issue: {finding.issue}",
        f"Count: {finding.count}",
    ]
    if finding.issue == _ISSUE_POSITIVE_INF and finding.count == 1:
        lines.append("Timestamp:")
        lines.append(_format_open_time_utc(finding.first_open_time))
    else:
        lines.append("Range:")
        lines.append(_format_open_time(finding.first_open_time))
        lines.append("->")
        lines.append(_format_open_time(finding.last_open_time))
    return "\n".join(lines)


def _format_open_time(open_time_ms: int) -> str:
    """Format epoch milliseconds as a UTC calendar date."""
    moment = datetime.fromtimestamp(open_time_ms / 1000.0, tz=UTC)
    return moment.date().isoformat()


def _format_open_time_utc(open_time_ms: int) -> str:
    """Format epoch milliseconds as ``YYYY-MM-DD HH:MM UTC``."""
    moment = datetime.fromtimestamp(open_time_ms / 1000.0, tz=UTC)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _invalid_numeric_title(kind: InvalidNumericKind) -> str:
    """Return a human-readable title for an invalid numeric kind."""
    mapping: dict[InvalidNumericKind, str] = {
        InvalidNumericKind.POSITIVE_INFINITY: "Positive infinity",
        InvalidNumericKind.NEGATIVE_INFINITY: "Negative infinity",
        InvalidNumericKind.OVERFLOW: "Overflow",
        InvalidNumericKind.UNDERFLOW: "Underflow",
        InvalidNumericKind.NON_FINITE: "Non-finite",
    }
    return mapping[kind]
