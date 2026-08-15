"""CQROS merged portfolio risk decision dataset verification.

Purpose:
    Inspect canonical portfolio-risk frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged portfolio-risk columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps, invalid
      enum values, invalid booleans, cooldown inconsistencies, and
      non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.portfolio_risk.exceptions``,
    ``cqros.portfolio_risk.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``PortfolioRiskVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.portfolio_risk.exceptions import PortfolioRiskValidationError
from cqros.portfolio_risk.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PortfolioRiskState,
    ShutdownReason,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PortfolioRiskVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PRISK-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PRISK-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_STATE: Final[str] = "portfolio_risk_state"
_COL_ALLOW: Final[str] = "allow_new_entries"
_COL_REASON: Final[str] = "shutdown_reason"
_COL_COOLDOWN: Final[str] = "cooldown_until"
_COL_EQUITY: Final[str] = "equity"
_COL_GROSS: Final[str] = "gross_exposure"
_COL_NET: Final[str] = "net_exposure"
_COL_DAILY_REALIZED: Final[str] = "daily_realized_pnl"
_COL_DAILY_UNREALIZED: Final[str] = "daily_unrealized_pnl"
_COL_DAILY_TOTAL: Final[str] = "daily_total_pnl"
_COL_DAILY_RETURN: Final[str] = "daily_return_pct"
_COL_DAILY_DRAWDOWN: Final[str] = "daily_drawdown_pct"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# cooldown_until is intentionally nullable and excluded from null checks.
_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column != _COL_COOLDOWN
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_EQUITY,
    _COL_GROSS,
    _COL_NET,
    _COL_DAILY_REALIZED,
    _COL_DAILY_UNREALIZED,
    _COL_DAILY_TOTAL,
    _COL_DAILY_RETURN,
    _COL_DAILY_DRAWDOWN,
)

_ALLOWED_STATES: Final[tuple[str, ...]] = values(PortfolioRiskState)
_ALLOWED_REASONS: Final[tuple[str, ...]] = values(ShutdownReason)

_WARN_DUPLICATES: Final[str] = "Duplicate portfolio-risk primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATE: Final[str] = "Empty portfolio_risk_state values detected."
_WARN_INVALID_STATE: Final[str] = "Invalid PortfolioRiskState values detected."
_WARN_INVALID_REASON: Final[str] = "Invalid ShutdownReason values detected."
_WARN_INVALID_BOOLEAN: Final[str] = "Invalid allow_new_entries boolean values detected."
_WARN_COOLDOWN: Final[str] = "Inconsistent cooldown_until values detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "model_name",
    "model_version",
    "optimizer",
    "policy",
)


class PortfolioRiskVerifier(BaseVerifier):
    """Deterministic canonical portfolio-risk verifier that reports findings only.

    Inspects structural quality of a canonical portfolio-risk frame against
    ``cqros.portfolio_risk.schema`` / ``MERGED_PORTFOLIO_RISK_SCHEMA`` and the
    canonical state / reason enumerations. Does not clean rows, fill gaps,
    sort timestamps, mutate values, access storage, or apply risk logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical portfolio-risk DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PortfolioRiskValidationError: If any required column is missing or
                column dtypes do not match the merged portfolio-risk schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame, _COL_OPEN_TIME)
        empty_state_rows = self._count_empty_string_rows(frame, _COL_STATE)
        invalid_state_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATE,
            _ALLOWED_STATES,
        )
        invalid_reason_rows = self._count_invalid_enum_rows(
            frame,
            _COL_REASON,
            _ALLOWED_REASONS,
        )
        invalid_boolean_rows = self._count_invalid_boolean_rows(frame)
        inconsistent_cooldown_rows = self._count_inconsistent_cooldown_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_state_rows=empty_state_rows,
            invalid_state_rows=invalid_state_rows,
            invalid_reason_rows=invalid_reason_rows,
            invalid_boolean_rows=invalid_boolean_rows,
            inconsistent_cooldown_rows=inconsistent_cooldown_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_state_rows == 0
            and invalid_state_rows == 0
            and invalid_reason_rows == 0
            and invalid_boolean_rows == 0
            and inconsistent_cooldown_rows == 0
            and incomplete_lineage_rows == 0
            and is_sorted
            and is_canonical_order
        )
        return VerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            warnings=warnings,
            passed=passed,
        )

    def _validate_required_columns(
        self,
        frame: pl.DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Raise when any required column is absent from ``frame``."""
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise PortfolioRiskValidationError(
                f"missing required columns: {list(missing)}",
                error_code=ERROR_REQUIRED_COLUMNS,
                details={
                    "missing_columns": missing,
                    "required_columns": tuple(required_columns),
                    "available_columns": tuple(frame.columns),
                },
            )

    def _validate_column_dtypes(self, frame: pl.DataFrame) -> None:
        """Raise when any required column dtype differs from the schema."""
        mismatched: list[dict[str, object]] = []
        for column in REQUIRED_COLUMNS:
            expected = COLUMN_DTYPES[column]
            actual = frame.schema[column]
            if actual != expected:
                mismatched.append(
                    {
                        "column": column,
                        "expected": str(expected),
                        "actual": str(actual),
                    }
                )
        if mismatched:
            raise PortfolioRiskValidationError(
                "merged portfolio-risk schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first primary-key occurrence."""
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_invalid_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows with NULL timestamp values in ``timestamp_column``."""
        if frame.height == 0:
            return 0
        expected = COLUMN_DTYPES[timestamp_column]
        actual = frame.schema[timestamp_column]
        if actual != expected:
            return frame.height
        return int(frame.select(pl.col(timestamp_column).is_null().sum()).item())

    def _count_empty_string_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows containing an empty string in ``column``."""
        if frame.height == 0:
            return 0
        return int(frame.select((pl.col(column) == "").sum()).item())

    def _count_invalid_enum_rows(
        self,
        frame: pl.DataFrame,
        column: str,
        allowed: Sequence[str],
    ) -> int:
        """Return rows whose ``column`` value is outside ``allowed``."""
        if frame.height == 0:
            return 0
        allowed_list = list(allowed)
        invalid_mask = pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed_list)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_invalid_boolean_rows(self, frame: pl.DataFrame) -> int:
        """Return rows whose ``allow_new_entries`` value is null."""
        if frame.height == 0:
            return 0
        return int(frame.select(pl.col(_COL_ALLOW).is_null().sum()).item())

    def _count_inconsistent_cooldown_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with cooldown_until inconsistent with shutdown_reason."""
        if frame.height == 0:
            return 0
        requires_cooldown = pl.col(_COL_REASON).is_in(
            [
                ShutdownReason.DAILY_LOSS_LIMIT.value,
                ShutdownReason.COOLDOWN.value,
            ]
        )
        missing_when_required = requires_cooldown & pl.col(_COL_COOLDOWN).is_null()
        present_when_none = (pl.col(_COL_REASON) == ShutdownReason.NONE.value) & pl.col(
            _COL_COOLDOWN
        ).is_not_null()
        present_when_exposure = (
            pl.col(_COL_REASON) == ShutdownReason.EXPOSURE_LIMIT.value
        ) & pl.col(_COL_COOLDOWN).is_not_null()
        # Daily-loss and cooldown rows require cooldown_until strictly after
        # open_time (shutdown expiry is open_time + cooldown hours).
        cooldown_not_after_open = pl.col(_COL_COOLDOWN).is_not_null() & (
            pl.col(_COL_COOLDOWN) <= pl.col(_COL_OPEN_TIME)
        )
        invalid_mask = (
            missing_when_required
            | present_when_none
            | present_when_exposure
            | cooldown_not_after_open
        )
        return int(frame.select(invalid_mask.sum()).item())

    def _count_incomplete_lineage_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank lineage metadata fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *((pl.col(column).is_null()) | (pl.col(column) == "") for column in _LINEAGE_COLUMNS)
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with non-finite numerics."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        return int(frame.select(non_finite.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_state_rows: int,
    invalid_state_rows: int,
    invalid_reason_rows: int,
    invalid_boolean_rows: int,
    inconsistent_cooldown_rows: int,
    invalid_numeric_rows: int,
    incomplete_lineage_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
) -> tuple[str, ...]:
    """Return deterministic warnings for non-zero counters and structure fails."""
    warnings: list[str] = []
    if not is_canonical_order:
        warnings.append(_WARN_COLUMN_ORDER)
    if duplicate_timestamp_rows > 0:
        warnings.append(_WARN_DUPLICATES)
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if empty_state_rows > 0:
        warnings.append(_WARN_EMPTY_STATE)
    if invalid_state_rows > 0:
        warnings.append(_WARN_INVALID_STATE)
    if invalid_reason_rows > 0:
        warnings.append(_WARN_INVALID_REASON)
    if invalid_boolean_rows > 0:
        warnings.append(_WARN_INVALID_BOOLEAN)
    if inconsistent_cooldown_rows > 0:
        warnings.append(_WARN_COOLDOWN)
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
