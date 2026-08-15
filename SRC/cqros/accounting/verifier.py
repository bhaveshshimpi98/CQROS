"""CQROS merged portfolio accounting dataset verification.

Purpose:
    Inspect canonical accounting frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required merged-accounting columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps, invalid
      ``position_status`` enum values, negative quantities, negative equity,
      and non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.accounting.exceptions``,
    ``cqros.accounting.schema``, ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``AccountingVerifier``, ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.accounting.exceptions import AccountingValidationError
from cqros.accounting.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PositionStatus,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "AccountingVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "ACC-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "ACC-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_POSITION_STATUS: Final[str] = "position_status"
_COL_QUANTITY: Final[str] = "quantity"
_COL_EQUITY: Final[str] = "equity"
_COL_AVERAGE_ENTRY_PRICE: Final[str] = "average_entry_price"
_COL_MARK_PRICE: Final[str] = "mark_price"
_COL_POSITION_VALUE: Final[str] = "position_value"
_COL_MARKET_VALUE: Final[str] = "market_value"
_COL_CASH: Final[str] = "cash"
_COL_REALIZED_PNL: Final[str] = "realized_pnl"
_COL_UNREALIZED_PNL: Final[str] = "unrealized_pnl"
_COL_TOTAL_PNL: Final[str] = "total_pnl"
_COL_GROSS_EXPOSURE: Final[str] = "gross_exposure"
_COL_NET_EXPOSURE: Final[str] = "net_exposure"
_COL_RETURN_PCT: Final[str] = "return_pct"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_QUANTITY,
    _COL_AVERAGE_ENTRY_PRICE,
    _COL_MARK_PRICE,
    _COL_POSITION_VALUE,
    _COL_MARKET_VALUE,
    _COL_CASH,
    _COL_REALIZED_PNL,
    _COL_UNREALIZED_PNL,
    _COL_TOTAL_PNL,
    _COL_GROSS_EXPOSURE,
    _COL_NET_EXPOSURE,
    _COL_EQUITY,
    _COL_RETURN_PCT,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(PositionStatus)

_WARN_DUPLICATES: Final[str] = "Duplicate accounting primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty position_status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid PositionStatus values detected."
_WARN_NEGATIVE_QUANTITY: Final[str] = "Negative quantity values detected."
_WARN_NEGATIVE_EQUITY: Final[str] = "Negative equity values detected."
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


class AccountingVerifier(BaseVerifier):
    """Deterministic canonical accounting verifier that reports findings only.

    Inspects structural quality of a canonical accounting frame against
    ``cqros.accounting.schema`` / ``MERGED_ACCOUNTING_SCHEMA`` and the
    canonical position-status enumeration. Does not clean rows, fill gaps,
    sort timestamps, mutate values, access storage, or apply accounting logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical accounting DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            AccountingValidationError: If any required column is missing or
                column dtypes do not match the merged accounting schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame, _COL_OPEN_TIME)
        empty_status_rows = self._count_empty_string_rows(frame, _COL_POSITION_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_POSITION_STATUS,
            _ALLOWED_STATUSES,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
            frame=frame,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_status_rows == 0
            and invalid_status_rows == 0
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
            raise AccountingValidationError(
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
            raise AccountingValidationError(
                "merged accounting schema dtype mismatch",
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

    def _count_incomplete_lineage_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank lineage metadata fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *((pl.col(column).is_null()) | (pl.col(column) == "") for column in _LINEAGE_COLUMNS)
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative quantity/equity or non-finite numerics."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        invalid_mask = (pl.col(_COL_QUANTITY) < 0.0) | (pl.col(_COL_EQUITY) < 0.0) | non_finite
        return int(frame.select(invalid_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_status_rows: int,
    invalid_status_rows: int,
    invalid_numeric_rows: int,
    incomplete_lineage_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
    frame: pl.DataFrame,
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
    if empty_status_rows > 0:
        warnings.append(_WARN_EMPTY_STATUS)
    if invalid_status_rows > 0:
        warnings.append(_WARN_INVALID_STATUS)
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if frame.height > 0:
        if int(frame.select((pl.col(_COL_QUANTITY) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_NEGATIVE_QUANTITY)
        if int(frame.select((pl.col(_COL_EQUITY) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_NEGATIVE_EQUITY)
        if (
            int(
                frame.select(
                    pl.any_horizontal(
                        *(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_NON_FINITE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
