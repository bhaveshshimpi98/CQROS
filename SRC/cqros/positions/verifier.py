"""CQROS merged position dataset verification.

Purpose:
    Inspect canonical position frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required merged-position columns and expected dtypes
    - Validate canonical column order
    - Count duplicate ``position_id`` values, nulls, NaNs, invalid
      timestamps, invalid ``status`` / ``side`` enum values, negative
      quantities, negative average entry prices, and non-finite PnL
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.positions.exceptions``,
    ``cqros.positions.schema``, ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``PositionVerifier``, ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.positions.exceptions import PositionValidationError
from cqros.positions.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PositionSide,
    PositionStatus,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PositionVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "POS-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "POS-VERIFICATION-002"

_COL_OPENED_AT: Final[str] = "opened_at"
_COL_UPDATED_AT: Final[str] = "updated_at"
_COL_CLOSED_AT: Final[str] = "closed_at"
_COL_STATUS: Final[str] = "status"
_COL_SIDE: Final[str] = "side"
_COL_QUANTITY: Final[str] = "quantity"
_COL_AVERAGE_ENTRY_PRICE: Final[str] = "average_entry_price"
_COL_REALIZED_PNL: Final[str] = "realized_pnl"
_COL_UNREALIZED_PNL: Final[str] = "unrealized_pnl"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column != _COL_CLOSED_AT
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_QUANTITY,
    _COL_AVERAGE_ENTRY_PRICE,
    "market_price",
    _COL_REALIZED_PNL,
    _COL_UNREALIZED_PNL,
    "fees_paid",
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(PositionStatus)
_ALLOWED_SIDES: Final[tuple[str, ...]] = values(PositionSide)

_WARN_DUPLICATES: Final[str] = "Duplicate position ids detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid PositionStatus values detected."
_WARN_EMPTY_SIDE: Final[str] = "Empty side values detected."
_WARN_INVALID_SIDE: Final[str] = "Invalid PositionSide values detected."
_WARN_NEGATIVE_QUANTITY: Final[str] = "Negative quantity values detected."
_WARN_NEGATIVE_ENTRY: Final[str] = "Negative average_entry_price values detected."
_WARN_NON_FINITE_PNL: Final[str] = "Non-finite PnL values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by opened_at."


class PositionVerifier(BaseVerifier):
    """Deterministic canonical position verifier that reports findings only.

    Inspects structural quality of a canonical position frame against
    ``cqros.positions.schema`` / ``MERGED_POSITION_SCHEMA`` and the canonical
    position enumerations. Does not clean rows, fill gaps, sort timestamps,
    mutate values, access storage, or apply accounting logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical position DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PositionValidationError: If any required column is missing or
                column dtypes do not match the merged position schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = (
            self._count_invalid_timestamp_rows(frame, _COL_OPENED_AT)
            + self._count_invalid_timestamp_rows(frame, _COL_UPDATED_AT)
            + self._count_invalid_closed_at_rows(frame)
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_side_rows = self._count_empty_string_rows(frame, _COL_SIDE)
        invalid_side_rows = self._count_invalid_enum_rows(
            frame,
            _COL_SIDE,
            _ALLOWED_SIDES,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPENED_AT)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            empty_side_rows=empty_side_rows,
            invalid_side_rows=invalid_side_rows,
            invalid_numeric_rows=invalid_numeric_rows,
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
            and empty_side_rows == 0
            and invalid_side_rows == 0
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
            raise PositionValidationError(
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
            raise PositionValidationError(
                "merged position schema dtype mismatch",
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

    def _count_invalid_closed_at_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with invalid ``closed_at`` relative to ``status``."""
        if frame.height == 0:
            return 0
        expected = COLUMN_DTYPES[_COL_CLOSED_AT]
        actual = frame.schema[_COL_CLOSED_AT]
        if actual != expected:
            return frame.height
        open_with_closed = (pl.col(_COL_STATUS) == PositionStatus.OPEN.value) & pl.col(
            _COL_CLOSED_AT
        ).is_not_null()
        closed_without = (pl.col(_COL_STATUS) == PositionStatus.CLOSED.value) & pl.col(
            _COL_CLOSED_AT
        ).is_null()
        return int(frame.select((open_with_closed | closed_without).sum()).item())

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

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative quantity/entry or non-finite PnL."""
        if frame.height == 0:
            return 0
        invalid_mask = (
            (pl.col(_COL_QUANTITY) < 0.0)
            | (pl.col(_COL_AVERAGE_ENTRY_PRICE) < 0.0)
            | ~pl.col(_COL_REALIZED_PNL).is_finite()
            | ~pl.col(_COL_UNREALIZED_PNL).is_finite()
        )
        return int(frame.select(invalid_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_status_rows: int,
    invalid_status_rows: int,
    empty_side_rows: int,
    invalid_side_rows: int,
    invalid_numeric_rows: int,
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
    if empty_side_rows > 0:
        warnings.append(_WARN_EMPTY_SIDE)
    if invalid_side_rows > 0:
        warnings.append(_WARN_INVALID_SIDE)
    if frame.height > 0:
        if int(frame.select((pl.col(_COL_QUANTITY) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_NEGATIVE_QUANTITY)
        if int(frame.select((pl.col(_COL_AVERAGE_ENTRY_PRICE) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_NEGATIVE_ENTRY)
        if (
            int(
                frame.select(
                    (
                        ~pl.col(_COL_REALIZED_PNL).is_finite()
                        | ~pl.col(_COL_UNREALIZED_PNL).is_finite()
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_NON_FINITE_PNL)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE_PNL)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
