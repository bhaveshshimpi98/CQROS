"""CQROS walk-forward evaluation dataset verification.

Purpose:
    Inspect canonical walk-forward frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required walk-forward columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, empty strategy/model identity fields,
      non-positive row counts, negative selected-factor counts, non-finite
      scores, and invalid train/test window ordering
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.walk_forward.exceptions``,
    ``cqros.walk_forward.schema``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``WalkForwardVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.walk_forward.exceptions import WalkForwardError
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    walk_forward_status_values,
)

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "WalkForwardVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "WF-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "WF-VERIFICATION-002"

_COL_TRAIN_START: Final[str] = "train_start"
_COL_TRAIN_END: Final[str] = "train_end"
_COL_TEST_START: Final[str] = "test_start"
_COL_TEST_END: Final[str] = "test_end"
_COL_STATUS: Final[str] = "status"
_COL_STRATEGY_NAME: Final[str] = "strategy_name"
_COL_STRATEGY_VERSION: Final[str] = "strategy_version"
_COL_MODEL_VERSION: Final[str] = "model_version"
_COL_TRAIN_ROWS: Final[str] = "train_rows"
_COL_TEST_ROWS: Final[str] = "test_rows"
_COL_SELECTED_FACTORS: Final[str] = "selected_factors"
_COL_TRAIN_SCORE: Final[str] = "train_score"
_COL_TEST_SCORE: Final[str] = "test_score"
_COL_OVERFIT_GAP: Final[str] = "overfit_gap"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TRAIN_START,
    _COL_TRAIN_END,
    _COL_TEST_START,
    _COL_TEST_END,
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TRAIN_SCORE,
    _COL_TEST_SCORE,
    _COL_OVERFIT_GAP,
)

_NON_EMPTY_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_STRATEGY_NAME,
    _COL_STRATEGY_VERSION,
    _COL_MODEL_VERSION,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = walk_forward_status_values()

_WARN_DUPLICATES: Final[str] = "Duplicate walk-forward primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid WalkForwardStatus values detected."
_WARN_EMPTY_IDENTITY_FIELDS: Final[str] = (
    "Empty required strategy or model identity fields detected."
)
_WARN_TRAIN_ROWS: Final[str] = "train_rows values less than or equal to 0 detected."
_WARN_TEST_ROWS: Final[str] = "test_rows values less than or equal to 0 detected."
_WARN_SELECTED_FACTORS: Final[str] = "selected_factors values less than 0 detected."
_WARN_WINDOW_ORDER: Final[str] = "Invalid train/test window ordering detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by train_start."


class WalkForwardVerifier(BaseVerifier):
    """Deterministic canonical walk-forward verifier that reports findings only.

    Inspects structural quality of a canonical walk-forward frame against
    ``cqros.walk_forward.schema`` / ``WALK_FORWARD_SCHEMA`` and the
    canonical status enumeration. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or apply walk-forward logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical walk-forward DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            WalkForwardError: If any required column is missing or column
                dtypes do not match the walk-forward schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame)
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_identity_field_rows = self._count_empty_identity_field_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_TRAIN_START)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            empty_identity_field_rows=empty_identity_field_rows,
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
            and empty_identity_field_rows == 0
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
            raise WalkForwardError(
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
            raise WalkForwardError(
                "walk-forward schema dtype mismatch",
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

    def _count_invalid_timestamp_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with NULL values in any fold-window timestamp column."""
        if frame.height == 0:
            return 0
        for timestamp_column in _TIMESTAMP_COLUMNS:
            expected = COLUMN_DTYPES[timestamp_column]
            actual = frame.schema[timestamp_column]
            if actual != expected:
                return frame.height
        null_mask = pl.any_horizontal(*(pl.col(column).is_null() for column in _TIMESTAMP_COLUMNS))
        return int(frame.select(null_mask.sum()).item())

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

    def _count_empty_identity_field_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank required strategy or model identity fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *(
                (pl.col(column).is_null()) | (pl.col(column) == "")
                for column in _NON_EMPTY_IDENTITY_COLUMNS
            )
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with domain or non-finite numeric violations."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        train_rows_invalid = pl.col(_COL_TRAIN_ROWS) <= 0
        test_rows_invalid = pl.col(_COL_TEST_ROWS) <= 0
        selected_factors_invalid = pl.col(_COL_SELECTED_FACTORS) < 0
        train_window_invalid = pl.col(_COL_TRAIN_START) > pl.col(_COL_TRAIN_END)
        test_window_invalid = pl.col(_COL_TEST_START) > pl.col(_COL_TEST_END)
        train_before_test_invalid = pl.col(_COL_TRAIN_END) > pl.col(_COL_TEST_START)
        invalid_mask = (
            non_finite
            | train_rows_invalid
            | test_rows_invalid
            | selected_factors_invalid
            | train_window_invalid
            | test_window_invalid
            | train_before_test_invalid
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
    empty_identity_field_rows: int,
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
    if empty_identity_field_rows > 0:
        warnings.append(_WARN_EMPTY_IDENTITY_FIELDS)
    if frame.height > 0:
        if int(frame.select((pl.col(_COL_TRAIN_ROWS) <= 0).sum()).item()) > 0:
            warnings.append(_WARN_TRAIN_ROWS)
        if int(frame.select((pl.col(_COL_TEST_ROWS) <= 0).sum()).item()) > 0:
            warnings.append(_WARN_TEST_ROWS)
        if int(frame.select((pl.col(_COL_SELECTED_FACTORS) < 0).sum()).item()) > 0:
            warnings.append(_WARN_SELECTED_FACTORS)
        window_invalid = int(
            frame.select(
                (
                    (pl.col(_COL_TRAIN_START) > pl.col(_COL_TRAIN_END))
                    | (pl.col(_COL_TEST_START) > pl.col(_COL_TEST_END))
                    | (pl.col(_COL_TRAIN_END) > pl.col(_COL_TEST_START))
                ).sum()
            ).item()
        )
        if window_invalid > 0:
            warnings.append(_WARN_WINDOW_ORDER)
        non_finite = int(
            frame.select(
                pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)).sum()
            ).item()
        )
        if non_finite > 0:
            warnings.append(_WARN_NON_FINITE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
