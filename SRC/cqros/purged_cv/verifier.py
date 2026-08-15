"""CQROS purged cross-validation dataset verification.

Purpose:
    Inspect canonical purged-CV frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required purged-CV columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, empty strategy identity fields,
      negative purge/embargo/row counts, non-finite scores, and invalid
      intra-window timestamp ordering
    - Require rows sorted by ``fold_id`` (engine emission order)
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Notes:
    Purged-CV training is non-contiguous: retained train observations may
    sit both before and after the test block (outside purge/embargo zones).
    ``train_start_time`` / ``train_end_time`` are the chronological extent
    of retained training rows, so ``train_end_time >= test_start_time`` is
    valid and must not be treated as overlap.

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.purged_cv.exceptions``,
    ``cqros.purged_cv.schema``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``PurgedCVVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.purged_cv.exceptions import PurgedCVError
from cqros.purged_cv.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    purged_cv_status_values,
)

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PurgedCVVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PCV-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PCV-VERIFICATION-002"

_COL_TRAIN_START: Final[str] = "train_start_time"
_COL_TRAIN_END: Final[str] = "train_end_time"
_COL_TEST_START: Final[str] = "test_start_time"
_COL_TEST_END: Final[str] = "test_end_time"
_COL_STATUS: Final[str] = "status"
_COL_STRATEGY_NAME: Final[str] = "strategy_name"
_COL_STRATEGY_VERSION: Final[str] = "strategy_version"
_COL_TIMEFRAME: Final[str] = "timeframe"
_COL_FOLD_ID: Final[str] = "fold_id"
_COL_PURGE_SIZE: Final[str] = "purge_size"
_COL_EMBARGO_SIZE: Final[str] = "embargo_size"
_COL_TRAIN_ROWS: Final[str] = "train_rows"
_COL_TEST_ROWS: Final[str] = "test_rows"
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
    _COL_TIMEFRAME,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = purged_cv_status_values()

_WARN_DUPLICATES: Final[str] = "Duplicate purged-CV primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid PurgedCVStatus values detected."
_WARN_EMPTY_IDENTITY_FIELDS: Final[str] = "Empty required strategy identity fields detected."
_WARN_PURGE_SIZE: Final[str] = "purge_size values less than 0 detected."
_WARN_EMBARGO_SIZE: Final[str] = "embargo_size values less than 0 detected."
_WARN_TRAIN_ROWS: Final[str] = "train_rows values less than 0 detected."
_WARN_TEST_ROWS: Final[str] = "test_rows values less than 0 detected."
_WARN_WINDOW_ORDER: Final[str] = "Invalid train/test window ordering detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by fold_id."


class PurgedCVVerifier(BaseVerifier):
    """Deterministic canonical purged-CV verifier that reports findings only.

    Inspects structural quality of a canonical purged-CV frame against
    ``cqros.purged_cv.schema`` / ``PURGED_CV_SCHEMA`` and the
    canonical status enumeration. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or apply purge logic.

    Window checks validate each side independently
    (``train_start_time <= train_end_time`` and
    ``test_start_time <= test_end_time``). They do not require training
    extents to precede the test block, because purged-CV training may
    retain observations on both sides of the test fold.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical purged-CV DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PurgedCVError: If any required column is missing or column
                dtypes do not match the purged-CV schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_fold_timestamp_rows(frame)
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_identity_field_rows = self._count_empty_identity_field_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_FOLD_ID)
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
            raise PurgedCVError(
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
            raise PurgedCVError(
                "purged-CV schema dtype mismatch",
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

    def _count_invalid_fold_timestamp_rows(self, frame: pl.DataFrame) -> int:
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
        """Return rows with blank required strategy identity fields."""
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
        """Return rows with domain or non-finite numeric violations.

        Intra-window ordering is checked independently for train and test
        extents. Cross-window ``train_end_time`` vs ``test_start_time``
        comparisons are intentionally omitted: purged-CV training may
        retain observations after the test block.
        """
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(
            *(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in _VALUE_COLUMNS
            )
        )
        purge_size_invalid = pl.col(_COL_PURGE_SIZE) < 0
        embargo_size_invalid = pl.col(_COL_EMBARGO_SIZE) < 0
        train_rows_invalid = pl.col(_COL_TRAIN_ROWS) < 0
        test_rows_invalid = pl.col(_COL_TEST_ROWS) < 0
        train_window_invalid = pl.col(_COL_TRAIN_START) > pl.col(_COL_TRAIN_END)
        test_window_invalid = pl.col(_COL_TEST_START) > pl.col(_COL_TEST_END)
        invalid_mask = (
            non_finite
            | purge_size_invalid
            | embargo_size_invalid
            | train_rows_invalid
            | test_rows_invalid
            | train_window_invalid
            | test_window_invalid
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
        if int(frame.select((pl.col(_COL_PURGE_SIZE) < 0).sum()).item()) > 0:
            warnings.append(_WARN_PURGE_SIZE)
        if int(frame.select((pl.col(_COL_EMBARGO_SIZE) < 0).sum()).item()) > 0:
            warnings.append(_WARN_EMBARGO_SIZE)
        if int(frame.select((pl.col(_COL_TRAIN_ROWS) < 0).sum()).item()) > 0:
            warnings.append(_WARN_TRAIN_ROWS)
        if int(frame.select((pl.col(_COL_TEST_ROWS) < 0).sum()).item()) > 0:
            warnings.append(_WARN_TEST_ROWS)
        window_invalid = int(
            frame.select(
                (
                    (pl.col(_COL_TRAIN_START) > pl.col(_COL_TRAIN_END))
                    | (pl.col(_COL_TEST_START) > pl.col(_COL_TEST_END))
                ).sum()
            ).item()
        )
        if window_invalid > 0:
            warnings.append(_WARN_WINDOW_ORDER)
        non_finite = int(
            frame.select(
                pl.any_horizontal(
                    *(
                        pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                        for column in _VALUE_COLUMNS
                    )
                ).sum()
            ).item()
        )
        if non_finite > 0:
            warnings.append(_WARN_NON_FINITE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
