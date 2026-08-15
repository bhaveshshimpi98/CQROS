"""CQROS factor validation metrics dataset verification.

Purpose:
    Inspect canonical factor validation frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required factor validation columns and expected dtypes against
      ``schema.py`` (``REQUIRED_COLUMNS``, ``COLUMN_DTYPES``,
      ``FACTOR_VALIDATION_SCHEMA``, ``CANONICAL_COLUMN_ORDER``)
    - Validate canonical column order
    - Count duplicate primary keys, nulls in identity/lineage/status fields,
      NaNs, invalid timestamps, invalid status enum values, empty factor
      identity fields, and domain/non-finite numeric violations already
      enforced structurally
    - Allow null float validation metrics and ``ic_observations`` because the
      engine contract emits them for insufficient samples / unavailable
      horizons
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame
    - Never perform statistical or business-meaning validation (that belongs
      to the factor validation engine)

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.factor_validation.exceptions``,
    ``cqros.factor_validation.schema``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Notes:
    ``cqros.factor_validation.schema`` is the single source of truth for the
    columnar contract. This verifier performs structural validation only.
    Statistical validation and metric interpretation belong to the engine.

Public API:
    ``FactorValidationVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.factor_validation.exceptions import FactorValidationError
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_VALIDATION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    factor_validation_status_values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorValidationVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "FVAL-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "FVAL-VERIFICATION-002"

_COL_VALIDATION_TIME: Final[str] = "validation_time"
_COL_STATUS: Final[str] = "status"
_COL_FACTOR_NAME: Final[str] = "factor_name"
_COL_FACTOR_VERSION: Final[str] = "factor_version"
_COL_FACTOR_CATEGORY: Final[str] = "factor_category"
_COL_IC_P_VALUE: Final[str] = "ic_p_value"
_COL_IC_DECAY: Final[str] = "ic_decay"
_COL_TURNOVER: Final[str] = "turnover"
_COL_OBSERVATIONS: Final[str] = "observations"
_COL_IC_OBSERVATIONS: Final[str] = "ic_observations"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Identity, lineage, window, and status columns that must always be present.
# Float validation metrics and ``ic_observations`` may be null under the engine
# contract (for example insufficient observations -> FAIL with null metrics,
# or unavailable IC Decay horizons).
_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in REQUIRED_COLUMNS
    if COLUMN_DTYPES[column] != pl.Float64 and column != _COL_IC_OBSERVATIONS
)

# All Float64 metric columns from the schema contract (schema-driven).
_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if COLUMN_DTYPES[column] == pl.Float64
)

_STATUS_PASS: Final[str] = "PASS"

_NON_EMPTY_FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FACTOR_NAME,
    _COL_FACTOR_VERSION,
    _COL_FACTOR_CATEGORY,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = factor_validation_status_values()

_WARN_DUPLICATES: Final[str] = "Duplicate factor validation primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid FactorValidationStatus values detected."
_WARN_EMPTY_FACTOR_FIELDS: Final[str] = "Empty required factor identity fields detected."
_WARN_IC_P_VALUE: Final[str] = "ic_p_value values outside [0, 1] detected."
_WARN_IC_DECAY: Final[str] = "Negative ic_decay values detected."
_WARN_TURNOVER: Final[str] = "Negative turnover values detected."
_WARN_OBSERVATIONS: Final[str] = "observations values less than or equal to 0 detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by validation_time."


class FactorValidationVerifier(BaseVerifier):
    """Deterministic canonical factor validation verifier that reports findings only.

    Inspects structural quality of a canonical factor validation frame against
    ``cqros.factor_validation.schema`` (``FACTOR_VALIDATION_SCHEMA`` is the
    single source of truth) and the canonical status enumeration. Performs
    structural validation only. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or compute statistics.
    Statistical and business-meaning validation belong to the engine.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical factor validation DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            FactorValidationError: If any required column is missing or
                column dtypes do not match ``FACTOR_VALIDATION_SCHEMA``.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_VALIDATION_TIME,
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_factor_field_rows = self._count_empty_factor_field_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_VALIDATION_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            empty_factor_field_rows=empty_factor_field_rows,
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
            and empty_factor_field_rows == 0
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
            raise FactorValidationError(
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
        for column in FACTOR_VALIDATION_SCHEMA.names():
            expected = FACTOR_VALIDATION_SCHEMA[column]
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
            raise FactorValidationError(
                "factor validation schema dtype mismatch",
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
        expected = FACTOR_VALIDATION_SCHEMA[timestamp_column]
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

    def _count_empty_factor_field_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank required factor identity fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *(
                (pl.col(column).is_null()) | (pl.col(column) == "")
                for column in _NON_EMPTY_FACTOR_COLUMNS
            )
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with domain or non-finite numeric violations.

        Non-finite checks cover every Float64 schema column, but only where
        values are present. Domain bounds remain limited to previously
        enforced structural ranges (``ic_p_value``, ``ic_decay``,
        ``turnover``, ``observations``). ``observations <= 0`` is invalid
        only for ``PASS`` rows because the engine intentionally emits
        ``FAIL`` with zero observations when pairs are insufficient.
        Metric interpretation such as ``monotonicity_score`` range belongs
        to the engine, not this verifier.
        """
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(
            *(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in _VALUE_COLUMNS
            )
        )
        ic_p_value_invalid = pl.col(_COL_IC_P_VALUE).is_not_null() & (
            (pl.col(_COL_IC_P_VALUE) < 0.0) | (pl.col(_COL_IC_P_VALUE) > 1.0)
        )
        ic_decay_invalid = pl.col(_COL_IC_DECAY).is_not_null() & (pl.col(_COL_IC_DECAY) < 0.0)
        turnover_invalid = pl.col(_COL_TURNOVER).is_not_null() & (pl.col(_COL_TURNOVER) < 0.0)
        observations_invalid = (pl.col(_COL_STATUS) == _STATUS_PASS) & (
            pl.col(_COL_OBSERVATIONS) <= 0
        )
        invalid_mask = (
            non_finite
            | ic_p_value_invalid
            | ic_decay_invalid
            | turnover_invalid
            | observations_invalid
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
    empty_factor_field_rows: int,
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
    if empty_factor_field_rows > 0:
        warnings.append(_WARN_EMPTY_FACTOR_FIELDS)
    if frame.height > 0:
        if (
            int(
                frame.select(
                    (
                        pl.col(_COL_IC_P_VALUE).is_not_null()
                        & ((pl.col(_COL_IC_P_VALUE) < 0.0) | (pl.col(_COL_IC_P_VALUE) > 1.0))
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_IC_P_VALUE)
        if (
            int(
                frame.select(
                    (pl.col(_COL_IC_DECAY).is_not_null() & (pl.col(_COL_IC_DECAY) < 0.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_IC_DECAY)
        if (
            int(
                frame.select(
                    (pl.col(_COL_TURNOVER).is_not_null() & (pl.col(_COL_TURNOVER) < 0.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_TURNOVER)
        if (
            int(
                frame.select(
                    ((pl.col(_COL_STATUS) == _STATUS_PASS) & (pl.col(_COL_OBSERVATIONS) <= 0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_OBSERVATIONS)
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
