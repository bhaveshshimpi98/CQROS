"""CQROS processing verification report value object.

Purpose:
    Provide an immutable summary of a single dataset verification pass.

Responsibilities:
    - Capture row-level verification counters and warnings
    - Validate constructor invariants for non-negative counters and warning
      shape
    - Preserve the verifier-supplied ``passed`` flag without deriving it

Dependencies:
    Python standard library and ``cqros.processing.verification.exceptions``.

Public API:
    ``VerificationReport``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from cqros.processing.verification.exceptions import (
    ERROR_COUNTER_INVALID,
    ERROR_ROWS_CHECKED_INVALID,
    ERROR_WARNINGS_INVALID,
    ProcessingValidationError,
)

__all__ = ["VerificationReport"]

_COUNTER_FIELDS: Final[tuple[str, ...]] = (
    "duplicate_timestamp_rows",
    "null_rows",
    "nan_rows",
    "invalid_timestamp_rows",
    "invalid_numeric_rows",
)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Immutable summary of a dataset verification pass.

    Counters describe how many rows matched each check category. ``passed``
    is supplied by the verifier and is never derived from the counters.

    Attributes:
        rows_checked: Total rows inspected by the verifier.
        duplicate_timestamp_rows: Rows with duplicate timestamps.
        null_rows: Rows containing null values in mandatory fields.
        nan_rows: Rows containing NaN values in mandatory numeric fields.
        invalid_timestamp_rows: Rows with invalid timestamp values.
        invalid_numeric_rows: Rows with invalid numeric values.
        warnings: Deterministic human-readable warnings from the pass.
        passed: Verifier-supplied overall pass/fail outcome.
    """

    rows_checked: int
    duplicate_timestamp_rows: int
    null_rows: int
    nan_rows: int
    invalid_timestamp_rows: int
    invalid_numeric_rows: int
    warnings: tuple[str, ...]
    passed: bool

    def __post_init__(self) -> None:
        """Validate constructor invariants.

        Raises:
            ProcessingValidationError: If any counter is negative or not an
                integer, or if ``warnings`` is not ``tuple[str, ...]``.
        """
        _require_non_negative_int(
            self.rows_checked,
            parameter="rows_checked",
            error_code=ERROR_ROWS_CHECKED_INVALID,
        )
        for field_name in _COUNTER_FIELDS:
            _require_non_negative_int(
                getattr(self, field_name),
                parameter=field_name,
                error_code=ERROR_COUNTER_INVALID,
            )
        _require_warnings_tuple(self.warnings)


def _require_non_negative_int(value: object, *, parameter: str, error_code: str) -> None:
    """Raise when ``value`` is not a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProcessingValidationError(
            f"{parameter} must be a non-negative integer",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _require_warnings_tuple(value: object) -> None:
    """Raise when ``value`` is not ``tuple[str, ...]``."""
    if not isinstance(value, tuple):
        raise ProcessingValidationError(
            "warnings must be tuple[str, ...]",
            error_code=ERROR_WARNINGS_INVALID,
            details={"parameter": "warnings", "value": value},
        )
    warnings_tuple = cast(tuple[object, ...], value)
    if not all(isinstance(item, str) for item in warnings_tuple):
        raise ProcessingValidationError(
            "warnings must be tuple[str, ...]",
            error_code=ERROR_WARNINGS_INVALID,
            details={"parameter": "warnings", "value": warnings_tuple},
        )
