"""CQROS alpha dataset verification.

Purpose:
    Perform structural validation of canonical Alpha frames against
    ``ALPHA_SCHEMA`` without cleaning or mutating input data.

Responsibilities:
    - Reject non-DataFrame and empty inputs
    - Validate required columns against ``REQUIRED_COLUMNS``
    - Validate canonical column order against ``CANONICAL_COLUMN_ORDER``
    - Validate schema equality against ``ALPHA_SCHEMA``
    - Raise ``AlphaError`` on structural failure
    - Return the validated frame unchanged
    - Remain free of ranking, statistics, model checks, and research logic

Dependencies:
    ``polars``, ``cqros.alpha.exceptions``, and ``cqros.alpha.schema``.

Public API:
    ``AlphaVerifier``, ``ERROR_FRAME_TYPE``, ``ERROR_FRAME_EMPTY``,
    ``ERROR_REQUIRED_COLUMNS``, ``ERROR_COLUMN_ORDER``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.alpha.exceptions import AlphaError
from cqros.alpha.schema import (
    ALPHA_SCHEMA,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    REQUIRED_COLUMNS,
)

__all__ = [
    "ERROR_COLUMN_ORDER",
    "ERROR_FRAME_EMPTY",
    "ERROR_FRAME_TYPE",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "AlphaVerifier",
]

ERROR_FRAME_TYPE: Final[str] = "ALPHA-VERIFICATION-001"
ERROR_FRAME_EMPTY: Final[str] = "ALPHA-VERIFICATION-002"
ERROR_REQUIRED_COLUMNS: Final[str] = "ALPHA-VERIFICATION-003"
ERROR_COLUMN_ORDER: Final[str] = "ALPHA-VERIFICATION-004"
ERROR_SCHEMA_MISMATCH: Final[str] = "ALPHA-VERIFICATION-005"


class AlphaVerifier:
    """Structural verifier for canonical Alpha DataFrames.

    Validates frame type, non-emptiness, required columns, canonical column
    order, and schema equality against ``ALPHA_SCHEMA``. Does not mutate the
    input, access storage, or apply alpha research logic.
    """

    __slots__ = ()

    def verify(self, frame: object) -> pl.DataFrame:
        """Verify ``frame`` structurally and return it unchanged.

        Args:
            frame: Input canonical Alpha DataFrame. Must not be mutated.

        Returns:
            The same ``frame`` instance after structural checks succeed.

        Raises:
            AlphaError: If ``frame`` is not a Polars DataFrame, is empty, is
                missing required columns, has non-canonical column order, or
                does not match ``ALPHA_SCHEMA``.
        """
        validated = _require_dataframe(frame)
        _require_non_empty(validated)
        _require_required_columns(validated)
        _require_canonical_column_order(validated)
        _require_schema_equality(validated)
        return validated


def _require_dataframe(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise AlphaError(
            "alpha frame must be a polars DataFrame",
            error_code=ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame


def _require_non_empty(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` contains no rows."""
    if frame.height == 0:
        raise AlphaError(
            "alpha frame must contain at least one row",
            error_code=ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )


def _require_required_columns(frame: pl.DataFrame) -> None:
    """Raise when any required column is absent from ``frame``."""
    missing = tuple(name for name in REQUIRED_COLUMNS if name not in frame.columns)
    if missing:
        raise AlphaError(
            f"missing required columns: {list(missing)}",
            error_code=ERROR_REQUIRED_COLUMNS,
            details={
                "missing_columns": missing,
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_canonical_column_order(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` column order differs from canonical order."""
    actual_order = tuple(frame.columns)
    if actual_order != CANONICAL_COLUMN_ORDER:
        raise AlphaError(
            "alpha frame column order does not match canonical order",
            error_code=ERROR_COLUMN_ORDER,
            details={
                "expected_order": CANONICAL_COLUMN_ORDER,
                "actual_order": actual_order,
            },
        )


def _require_schema_equality(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` schema differs from ``ALPHA_SCHEMA``."""
    if frame.schema == ALPHA_SCHEMA:
        return

    mismatched: list[dict[str, object]] = []
    for column in CANONICAL_COLUMN_ORDER:
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

    raise AlphaError(
        "alpha schema mismatch",
        error_code=ERROR_SCHEMA_MISMATCH,
        details={
            "expected_schema": str(ALPHA_SCHEMA),
            "actual_schema": str(frame.schema),
            "mismatched_columns": tuple(item["column"] for item in mismatched),
            "mismatches": tuple(mismatched),
        },
    )
