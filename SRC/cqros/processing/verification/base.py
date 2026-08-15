"""CQROS processing verification shared helper base.

Purpose:
    Provide reusable, non-mutating verification helpers used by every
    dataset verifier.

Responsibilities:
    - Validate required columns
    - Count duplicate timestamps, null rows, NaN rows, and invalid timestamps
    - Detect monotonic (non-decreasing) timestamp order
    - Report findings only; never clean, sort, or mutate frames

Dependencies:
    ``polars``, the Python standard library, and
    ``cqros.processing.verification.exceptions``.

Public API:
    ``BaseVerifier``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ProcessingValidationError,
)

__all__ = ["BaseVerifier"]

_ERROR_REQUIRED_COLUMNS: Final[str] = ERROR_REQUIRED_COLUMNS


class BaseVerifier:
    """Concrete helper base shared by dataset verification implementations.

    Methods inspect frames and return findings. They never mutate inputs,
    never sort, and never remove or rewrite rows. Dataset verifiers compose
    these helpers and build ``VerificationReport`` instances themselves.
    """

    def _validate_required_columns(
        self,
        frame: pl.DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Raise when any required column is absent from ``frame``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            required_columns: Column names that must be present.

        Raises:
            ProcessingValidationError: If one or more required columns are
                missing.
        """
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise ProcessingValidationError(
                f"missing required columns: {list(missing)}",
                error_code=_ERROR_REQUIRED_COLUMNS,
                details={
                    "missing_columns": missing,
                    "required_columns": tuple(required_columns),
                    "available_columns": tuple(frame.columns),
                },
            )

    def _count_duplicate_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows beyond the first occurrence of each timestamp.

        Uses keep-first semantics: one row per distinct timestamp value is
        retained conceptually; all later duplicates are counted.

        Args:
            frame: Input DataFrame. Must not be mutated.
            timestamp_column: Timestamp column name.

        Returns:
            Count of duplicate timestamp rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.col(timestamp_column).n_unique()).item())
        return frame.height - unique_count

    def _count_null_rows(
        self,
        frame: pl.DataFrame,
        columns: Sequence[str],
    ) -> int:
        """Return rows containing at least one NULL among ``columns``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            columns: Columns inspected for null values.

        Returns:
            Count of rows with one or more nulls in the given columns.
        """
        if frame.height == 0 or not columns:
            return 0
        null_mask = pl.any_horizontal(*(pl.col(name).is_null() for name in columns))
        return int(frame.select(null_mask.sum()).item())

    def _count_nan_rows(
        self,
        frame: pl.DataFrame,
        numeric_columns: Sequence[str],
    ) -> int:
        """Return rows containing at least one NaN among floating columns.

        Non-floating columns listed in ``numeric_columns`` are ignored.

        Args:
            frame: Input DataFrame. Must not be mutated.
            numeric_columns: Candidate numeric column names.

        Returns:
            Count of rows with one or more NaN values in floating columns.
        """
        if frame.height == 0 or not numeric_columns:
            return 0
        floating_columns = tuple(
            name
            for name in numeric_columns
            if name in frame.schema and frame.schema[name].is_float()
        )
        if not floating_columns:
            return 0
        nan_mask = pl.any_horizontal(*(pl.col(name).is_nan() for name in floating_columns))
        return int(frame.select(nan_mask.sum()).item())

    def _count_invalid_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows with NULL, non-positive, or non-integer timestamps.

        When ``timestamp_column`` does not have an integer dtype, every row is
        counted as invalid. Timestamps are never modified.

        Args:
            frame: Input DataFrame. Must not be mutated.
            timestamp_column: Timestamp column name.

        Returns:
            Count of invalid timestamp rows.
        """
        if frame.height == 0:
            return 0
        dtype = frame.schema[timestamp_column]
        if not dtype.is_integer():
            return frame.height
        invalid_mask = pl.col(timestamp_column).is_null() | (pl.col(timestamp_column) <= 0)
        return int(frame.select(invalid_mask.sum()).item())

    def _is_sorted(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> bool:
        """Return whether timestamps are monotonically non-decreasing.

        Equal adjacent timestamps are allowed; duplicates are reported by
        ``_count_duplicate_timestamp_rows``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            timestamp_column: Timestamp column name.

        Returns:
            ``True`` when timestamps are sorted ascending with equals allowed.
        """
        if frame.height <= 1:
            return True
        return bool(frame.get_column(timestamp_column).is_sorted())
