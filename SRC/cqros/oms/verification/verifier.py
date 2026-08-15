"""CQROS merged order dataset verification.

Purpose:
    Inspect canonical OMS order frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required merged-order columns and expected dtypes
    - Validate canonical column order
    - Count duplicate ``(symbol, timeframe, open_time, order_id)`` keys,
      nulls in non-nullable columns, NaNs, invalid ``open_time`` values,
      invalid ``side`` / ``order_type`` / ``status`` values, empty identity
      and categorical strings, and non-finite numeric order fields
    - Accept NULL ``limit_price``, ``stop_price``, and ``average_fill_price``
      (required to exist; nullable by order type / fill state)
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.oms.enums``,
    ``cqros.oms.schema``, ``cqros.oms.verification.exceptions``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``OrderVerifier``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.oms.enums import OrderSide, OrderStatus, OrderType, values
from cqros.oms.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.oms.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    OMSValidationError,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["OrderVerifier"]

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_ORDER_ID: Final[str] = "order_id"
_COL_PARENT_ORDER_ID: Final[str] = "parent_order_id"
_COL_SIDE: Final[str] = "side"
_COL_ORDER_TYPE: Final[str] = "order_type"
_COL_STATUS: Final[str] = "status"
_COL_QUANTITY: Final[str] = "quantity"
_COL_LIMIT_PRICE: Final[str] = "limit_price"
_COL_STOP_PRICE: Final[str] = "stop_price"
_COL_FILLED_QUANTITY: Final[str] = "filled_quantity"
_COL_AVERAGE_FILL_PRICE: Final[str] = "average_fill_price"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Present in REQUIRED_COLUMNS / MERGED_ORDER_SCHEMA but intentionally nullable:
# market orders omit limit/stop prices; unfilled orders omit average fill price.
_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        _COL_LIMIT_PRICE,
        _COL_STOP_PRICE,
        _COL_AVERAGE_FILL_PRICE,
    }
)

# Required columns whose NULL values are hard verification failures.
_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in _NULLABLE_COLUMNS
)

# Non-primary-key columns inspected for NaN among floating dtypes.
_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in PRIMARY_KEY_COLUMNS
)

_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_QUANTITY,
    _COL_LIMIT_PRICE,
    _COL_STOP_PRICE,
    _COL_FILLED_QUANTITY,
    _COL_AVERAGE_FILL_PRICE,
)

_ALLOWED_SIDES: Final[tuple[str, ...]] = values(OrderSide)
_ALLOWED_ORDER_TYPES: Final[tuple[str, ...]] = values(OrderType)
_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(OrderStatus)

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_ORDER_ID: Final[str] = "Empty order_id values detected."
_WARN_EMPTY_PARENT_ORDER_ID: Final[str] = "Empty parent_order_id values detected."
_WARN_EMPTY_SIDE: Final[str] = "Empty side values detected."
_WARN_EMPTY_ORDER_TYPE: Final[str] = "Empty order_type values detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_SIDE: Final[str] = "Invalid OrderSide values detected."
_WARN_INVALID_ORDER_TYPE: Final[str] = "Invalid OrderType values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid OrderStatus values detected."
_WARN_INVALID_QUANTITY: Final[str] = "Invalid quantity values detected."
_WARN_INVALID_LIMIT_PRICE: Final[str] = "Invalid limit_price values detected."
_WARN_INVALID_STOP_PRICE: Final[str] = "Invalid stop_price values detected."
_WARN_INVALID_FILLED_QUANTITY: Final[str] = "Invalid filled_quantity values detected."
_WARN_INVALID_AVERAGE_FILL_PRICE: Final[str] = "Invalid average_fill_price values detected."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."


class OrderVerifier(BaseVerifier):
    """Deterministic canonical OMS-order verifier that reports findings only.

    Inspects structural quality of a canonical order frame against
    ``cqros.oms.schema`` / ``MERGED_ORDER_SCHEMA`` and the canonical OMS
    enumerations. Does not clean rows, fill gaps, sort timestamps, mutate
    values, access storage, generate orders, validate lifecycle transitions,
    or apply execution logic. Business-rule and EMS checks are intentionally
    out of scope. NULL values in ``limit_price``, ``stop_price``, and
    ``average_fill_price`` are accepted; NULL values in all other required
    columns remain hard failures.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical OMS order DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            OMSValidationError: If any required column is missing or column
                dtypes do not match the merged order schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        )
        empty_order_id_rows = self._count_empty_string_rows(frame, _COL_ORDER_ID)
        empty_parent_order_id_rows = self._count_empty_string_rows(
            frame,
            _COL_PARENT_ORDER_ID,
        )
        empty_side_rows = self._count_empty_string_rows(frame, _COL_SIDE)
        empty_order_type_rows = self._count_empty_string_rows(frame, _COL_ORDER_TYPE)
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_side_rows = self._count_invalid_enum_rows(
            frame,
            _COL_SIDE,
            _ALLOWED_SIDES,
        )
        invalid_order_type_rows = self._count_invalid_enum_rows(
            frame,
            _COL_ORDER_TYPE,
            _ALLOWED_ORDER_TYPES,
        )
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        invalid_quantity_rows = self._count_non_finite_rows(frame, _COL_QUANTITY)
        invalid_limit_price_rows = self._count_non_finite_rows(frame, _COL_LIMIT_PRICE)
        invalid_stop_price_rows = self._count_non_finite_rows(frame, _COL_STOP_PRICE)
        invalid_filled_quantity_rows = self._count_non_finite_rows(
            frame,
            _COL_FILLED_QUANTITY,
        )
        invalid_average_fill_price_rows = self._count_non_finite_rows(
            frame,
            _COL_AVERAGE_FILL_PRICE,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_order_id_rows=empty_order_id_rows,
            empty_parent_order_id_rows=empty_parent_order_id_rows,
            empty_side_rows=empty_side_rows,
            empty_order_type_rows=empty_order_type_rows,
            empty_status_rows=empty_status_rows,
            invalid_side_rows=invalid_side_rows,
            invalid_order_type_rows=invalid_order_type_rows,
            invalid_status_rows=invalid_status_rows,
            invalid_quantity_rows=invalid_quantity_rows,
            invalid_limit_price_rows=invalid_limit_price_rows,
            invalid_stop_price_rows=invalid_stop_price_rows,
            invalid_filled_quantity_rows=invalid_filled_quantity_rows,
            invalid_average_fill_price_rows=invalid_average_fill_price_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
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
        """Raise when any required column is absent from ``frame``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            required_columns: Column names that must be present.

        Raises:
            OMSValidationError: If one or more required columns are missing.
        """
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise OMSValidationError(
                f"missing required columns: {list(missing)}",
                error_code=ERROR_REQUIRED_COLUMNS,
                details={
                    "missing_columns": missing,
                    "required_columns": tuple(required_columns),
                    "available_columns": tuple(frame.columns),
                },
            )

    def _validate_column_dtypes(self, frame: pl.DataFrame) -> None:
        """Raise when any required column dtype differs from the schema.

        Args:
            frame: Input DataFrame. Must not be mutated.

        Raises:
            OMSValidationError: If one or more column dtypes do not match
                ``COLUMN_DTYPES`` / ``MERGED_ORDER_SCHEMA``.
        """
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
            raise OMSValidationError(
                "merged order schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first primary-key occurrence.

        Uses keep-first semantics over
        ``(symbol, timeframe, open_time, order_id)``.

        Args:
            frame: Input order DataFrame. Must not be mutated.

        Returns:
            Count of duplicate primary-key rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_invalid_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows with NULL ``open_time`` values.

        Order ``open_time`` is ``Datetime("us", "UTC")``. BaseVerifier's
        integer-only timestamp check is overridden so valid UTC datetimes are
        not treated as globally invalid.

        Args:
            frame: Input order DataFrame. Must not be mutated.
            timestamp_column: Timestamp column name.

        Returns:
            Count of invalid timestamp rows.
        """
        if frame.height == 0:
            return 0
        expected = COLUMN_DTYPES[timestamp_column]
        actual = frame.schema[timestamp_column]
        if actual != expected:
            return frame.height
        return int(frame.select(pl.col(timestamp_column).is_null().sum()).item())

    def _count_empty_string_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows containing an empty string in ``column``.

        Args:
            frame: Input order DataFrame. Must not be mutated.
            column: String column name to inspect.

        Returns:
            Count of rows with an empty string value in ``column``.
        """
        if frame.height == 0:
            return 0
        return int(frame.select((pl.col(column) == "").sum()).item())

    def _count_invalid_enum_rows(
        self,
        frame: pl.DataFrame,
        column: str,
        allowed: Sequence[str],
    ) -> int:
        """Return rows whose ``column`` value is outside ``allowed``.

        NULL values are counted by ``_count_null_rows`` and excluded here.

        Args:
            frame: Input order DataFrame. Must not be mutated.
            column: Categorical string column name.
            allowed: Canonical enumeration string values.

        Returns:
            Count of rows with unexpected non-null values.
        """
        if frame.height == 0:
            return 0
        allowed_list = list(allowed)
        invalid_mask = pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed_list)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_non_finite_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows with infinite values in a numeric column.

        NaN values are reported through ``_count_nan_rows``. NULL values in
        non-nullable columns are reported through ``_count_null_rows``. NULL
        values in nullable price columns are valid and excluded here.

        Args:
            frame: Input order DataFrame. Must not be mutated.
            column: Floating-point column name to inspect.

        Returns:
            Count of rows with infinite values in ``column``.
        """
        if frame.height == 0:
            return 0
        if column not in frame.schema or not frame.schema[column].is_float():
            return 0
        return int(frame.select(pl.col(column).is_infinite().sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with empty strings, invalid enums, or infinite numerics.

        Args:
            frame: Input order DataFrame. Must not be mutated.

        Returns:
            Count of rows with one or more invalid categorical/string/numeric
            values.
        """
        if frame.height == 0:
            return 0
        side_invalid = pl.col(_COL_SIDE).is_not_null() & ~pl.col(_COL_SIDE).is_in(
            list(_ALLOWED_SIDES)
        )
        order_type_invalid = pl.col(_COL_ORDER_TYPE).is_not_null() & ~pl.col(_COL_ORDER_TYPE).is_in(
            list(_ALLOWED_ORDER_TYPES)
        )
        status_invalid = pl.col(_COL_STATUS).is_not_null() & ~pl.col(_COL_STATUS).is_in(
            list(_ALLOWED_STATUSES)
        )
        empty_order_id = pl.col(_COL_ORDER_ID) == ""
        empty_parent = pl.col(_COL_PARENT_ORDER_ID) == ""
        empty_side = pl.col(_COL_SIDE) == ""
        empty_order_type = pl.col(_COL_ORDER_TYPE) == ""
        empty_status = pl.col(_COL_STATUS) == ""
        numeric_invalid = pl.any_horizontal(
            *(pl.col(name).is_infinite() for name in _NUMERIC_COLUMNS)
        )
        return int(
            frame.select(
                (
                    side_invalid
                    | order_type_invalid
                    | status_invalid
                    | empty_order_id
                    | empty_parent
                    | empty_side
                    | empty_order_type
                    | empty_status
                    | numeric_invalid
                ).sum()
            ).item()
        )


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_order_id_rows: int,
    empty_parent_order_id_rows: int,
    empty_side_rows: int,
    empty_order_type_rows: int,
    empty_status_rows: int,
    invalid_side_rows: int,
    invalid_order_type_rows: int,
    invalid_status_rows: int,
    invalid_quantity_rows: int,
    invalid_limit_price_rows: int,
    invalid_stop_price_rows: int,
    invalid_filled_quantity_rows: int,
    invalid_average_fill_price_rows: int,
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
    if empty_order_id_rows > 0:
        warnings.append(_WARN_EMPTY_ORDER_ID)
    if empty_parent_order_id_rows > 0:
        warnings.append(_WARN_EMPTY_PARENT_ORDER_ID)
    if empty_side_rows > 0:
        warnings.append(_WARN_EMPTY_SIDE)
    if empty_order_type_rows > 0:
        warnings.append(_WARN_EMPTY_ORDER_TYPE)
    if empty_status_rows > 0:
        warnings.append(_WARN_EMPTY_STATUS)
    if invalid_side_rows > 0:
        warnings.append(_WARN_INVALID_SIDE)
    if invalid_order_type_rows > 0:
        warnings.append(_WARN_INVALID_ORDER_TYPE)
    if invalid_status_rows > 0:
        warnings.append(_WARN_INVALID_STATUS)
    if invalid_quantity_rows > 0:
        warnings.append(_WARN_INVALID_QUANTITY)
    if invalid_limit_price_rows > 0:
        warnings.append(_WARN_INVALID_LIMIT_PRICE)
    if invalid_stop_price_rows > 0:
        warnings.append(_WARN_INVALID_STOP_PRICE)
    if invalid_filled_quantity_rows > 0:
        warnings.append(_WARN_INVALID_FILLED_QUANTITY)
    if invalid_average_fill_price_rows > 0:
        warnings.append(_WARN_INVALID_AVERAGE_FILL_PRICE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
