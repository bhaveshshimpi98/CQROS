"""CQROS factor dataset verification.

Purpose:
    Inspect canonical long-format factor frames and report structural
    findings without cleaning or mutating input data.

Responsibilities:
    - Validate required factor columns and expected dtypes against
      ``FACTOR_SCHEMA``
    - Validate canonical column order
    - Count duplicate ``(symbol, timeframe, open_time, factor_name)`` keys,
      nulls, NaNs, infinite values, and invalid ``open_time`` values
    - Classify ``factor_value`` NULLs as warmup, domain, or unexpected
    - Classify invalid numerics as ``+inf``, ``-inf``, overflow, underflow,
      and non-finite
    - Validate metadata completeness and
      ``factor_name`` / ``factor_version`` / ``factor_category`` consistency
    - Validate ``status`` against ``FactorStatus``
    - Emit deterministic structured diagnostics and a pass/fail outcome
    - Never sort, clean, repair, mutate, recompute, or persist frames

Dependencies:
    ``polars``, the Python standard library, ``cqros.factors.schema``,
    ``cqros.factors.verification.diagnostics``,
    ``cqros.factors.verification.domain_null_metadata``,
    ``cqros.factors.verification.exceptions``, and
    ``cqros.processing.verification.base``.

Public API:
    ``FactorVerifier``
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    factor_status_values,
)
from cqros.factors.verification.diagnostics import (
    FactorInvalidNumericDiagnostic,
    FactorNullDiagnostic,
    FactorVerificationDiagnostics,
    FactorVerificationReport,
    FactorWarningDiagnostic,
    InvalidNumericKind,
    NullClassification,
)
from cqros.factors.verification.domain_null_metadata import factor_allows_domain_nulls
from cqros.factors.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorValidationError,
)
from cqros.processing.verification.base import BaseVerifier

__all__ = ["FactorVerifier"]

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_FACTOR_NAME: Final[str] = "factor_name"
_COL_FACTOR_VERSION: Final[str] = "factor_version"
_COL_FACTOR_CATEGORY: Final[str] = "factor_category"
_COL_FACTOR_VALUE: Final[str] = "factor_value"
_COL_STATUS: Final[str] = "status"

# Long-format uniqueness: bar identity plus factor identity.
_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    *PRIMARY_KEY_COLUMNS,
    _COL_FACTOR_NAME,
)

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_OTHER_NULL_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column != _COL_FACTOR_VALUE
)

_NAN_CHECK_COLUMNS: Final[tuple[str, ...]] = (_COL_FACTOR_VALUE,)

_NON_EMPTY_METADATA_COLUMNS: Final[tuple[str, ...]] = METADATA_COLUMNS

_ALLOWED_STATUSES: Final[tuple[str, ...]] = factor_status_values()

_FLOAT_MAX: Final[float] = sys.float_info.max
_FLOAT_MIN_NORMAL: Final[float] = sys.float_info.min

_WARN_DUPLICATES: Final[str] = "DUPLICATE_KEYS"
_WARN_TIMESTAMPS: Final[str] = "INVALID_TIMESTAMPS"
_WARN_UNSORTED: Final[str] = "UNSORTED_TIMESTAMPS"
_WARN_COLUMN_ORDER: Final[str] = "COLUMN_ORDER"
_WARN_EMPTY_METADATA: Final[str] = "EMPTY_METADATA"
_WARN_FACTOR_NAME: Final[str] = "EMPTY_FACTOR_NAME"
_WARN_FACTOR_VERSION: Final[str] = "INCONSISTENT_FACTOR_VERSION"
_WARN_FACTOR_CATEGORY: Final[str] = "INCONSISTENT_FACTOR_CATEGORY"
_WARN_EMPTY_STATUS: Final[str] = "EMPTY_STATUS"
_WARN_INVALID_STATUS: Final[str] = "INVALID_STATUS"


class FactorVerifier(BaseVerifier):
    """Deterministic factor-dataset verifier that reports findings only.

    Inspects structural quality of a long-format factor frame against
    ``cqros.factors.schema`` / ``FACTOR_SCHEMA``. Does not clean rows, fill
    gaps, sort timestamps, mutate values, access storage, or recompute
    factors.

    Warmup-only leading ``factor_value`` NULLs and allowlisted domain NULLs
    are informational and do not fail verification. Unexpected NULLs and
    infinite values fail.
    """

    def verify(self, frame: pl.DataFrame) -> FactorVerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input long-format factor DataFrame. Must not be mutated.

        Returns:
            A ``FactorVerificationReport`` describing counters, structured
            diagnostics, warnings, and overall pass/fail status.

        Raises:
            FactorValidationError: If any required column is missing or
                column dtypes do not match ``FACTOR_SCHEMA``.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _NAN_CHECK_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        )
        (
            null_diagnostics,
            warmup_null_rows,
            domain_null_rows,
            unexpected_factor_null_rows,
        ) = self._classify_factor_value_nulls(frame)
        other_null_rows = self._count_null_rows(frame, _OTHER_NULL_COLUMNS)
        unexpected_null_rows = unexpected_factor_null_rows + other_null_rows

        invalid_numeric_diagnostics, inf_counts = self._classify_invalid_numerics(frame)
        positive_inf_rows = inf_counts[InvalidNumericKind.POSITIVE_INFINITY]
        negative_inf_rows = inf_counts[InvalidNumericKind.NEGATIVE_INFINITY]
        overflow_rows = inf_counts[InvalidNumericKind.OVERFLOW]
        underflow_rows = inf_counts[InvalidNumericKind.UNDERFLOW]
        non_finite_rows = inf_counts[InvalidNumericKind.NON_FINITE]
        invalid_numeric_rows = (
            positive_inf_rows + negative_inf_rows + overflow_rows + underflow_rows
        )

        empty_metadata_rows = self._count_empty_metadata_rows(frame)
        inconsistent_name_rows = self._count_empty_string_rows(frame, _COL_FACTOR_NAME)
        inconsistent_version_rows = self._count_inconsistent_metadata_rows(
            frame,
            identity_column=_COL_FACTOR_NAME,
            attribute_column=_COL_FACTOR_VERSION,
        )
        inconsistent_category_rows = self._count_inconsistent_metadata_rows(
            frame,
            identity_column=_COL_FACTOR_NAME,
            attribute_column=_COL_FACTOR_CATEGORY,
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warning_diagnostics = _build_warning_diagnostics(
            null_diagnostics=null_diagnostics,
            invalid_numeric_diagnostics=invalid_numeric_diagnostics,
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_metadata_rows=empty_metadata_rows,
            inconsistent_name_rows=inconsistent_name_rows,
            inconsistent_version_rows=inconsistent_version_rows,
            inconsistent_category_rows=inconsistent_category_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
            other_null_rows=other_null_rows,
        )
        warnings = tuple(_format_warning(item) for item in warning_diagnostics)
        diagnostics = FactorVerificationDiagnostics(
            null_diagnostics=null_diagnostics,
            invalid_numeric_diagnostics=invalid_numeric_diagnostics,
            warning_diagnostics=warning_diagnostics,
        )

        # Warmup and domain factor_value NULLs are informational and do not fail.
        passed = (
            duplicate_timestamp_rows == 0
            and unexpected_null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_metadata_rows == 0
            and inconsistent_name_rows == 0
            and inconsistent_version_rows == 0
            and inconsistent_category_rows == 0
            and empty_status_rows == 0
            and invalid_status_rows == 0
            and is_sorted
            and is_canonical_order
        )
        return FactorVerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            warmup_null_rows=warmup_null_rows,
            domain_null_rows=domain_null_rows,
            unexpected_null_rows=unexpected_null_rows,
            positive_inf_rows=positive_inf_rows,
            negative_inf_rows=negative_inf_rows,
            non_finite_rows=non_finite_rows,
            warnings=warnings,
            diagnostics=diagnostics,
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
            FactorValidationError: If one or more required columns are missing.
        """
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
        """Raise when any required column dtype differs from the schema.

        Args:
            frame: Input DataFrame. Must not be mutated.

        Raises:
            FactorValidationError: If one or more column dtypes do not match
                ``COLUMN_DTYPES``.
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
            raise FactorValidationError(
                "factor schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first long-format primary-key occurrence.

        Uses keep-first semantics over
        ``(symbol, timeframe, open_time, factor_name)``.

        Args:
            frame: Input factor DataFrame. Must not be mutated.

        Returns:
            Count of duplicate primary-key rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _classify_factor_value_nulls(
        self,
        frame: pl.DataFrame,
    ) -> tuple[tuple[FactorNullDiagnostic, ...], int, int, int]:
        """Classify per-factor ``factor_value`` NULLs as warmup, domain, or unexpected.

        Warmup NULLs occupy only the initial consecutive rows for a factor.
        Post-warmup NULLs on allowlisted factors are domain NULLs. Any other
        NULL after a valid observation is unexpected.

        Args:
            frame: Input factor DataFrame. Must not be mutated.

        Returns:
            Tuple of
            ``(diagnostics, warmup_null_rows, domain_null_rows,
            unexpected_null_rows)``.
        """
        if frame.height == 0:
            return (), 0, 0, 0

        diagnostics: list[FactorNullDiagnostic] = []
        warmup_total = 0
        domain_total = 0
        unexpected_total = 0

        factor_names = frame.get_column(_COL_FACTOR_NAME).unique().sort().to_list()
        for factor_name in factor_names:
            factor_frame = (
                frame.filter(pl.col(_COL_FACTOR_NAME) == factor_name)
                .select(_COL_OPEN_TIME, _COL_FACTOR_VALUE)
                .sort(_COL_OPEN_TIME)
            )
            values = factor_frame.get_column(_COL_FACTOR_VALUE).to_list()
            times = factor_frame.get_column(_COL_OPEN_TIME).to_list()
            null_indices = [index for index, value in enumerate(values) if value is None]
            if not null_indices:
                continue

            first_valid_index: int | None = None
            for index, value in enumerate(values):
                if value is not None:
                    first_valid_index = index
                    break

            appears_after_valid = False
            if first_valid_index is not None:
                appears_after_valid = any(index > first_valid_index for index in null_indices)

            only_at_beginning = not appears_after_valid and all(
                index < (first_valid_index if first_valid_index is not None else len(values))
                for index in null_indices
            )
            # Leading consecutive run: every NULL index must be before first
            # valid and form a prefix of the series.
            if only_at_beginning and first_valid_index is not None:
                only_at_beginning = null_indices == list(range(first_valid_index))
            elif only_at_beginning and first_valid_index is None:
                only_at_beginning = null_indices == list(range(len(values)))

            factor_name_str = str(factor_name)
            if only_at_beginning and not appears_after_valid:
                classification = NullClassification.WARMUP_NULLS
            elif factor_allows_domain_nulls(factor_name_str):
                classification = NullClassification.DOMAIN_NULLS
            else:
                classification = NullClassification.UNEXPECTED_NULLS
            count = len(null_indices)
            diagnostic = FactorNullDiagnostic(
                factor_name=factor_name_str,
                count=count,
                first_open_time=int(times[null_indices[0]]),
                last_open_time=int(times[null_indices[-1]]),
                only_at_beginning=only_at_beginning,
                appears_after_valid=appears_after_valid,
                classification=classification,
            )
            diagnostics.append(diagnostic)
            if classification == NullClassification.WARMUP_NULLS:
                warmup_total += count
            elif classification == NullClassification.DOMAIN_NULLS:
                domain_total += count
            else:
                unexpected_total += count

        diagnostics.sort(key=lambda item: (item.factor_name, item.first_open_time))
        return tuple(diagnostics), warmup_total, domain_total, unexpected_total

    def _classify_invalid_numerics(
        self,
        frame: pl.DataFrame,
    ) -> tuple[
        tuple[FactorInvalidNumericDiagnostic, ...],
        dict[InvalidNumericKind, int],
    ]:
        """Classify invalid ``factor_value`` numerics by kind.

        Args:
            frame: Input factor DataFrame. Must not be mutated.

        Returns:
            Tuple of diagnostics and per-kind row counts.
        """
        counts: dict[InvalidNumericKind, int] = {kind: 0 for kind in InvalidNumericKind}
        if frame.height == 0:
            return (), counts
        if _COL_FACTOR_VALUE not in frame.schema or not frame.schema[_COL_FACTOR_VALUE].is_float():
            return (), counts

        diagnostics: list[FactorInvalidNumericDiagnostic] = []
        rows = frame.select(
            _COL_FACTOR_NAME,
            _COL_OPEN_TIME,
            _COL_FACTOR_VALUE,
        ).iter_rows(named=True)
        for row in rows:
            value = row[_COL_FACTOR_VALUE]
            kind = _classify_numeric_value(value)
            if kind is None:
                continue
            counts[kind] += 1
            diagnostics.append(
                FactorInvalidNumericDiagnostic(
                    factor_name=str(row[_COL_FACTOR_NAME]),
                    open_time=int(row[_COL_OPEN_TIME]),
                    kind=kind,
                    count=1,
                )
            )

        diagnostics.sort(key=lambda item: (item.factor_name, item.open_time, item.kind.value))
        return tuple(diagnostics), counts

    def _count_empty_string_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows containing an empty string in ``column``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            column: String column name.

        Returns:
            Count of rows with an empty string in ``column``.
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

        Args:
            frame: Input DataFrame. Must not be mutated.
            column: Enumeration column name.
            allowed: Allowed string values.

        Returns:
            Count of rows with disallowed enum values.
        """
        if frame.height == 0:
            return 0
        allowed_list = list(allowed)
        invalid_mask = pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed_list)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_empty_metadata_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank required factor metadata fields.

        Args:
            frame: Input factor DataFrame. Must not be mutated.

        Returns:
            Count of rows with null or empty metadata fields.
        """
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *(
                (pl.col(column).is_null()) | (pl.col(column) == "")
                for column in _NON_EMPTY_METADATA_COLUMNS
            )
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_inconsistent_metadata_rows(
        self,
        frame: pl.DataFrame,
        *,
        identity_column: str,
        attribute_column: str,
    ) -> int:
        """Return rows belonging to identities with multiple attribute values.

        Args:
            frame: Input factor DataFrame. Must not be mutated.
            identity_column: Grouping identity column (``factor_name``).
            attribute_column: Attribute that must be unique per identity.

        Returns:
            Count of rows whose identity maps to more than one distinct
            attribute value.
        """
        if frame.height == 0:
            return 0
        counts = (
            frame.group_by(identity_column)
            .agg(pl.col(attribute_column).n_unique().alias("_n_unique"))
            .filter(pl.col("_n_unique") > 1)
            .select(identity_column)
        )
        if counts.height == 0:
            return 0
        inconsistent = frame.join(counts, on=identity_column, how="inner")
        return inconsistent.height


def _classify_numeric_value(value: object) -> InvalidNumericKind | None:
    """Return the invalid-numeric kind for ``value``, or ``None`` if valid."""
    if value is None:
        return None
    if not isinstance(value, float):
        return None
    if math.isnan(value):
        return InvalidNumericKind.NON_FINITE
    if math.isinf(value):
        if value > 0:
            return InvalidNumericKind.POSITIVE_INFINITY
        return InvalidNumericKind.NEGATIVE_INFINITY
    absolute = abs(value)
    if absolute >= _FLOAT_MAX:
        return InvalidNumericKind.OVERFLOW
    if value != 0.0 and absolute < _FLOAT_MIN_NORMAL:
        return InvalidNumericKind.UNDERFLOW
    return None


def _build_warning_diagnostics(
    *,
    null_diagnostics: tuple[FactorNullDiagnostic, ...],
    invalid_numeric_diagnostics: tuple[FactorInvalidNumericDiagnostic, ...],
    duplicate_timestamp_rows: int,
    invalid_timestamp_rows: int,
    empty_metadata_rows: int,
    inconsistent_name_rows: int,
    inconsistent_version_rows: int,
    inconsistent_category_rows: int,
    empty_status_rows: int,
    invalid_status_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
    other_null_rows: int,
) -> tuple[FactorWarningDiagnostic, ...]:
    """Return deterministic structured warnings for verification findings."""
    warnings: list[FactorWarningDiagnostic] = []

    if not is_canonical_order:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_COLUMN_ORDER,
                factor_name=None,
                count=1,
            )
        )
    if duplicate_timestamp_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_DUPLICATES,
                factor_name=None,
                count=duplicate_timestamp_rows,
            )
        )

    for item in null_diagnostics:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=item.classification.value,
                factor_name=item.factor_name,
                count=item.count,
            )
        )
    if other_null_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=NullClassification.UNEXPECTED_NULLS.value,
                factor_name=None,
                count=other_null_rows,
            )
        )

    if invalid_timestamp_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_TIMESTAMPS,
                factor_name=None,
                count=invalid_timestamp_rows,
            )
        )

    numeric_totals: dict[tuple[str, InvalidNumericKind], int] = {}
    for item in invalid_numeric_diagnostics:
        key = (item.factor_name, item.kind)
        numeric_totals[key] = numeric_totals.get(key, 0) + item.count
    for (factor_name, kind), count in sorted(
        numeric_totals.items(),
        key=lambda pair: (pair[0][0], pair[0][1].value),
    ):
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_numeric_warning_type(kind),
                factor_name=factor_name,
                count=count,
            )
        )

    if empty_metadata_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_EMPTY_METADATA,
                factor_name=None,
                count=empty_metadata_rows,
            )
        )
    if inconsistent_name_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_FACTOR_NAME,
                factor_name=None,
                count=inconsistent_name_rows,
            )
        )
    if inconsistent_version_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_FACTOR_VERSION,
                factor_name=None,
                count=inconsistent_version_rows,
            )
        )
    if inconsistent_category_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_FACTOR_CATEGORY,
                factor_name=None,
                count=inconsistent_category_rows,
            )
        )
    if empty_status_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_EMPTY_STATUS,
                factor_name=None,
                count=empty_status_rows,
            )
        )
    if invalid_status_rows > 0:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_INVALID_STATUS,
                factor_name=None,
                count=invalid_status_rows,
            )
        )
    if not is_sorted:
        warnings.append(
            FactorWarningDiagnostic(
                warning_type=_WARN_UNSORTED,
                factor_name=None,
                count=1,
            )
        )
    return tuple(warnings)


def _numeric_warning_type(kind: InvalidNumericKind) -> str:
    """Map an invalid numeric kind to a warning type string."""
    mapping: dict[InvalidNumericKind, str] = {
        InvalidNumericKind.POSITIVE_INFINITY: "POSITIVE_INFINITY",
        InvalidNumericKind.NEGATIVE_INFINITY: "NEGATIVE_INFINITY",
        InvalidNumericKind.OVERFLOW: "OVERFLOW",
        InvalidNumericKind.UNDERFLOW: "UNDERFLOW",
        InvalidNumericKind.NON_FINITE: "NON_FINITE",
    }
    return mapping[kind]


def _format_warning(item: FactorWarningDiagnostic) -> str:
    """Format a structured warning as ``TYPE factor=... count=...``."""
    factor_label = item.factor_name if item.factor_name is not None else "*"
    return f"{item.warning_type} factor={factor_label} count={item.count}"
