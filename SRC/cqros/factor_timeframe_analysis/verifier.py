"""CQROS factor timeframe analysis metrics dataset verification.

Purpose:
    Inspect canonical factor timeframe analysis frames and report
    structural findings without cleaning or mutating input data.

Responsibilities:
    - Validate required timeframe analysis columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, empty factor identity fields,
      non-finite scores, non-positive timeframe ranks, out-of-range
      confidence and stability values, selected/status inconsistencies,
      and empty source_selection_version values
    - Optionally cross-validate a FTA frame against its source Factor
      Selection frame via ``verify_against_selection``
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.factor_timeframe_analysis.exceptions``,
    ``cqros.factor_timeframe_analysis.schema``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``FactorTimeframeAnalysisVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.factor_timeframe_analysis.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TimeframeAnalysisStatus,
    timeframe_analysis_status_values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorTimeframeAnalysisVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "FTA-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "FTA-VERIFICATION-002"

_COL_ANALYSIS_TIME: Final[str] = "analysis_time"
_COL_STATUS: Final[str] = "status"
_COL_FACTOR_NAME: Final[str] = "factor_name"
_COL_FACTOR_VERSION: Final[str] = "factor_version"
_COL_FACTOR_CATEGORY: Final[str] = "factor_category"
_COL_BEST_TIMEFRAME: Final[str] = "best_timeframe"
_COL_BEST_SELECTION_SCORE: Final[str] = "best_selection_score"
_COL_TIMEFRAME_RANK: Final[str] = "timeframe_rank"
_COL_TIMEFRAME_STABILITY: Final[str] = "timeframe_stability"
_COL_TIMEFRAME_CONFIDENCE: Final[str] = "timeframe_confidence"
_COL_SELECTED: Final[str] = "selected"
_COL_SOURCE_SELECTION_VERSION: Final[str] = "source_selection_version"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Nullable by design when only one timeframe is available.
_NULLABLE_METRIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "winner_margin",
        "score_gap",
    }
)

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in _NULLABLE_METRIC_COLUMNS
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_BEST_SELECTION_SCORE,
    _COL_TIMEFRAME_STABILITY,
    _COL_TIMEFRAME_CONFIDENCE,
)

_NON_EMPTY_FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FACTOR_NAME,
    _COL_FACTOR_VERSION,
    _COL_FACTOR_CATEGORY,
    _COL_BEST_TIMEFRAME,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = timeframe_analysis_status_values()

_PASS_STATUS: Final[str] = TimeframeAnalysisStatus.PASS.value

_WARN_DUPLICATES: Final[str] = "Duplicate factor timeframe analysis primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid TimeframeAnalysisStatus values detected."
_WARN_EMPTY_FACTOR_FIELDS: Final[str] = "Empty required factor identity fields detected."
_WARN_TIMEFRAME_RANK: Final[str] = "timeframe_rank values less than or equal to 0 detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_CONFIDENCE_RANGE: Final[str] = "timeframe_confidence values outside [0, 1] detected."
_WARN_STABILITY_RANGE: Final[str] = "timeframe_stability values outside [0, 1] detected."
_WARN_SELECTED_INCONSISTENCY: Final[str] = (
    "selected/status inconsistency: selected must be True iff status is PASS."
)
_WARN_EMPTY_SOURCE_SELECTION_VERSION: Final[str] = "Empty source_selection_version values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = (
    "Frame is not sorted by factor_name, factor_version (engine canonical order)."
)

# Cross-frame warning constants for verify_against_selection.
_WARN_SELECTION_ORIGIN: Final[str] = (
    "FTA selected factors not found in Factor Selection selected rows."
)
_WARN_SELECTION_SCORE_MISMATCH: Final[str] = (
    "best_selection_score does not match max selection_score among selection selected rows."
)
_WARN_BEST_TIMEFRAME_NOT_IN_SELECTION: Final[str] = (
    "best_timeframe not among source Factor Selection timeframes for the factor."
)


class FactorTimeframeAnalysisVerifier(BaseVerifier):
    """Deterministic canonical timeframe analysis verifier that reports findings only.

    Inspects structural quality of a canonical timeframe analysis frame
    against ``cqros.factor_timeframe_analysis.schema`` /
    ``TIMEFRAME_ANALYSIS_SCHEMA`` and the canonical status enumeration.
    Does not clean rows, fill gaps, sort timestamps, mutate values, access
    storage, or apply analysis logic.

    Use ``verify`` for structural-only inspection. Use
    ``verify_against_selection`` to additionally cross-validate a FTA frame
    against the Factor Selection frame it was derived from.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Checks include: required columns and dtypes, duplicate primary keys,
        nulls, NaNs, invalid timestamps, invalid status values, empty factor
        identity fields, non-finite numeric values, non-positive timeframe
        ranks, out-of-range confidence/stability values (outside [0, 1]),
        selected/status inconsistency, empty source_selection_version values,
        canonical column order, and sort order.

        Args:
            frame: Input canonical timeframe analysis DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            FactorTimeframeAnalysisError: If any required column is missing
                or column dtypes do not match the timeframe analysis schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_ANALYSIS_TIME,
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_factor_field_rows = self._count_empty_factor_field_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        selected_inconsistency_rows = self._count_selected_inconsistency_rows(frame)
        empty_source_version_rows = self._count_empty_string_rows(
            frame, _COL_SOURCE_SELECTION_VERSION
        )
        is_sorted = _is_sorted_by_factor_identity(frame)
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
            selected_inconsistency_rows=selected_inconsistency_rows,
            empty_source_version_rows=empty_source_version_rows,
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
            and selected_inconsistency_rows == 0
            and empty_source_version_rows == 0
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

    def verify_against_selection(
        self,
        fta_frame: pl.DataFrame,
        selection_frame: pl.DataFrame,
    ) -> VerificationReport:
        """Verify ``fta_frame`` structurally and cross-check against ``selection_frame``.

        First performs a full structural ``verify(fta_frame)`` pass. Then runs
        cross-frame consistency checks against ``selection_frame``:

        - Every FTA ``selected==True`` factor exists in selection
          ``selected==True`` rows.
        - ``best_selection_score`` matches the maximum ``selection_score``
          among the selection ``selected==True`` rows for that factor/version
          across all timeframes (within floating-point tolerance 1e-9).
        - ``best_timeframe`` is among the source selection timeframes for that
          factor/version.
        - No FTA ``selected==False`` factors appear as cross-frame selected.

        Args:
            fta_frame: Canonical FTA DataFrame. Structurally verified first.
            selection_frame: Factor Selection DataFrame whose ``selected``,
                ``selection_score``, and ``timeframe`` columns are consulted.
                Must contain ``factor_name``, ``factor_version``, ``timeframe``,
                ``selection_score``, and ``selected`` columns.

        Returns:
            A ``VerificationReport`` where ``warnings`` accumulates both
            structural and cross-frame findings, and ``passed`` is ``True``
            only when both structural and cross-frame checks pass.

        Raises:
            FactorTimeframeAnalysisError: If structural column or dtype checks
                fail (propagated from ``verify``).
        """
        structural = self.verify(fta_frame)
        cross_warnings = _cross_validate_against_selection(fta_frame, selection_frame)
        combined_warnings = structural.warnings + cross_warnings
        cross_passed = len(cross_warnings) == 0
        return VerificationReport(
            rows_checked=structural.rows_checked,
            duplicate_timestamp_rows=structural.duplicate_timestamp_rows,
            null_rows=structural.null_rows,
            nan_rows=structural.nan_rows,
            invalid_timestamp_rows=structural.invalid_timestamp_rows,
            invalid_numeric_rows=structural.invalid_numeric_rows,
            warnings=combined_warnings,
            passed=structural.passed and cross_passed,
        )

    def _validate_required_columns(
        self,
        frame: pl.DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Raise when any required column is absent from ``frame``."""
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise FactorTimeframeAnalysisError(
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
            raise FactorTimeframeAnalysisError(
                "factor timeframe analysis schema dtype mismatch",
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

        Checks:
        - Non-finite values in ``best_selection_score``, ``timeframe_stability``,
          ``timeframe_confidence``
        - ``timeframe_rank`` <= 0
        - ``timeframe_confidence`` outside [0, 1] (non-null, finite only)
        - ``timeframe_stability`` outside [0, 1] (non-null, finite only)
        """
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(
            *(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in _VALUE_COLUMNS
            )
        )
        timeframe_rank_invalid = pl.col(_COL_TIMEFRAME_RANK) <= 0
        # Range checks apply only to finite, non-null values; non-finite
        # violations are already captured above.
        confidence_out_of_range = (
            pl.col(_COL_TIMEFRAME_CONFIDENCE).is_not_null()
            & pl.col(_COL_TIMEFRAME_CONFIDENCE).is_finite()
            & (
                (pl.col(_COL_TIMEFRAME_CONFIDENCE) < 0.0)
                | (pl.col(_COL_TIMEFRAME_CONFIDENCE) > 1.0)
            )
        )
        stability_out_of_range = (
            pl.col(_COL_TIMEFRAME_STABILITY).is_not_null()
            & pl.col(_COL_TIMEFRAME_STABILITY).is_finite()
            & ((pl.col(_COL_TIMEFRAME_STABILITY) < 0.0) | (pl.col(_COL_TIMEFRAME_STABILITY) > 1.0))
        )
        invalid_mask = (
            non_finite | timeframe_rank_invalid | confidence_out_of_range | stability_out_of_range
        )
        return int(frame.select(invalid_mask.sum()).item())

    def _count_selected_inconsistency_rows(self, frame: pl.DataFrame) -> int:
        """Return rows where ``selected`` and ``status`` are inconsistent.

        ``selected`` must be ``True`` when and only when ``status`` is ``PASS``.
        Null ``selected`` values are treated as ``False`` for this check.
        """
        if frame.height == 0:
            return 0
        is_pass = pl.col(_COL_STATUS) == _PASS_STATUS
        is_selected = pl.col(_COL_SELECTED).fill_null(value=False)
        inconsistent = is_pass != is_selected
        return int(frame.select(inconsistent.sum()).item())


def _cross_validate_against_selection(
    fta_frame: pl.DataFrame,
    selection_frame: pl.DataFrame,
) -> tuple[str, ...]:
    """Run cross-frame consistency checks and return warning strings.

    Args:
        fta_frame: Canonical FTA DataFrame (already structurally verified).
        selection_frame: Source Factor Selection DataFrame.

    Returns:
        Tuple of warning strings for each cross-frame violation found.
        Empty tuple when all checks pass.
    """
    warnings: list[str] = []
    fta_selected = fta_frame.filter(pl.col(_COL_SELECTED).fill_null(value=False))
    if fta_selected.height == 0:
        return ()

    sel_selected = selection_frame.filter(
        pl.col("selected").fill_null(value=False)  # type: ignore[arg-type]
    )

    # Check: every FTA selected factor exists in selection selected rows.
    fta_factor_keys = (
        fta_selected.select(["factor_name", "factor_version"]).unique().sort(["factor_name"])
    )
    sel_factor_keys = (
        sel_selected.select(["factor_name", "factor_version"]).unique()
        if sel_selected.height > 0
        else selection_frame.select(["factor_name", "factor_version"]).clear()
    )
    unmatched_origin = fta_factor_keys.join(
        sel_factor_keys,
        on=["factor_name", "factor_version"],
        how="anti",
    )
    if unmatched_origin.height > 0:
        warnings.append(_WARN_SELECTION_ORIGIN)

    if sel_selected.height > 0:
        # Check: best_selection_score matches max selection_score per factor.
        max_scores = sel_selected.group_by(["factor_name", "factor_version"]).agg(
            pl.col("selection_score").max().alias("_max_sel_score")
        )
        score_joined = fta_selected.join(
            max_scores,
            on=["factor_name", "factor_version"],
            how="left",
        )
        score_mismatch = score_joined.filter(
            pl.col(_COL_BEST_SELECTION_SCORE).is_not_null()
            & pl.col("_max_sel_score").is_not_null()
            & ((pl.col(_COL_BEST_SELECTION_SCORE) - pl.col("_max_sel_score")).abs() > 1e-9)
        )
        if score_mismatch.height > 0:
            warnings.append(_WARN_SELECTION_SCORE_MISMATCH)

        # Check: best_timeframe is among source selection timeframes per factor.
        # Build (factor_name, factor_version, timeframe) keys from sel_selected.
        # Then for each FTA selected row check (factor_name, factor_version,
        # best_timeframe) exists in that set.
        sel_tf_keys = sel_selected.select(["factor_name", "factor_version", "timeframe"]).unique()
        fta_tf_lookup = fta_selected.select(
            pl.col("factor_name"),
            pl.col("factor_version"),
            pl.col(_COL_BEST_TIMEFRAME).alias("timeframe"),
        ).unique()
        unmatched_tf = fta_tf_lookup.join(
            sel_tf_keys,
            on=["factor_name", "factor_version", "timeframe"],
            how="anti",
        )
        if unmatched_tf.height > 0:
            warnings.append(_WARN_BEST_TIMEFRAME_NOT_IN_SELECTION)

    return tuple(warnings)


def _is_sorted_by_factor_identity(frame: pl.DataFrame) -> bool:
    """Return whether ``frame`` is sorted by factor_name then factor_version."""
    if frame.height <= 1:
        return True
    ordered = frame.sort(_COL_FACTOR_NAME, _COL_FACTOR_VERSION, maintain_order=True)
    return ordered.select(_COL_FACTOR_NAME, _COL_FACTOR_VERSION).equals(
        frame.select(_COL_FACTOR_NAME, _COL_FACTOR_VERSION)
    )


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
    selected_inconsistency_rows: int,
    empty_source_version_rows: int,
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
    if selected_inconsistency_rows > 0:
        warnings.append(_WARN_SELECTED_INCONSISTENCY)
    if empty_source_version_rows > 0:
        warnings.append(_WARN_EMPTY_SOURCE_SELECTION_VERSION)
    if frame.height > 0:
        if int(frame.select((pl.col(_COL_TIMEFRAME_RANK) <= 0).sum()).item()) > 0:
            warnings.append(_WARN_TIMEFRAME_RANK)
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
        confidence_violations = int(
            frame.select(
                (
                    pl.col(_COL_TIMEFRAME_CONFIDENCE).is_not_null()
                    & pl.col(_COL_TIMEFRAME_CONFIDENCE).is_finite()
                    & (
                        (pl.col(_COL_TIMEFRAME_CONFIDENCE) < 0.0)
                        | (pl.col(_COL_TIMEFRAME_CONFIDENCE) > 1.0)
                    )
                ).sum()
            ).item()
        )
        if confidence_violations > 0:
            warnings.append(_WARN_CONFIDENCE_RANGE)
        stability_violations = int(
            frame.select(
                (
                    pl.col(_COL_TIMEFRAME_STABILITY).is_not_null()
                    & pl.col(_COL_TIMEFRAME_STABILITY).is_finite()
                    & (
                        (pl.col(_COL_TIMEFRAME_STABILITY) < 0.0)
                        | (pl.col(_COL_TIMEFRAME_STABILITY) > 1.0)
                    )
                ).sum()
            ).item()
        )
        if stability_violations > 0:
            warnings.append(_WARN_STABILITY_RANGE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
