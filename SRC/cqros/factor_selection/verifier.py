"""CQROS factor selection metrics dataset verification.

Purpose:
    Inspect canonical factor selection frames and report structural and
    ranking findings without cleaning or mutating input data.

Responsibilities:
    - Validate required factor selection columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, empty factor identity fields, empty
      selection reason, non-finite selection scores, and non-positive
      selection ranks
    - Validate ranking uniqueness, rank origin, Top-N selection counts,
      rank ordering, selection reasons, and selected/status consistency
      within each timeframe using the configured ``top_n``
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.factor_selection.engine``,
    ``cqros.factor_selection.exceptions``,
    ``cqros.factor_selection.schema``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``FactorSelectionVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.factor_selection.engine import DEFAULT_TOP_N, require_top_n
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.redundancy import (
    DEFAULT_CANDIDATE_N,
    REASON_OUTSIDE_CANDIDATE_N,
    REASON_OUTSIDE_TOP_N,
    REASON_REDUNDANT,
    REASON_TOP_N,
    require_candidate_n,
)
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    FactorSelectionStatus,
    factor_selection_status_values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorSelectionVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "FSEL-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "FSEL-VERIFICATION-002"

_COL_SELECTION_TIME: Final[str] = "selection_time"
_COL_STATUS: Final[str] = "status"
_COL_SELECTED: Final[str] = "selected"
_COL_FACTOR_NAME: Final[str] = "factor_name"
_COL_FACTOR_VERSION: Final[str] = "factor_version"
_COL_FACTOR_CATEGORY: Final[str] = "factor_category"
_COL_SELECTION_SCORE: Final[str] = "selection_score"
_COL_SELECTION_RANK: Final[str] = "selection_rank"
_COL_SELECTION_REASON: Final[str] = "selection_reason"
_COL_SELECTION_IC: Final[str] = "selection_ic"
_COL_SELECTED_DIRECTION: Final[str] = "selected_direction"
_COL_ORIENTATION_POLICY: Final[str] = "orientation_policy"
_COL_TIMEFRAME: Final[str] = "timeframe"

_REASON_TOP_N: Final[str] = REASON_TOP_N
_REASON_OUTSIDE_TOP_N: Final[str] = REASON_OUTSIDE_TOP_N
_REASON_REDUNDANT: Final[str] = REASON_REDUNDANT
_REASON_OUTSIDE_CANDIDATE_N: Final[str] = REASON_OUTSIDE_CANDIDATE_N
_ALLOWED_REASONS: Final[tuple[str, ...]] = (
    REASON_TOP_N,
    REASON_OUTSIDE_TOP_N,
    REASON_REDUNDANT,
    REASON_OUTSIDE_CANDIDATE_N,
)

_WARN_TOP_N_SELECTION: Final[str] = (
    "Top-N selected flags do not match redundancy-aware selection rules."
)
_WARN_TOP_N_COUNT: Final[str] = "Selected row count exceeds configured Top-N per timeframe."
_WARN_CANDIDATE_REASON: Final[str] = (
    "Ranks outside candidate_n do not use selection_reason=outside_candidate_n."
)
_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_VALUE_COLUMNS: Final[tuple[str, ...]] = (_COL_SELECTION_SCORE, _COL_SELECTION_IC)

_NON_EMPTY_FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FACTOR_NAME,
    _COL_FACTOR_VERSION,
    _COL_FACTOR_CATEGORY,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = factor_selection_status_values()

_WARN_DUPLICATES: Final[str] = "Duplicate factor selection primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid FactorSelectionStatus values detected."
_WARN_EMPTY_FACTOR_FIELDS: Final[str] = "Empty required factor identity fields detected."
_WARN_EMPTY_SELECTION_REASON: Final[str] = "Empty selection_reason values detected."
_WARN_SELECTION_RANK: Final[str] = "selection_rank values less than or equal to 0 detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by selection_time."
_WARN_RANK_UNIQUENESS: Final[str] = "Duplicate selection_rank values detected within a timeframe."
_WARN_RANK_ORIGIN: Final[str] = "selection_rank sequences do not begin at 1 within a timeframe."
_WARN_RANK_ORDER: Final[str] = (
    "selection_rank ordering does not follow selection_score descending with "
    "deterministic factor_name/factor_version tie-breaks."
)
_WARN_SELECTED_STATUS: Final[str] = "selected and status values are inconsistent."
_WARN_SELECTION_REASON: Final[str] = "selection_reason values are inconsistent with selected."
_WARN_MISSING_SELECTION_SCORE: Final[str] = "selection_score values are missing or non-finite."


class FactorSelectionVerifier(BaseVerifier):
    """Deterministic canonical factor selection verifier that reports findings only.

    Inspects structural quality and ranking consistency of a canonical factor
    selection frame against ``cqros.factor_selection.schema`` /
    ``FACTOR_SELECTION_SCHEMA`` and ranking-engine invariants for the
    configured ``top_n`` / ``candidate_n``. Does not clean rows, fill gaps,
    sort timestamps, mutate values, access storage, or apply selection scoring.

    Args:
        top_n: Final Top-N limit used when the frame was generated.
        candidate_n: Candidate pool size used when the frame was generated.
    """

    __slots__ = ("_candidate_n", "_top_n")

    _top_n: int
    _candidate_n: int

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        *,
        candidate_n: int = DEFAULT_CANDIDATE_N,
    ) -> None:
        """Initialize the verifier with Top-N and candidate-pool limits."""
        self._top_n = require_top_n(top_n)
        self._candidate_n = require_candidate_n(candidate_n)
        if self._candidate_n < self._top_n:
            raise FactorSelectionError(
                "candidate_n must be greater than or equal to top_n",
                error_code="FSEL_CANDIDATE_N_LT_TOP_N",
                details={"candidate_n": self._candidate_n, "top_n": self._top_n},
            )

    @property
    def top_n(self) -> int:
        """Return the configured Top-N selection limit used for verification."""
        return self._top_n

    @property
    def candidate_n(self) -> int:
        """Return the configured candidate pool size used for verification."""
        return self._candidate_n

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical factor selection DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            FactorSelectionError: If any required column is missing or
                column dtypes do not match the factor selection schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_SELECTION_TIME,
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        empty_factor_field_rows = self._count_empty_factor_field_rows(frame)
        empty_selection_reason_rows = self._count_empty_string_rows(
            frame,
            _COL_SELECTION_REASON,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        ranking_violation_rows = self._count_ranking_violation_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_SELECTION_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            empty_factor_field_rows=empty_factor_field_rows,
            empty_selection_reason_rows=empty_selection_reason_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            ranking_violation_rows=ranking_violation_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
            frame=frame,
            top_n=self._top_n,
            candidate_n=self._candidate_n,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and ranking_violation_rows == 0
            and empty_status_rows == 0
            and invalid_status_rows == 0
            and empty_factor_field_rows == 0
            and empty_selection_reason_rows == 0
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
            raise FactorSelectionError(
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
            raise FactorSelectionError(
                "factor selection schema dtype mismatch",
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
        """Return rows with domain or non-finite numeric violations."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        selection_rank_invalid = pl.col(_COL_SELECTION_RANK) <= 0
        direction_invalid = ~pl.col(_COL_SELECTED_DIRECTION).is_in([-1, 1])
        policy_invalid = (pl.col(_COL_ORIENTATION_POLICY).is_null()) | (
            pl.col(_COL_ORIENTATION_POLICY) == ""
        )
        invalid_mask = non_finite | selection_rank_invalid | direction_invalid | policy_invalid
        return int(frame.select(invalid_mask.sum()).item())

    def _count_ranking_violation_rows(self, frame: pl.DataFrame) -> int:
        """Return rows that violate ranking-engine selection invariants."""
        if frame.height == 0:
            return 0

        selected_status = FactorSelectionStatus.SELECTED.value
        rejected_status = FactorSelectionStatus.REJECTED.value
        top_n = self._top_n
        candidate_n = self._candidate_n

        rank_unique_violation = pl.col(_COL_SELECTION_RANK).n_unique().over(
            _COL_TIMEFRAME
        ) != pl.len().over(_COL_TIMEFRAME)
        rank_origin_violation = (pl.col(_COL_SELECTION_RANK).min().over(_COL_TIMEFRAME) != 1) | (
            pl.col(_COL_SELECTION_RANK).max().over(_COL_TIMEFRAME) != pl.len().over(_COL_TIMEFRAME)
        )
        selected_count_violation = (
            pl.col(_COL_SELECTED).cast(pl.Int64).sum().over(_COL_TIMEFRAME) > top_n
        )
        selected_status_violation = (
            pl.col(_COL_SELECTED) & (pl.col(_COL_STATUS) != selected_status)
        ) | ((~pl.col(_COL_SELECTED)) & (pl.col(_COL_STATUS) != rejected_status))
        allowed_reasons = list(_ALLOWED_REASONS)
        invalid_reason_violation = ~pl.col(_COL_SELECTION_REASON).is_in(allowed_reasons)
        selected_reason_violation = pl.col(_COL_SELECTED) & (
            pl.col(_COL_SELECTION_REASON) != _REASON_TOP_N
        )
        rejected_top_n_reason_violation = (~pl.col(_COL_SELECTED)) & (
            pl.col(_COL_SELECTION_REASON) == _REASON_TOP_N
        )
        outside_candidate_violation = (pl.col(_COL_SELECTION_RANK) > candidate_n) & (
            pl.col(_COL_SELECTION_REASON) != _REASON_OUTSIDE_CANDIDATE_N
        )
        outside_candidate_selected_violation = (
            pl.col(_COL_SELECTION_REASON) == _REASON_OUTSIDE_CANDIDATE_N
        ) & pl.col(_COL_SELECTED)
        missing_score_violation = pl.col(_COL_SELECTION_SCORE).is_null() | (
            ~pl.col(_COL_SELECTION_SCORE).is_finite()
        )

        expression_violations = int(
            frame.select(
                (
                    rank_unique_violation
                    | rank_origin_violation
                    | selected_count_violation
                    | selected_status_violation
                    | invalid_reason_violation
                    | selected_reason_violation
                    | rejected_top_n_reason_violation
                    | outside_candidate_violation
                    | outside_candidate_selected_violation
                    | missing_score_violation
                ).sum()
            ).item()
        )
        order_violations = self._count_rank_order_violation_rows(frame)
        survivor_violations = self._count_survivor_top_n_violation_rows(frame)
        return expression_violations + order_violations + survivor_violations

    def _count_survivor_top_n_violation_rows(self, frame: pl.DataFrame) -> int:
        """Return violations of first-top_n-among-survivors selection semantics."""
        if frame.height == 0:
            return 0
        survivors = frame.filter(
            pl.col(_COL_SELECTION_REASON).is_in([_REASON_TOP_N, _REASON_OUTSIDE_TOP_N])
        )
        if survivors.height == 0:
            return 0
        checked = survivors.sort(
            _COL_TIMEFRAME,
            _COL_SELECTION_RANK,
            _COL_FACTOR_NAME,
            _COL_FACTOR_VERSION,
            maintain_order=True,
        ).with_columns((pl.int_range(pl.len()).over(_COL_TIMEFRAME) + 1).alias("_survivor_order"))
        expected_selected = pl.col("_survivor_order") <= self._top_n
        return int(checked.select((pl.col(_COL_SELECTED) != expected_selected).sum()).item())

    def _count_rank_order_violation_rows(self, frame: pl.DataFrame) -> int:
        """Return rows whose ranks disagree with score/name/version ordering."""
        if frame.height == 0:
            return 0
        expected = (
            frame.select(
                _COL_TIMEFRAME,
                _COL_FACTOR_NAME,
                _COL_FACTOR_VERSION,
                _COL_SELECTION_SCORE,
                _COL_SELECTION_RANK,
            )
            .sort(
                _COL_TIMEFRAME,
                _COL_SELECTION_SCORE,
                _COL_FACTOR_NAME,
                _COL_FACTOR_VERSION,
                descending=[False, True, False, False],
                maintain_order=True,
            )
            .with_columns(
                (pl.int_range(pl.len()).over(_COL_TIMEFRAME) + 1)
                .cast(pl.Int32)
                .alias("_expected_rank")
            )
        )
        return int(
            expected.select((pl.col(_COL_SELECTION_RANK) != pl.col("_expected_rank")).sum()).item()
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
    empty_selection_reason_rows: int,
    invalid_numeric_rows: int,
    ranking_violation_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
    frame: pl.DataFrame,
    top_n: int,
    candidate_n: int,
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
    if empty_selection_reason_rows > 0:
        warnings.append(_WARN_EMPTY_SELECTION_REASON)
    if frame.height > 0:
        if int(frame.select((pl.col(_COL_SELECTION_RANK) <= 0).sum()).item()) > 0:
            warnings.append(_WARN_SELECTION_RANK)
        non_finite = int(
            frame.select(
                pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)).sum()
            ).item()
        )
        if non_finite > 0:
            warnings.append(_WARN_NON_FINITE)
        if ranking_violation_rows > 0:
            selected_status = FactorSelectionStatus.SELECTED.value
            rejected_status = FactorSelectionStatus.REJECTED.value
            if (
                int(
                    frame.select(
                        (
                            pl.col(_COL_SELECTION_RANK).n_unique().over(_COL_TIMEFRAME)
                            != pl.len().over(_COL_TIMEFRAME)
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_RANK_UNIQUENESS)
            if (
                int(
                    frame.select(
                        (
                            (pl.col(_COL_SELECTION_RANK).min().over(_COL_TIMEFRAME) != 1)
                            | (
                                pl.col(_COL_SELECTION_RANK).max().over(_COL_TIMEFRAME)
                                != pl.len().over(_COL_TIMEFRAME)
                            )
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_RANK_ORIGIN)
            if (
                int(
                    frame.select(
                        (
                            pl.col(_COL_SELECTED).cast(pl.Int64).sum().over(_COL_TIMEFRAME) > top_n
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_TOP_N_COUNT)
            if (
                int(
                    frame.select(
                        (
                            (pl.col(_COL_SELECTION_RANK) > candidate_n)
                            & (pl.col(_COL_SELECTION_REASON) != _REASON_OUTSIDE_CANDIDATE_N)
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_CANDIDATE_REASON)
            expected = (
                frame.select(
                    _COL_TIMEFRAME,
                    _COL_FACTOR_NAME,
                    _COL_FACTOR_VERSION,
                    _COL_SELECTION_SCORE,
                    _COL_SELECTION_RANK,
                )
                .sort(
                    _COL_TIMEFRAME,
                    _COL_SELECTION_SCORE,
                    _COL_FACTOR_NAME,
                    _COL_FACTOR_VERSION,
                    descending=[False, True, False, False],
                    maintain_order=True,
                )
                .with_columns(
                    (pl.int_range(pl.len()).over(_COL_TIMEFRAME) + 1)
                    .cast(pl.Int32)
                    .alias("_expected_rank")
                )
            )
            if (
                int(
                    expected.select(
                        (pl.col(_COL_SELECTION_RANK) != pl.col("_expected_rank")).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_RANK_ORDER)
            if (
                int(
                    frame.select(
                        (
                            (pl.col(_COL_SELECTED) & (pl.col(_COL_STATUS) != selected_status))
                            | ((~pl.col(_COL_SELECTED)) & (pl.col(_COL_STATUS) != rejected_status))
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_SELECTED_STATUS)
            if (
                int(
                    frame.select(
                        (
                            (
                                pl.col(_COL_SELECTED)
                                & (pl.col(_COL_SELECTION_REASON) != _REASON_TOP_N)
                            )
                            | (
                                (~pl.col(_COL_SELECTED))
                                & (pl.col(_COL_SELECTION_REASON) == _REASON_TOP_N)
                            )
                            | (~pl.col(_COL_SELECTION_REASON).is_in(list(_ALLOWED_REASONS)))
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_SELECTION_REASON)
            if (
                int(
                    frame.select(
                        (
                            pl.col(_COL_SELECTION_SCORE).is_null()
                            | (~pl.col(_COL_SELECTION_SCORE).is_finite())
                        ).sum()
                    ).item()
                )
                > 0
            ):
                warnings.append(_WARN_MISSING_SELECTION_SCORE)
            survivors = frame.filter(
                pl.col(_COL_SELECTION_REASON).is_in([_REASON_TOP_N, _REASON_OUTSIDE_TOP_N])
            )
            if survivors.height > 0:
                checked = survivors.sort(
                    _COL_TIMEFRAME,
                    _COL_SELECTION_RANK,
                    _COL_FACTOR_NAME,
                    _COL_FACTOR_VERSION,
                    maintain_order=True,
                ).with_columns(
                    (pl.int_range(pl.len()).over(_COL_TIMEFRAME) + 1).alias("_survivor_order")
                )
                if (
                    int(
                        checked.select(
                            (pl.col(_COL_SELECTED) != (pl.col("_survivor_order") <= top_n)).sum()
                        ).item()
                    )
                    > 0
                ):
                    warnings.append(_WARN_TOP_N_SELECTION)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)
