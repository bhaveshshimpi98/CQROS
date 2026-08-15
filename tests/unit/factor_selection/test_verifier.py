"""Unit tests for CQROS ``FactorSelectionVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factor_selection import (
    FactorSelectionError,
    FactorSelectionStatus,
    FactorSelectionVerifier,
)
from cqros.factor_selection.engine import DEFAULT_TOP_N
from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.factor_selection.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_TIMEFRAME = "1h"
_SELECTION_TIME = 1_718_452_800_000
_REASON_TOP_N = "top_n"
_REASON_OUTSIDE_TOP_N = "outside_top_n"


def _canonical_frame(
    *,
    factor_name: str = _FACTOR_NAME,
    factor_version: str = _FACTOR_VERSION,
    timeframe: str = _TIMEFRAME,
    selection_time: int = _SELECTION_TIME,
    factor_category: str = _FACTOR_CATEGORY,
    selected: bool = True,
    selection_score: float = 0.12,
    selection_rank: int = 1,
    selection_reason: str = _REASON_TOP_N,
    selection_ic: float = 0.08,
    selected_direction: int = 1,
    orientation_policy: str = FACTOR_ORIENTATION_POLICY,
    status: str = FactorSelectionStatus.SELECTED.value,
) -> pl.DataFrame:
    """Build a canonical factor selection frame that should pass verification."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [factor_version],
            "timeframe": [timeframe],
            "selection_time": [selection_time],
            "factor_category": [factor_category],
            "selected": [selected],
            "selection_score": [selection_score],
            "selection_rank": [selection_rank],
            "selection_reason": [selection_reason],
            "selection_ic": [selection_ic],
            "selected_direction": [selected_direction],
            "orientation_policy": [orientation_policy],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed factor selection frame passes all verifier checks."""
    verifier = FactorSelectionVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_rejected_status_frame() -> None:
    """A REJECTED outside-top-N frame passes structural and ranking checks."""
    verifier = FactorSelectionVerifier()
    selected_row = _canonical_frame(
        factor_name="selected_factor",
        selection_rank=1,
        selection_score=0.90,
        selected=True,
        selection_reason=_REASON_TOP_N,
        status=FactorSelectionStatus.SELECTED.value,
    )
    rejected_row = _canonical_frame(
        factor_name="rejected_factor",
        selection_rank=DEFAULT_TOP_N + 1,
        selection_score=0.10,
        selected=False,
        selection_reason=_REASON_OUTSIDE_TOP_N,
        status=FactorSelectionStatus.REJECTED.value,
    )
    # Dense ranks 1..N require filler rows between 1 and DEFAULT_TOP_N + 1.
    fillers = [
        _canonical_frame(
            factor_name=f"filler_{index:02d}",
            selection_rank=index,
            selection_score=0.80 - (index * 0.01),
            selected=True,
            selection_reason=_REASON_TOP_N,
            status=FactorSelectionStatus.SELECTED.value,
        )
        for index in range(2, DEFAULT_TOP_N + 1)
    ]
    frame = pl.concat([selected_row, *fillers, rejected_row])
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == DEFAULT_TOP_N + 1


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = FactorSelectionVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


# ---------------------------------------------------------------------------
# Missing / mismatched columns
# ---------------------------------------------------------------------------


def test_verifier_raises_on_missing_required_column() -> None:
    """Missing required columns raise FactorSelectionError."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().drop("selection_score")
    with pytest.raises(FactorSelectionError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises FactorSelectionError with ERROR_SCHEMA_MISMATCH."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.col("selection_rank").cast(pl.Float64))
    with pytest.raises(FactorSelectionError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise FactorSelectionError for invalid dtypes."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.col("selection_score").cast(pl.String))
    with pytest.raises(FactorSelectionError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_non_boolean_selected() -> None:
    """Non-Boolean selected dtype raises FactorSelectionError schema mismatch."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.col("selected").cast(pl.String))
    with pytest.raises(FactorSelectionError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate primary keys emit a warning and FAIL the report."""
    verifier = FactorSelectionVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_selection_score() -> None:
    """Rows with NULL selection_score are counted in null_rows."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.Series("selection_score", [None], dtype=pl.Float64))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.String))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("FactorSelectionStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required string fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_factor_name() -> None:
    """Empty factor_name emits a factor-identity warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(factor_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_factor_version() -> None:
    """Empty factor_version emits a factor-identity warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(factor_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_factor_category() -> None:
    """Empty factor_category emits a factor-identity warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(factor_category="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_selection_reason() -> None:
    """Empty selection_reason emits a selection-reason warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_reason="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("selection_reason" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Numeric range and finiteness checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_finite_selection_score() -> None:
    """Non-finite selection_score emits a numeric warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("finite" in w.lower() or "NaN" in w for w in report.warnings)


def test_verifier_warns_on_nan_selection_score() -> None:
    """NaN selection_score emits a numeric or NaN warning."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_score=math.nan)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0 or report.nan_rows > 0


def test_verifier_warns_on_non_positive_selection_rank() -> None:
    """selection_rank <= 0 emits a selection_rank warning."""
    verifier = FactorSelectionVerifier()
    for bad_rank in (0, -1):
        frame = _canonical_frame(selection_rank=bad_rank)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for selection_rank={bad_rank}"
        assert any("selection_rank" in w for w in report.warnings)


def test_verifier_passes_boundary_numeric_values() -> None:
    """Boundary values for range-checked metrics are valid."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_score=0.0, selection_rank=1)
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL selection_time values are counted as invalid timestamps."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("selection_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = FactorSelectionVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_selection_time() -> None:
    """Frame not sorted by selection_time emits an unsorted warning."""
    verifier = FactorSelectionVerifier()
    t0 = _SELECTION_TIME
    t1 = _SELECTION_TIME + 3_600_000
    row_a = _canonical_frame(selection_time=t1, factor_name="momentum")
    row_b = _canonical_frame(selection_time=t0, factor_name="rsi")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = FactorSelectionVerifier()
    t0 = _SELECTION_TIME
    t1 = _SELECTION_TIME + 3_600_000
    row_a = _canonical_frame(
        selection_time=t0,
        factor_name="momentum",
        selection_rank=1,
        selection_score=0.90,
    )
    row_b = _canonical_frame(
        selection_time=t1,
        factor_name="rsi",
        selection_rank=2,
        selection_score=0.50,
    )
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()


# ---------------------------------------------------------------------------
# Ranking verification
# ---------------------------------------------------------------------------


def test_verifier_passes_ranked_top_n_frame() -> None:
    """A correctly ranked top-N frame passes ranking verification."""
    verifier = FactorSelectionVerifier()
    row_a = _canonical_frame(
        factor_name="alpha",
        selection_rank=1,
        selection_score=0.90,
        selected=True,
        selection_reason=_REASON_TOP_N,
        status=FactorSelectionStatus.SELECTED.value,
    )
    row_b = _canonical_frame(
        factor_name="beta",
        selection_rank=2,
        selection_score=0.50,
        selected=True,
        selection_reason=_REASON_TOP_N,
        status=FactorSelectionStatus.SELECTED.value,
    )
    report = verifier.verify(pl.concat([row_a, row_b]))
    assert report.passed is True
    assert report.rows_checked == 2


def test_verifier_warns_on_duplicate_ranks_within_timeframe() -> None:
    """Duplicate selection_rank values within a timeframe fail verification."""
    verifier = FactorSelectionVerifier()
    row_a = _canonical_frame(factor_name="alpha", selection_rank=1)
    row_b = _canonical_frame(factor_name="beta", selection_rank=1)
    report = verifier.verify(pl.concat([row_a, row_b]))
    assert report.passed is False
    assert any("Duplicate selection_rank" in warning for warning in report.warnings)


def test_verifier_warns_when_ranks_do_not_begin_at_one() -> None:
    """Rank sequences that do not begin at 1 fail verification."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(selection_rank=2, selected=True)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("begin at 1" in warning for warning in report.warnings)


def test_verifier_warns_on_top_n_selection_mismatch() -> None:
    """selected flags that disagree with Rank <= configured top_n fail verification."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(
        selection_rank=1,
        selected=False,
        selection_reason=_REASON_OUTSIDE_TOP_N,
        status=FactorSelectionStatus.REJECTED.value,
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("Top-N" in warning for warning in report.warnings)


def test_verifier_uses_configured_top_n() -> None:
    """Verifier evaluates selection against the injected top_n, not a hard-coded 20."""
    verifier = FactorSelectionVerifier(top_n=2)
    rows = [
        _canonical_frame(
            factor_name="a",
            selection_rank=1,
            selection_score=0.90,
            selected=True,
            selection_reason=_REASON_TOP_N,
            status=FactorSelectionStatus.SELECTED.value,
        ),
        _canonical_frame(
            factor_name="b",
            selection_rank=2,
            selection_score=0.80,
            selected=True,
            selection_reason=_REASON_TOP_N,
            status=FactorSelectionStatus.SELECTED.value,
        ),
        _canonical_frame(
            factor_name="c",
            selection_rank=3,
            selection_score=0.70,
            selected=False,
            selection_reason=_REASON_OUTSIDE_TOP_N,
            status=FactorSelectionStatus.REJECTED.value,
        ),
    ]
    report = verifier.verify(pl.concat(rows))
    assert report.passed is True
    assert verifier.top_n == 2


def test_verifier_rejects_invalid_top_n() -> None:
    """Verifier construction rejects non-positive or non-integer top_n."""
    with pytest.raises(FactorSelectionError) as exc_info:
        FactorSelectionVerifier(top_n=0)
    assert exc_info.value.error_code == "FSEL_TOP_N_INVALID"


def test_verifier_warns_on_selection_reason_mismatch() -> None:
    """selection_reason must agree with selected/rejected status."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(
        selected=True,
        selection_rank=1,
        selection_reason=_REASON_OUTSIDE_TOP_N,
        status=FactorSelectionStatus.SELECTED.value,
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("selection_reason" in warning for warning in report.warnings)


def test_verifier_warns_on_selected_status_inconsistency() -> None:
    """selected True with REJECTED status fails verification."""
    verifier = FactorSelectionVerifier()
    frame = _canonical_frame(
        selected=True,
        selection_rank=1,
        status=FactorSelectionStatus.REJECTED.value,
        selection_reason=_REASON_TOP_N,
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("selected and status" in warning for warning in report.warnings)


def test_verifier_accepts_outside_top_n_rejected_rows() -> None:
    """Rank beyond DEFAULT_TOP_N may be REJECTED with outside_top_n reason."""
    verifier = FactorSelectionVerifier()
    rows = [
        _canonical_frame(
            factor_name=f"factor_{index:02d}",
            selection_rank=index,
            selection_score=1.0 - (index * 0.01),
            selected=index <= DEFAULT_TOP_N,
            selection_reason=_REASON_TOP_N if index <= DEFAULT_TOP_N else _REASON_OUTSIDE_TOP_N,
            status=(
                FactorSelectionStatus.SELECTED.value
                if index <= DEFAULT_TOP_N
                else FactorSelectionStatus.REJECTED.value
            ),
        )
        for index in range(1, DEFAULT_TOP_N + 3)
    ]
    report = verifier.verify(pl.concat(rows))
    assert report.passed is True
    assert report.rows_checked == DEFAULT_TOP_N + 2
