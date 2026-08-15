"""Unit tests for CQROS ``WalkForwardVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.walk_forward import WalkForwardError, WalkForwardStatus, WalkForwardVerifier
from cqros.walk_forward.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.walk_forward.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_STRATEGY_NAME = "default_strategy"
_STRATEGY_VERSION = "v1"
_MODEL_VERSION = "v1"
_TIMEFRAME = "1h"
_TRAIN_START = 1_704_067_200_000
_TRAIN_END = 1_704_070_800_000
_TEST_START = 1_704_070_800_000
_TEST_END = 1_704_074_400_000


def _canonical_frame(
    *,
    strategy_name: str = _STRATEGY_NAME,
    strategy_version: str = _STRATEGY_VERSION,
    timeframe: str = _TIMEFRAME,
    fold_id: int = 1,
    train_start: int = _TRAIN_START,
    train_end: int = _TRAIN_END,
    test_start: int = _TEST_START,
    test_end: int = _TEST_END,
    train_rows: int = 1,
    test_rows: int = 1,
    selected_factors: int = 1,
    model_version: str = _MODEL_VERSION,
    train_score: float = 0.0,
    test_score: float = 0.0,
    overfit_gap: float = 0.0,
    status: str = WalkForwardStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical walk-forward frame that should pass verification."""
    return pl.DataFrame(
        {
            "strategy_name": [strategy_name],
            "strategy_version": [strategy_version],
            "timeframe": [timeframe],
            "fold_id": [fold_id],
            "train_start": [train_start],
            "train_end": [train_end],
            "test_start": [test_start],
            "test_end": [test_end],
            "train_rows": [train_rows],
            "test_rows": [test_rows],
            "selected_factors": [selected_factors],
            "model_version": [model_version],
            "train_score": [train_score],
            "test_score": [test_score],
            "overfit_gap": [overfit_gap],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed walk-forward frame passes all verifier checks."""
    verifier = WalkForwardVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_fail_status_frame() -> None:
    """A FAIL status frame passes structural verifier checks."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(status=WalkForwardStatus.FAIL.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = WalkForwardVerifier()
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
    """Missing required columns raise WalkForwardError."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().drop("train_score")
    with pytest.raises(WalkForwardError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises WalkForwardError with ERROR_SCHEMA_MISMATCH."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.col("fold_id").cast(pl.Float64))
    with pytest.raises(WalkForwardError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise WalkForwardError for invalid dtypes."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.col("train_score").cast(pl.String))
    with pytest.raises(WalkForwardError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate primary keys emit a warning and FAIL the report."""
    verifier = WalkForwardVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_train_score() -> None:
    """Rows with NULL train_score are counted in null_rows."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.Series("train_score", [None], dtype=pl.Float64))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.String))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("WalkForwardStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required string fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_strategy_name() -> None:
    """Empty strategy_name emits an identity-field warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(strategy_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_strategy_version() -> None:
    """Empty strategy_version emits an identity-field warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(strategy_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_model_version() -> None:
    """Empty model_version emits an identity-field warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(model_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Numeric range and finiteness checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_finite_train_score() -> None:
    """Non-finite train_score emits a numeric warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(train_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("finite" in w.lower() or "NaN" in w for w in report.warnings)


def test_verifier_warns_on_nan_test_score() -> None:
    """NaN test_score emits a numeric or NaN warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(test_score=math.nan)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0 or report.nan_rows > 0


def test_verifier_warns_on_non_finite_overfit_gap() -> None:
    """Non-finite overfit_gap emits a numeric warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(overfit_gap=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1


def test_verifier_warns_on_non_positive_train_rows() -> None:
    """train_rows <= 0 emits a train_rows warning."""
    verifier = WalkForwardVerifier()
    for bad_rows in (0, -1):
        frame = _canonical_frame(train_rows=bad_rows)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for train_rows={bad_rows}"
        assert any("train_rows" in w for w in report.warnings)


def test_verifier_warns_on_non_positive_test_rows() -> None:
    """test_rows <= 0 emits a test_rows warning."""
    verifier = WalkForwardVerifier()
    for bad_rows in (0, -1):
        frame = _canonical_frame(test_rows=bad_rows)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for test_rows={bad_rows}"
        assert any("test_rows" in w for w in report.warnings)


def test_verifier_warns_on_negative_selected_factors() -> None:
    """selected_factors < 0 emits a selected_factors warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(selected_factors=-1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("selected_factors" in w for w in report.warnings)


def test_verifier_warns_on_invalid_train_window_order() -> None:
    """train_start > train_end emits a window-ordering warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(train_start=_TRAIN_END, train_end=_TRAIN_START)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("window" in w.lower() for w in report.warnings)


def test_verifier_warns_on_invalid_test_window_order() -> None:
    """test_start > test_end emits a window-ordering warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(test_start=_TEST_END, test_end=_TEST_START)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("window" in w.lower() for w in report.warnings)


def test_verifier_warns_on_train_end_after_test_start() -> None:
    """train_end > test_start emits a window-ordering warning."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(
        train_end=_TEST_START + 3_600_000,
        test_start=_TEST_START,
    )
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("window" in w.lower() for w in report.warnings)


def test_verifier_passes_boundary_numeric_values() -> None:
    """Boundary values for range-checked metrics are valid."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(
        train_rows=1,
        test_rows=1,
        selected_factors=0,
        train_score=0.0,
        test_score=0.0,
        overfit_gap=0.0,
    )
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL train_start values are counted as invalid timestamps."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("train_start"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = WalkForwardVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_train_start() -> None:
    """Frame not sorted by train_start emits an unsorted warning."""
    verifier = WalkForwardVerifier()
    t0 = _TRAIN_START
    t1 = _TRAIN_START + 3_600_000
    row_a = _canonical_frame(
        train_start=t1,
        train_end=t1,
        test_start=t1,
        test_end=t1,
        fold_id=2,
        timeframe="4h",
    )
    row_b = _canonical_frame(
        train_start=t0,
        train_end=t0,
        test_start=t0,
        test_end=t0,
        fold_id=1,
        timeframe="1h",
    )
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = WalkForwardVerifier()
    t0 = _TRAIN_START
    t1 = _TRAIN_START + 3_600_000
    row_a = _canonical_frame(
        train_start=t0,
        train_end=t0,
        test_start=t0,
        test_end=t0,
        fold_id=1,
        timeframe="1h",
    )
    row_b = _canonical_frame(
        train_start=t1,
        train_end=t1,
        test_start=t1,
        test_end=t1,
        fold_id=1,
        timeframe="4h",
    )
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = WalkForwardVerifier()
    frame = _canonical_frame(train_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
