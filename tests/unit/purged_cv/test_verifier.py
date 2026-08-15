"""Unit tests for CQROS ``PurgedCVVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.purged_cv import PurgedCVError, PurgedCVStatus, PurgedCVVerifier
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.purged_cv.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_STRATEGY_NAME = "default_strategy"
_STRATEGY_VERSION = "v1"
_TIMEFRAME = "1h"
_TRAIN_START = 1_704_067_200_000
_TRAIN_END = 1_704_070_800_000
_TEST_START = 1_704_074_400_000
_TEST_END = 1_704_078_000_000


def _canonical_frame(
    *,
    strategy_name: str = _STRATEGY_NAME,
    strategy_version: str = _STRATEGY_VERSION,
    timeframe: str = _TIMEFRAME,
    fold_id: int = 1,
    train_start_time: int = _TRAIN_START,
    train_end_time: int = _TRAIN_END,
    test_start_time: int = _TEST_START,
    test_end_time: int = _TEST_END,
    purge_size: int = 5,
    embargo_size: int = 5,
    train_rows: int = 2,
    test_rows: int = 2,
    train_score: float = 0.10,
    test_score: float = 0.05,
    overfit_gap: float = 0.05,
    status: str = PurgedCVStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical purged-CV frame that should pass verification."""
    return pl.DataFrame(
        {
            "strategy_name": [strategy_name],
            "strategy_version": [strategy_version],
            "timeframe": [timeframe],
            "fold_id": [fold_id],
            "train_start_time": [train_start_time],
            "train_end_time": [train_end_time],
            "test_start_time": [test_start_time],
            "test_end_time": [test_end_time],
            "purge_size": [purge_size],
            "embargo_size": [embargo_size],
            "train_rows": [train_rows],
            "test_rows": [test_rows],
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
    """A correctly formed purged-CV frame passes all verifier checks."""
    verifier = PurgedCVVerifier()
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
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(status=PurgedCVStatus.FAIL.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = PurgedCVVerifier()
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
    """Missing required columns raise PurgedCVError."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().drop("train_score")
    with pytest.raises(PurgedCVError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises PurgedCVError with ERROR_SCHEMA_MISMATCH."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.col("fold_id").cast(pl.Float64))
    with pytest.raises(PurgedCVError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise PurgedCVError for invalid dtypes."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.col("train_score").cast(pl.String))
    with pytest.raises(PurgedCVError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate primary keys emit a warning and FAIL the report."""
    verifier = PurgedCVVerifier()
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
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.Series("train_score", [None], dtype=pl.Float64))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.String))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("PurgedCVStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required string fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_strategy_name() -> None:
    """Empty strategy_name emits an identity-field warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(strategy_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_strategy_version() -> None:
    """Empty strategy_version emits an identity-field warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(strategy_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_timeframe() -> None:
    """Empty timeframe emits an identity-field warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(timeframe="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("identity" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Numeric range and finiteness checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_finite_train_score() -> None:
    """Non-finite train_score emits a numeric warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(train_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("finite" in w.lower() or "NaN" in w for w in report.warnings)


def test_verifier_warns_on_nan_test_score() -> None:
    """NaN test_score emits a numeric or NaN warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(test_score=math.nan)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0 or report.nan_rows > 0


def test_verifier_warns_on_non_finite_overfit_gap() -> None:
    """Non-finite overfit_gap emits a numeric warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(overfit_gap=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1


def test_verifier_warns_on_negative_train_rows() -> None:
    """train_rows < 0 emits a train_rows warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(train_rows=-1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("train_rows" in w for w in report.warnings)


def test_verifier_warns_on_negative_test_rows() -> None:
    """test_rows < 0 emits a test_rows warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(test_rows=-1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("test_rows" in w for w in report.warnings)


def test_verifier_warns_on_negative_purge_size() -> None:
    """purge_size < 0 emits a purge_size warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(purge_size=-1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("purge_size" in w for w in report.warnings)


def test_verifier_warns_on_negative_embargo_size() -> None:
    """embargo_size < 0 emits an embargo_size warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(embargo_size=-1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("embargo_size" in w for w in report.warnings)


def test_verifier_warns_on_invalid_train_window_order() -> None:
    """train_start_time > train_end_time emits a window-ordering warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(train_start_time=_TRAIN_END, train_end_time=_TRAIN_START)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("window" in w.lower() for w in report.warnings)


def test_verifier_warns_on_invalid_test_window_order() -> None:
    """test_start_time > test_end_time emits a window-ordering warning."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(test_start_time=_TEST_END, test_end_time=_TEST_START)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("window" in w.lower() for w in report.warnings)


def test_verifier_passes_when_train_extent_spans_test_block() -> None:
    """Train extents wrapping the test block are valid for purged-CV."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(
        train_start_time=_TRAIN_START,
        train_end_time=_TEST_END + 3_600_000,
        test_start_time=_TEST_START,
        test_end_time=_TEST_END,
    )
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_when_train_end_equals_test_start() -> None:
    """train_end_time == test_start_time is not a structural window failure."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(
        train_end_time=_TEST_START,
        test_start_time=_TEST_START,
    )
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_boundary_numeric_values() -> None:
    """Boundary values for range-checked metrics are valid."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(
        purge_size=0,
        embargo_size=0,
        train_rows=0,
        test_rows=0,
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
    """Rows with NULL train_start_time values are counted as invalid timestamps."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("train_start_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = PurgedCVVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_fold_id() -> None:
    """Frame not sorted by fold_id emits an unsorted warning."""
    verifier = PurgedCVVerifier()
    t0 = _TRAIN_START
    t1 = _TRAIN_START + 3_600_000
    row_a = _canonical_frame(
        train_start_time=t1,
        train_end_time=t1,
        test_start_time=t1 + 3_600_000,
        test_end_time=t1 + 7_200_000,
        fold_id=2,
        timeframe="4h",
    )
    row_b = _canonical_frame(
        train_start_time=t0,
        train_end_time=t0,
        test_start_time=t0 + 3_600_000,
        test_end_time=t0 + 7_200_000,
        fold_id=1,
        timeframe="1h",
    )
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("fold_id" in w.lower() for w in report.warnings)


def test_verifier_passes_fold_id_sorted_despite_train_start_disorder() -> None:
    """fold_id order passes even when train_start_time is non-monotonic."""
    verifier = PurgedCVVerifier()
    # Mirrors engine emission: early folds may purge early train observations,
    # so fold 1 can have a later train_start_time than fold 2.
    fold_one = _canonical_frame(
        fold_id=1,
        train_start_time=_TRAIN_START + 3_600_000,
        train_end_time=_TEST_END + 3_600_000,
        test_start_time=_TRAIN_START,
        test_end_time=_TRAIN_START + 1_800_000,
        timeframe="1h",
    )
    fold_two = _canonical_frame(
        fold_id=2,
        train_start_time=_TRAIN_START,
        train_end_time=_TEST_END + 3_600_000,
        test_start_time=_TEST_START,
        test_end_time=_TEST_END,
        timeframe="1h",
    )
    frame = pl.concat([fold_one, fold_two])
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.warnings == ()


def test_verifier_passes_engine_generated_multi_fold_frame() -> None:
    """Engine-produced multi-fold ledgers pass without false-positive warnings."""
    times = [_TRAIN_START + index * 3_600_000 for index in range(20)]
    walk_forward = pl.DataFrame(
        {
            "strategy_name": [_STRATEGY_NAME] * 20,
            "strategy_version": [_STRATEGY_VERSION] * 20,
            "timeframe": [_TIMEFRAME] * 20,
            "test_start": times,
            "train_score": [float(index) for index in range(20)],
            "test_score": [float(index) for index in range(20)],
        }
    )
    frame = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1).build(walk_forward)
    report = PurgedCVVerifier().verify(frame)
    assert report.passed is True
    assert report.rows_checked == 5
    assert report.warnings == ()
    assert report.invalid_numeric_rows == 0


# ---------------------------------------------------------------------------
# Multiple rows and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = PurgedCVVerifier()
    t0 = _TRAIN_START
    t1 = _TRAIN_START + 3_600_000
    row_a = _canonical_frame(
        train_start_time=t0,
        train_end_time=t0,
        test_start_time=t0 + 3_600_000,
        test_end_time=t0 + 7_200_000,
        fold_id=1,
        timeframe="1h",
    )
    row_b = _canonical_frame(
        train_start_time=t1,
        train_end_time=t1,
        test_start_time=t1 + 3_600_000,
        test_end_time=t1 + 7_200_000,
        fold_id=1,
        timeframe="4h",
    )
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = PurgedCVVerifier()
    frame = _canonical_frame(train_score=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
