"""Unit tests for CQROS ``FactorValidationVerifier``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from cqros.factor_validation import (
    FactorValidationError,
    FactorValidationStatus,
    FactorValidationVerifier,
)
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_VALIDATION_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.factor_validation.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_TIMEFRAME = "1h"
_VALIDATION_TIME = 1_718_452_800_000
_VALIDATION_START_TIME = 1_718_366_400_000
_VALIDATION_END_TIME = 1_718_452_800_000
_DATASET_VERSION = "dataset-v1"
_LABEL_VERSION = "label-v1"


def _canonical_frame(
    *,
    factor_name: str = _FACTOR_NAME,
    factor_version: str = _FACTOR_VERSION,
    timeframe: str = _TIMEFRAME,
    validation_time: int = _VALIDATION_TIME,
    factor_category: str = _FACTOR_CATEGORY,
    dataset_version: str = _DATASET_VERSION,
    label_version: str = _LABEL_VERSION,
    validation_start_time: int = _VALIDATION_START_TIME,
    validation_end_time: int = _VALIDATION_END_TIME,
    information_coefficient: float = 0.0,
    rank_information_coefficient: float = 0.0,
    ic_information_ratio: float = 0.0,
    ic_std: float = 0.0,
    ic_p_value: float = 1.0,
    ic_t_stat: float = 0.0,
    ic_decay: float = 0.0,
    turnover: float = 0.0,
    monotonicity_score: float = 0.0,
    quantile_spread: float = 0.0,
    observations: int = 1,
    ic_observations: int = 1,
    status: str = FactorValidationStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical factor validation frame that should pass verification."""
    return pl.DataFrame(
        {
            "factor_name": [factor_name],
            "factor_version": [factor_version],
            "timeframe": [timeframe],
            "validation_time": [validation_time],
            "factor_category": [factor_category],
            "dataset_version": [dataset_version],
            "label_version": [label_version],
            "validation_start_time": [validation_start_time],
            "validation_end_time": [validation_end_time],
            "information_coefficient": [information_coefficient],
            "rank_information_coefficient": [rank_information_coefficient],
            "ic_information_ratio": [ic_information_ratio],
            "ic_std": [ic_std],
            "ic_p_value": [ic_p_value],
            "ic_t_stat": [ic_t_stat],
            "ic_decay": [ic_decay],
            "turnover": [turnover],
            "monotonicity_score": [monotonicity_score],
            "quantile_spread": [quantile_spread],
            "observations": [observations],
            "ic_observations": [ic_observations],
            "status": [status],
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed factor validation frame passes all verifier checks."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame()
    assert frame.columns == list(REQUIRED_COLUMNS)
    assert frame.columns == list(CANONICAL_COLUMN_ORDER)
    assert frame.schema == FACTOR_VALIDATION_SCHEMA
    report = verifier.verify(frame)
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
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(status=FactorValidationStatus.FAIL.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_on_skipped_status_frame() -> None:
    """A SKIPPED status frame passes structural verifier checks."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(status=FactorValidationStatus.SKIPPED.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = FactorValidationVerifier()
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
    """Missing required columns raise FactorValidationError."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().drop("information_coefficient")
    with pytest.raises(FactorValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises FactorValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.col("observations").cast(pl.Float64))
    with pytest.raises(FactorValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise FactorValidationError for invalid dtypes."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.col("information_coefficient").cast(pl.String))
    with pytest.raises(FactorValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate primary keys emit a warning and FAIL the report."""
    verifier = FactorValidationVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_allows_null_information_coefficient() -> None:
    """NULL metric floats are allowed by the engine contract and do not fail."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(
        pl.Series("information_coefficient", [None], dtype=pl.Float64)
    )
    report = verifier.verify(frame)
    assert report.null_rows == 0
    assert report.passed is True


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.String))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("FactorValidationStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Empty required factor identity fields
# ---------------------------------------------------------------------------


def test_verifier_warns_on_empty_factor_name() -> None:
    """Empty factor_name emits a factor-identity warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(factor_name="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_factor_version() -> None:
    """Empty factor_version emits a factor-identity warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(factor_version="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


def test_verifier_warns_on_empty_factor_category() -> None:
    """Empty factor_category emits a factor-identity warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(factor_category="")
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("factor identity" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Numeric range and finiteness checks
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_finite_information_coefficient() -> None:
    """Non-finite information_coefficient emits a numeric warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(information_coefficient=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any("finite" in w.lower() or "NaN" in w for w in report.warnings)


def test_verifier_warns_on_non_finite_rank_information_coefficient() -> None:
    """Non-finite rank_information_coefficient emits a numeric warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(rank_information_coefficient=math.nan)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0 or report.nan_rows > 0


def test_verifier_warns_on_ic_p_value_out_of_range() -> None:
    """ic_p_value values outside [0, 1] emit an ic_p_value warning."""
    verifier = FactorValidationVerifier()
    for bad_value in (-0.1, 1.1):
        frame = _canonical_frame(ic_p_value=bad_value)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for ic_p_value={bad_value}"
        assert any("ic_p_value" in w for w in report.warnings)


def test_verifier_warns_on_negative_ic_decay() -> None:
    """Negative ic_decay emits an ic_decay warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(ic_decay=-0.01)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("ic_decay" in w for w in report.warnings)


def test_verifier_warns_on_negative_turnover() -> None:
    """Negative turnover emits a turnover warning."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(turnover=-1.0)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("turnover" in w for w in report.warnings)


def test_verifier_warns_on_non_positive_observations() -> None:
    """observations <= 0 emits an observations warning."""
    verifier = FactorValidationVerifier()
    for bad_count in (0, -1):
        frame = _canonical_frame(observations=bad_count)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for observations={bad_count}"
        assert any("observations" in w for w in report.warnings)


def test_verifier_passes_boundary_numeric_values() -> None:
    """Boundary values for range-checked metrics are valid."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(
        ic_p_value=0.0,
        monotonicity_score=0.0,
        ic_decay=0.0,
        turnover=0.0,
        observations=1,
        ic_observations=1,
    )
    report = verifier.verify(frame)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL validation_time values are counted as invalid timestamps."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("validation_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = FactorValidationVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_validation_time() -> None:
    """Frame not sorted by validation_time emits an unsorted warning."""
    verifier = FactorValidationVerifier()
    t0 = _VALIDATION_TIME
    t1 = _VALIDATION_TIME + 3_600_000
    row_a = _canonical_frame(validation_time=t1, factor_name="momentum")
    row_b = _canonical_frame(validation_time=t0, factor_name="rsi")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = FactorValidationVerifier()
    t0 = _VALIDATION_TIME
    t1 = _VALIDATION_TIME + 3_600_000
    row_a = _canonical_frame(validation_time=t0, factor_name="momentum")
    row_b = _canonical_frame(validation_time=t1, factor_name="rsi")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = FactorValidationVerifier()
    frame = _canonical_frame(information_coefficient=math.inf)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
