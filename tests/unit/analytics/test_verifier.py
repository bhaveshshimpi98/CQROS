"""Unit tests for CQROS ``AnalyticsVerifier``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.analytics import AnalyticsStatus, AnalyticsValidationError, AnalyticsVerifier
from cqros.analytics.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.analytics.verifier import ERROR_REQUIRED_COLUMNS, ERROR_SCHEMA_MISMATCH

_MANAGER = "simple"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_OPEN_TIME = 1_718_452_800_000


def _canonical_frame(
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    open_time: int = _OPEN_TIME,
    manager: str = _MANAGER,
    rolling_return: float = 0.05,
    rolling_volatility: float = 0.0,
    rolling_max_drawdown: float = 0.0,
    rolling_win_rate: float = 0.0,
    rolling_sharpe: float | None = None,
    rolling_sortino: float | None = None,
    rolling_profit_factor: float | None = None,
    rolling_calmar: float | None = None,
    rolling_recovery_factor: float | None = None,
    benchmark_beta: float = 0.0,
    benchmark_correlation: float = 0.0,
    benchmark_tracking_error: float = 0.0,
    status: str = AnalyticsStatus.FINISHED.value,
) -> pl.DataFrame:
    """Build a canonical analytics frame that should pass verification."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timeframe": [timeframe],
            "open_time": [open_time],
            "manager": [manager],
            "rolling_return": [rolling_return],
            "rolling_volatility": [rolling_volatility],
            "rolling_sharpe": [rolling_sharpe],
            "rolling_sortino": [rolling_sortino],
            "rolling_max_drawdown": [rolling_max_drawdown],
            "rolling_win_rate": [rolling_win_rate],
            "rolling_profit_factor": [rolling_profit_factor],
            "rolling_expectancy": [0.0],
            "rolling_cagr": [0.0],
            "rolling_calmar": [rolling_calmar],
            "rolling_recovery_factor": [rolling_recovery_factor],
            "benchmark_return": [0.0],
            "benchmark_alpha": [0.0],
            "benchmark_beta": [benchmark_beta],
            "benchmark_correlation": [benchmark_correlation],
            "benchmark_tracking_error": [benchmark_tracking_error],
            "benchmark_information_ratio": [0.0],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


# ---------------------------------------------------------------------------
# Clean frame passes
# ---------------------------------------------------------------------------


def test_verifier_passes_on_canonical_frame() -> None:
    """A correctly formed analytics frame passes all verifier checks."""
    verifier = AnalyticsVerifier()
    report = verifier.verify(_canonical_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.duplicate_timestamp_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ()


def test_verifier_passes_on_active_status_frame() -> None:
    """An ACTIVE status frame passes all verifier checks."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(status=AnalyticsStatus.ACTIVE.value)
    report = verifier.verify(frame)
    assert report.passed is True
    assert report.rows_checked == 1


def test_verifier_passes_with_null_optional_metrics() -> None:
    """NULL optional ratio metrics are allowed and do not fail verification."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_profit_factor=None, rolling_recovery_factor=None)
    report = verifier.verify(frame)
    assert report.passed is True


def test_verifier_pass_summary() -> None:
    """PASS summary reports zero defect counters and empty warnings."""
    verifier = AnalyticsVerifier()
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
    """Missing required columns raise AnalyticsValidationError."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().drop("rolling_return")
    with pytest.raises(AnalyticsValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_raises_on_dtype_mismatch() -> None:
    """Wrong column dtype raises AnalyticsValidationError with ERROR_SCHEMA_MISMATCH."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(AnalyticsValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_raises_on_invalid_schema() -> None:
    """Schema mismatches raise AnalyticsValidationError for invalid dtypes."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.col("rolling_volatility").cast(pl.Int64))
    with pytest.raises(AnalyticsValidationError) as exc_info:
        verifier.verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Duplicate primary keys
# ---------------------------------------------------------------------------


def test_verifier_warns_on_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) emits a warning."""
    verifier = AnalyticsVerifier()
    row = _canonical_frame()
    duplicate = pl.concat([row, row])
    report = verifier.verify(duplicate)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any("Duplicate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Null rows
# ---------------------------------------------------------------------------


def test_verifier_counts_null_rows_in_rolling_return() -> None:
    """Rows with NULL rolling_return are counted in null_rows."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.Series("rolling_return", [None], dtype=pl.Float64))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


def test_verifier_counts_null_rows_in_status() -> None:
    """Rows with NULL status are counted in null_rows."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.Series("status", [None], dtype=pl.Utf8))
    report = verifier.verify(frame)
    assert report.null_rows == 1
    assert report.passed is False


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_invalid_status() -> None:
    """Non-canonical status values emit an invalid-status warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit("INVALID_STATUS").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("AnalyticsStatus" in w or "status" in w.lower() for w in report.warnings)


def test_verifier_warns_on_empty_status() -> None:
    """Empty string in status emits an empty-status warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("status"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("status" in w.lower() or "Empty" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------


def test_verifier_warns_on_rolling_max_drawdown_out_of_range() -> None:
    """rolling_max_drawdown values outside [0, 1] emit a max-drawdown warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_max_drawdown=1.5)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("rolling_max_drawdown" in w for w in report.warnings)


def test_verifier_warns_on_rolling_win_rate_out_of_range() -> None:
    """rolling_win_rate values outside [0, 1] emit a win-rate warning."""
    verifier = AnalyticsVerifier()
    for bad_rate in (-0.1, 1.1):
        frame = _canonical_frame(rolling_win_rate=bad_rate)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for rolling_win_rate={bad_rate}"
        assert any("rolling_win_rate" in w for w in report.warnings)


def test_verifier_warns_on_negative_rolling_volatility() -> None:
    """Negative rolling_volatility emits a volatility warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_volatility=-0.01)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("rolling_volatility" in w for w in report.warnings)


def test_verifier_warns_on_benchmark_correlation_out_of_range() -> None:
    """benchmark_correlation values outside [-1, 1] emit a correlation warning."""
    verifier = AnalyticsVerifier()
    for bad_corr in (-1.1, 1.1):
        frame = _canonical_frame(benchmark_correlation=bad_corr)
        report = verifier.verify(frame)
        assert report.passed is False, f"Expected failure for correlation={bad_corr}"
        assert any("benchmark_correlation" in w for w in report.warnings)


def test_verifier_warns_on_negative_benchmark_tracking_error() -> None:
    """Negative benchmark_tracking_error emits a tracking-error warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(benchmark_tracking_error=-0.01)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("benchmark_tracking_error" in w for w in report.warnings)


def test_verifier_range_validation_failures() -> None:
    """Range validation failures mark the report as FAIL with warnings."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_win_rate=1.5, rolling_max_drawdown=-0.1)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0
    assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Non-finite numeric values
# ---------------------------------------------------------------------------


def test_verifier_warns_on_nan_rolling_return() -> None:
    """NaN rolling_return triggers a non-finite value warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("nan")).alias("rolling_return"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.nan_rows > 0 or report.invalid_numeric_rows > 0


def test_verifier_warns_on_inf_benchmark_beta() -> None:
    """Infinite benchmark_beta triggers a non-finite value warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(benchmark_beta=float("inf"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0


def test_verifier_warns_on_inf_rolling_sharpe() -> None:
    """Infinite rolling_sharpe triggers a non-finite value warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_sharpe=float("inf"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0


def test_verifier_warns_on_inf_rolling_sortino() -> None:
    """Infinite rolling_sortino triggers a non-finite value warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_sortino=float("inf"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows > 0


def test_verifier_warns_on_invalid_numeric_values() -> None:
    """Invalid numeric values fail verification with numeric counters."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit(float("nan")).alias("rolling_cagr"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.nan_rows > 0 or report.invalid_numeric_rows > 0


# ---------------------------------------------------------------------------
# Invalid timestamps
# ---------------------------------------------------------------------------


def test_verifier_counts_invalid_timestamp_rows() -> None:
    """Rows with NULL open_time values are counted as invalid timestamps."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("open_time"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows > 0 or report.null_rows > 0


# ---------------------------------------------------------------------------
# Column order and sorting
# ---------------------------------------------------------------------------


def test_verifier_warns_on_non_canonical_column_order() -> None:
    """Frame with columns out of canonical order emits a column-order warning."""
    verifier = AnalyticsVerifier()
    columns = list(CANONICAL_COLUMN_ORDER)
    columns.reverse()
    frame = _canonical_frame().select(columns)
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("column order" in w.lower() or "Column" in w for w in report.warnings)


def test_verifier_warns_on_unsorted_open_time() -> None:
    """Frame not sorted by open_time emits an unsorted warning."""
    verifier = AnalyticsVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + 3_600_000
    row_a = _canonical_frame(open_time=t1, symbol="BTCUSDT")
    row_b = _canonical_frame(open_time=t0, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("sorted" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Multiple rows, lineage, boundaries, and FAIL summary
# ---------------------------------------------------------------------------


def test_verifier_aggregates_multiple_rows() -> None:
    """Verifier correctly aggregates checks across multiple rows."""
    verifier = AnalyticsVerifier()
    t0 = _OPEN_TIME
    t1 = _OPEN_TIME + 3_600_000
    row_a = _canonical_frame(open_time=t0, status=AnalyticsStatus.ACTIVE.value)
    row_b = _canonical_frame(open_time=t1, symbol="ETHUSDT")
    frame = pl.concat([row_a, row_b])
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_verifier_incomplete_lineage_emits_warning() -> None:
    """Blank manager (lineage column) emits an incomplete-lineage warning."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame().with_columns(pl.lit("").alias("manager"))
    report = verifier.verify(frame)
    assert report.passed is False
    assert any("lineage" in w.lower() or "Lineage" in w for w in report.warnings)


def test_verifier_passes_rolling_win_rate_boundary_values() -> None:
    """rolling_win_rate = 0.0 and 1.0 are valid boundary values."""
    verifier = AnalyticsVerifier()
    for boundary in (0.0, 1.0):
        frame = _canonical_frame(rolling_win_rate=boundary)
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for rolling_win_rate={boundary}"


def test_verifier_passes_rolling_max_drawdown_boundary_values() -> None:
    """rolling_max_drawdown = 0.0 and 1.0 are valid boundary values."""
    verifier = AnalyticsVerifier()
    for boundary in (0.0, 1.0):
        frame = _canonical_frame(rolling_max_drawdown=boundary)
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for rolling_max_drawdown={boundary}"


def test_verifier_passes_benchmark_correlation_boundary_values() -> None:
    """benchmark_correlation = -1.0 and 1.0 are valid boundary values."""
    verifier = AnalyticsVerifier()
    for boundary in (-1.0, 1.0):
        frame = _canonical_frame(benchmark_correlation=boundary)
        report = verifier.verify(frame)
        assert report.passed is True, f"Expected pass for correlation={boundary}"


def test_verifier_fail_summary() -> None:
    """FAIL summary reports defect counters and non-empty warnings."""
    verifier = AnalyticsVerifier()
    frame = _canonical_frame(rolling_win_rate=1.5)
    report = verifier.verify(frame)
    assert report.passed is False
    assert report.rows_checked == 1
    assert report.invalid_numeric_rows > 0
    assert report.warnings != ()
