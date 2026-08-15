"""Unit tests for CQROS FactorVerifier diagnostics."""

from __future__ import annotations

import math
import sys

import polars as pl
import pytest

from cqros.factors.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES, FactorStatus
from cqros.factors.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorValidationError,
    FactorVerifier,
    InvalidNumericKind,
    NullClassification,
    collect_global_failure_findings,
    format_factor_diagnostics,
    format_global_failure_report,
)
from cqros.factors.verification.domain_null_metadata import factor_allows_domain_nulls

_MS_PER_DAY: int = 86_400_000
_BASE_OPEN_TIME: int = 1_704_067_200_000  # 2024-01-01 UTC


def _frame(**overrides: object) -> pl.DataFrame:
    """Build a minimal canonical factor frame with optional overrides."""
    data: dict[str, object] = {
        "symbol": ["BTCUSDT"],
        "timeframe": ["1h"],
        "open_time": [_BASE_OPEN_TIME],
        "factor_name": ["momentum"],
        "factor_version": ["1.0.0"],
        "factor_category": ["price"],
        "factor_group": ["trend"],
        "factor_value": [0.1],
        "lookback": [20],
        "prediction_horizon": [1],
        "enabled": [True],
        "status": [FactorStatus.ACTIVE.value],
    }
    data.update(overrides)
    return pl.DataFrame(data, schema=dict(COLUMN_DTYPES)).select(list(CANONICAL_COLUMN_ORDER))


def _rows(
    *,
    factor_name: str,
    values: list[float | None],
    start_open_time: int = _BASE_OPEN_TIME,
) -> pl.DataFrame:
    """Build consecutive daily rows for one factor."""
    frames = [
        _frame(
            open_time=[start_open_time + (index * _MS_PER_DAY)],
            factor_name=[factor_name],
            factor_value=[value],
        )
        for index, value in enumerate(values)
    ]
    return pl.concat(frames)


def test_factor_allows_domain_nulls_metadata_lookup() -> None:
    """Domain-NULL permission is centralized in the metadata lookup."""
    assert factor_allows_domain_nulls("ease_of_movement") is True
    assert factor_allows_domain_nulls("volume_rate_of_change") is True
    assert factor_allows_domain_nulls("rsi") is False
    assert factor_allows_domain_nulls("bollinger_width") is False


def test_verify_passing_frame() -> None:
    """A clean canonical frame passes with zero counters."""
    report = FactorVerifier().verify(_frame())
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.warmup_null_rows == 0
    assert report.domain_null_rows == 0
    assert report.unexpected_null_rows == 0
    assert report.warnings == ()
    assert report.diagnostics.has_content is False


def test_verify_raises_on_missing_columns() -> None:
    """Missing required columns raise FactorValidationError."""
    frame = _frame().drop("factor_value")
    with pytest.raises(FactorValidationError) as exc_info:
        FactorVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verify_raises_on_dtype_mismatch() -> None:
    """Dtype mismatches raise FactorValidationError."""
    frame = _frame().with_columns(pl.col("open_time").cast(pl.Float64))
    with pytest.raises(FactorValidationError) as exc_info:
        FactorVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verify_detects_duplicate_keys() -> None:
    """Duplicate (symbol, timeframe, open_time, factor_name) rows fail."""
    frame = pl.concat([_frame(), _frame()])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 1
    assert any(warning.startswith("DUPLICATE_KEYS ") for warning in report.warnings)


def test_verify_allows_multiple_factors_same_bar() -> None:
    """Distinct factor_name values at the same bar are not duplicates."""
    second = _frame(factor_name=["rsi"], factor_value=[0.2])
    frame = pl.concat([_frame(), second])
    report = FactorVerifier().verify(frame)
    assert report.passed is True
    assert report.duplicate_timestamp_rows == 0
    assert report.rows_checked == 2


def test_verify_warmup_only_nulls_do_not_fail() -> None:
    """Leading consecutive factor_value NULLs are warmup and do not fail."""
    frame = _rows(factor_name="rsi", values=[None, None, None, 0.5, 0.6])
    report = FactorVerifier().verify(frame)
    assert report.passed is True
    assert report.warmup_null_rows == 3
    assert report.domain_null_rows == 0
    assert report.unexpected_null_rows == 0
    assert report.null_rows == 3
    assert len(report.diagnostics.null_diagnostics) == 1
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.factor_name == "rsi"
    assert diagnostic.count == 3
    assert diagnostic.only_at_beginning is True
    assert diagnostic.appears_after_valid is False
    assert diagnostic.classification == NullClassification.WARMUP_NULLS
    assert any(warning.startswith("WARMUP_NULLS factor=rsi count=3") for warning in report.warnings)


def test_verify_domain_nulls_do_not_fail() -> None:
    """Post-warmup NULLs on allowlisted factors are domain NULLs and pass."""
    frame = _rows(factor_name="ease_of_movement", values=[0.1, None, 0.2, None])
    report = FactorVerifier().verify(frame)
    assert report.passed is True
    assert report.warmup_null_rows == 0
    assert report.domain_null_rows == 2
    assert report.unexpected_null_rows == 0
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.factor_name == "ease_of_movement"
    assert diagnostic.appears_after_valid is True
    assert diagnostic.classification == NullClassification.DOMAIN_NULLS
    assert any(
        warning.startswith("DOMAIN_NULLS factor=ease_of_movement count=2")
        for warning in report.warnings
    )


def test_verify_volume_rate_of_change_domain_nulls_pass() -> None:
    """volume_rate_of_change post-warmup NULLs classify as domain NULLs."""
    frame = _rows(factor_name="volume_rate_of_change", values=[None, None, 0.5, None, 0.6])
    report = FactorVerifier().verify(frame)
    assert report.passed is True
    assert report.warmup_null_rows == 0
    assert report.domain_null_rows == 3
    assert report.unexpected_null_rows == 0
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.classification == NullClassification.DOMAIN_NULLS
    assert diagnostic.appears_after_valid is True


def test_verify_allowlisted_factor_warmup_nulls_remain_warmup() -> None:
    """Allowlisted factors still classify leading-only NULLs as warmup."""
    frame = _rows(factor_name="ease_of_movement", values=[None, None, 0.5, 0.6])
    report = FactorVerifier().verify(frame)
    assert report.passed is True
    assert report.warmup_null_rows == 2
    assert report.domain_null_rows == 0
    assert report.unexpected_null_rows == 0
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.classification == NullClassification.WARMUP_NULLS


def test_verify_unexpected_nulls_fail() -> None:
    """NULLs that reappear after valid observations fail verification."""
    frame = _rows(factor_name="bollinger_width", values=[0.1, None, 0.2, None])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.warmup_null_rows == 0
    assert report.domain_null_rows == 0
    assert report.unexpected_null_rows == 2
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.factor_name == "bollinger_width"
    assert diagnostic.appears_after_valid is True
    assert diagnostic.classification == NullClassification.UNEXPECTED_NULLS
    assert any(
        warning.startswith("UNEXPECTED_NULLS factor=bollinger_width count=2")
        for warning in report.warnings
    )


def test_verify_mixed_warmup_and_unexpected_nulls() -> None:
    """Warmup leading NULLs plus later NULLs classify as unexpected overall."""
    frame = _rows(factor_name="rsi", values=[None, None, 0.4, None, 0.5])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.warmup_null_rows == 0
    assert report.domain_null_rows == 0
    assert report.unexpected_null_rows == 3
    diagnostic = report.diagnostics.null_diagnostics[0]
    assert diagnostic.classification == NullClassification.UNEXPECTED_NULLS
    assert diagnostic.appears_after_valid is True
    assert diagnostic.only_at_beginning is False


def test_verify_multiple_factors_null_classifications() -> None:
    """Per-factor NULL classification is independent across factors."""
    rsi = _rows(factor_name="rsi", values=[None, None, 0.5])
    width = _rows(
        factor_name="bollinger_width",
        values=[0.1, None, 0.2],
        start_open_time=_BASE_OPEN_TIME,
    )
    eom = _rows(
        factor_name="ease_of_movement",
        values=[0.1, None, 0.2],
        start_open_time=_BASE_OPEN_TIME,
    )
    frame = pl.concat([rsi, width, eom]).sort(["open_time", "factor_name"])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.warmup_null_rows == 2
    assert report.domain_null_rows == 1
    assert report.unexpected_null_rows == 1
    by_name = {item.factor_name: item for item in report.diagnostics.null_diagnostics}
    assert by_name["rsi"].classification == NullClassification.WARMUP_NULLS
    assert by_name["bollinger_width"].classification == NullClassification.UNEXPECTED_NULLS
    assert by_name["ease_of_movement"].classification == NullClassification.DOMAIN_NULLS


def test_verify_detects_nan() -> None:
    """NaN factor values fail and are reported as non-finite."""
    frame = _frame(factor_name=["rsi"], factor_value=[float("nan")])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.nan_rows == 1
    assert report.non_finite_rows == 1
    assert any(
        item.kind == InvalidNumericKind.NON_FINITE
        for item in report.diagnostics.invalid_numeric_diagnostics
    )


def test_verify_detects_positive_infinity() -> None:
    """Positive infinity fails with structured +inf diagnostics."""
    frame = _frame(factor_name=["vwap_distance"], factor_value=[math.inf])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert report.positive_inf_rows == 1
    assert report.negative_inf_rows == 0
    diagnostic = report.diagnostics.invalid_numeric_diagnostics[0]
    assert diagnostic.factor_name == "vwap_distance"
    assert diagnostic.kind == InvalidNumericKind.POSITIVE_INFINITY
    assert any(
        warning.startswith("POSITIVE_INFINITY factor=vwap_distance count=1")
        for warning in report.warnings
    )


def test_verify_detects_negative_infinity() -> None:
    """Negative infinity fails with structured -inf diagnostics."""
    frame = _frame(factor_name=["vwap_distance"], factor_value=[-math.inf])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert report.negative_inf_rows == 1
    assert report.positive_inf_rows == 0
    diagnostic = report.diagnostics.invalid_numeric_diagnostics[0]
    assert diagnostic.kind == InvalidNumericKind.NEGATIVE_INFINITY


def test_verify_detects_underflow() -> None:
    """Subnormal floats are classified as underflow and fail."""
    subnormal = sys.float_info.min / 2.0
    frame = _frame(factor_value=[subnormal])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.invalid_numeric_rows == 1
    assert any(
        item.kind == InvalidNumericKind.UNDERFLOW
        for item in report.diagnostics.invalid_numeric_diagnostics
    )


def test_verify_detects_invalid_timestamp() -> None:
    """Non-positive timestamps fail verification."""
    frame = _frame(open_time=[0], factor_value=[0.1])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert report.invalid_timestamp_rows == 1


def test_verify_detects_metadata_and_consistency_issues() -> None:
    """Empty metadata and inconsistent version/category fail verification."""
    row_a = _frame(factor_version=["1.0.0"], factor_category=["price"])
    row_b = _frame(
        open_time=[_BASE_OPEN_TIME + _MS_PER_DAY],
        factor_version=["2.0.0"],
        factor_category=["volume"],
    )
    empty = _frame(
        open_time=[_BASE_OPEN_TIME + (2 * _MS_PER_DAY)],
        factor_name=["other"],
        factor_version=[""],
        factor_category=["price"],
        factor_group=["trend"],
    )
    frame = pl.concat([row_a, row_b, empty])
    report = FactorVerifier().verify(frame)
    assert report.passed is False
    assert any(warning.startswith("EMPTY_METADATA ") for warning in report.warnings)
    assert any(warning.startswith("INCONSISTENT_FACTOR_VERSION ") for warning in report.warnings)
    assert any(warning.startswith("INCONSISTENT_FACTOR_CATEGORY ") for warning in report.warnings)


def test_verify_detects_invalid_status() -> None:
    """Invalid FactorStatus values fail verification."""
    report = FactorVerifier().verify(_frame(status=["UNKNOWN"]))
    assert report.passed is False
    assert any(warning.startswith("INVALID_STATUS ") for warning in report.warnings)


def test_format_factor_diagnostics_debug_output() -> None:
    """Debug formatter distinguishes warmup, domain, unexpected, and infinity."""
    rsi = _rows(factor_name="rsi", values=[None, None, 0.5])
    width = _rows(factor_name="bollinger_width", values=[0.1, None])
    eom = _rows(factor_name="ease_of_movement", values=[0.1, None])
    inf_row = _frame(
        open_time=[_BASE_OPEN_TIME + (2 * _MS_PER_DAY)],
        factor_name=["vwap_distance"],
        factor_value=[math.inf],
    )
    frame = pl.concat([rsi, width, eom, inf_row]).sort(["open_time", "factor_name"])
    report = FactorVerifier().verify(frame)
    text = format_factor_diagnostics(report.diagnostics)
    assert "Factor: rsi" in text
    assert "Issue: Warmup NULLs" in text
    assert "Factor: bollinger_width" in text
    assert "Issue: Unexpected NULL" in text
    assert "Factor: ease_of_movement" in text
    assert "Issue: Domain NULL" in text
    assert "Factor: vwap_distance" in text
    assert "Positive infinity" in text
    assert "--------------------------------" in text
    assert "->" in text
    assert "\u2192" not in text
    assert text.isascii()


def test_collect_and_format_global_failure_report() -> None:
    """Global FAIL report locates Unexpected NULL and +Inf by partition."""
    width = _rows(factor_name="bollinger_width", values=[0.1, None])
    eom = _rows(factor_name="ease_of_movement", values=[0.1, None])
    inf_row = _frame(
        open_time=[_BASE_OPEN_TIME + (2 * _MS_PER_DAY)],
        factor_name=["vwap_distance"],
        factor_value=[math.inf],
    )
    frame = pl.concat([width, eom, inf_row]).sort(["open_time", "factor_name"])
    report = FactorVerifier().verify(frame)
    findings = collect_global_failure_findings(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        diagnostics=report.diagnostics,
    )
    assert len(findings) == 2
    assert findings[0].issue == "Unexpected NULL"
    assert findings[0].factor_name == "bollinger_width"
    assert findings[1].issue == "+Inf"
    assert findings[1].factor_name == "vwap_distance"

    text = format_global_failure_report(findings)
    assert "CQROS Factor Failure Report" in text
    assert "Symbol: BTCUSDT" in text
    assert "Timeframe: 1h" in text
    assert "Year: 2024" in text
    assert "Factor: bollinger_width" in text
    assert "Issue: Unexpected NULL" in text
    assert "Range:" in text
    assert "->" in text
    assert "\u2192" not in text
    assert text.isascii()
    assert "Factor: vwap_distance" in text
    assert "Issue: +Inf" in text
    assert "Timestamp:" in text
    assert "UTC" in text
    assert "Affected partitions: 1" in text
    assert "Affected symbols: 1" in text
    assert "Affected factors: 2" in text
    # Warmup and domain findings must not appear in the global FAIL report.
    warmup = _rows(factor_name="rsi", values=[None, None, 0.5])
    warmup_report = FactorVerifier().verify(warmup)
    warmup_findings = collect_global_failure_findings(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        diagnostics=warmup_report.diagnostics,
    )
    assert warmup_findings == ()
    domain = _rows(factor_name="volume_rate_of_change", values=[0.1, None, 0.2])
    domain_report = FactorVerifier().verify(domain)
    domain_findings = collect_global_failure_findings(
        symbol="BTCUSDT",
        timeframe="1h",
        year=2024,
        diagnostics=domain_report.diagnostics,
    )
    assert domain_findings == ()
    assert format_global_failure_report(()) == ""
