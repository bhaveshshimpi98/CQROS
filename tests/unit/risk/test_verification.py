"""Unit tests for CQROS risk ``RiskVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import inf, nan
from typing import cast

import polars as pl
import pytest

from cqros.processing.verification.interfaces import DataVerifier
from cqros.risk import RiskDecision, RiskVerifier
from cqros.risk.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_RISK_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.risk.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    RiskValidationError,
    VerificationReport,
)
from cqros.risk.verification import RiskVerifier as RiskVerifierFromPackage
from cqros.risk.verification.verifier import RiskVerifier as RiskVerifierFromModule

_START = datetime(2024, 1, 1, tzinfo=UTC)
_INTERVAL = timedelta(hours=1)
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER = "equal_weight"
_POLICY = "fixed_risk"


def _open_times(count: int = 3) -> list[datetime]:
    """Build monotonically increasing UTC open_time values."""
    return [_START + (index * _INTERVAL) for index in range(count)]


def _risk_frame(
    *,
    symbols: list[str | None] | None = None,
    timeframes: list[str | None] | None = None,
    open_times: list[datetime | None] | None = None,
    model_names: list[str | None] | None = None,
    model_versions: list[str | None] | None = None,
    optimizers: list[str | None] | None = None,
    policies: list[str | None] | None = None,
    signals: list[str | None] | None = None,
    target_weights: list[float | None] | None = None,
    approved_weights: list[float | None] | None = None,
    decisions: list[str | None] | None = None,
    reasons: list[str | None] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged risk-decision verification frame."""
    times = open_times if open_times is not None else _open_times()
    row_count = len(times)
    default_signals = ["BUY", "SELL", "HOLD"]
    while len(default_signals) < row_count:
        default_signals.append("HOLD")
    default_weights = [0.5, -0.5, 0.0]
    while len(default_weights) < row_count:
        default_weights.append(0.0)
    default_decisions = [
        RiskDecision.APPROVE.value,
        RiskDecision.RESIZE.value,
        RiskDecision.REJECT.value,
    ]
    while len(default_decisions) < row_count:
        default_decisions.append(RiskDecision.APPROVE.value)
    default_reasons = ["ok", "resized", "rejected"]
    while len(default_reasons) < row_count:
        default_reasons.append("ok")
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": times,
        "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
        "model_version": (
            model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
        ),
        "optimizer": (optimizers if optimizers is not None else [_OPTIMIZER] * row_count),
        "policy": policies if policies is not None else [_POLICY] * row_count,
        "signal": signals if signals is not None else default_signals[:row_count],
        "target_weight": (
            target_weights if target_weights is not None else default_weights[:row_count]
        ),
        "approved_weight": (
            approved_weights if approved_weights is not None else default_weights[:row_count]
        ),
        "decision": (decisions if decisions is not None else default_decisions[:row_count]),
        "reason": reasons if reasons is not None else default_reasons[:row_count],
    }
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=dict(COLUMN_DTYPES))
    return frame.select(list(order))


def _verifier() -> RiskVerifier:
    """Build a RiskVerifier instance."""
    return RiskVerifier()


def _assert_clean_pass(report: VerificationReport, *, rows: int) -> None:
    """Assert a fully passing report for ``rows`` checked."""
    assert report == VerificationReport(
        rows_checked=rows,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warnings=(),
        passed=True,
    )


def test_package_exports_risk_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert RiskVerifier is RiskVerifierFromModule
    assert RiskVerifierFromPackage is RiskVerifierFromModule


def test_risk_verifier_satisfies_data_verifier_protocol() -> None:
    """RiskVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged risk frame passes verification."""
    report = _verifier().verify(_risk_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=MERGED_RISK_SCHEMA).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_canonical_schema_columns() -> None:
    """Successful verification inspects the canonical Risk Decision column set."""
    frame = _risk_frame()
    assert frame.columns == list(CANONICAL_COLUMN_ORDER)
    assert frame.columns == [
        "symbol",
        "timeframe",
        "open_time",
        "model_name",
        "model_version",
        "optimizer",
        "policy",
        "signal",
        "target_weight",
        "approved_weight",
        "decision",
        "reason",
    ]
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_risk_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time) keys fail verification."""
    times = [_START, _START, _START + _INTERVAL]
    frame = _risk_frame(open_times=times)
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_symbols() -> None:
    """Identical open_time values for distinct symbols do not count as duplicates."""
    frame = _risk_frame(
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        open_times=[_START, _START, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise RiskValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "model_name" in missing
    assert "optimizer" in missing
    assert "policy" in missing
    assert "approved_weight" in missing
    assert "decision" in missing
    assert "reason" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_missing_optimizer_column_raises() -> None:
    """Missing optimizer produces the same hard schema failure as any required column."""
    frame = _risk_frame().drop("optimizer")
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert missing == ("optimizer",)


def test_missing_policy_column_raises() -> None:
    """Missing policy produces the same hard schema failure as any required column."""
    frame = _risk_frame().drop("policy")
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert missing == ("policy",)


def test_null_values() -> None:
    """NULL values in required columns are counted and fail verification."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE.value, None, RiskDecision.REJECT.value],
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _risk_frame(symbols=["BTCUSDT", None, "BTCUSDT"])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_timestamp_rows() -> None:
    """NULL open_time values count as both null and invalid timestamp rows."""
    frame = _risk_frame(open_times=[_START, None, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid timestamps detected." in report.warnings


def test_nan_weight_rows() -> None:
    """NaN weight values are counted and fail verification."""
    frame = _risk_frame(
        target_weights=[0.5, nan, 0.0],
        approved_weights=[0.5, 0.4, nan],
    )
    report = _verifier().verify(frame)
    assert report.nan_rows == 2
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_infinite_target_weight_rows() -> None:
    """Infinite target_weight values are counted and fail verification."""
    frame = _risk_frame(target_weights=[0.5, inf, -inf])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.passed is False
    assert "Invalid target weight values detected." in report.warnings


def test_infinite_approved_weight_rows() -> None:
    """Infinite approved_weight values are counted and fail verification."""
    frame = _risk_frame(approved_weights=[0.5, inf, 0.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid approved weight values detected." in report.warnings


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _risk_frame(
        open_times=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by open_time.",)


def test_invalid_decision_values() -> None:
    """Decision values outside APPROVE/RESIZE/REJECT fail verification."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE.value, "ALLOW", RiskDecision.REJECT.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid decision values detected." in report.warnings


def test_invalid_decision_values_use_risk_decision_enum() -> None:
    """Only RiskDecision enum values are accepted in the decision column."""
    frame = _risk_frame(
        decisions=[
            RiskDecision.APPROVE.value,
            RiskDecision.RESIZE.value,
            RiskDecision.REJECT.value,
        ],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)

    frame = _risk_frame(decisions=["approve", "resize", "reject"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 3
    assert report.passed is False
    assert "Invalid decision values detected." in report.warnings


def test_empty_decision_values() -> None:
    """Empty decision strings fail verification."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE.value, "", RiskDecision.REJECT.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty decision values detected." in report.warnings
    assert "Invalid decision values detected." in report.warnings


def test_empty_reason_values() -> None:
    """Empty reason strings fail verification."""
    frame = _risk_frame(reasons=["ok", "", "rejected"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty reason values detected." in report.warnings


def test_incorrect_column_order() -> None:
    """Wrong column order fails verification with a deterministic warning."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _risk_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises RiskValidationError schema mismatch."""
    frame = _risk_frame().with_columns(pl.col("open_time").cast(pl.Int64))
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_approved_weight() -> None:
    """Non-Float64 approved_weight columns raise RiskValidationError."""
    frame = _risk_frame().with_columns(pl.col("approved_weight").cast(pl.Float32))
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "approved_weight" in mismatched


def test_dtype_mismatch_decision_column() -> None:
    """Non-Utf8 decision columns raise RiskValidationError."""
    frame = _risk_frame().with_columns(pl.col("decision").cast(pl.Categorical))
    with pytest.raises(RiskValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "decision" in mismatched


def test_does_not_enforce_policy_rules() -> None:
    """Verifier does not enforce institutional risk-policy thresholds."""
    frame = _risk_frame(
        decisions=[
            RiskDecision.APPROVE.value,
            RiskDecision.APPROVE.value,
            RiskDecision.APPROVE.value,
        ],
        target_weights=[0.9, 0.9, 0.9],
        approved_weights=[0.9, 0.9, 0.9],
        reasons=["ok", "ok", "ok"],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _risk_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        decisions=[RiskDecision.APPROVE.value, None, "ALLOW"],
        reasons=["ok", "", "bad"],
        target_weights=[0.5, nan, inf],
        approved_weights=[0.5, 0.4, -inf],
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_report_values_combined() -> None:
    """Combined failures populate report fields correctly."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _risk_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        decisions=[RiskDecision.APPROVE.value, None, "ALLOW"],
        reasons=["ok", "", "bad"],
        target_weights=[0.5, nan, inf],
        approved_weights=[0.5, 0.4, -inf],
        column_order=reordered,
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows >= 1
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Invalid decision values detected." in report.warnings
    assert "Empty reason values detected." in report.warnings
    assert "Invalid target weight values detected." in report.warnings
    assert "Invalid approved weight values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)


def test_risk_verifier_exported_from_root_package() -> None:
    """Risk package re-exports RiskVerifier."""
    from cqros import risk as risk_package

    assert "RiskVerifier" in risk_package.__all__
    assert risk_package.RiskVerifier is RiskVerifierFromModule
