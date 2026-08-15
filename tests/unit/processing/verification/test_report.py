"""Unit tests for CQROS processing ``VerificationReport``."""

from __future__ import annotations

import pytest

from cqros.processing.verification import VerificationReport
from cqros.processing.verification.exceptions import (
    ERROR_COUNTER_INVALID,
    ERROR_ROWS_CHECKED_INVALID,
    ERROR_WARNINGS_INVALID,
    ProcessingValidationError,
)
from cqros.processing.verification.report import VerificationReport as ReportFromModule


def _valid_report(**overrides: object) -> VerificationReport:
    """Build a valid VerificationReport with optional field overrides."""
    values: dict[str, object] = {
        "rows_checked": 10,
        "duplicate_timestamp_rows": 0,
        "null_rows": 0,
        "nan_rows": 0,
        "invalid_timestamp_rows": 0,
        "invalid_numeric_rows": 0,
        "warnings": (),
        "passed": True,
    }
    values.update(overrides)
    return VerificationReport(**values)  # type: ignore[arg-type]


def test_valid_construction() -> None:
    """A fully valid report stores all supplied fields."""
    report = VerificationReport(
        rows_checked=5,
        duplicate_timestamp_rows=1,
        null_rows=2,
        nan_rows=0,
        invalid_timestamp_rows=1,
        invalid_numeric_rows=0,
        warnings=("duplicate timestamps", "null rows"),
        passed=True,
    )
    assert report.rows_checked == 5
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 2
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 1
    assert report.invalid_numeric_rows == 0
    assert report.warnings == ("duplicate timestamps", "null rows")
    assert report.passed is True


def test_zero_rows_checked_is_valid() -> None:
    """Zero rows_checked is an allowed boundary."""
    report = _valid_report(rows_checked=0)
    assert report.rows_checked == 0


def test_report_is_frozen() -> None:
    """VerificationReport rejects attribute mutation."""
    report = _valid_report()
    with pytest.raises(AttributeError):
        report.passed = False  # type: ignore[misc]


def test_package_and_module_export_same_type() -> None:
    """Package re-export matches the report module symbol."""
    assert VerificationReport is ReportFromModule


@pytest.mark.parametrize(
    "field_name",
    (
        "duplicate_timestamp_rows",
        "null_rows",
        "nan_rows",
        "invalid_timestamp_rows",
        "invalid_numeric_rows",
    ),
)
def test_negative_counters_raise(field_name: str) -> None:
    """Negative counter fields raise ProcessingValidationError."""
    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(**{field_name: -1})
    assert exc_info.value.error_code == ERROR_COUNTER_INVALID
    assert dict(exc_info.value.details)["parameter"] == field_name


def test_negative_rows_checked_raises() -> None:
    """Negative rows_checked raises ProcessingValidationError."""
    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(rows_checked=-1)
    assert exc_info.value.error_code == ERROR_ROWS_CHECKED_INVALID
    assert dict(exc_info.value.details)["parameter"] == "rows_checked"


@pytest.mark.parametrize("bad_value", (True, 1.5, "0", None))
def test_non_integer_rows_checked_raises(bad_value: object) -> None:
    """Non-integer rows_checked values are rejected."""
    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(rows_checked=bad_value)
    assert exc_info.value.error_code == ERROR_ROWS_CHECKED_INVALID


@pytest.mark.parametrize("bad_value", (True, 1.5, "1", None))
def test_non_integer_counters_raise(bad_value: object) -> None:
    """Non-integer counter values are rejected."""
    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(null_rows=bad_value)
    assert exc_info.value.error_code == ERROR_COUNTER_INVALID


def test_warnings_must_be_tuple_of_str() -> None:
    """warnings must be tuple[str, ...]; lists and mixed tuples fail."""
    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(warnings=["a"])  # type: ignore[arg-type]
    assert exc_info.value.error_code == ERROR_WARNINGS_INVALID

    with pytest.raises(ProcessingValidationError) as exc_info:
        _valid_report(warnings=("ok", 1))  # type: ignore[arg-type]
    assert exc_info.value.error_code == ERROR_WARNINGS_INVALID


def test_empty_warnings_tuple_is_valid() -> None:
    """An empty warnings tuple is accepted."""
    report = _valid_report(warnings=())
    assert report.warnings == ()


def test_passed_true() -> None:
    """passed=True is stored as supplied by the verifier."""
    report = _valid_report(passed=True)
    assert report.passed is True


def test_passed_false() -> None:
    """passed=False is stored as supplied and not derived from counters."""
    report = _valid_report(
        duplicate_timestamp_rows=3,
        null_rows=1,
        passed=False,
    )
    assert report.passed is False
    assert report.duplicate_timestamp_rows == 3
    assert report.null_rows == 1


def test_passed_true_with_nonzero_counters_is_allowed() -> None:
    """passed is never auto-computed from counters."""
    report = _valid_report(null_rows=5, passed=True)
    assert report.passed is True
    assert report.null_rows == 5
