"""Unit tests for CQROS processing verification protocol surface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from cqros.processing.verification import DataVerifier, VerificationReport
from cqros.processing.verification.interfaces import DataVerifier as DataVerifierFromModule


class _ConformingVerifier:
    """Minimal DataVerifier-shaped stub for protocol conformance tests."""

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        return VerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamp_rows=0,
            invalid_numeric_rows=0,
            warnings=(),
            passed=True,
        )


class _IncompleteVerifier:
    """Stub missing ``verify`` so DataVerifier conformance must fail."""

    def inspect(self, frame: pl.DataFrame) -> VerificationReport:
        return VerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamp_rows=0,
            invalid_numeric_rows=0,
            warnings=(),
            passed=True,
        )


def test_data_verifier_is_runtime_checkable_protocol() -> None:
    """DataVerifier is a runtime-checkable Protocol."""
    assert isinstance(DataVerifier, type)
    assert issubclass(DataVerifier, Protocol)
    assert getattr(DataVerifier, "_is_runtime_protocol", False) is True


def test_package_exports_data_verifier() -> None:
    """Package re-export matches the interfaces module symbol."""
    assert DataVerifier is DataVerifierFromModule


def test_conforming_verifier_satisfies_protocol() -> None:
    """A structurally complete stub satisfies DataVerifier."""
    verifier = _ConformingVerifier()
    assert isinstance(verifier, DataVerifier)
    frame = pl.DataFrame({"a": [1, 2]})
    report = verifier.verify(frame)
    assert report.rows_checked == 2
    assert report.passed is True


def test_incomplete_verifier_does_not_satisfy_protocol() -> None:
    """Missing verify prevents DataVerifier conformance."""
    assert not isinstance(_IncompleteVerifier(), DataVerifier)


def test_data_verifier_is_runtime_checkable_decorator() -> None:
    """DataVerifier carries the runtime_checkable marker."""
    assert runtime_checkable is not None
    assert isinstance(_ConformingVerifier(), DataVerifier)
