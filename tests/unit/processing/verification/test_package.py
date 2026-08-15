"""Unit tests for CQROS processing verification package exports."""

from __future__ import annotations

import cqros.processing.verification as verification_package
from cqros.processing.verification import (
    DataVerifier,
    FundingVerifier,
    LongShortVerifier,
    OHLCVVerifier,
    OpenInterestVerifier,
    TakerVolumeVerifier,
    VerificationReport,
    VerificationRunner,
    VerificationSummary,
    VerificationTaskResult,
)


def test_package_exports() -> None:
    """verification package exports the public verification surface."""
    assert verification_package.__all__ == [
        "DataVerifier",
        "FundingVerifier",
        "LongShortVerifier",
        "OHLCVVerifier",
        "OpenInterestVerifier",
        "TakerVolumeVerifier",
        "VerificationReport",
        "VerificationRunner",
        "VerificationSummary",
        "VerificationTaskResult",
    ]
    assert verification_package.VerificationReport is VerificationReport
    assert verification_package.DataVerifier is DataVerifier
    assert verification_package.FundingVerifier is FundingVerifier
    assert verification_package.LongShortVerifier is LongShortVerifier
    assert verification_package.OHLCVVerifier is OHLCVVerifier
    assert verification_package.OpenInterestVerifier is OpenInterestVerifier
    assert verification_package.TakerVolumeVerifier is TakerVolumeVerifier
    assert verification_package.VerificationRunner is VerificationRunner
    assert verification_package.VerificationSummary is VerificationSummary
    assert verification_package.VerificationTaskResult is VerificationTaskResult
