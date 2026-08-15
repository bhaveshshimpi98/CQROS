"""CQROS processing verification package public API.

Purpose:
    Expose the verification foundation, concrete dataset verifiers, and the
    processed-market verification runner.

Responsibilities:
    - Re-export ``VerificationReport``, ``DataVerifier``, implemented dataset
      verifiers, and runner orchestration types
    - Remain free of unverified dataset verifier implementations

Public API:
    ``VerificationReport``, ``DataVerifier``, dataset verifiers,
    ``VerificationRunner``, ``VerificationSummary``, and
    ``VerificationTaskResult``
"""

from cqros.processing.verification.funding import FundingVerifier
from cqros.processing.verification.interfaces import DataVerifier
from cqros.processing.verification.long_short import LongShortVerifier
from cqros.processing.verification.ohlcv import OHLCVVerifier
from cqros.processing.verification.open_interest import OpenInterestVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.processing.verification.runner import (
    VerificationRunner,
    VerificationSummary,
    VerificationTaskResult,
)
from cqros.processing.verification.taker_volume import TakerVolumeVerifier

__all__ = [
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
