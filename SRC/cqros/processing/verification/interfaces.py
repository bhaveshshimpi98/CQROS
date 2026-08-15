"""CQROS processing verification public interfaces.

Purpose:
    Define structural contracts for dataset verifiers so every verification
    implementation shares one public surface.

Responsibilities:
    - Expose ``DataVerifier`` as the shared verification contract
    - Remain free of verification logic, storage, and orchestration

Dependencies:
    ``polars``, the Python standard library, and
    ``cqros.processing.verification.report``.

Public API:
    ``DataVerifier``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from cqros.processing.verification.report import VerificationReport

__all__ = [
    "DataVerifier",
]


@runtime_checkable
class DataVerifier(Protocol):
    """Structural contract for a deterministic dataset verification pass.

    Implementations inspect a DataFrame and return an immutable
    ``VerificationReport``. Verifiers must not mutate the caller-supplied
    frame and must not perform cleaning, feature engineering, or I/O.
    """

    def verify(
        self,
        frame: pl.DataFrame,
    ) -> VerificationReport:
        """Verify ``frame`` without mutating it.

        Args:
            frame: Input market DataFrame. Must not be mutated.

        Returns:
            An immutable ``VerificationReport`` describing the pass.
        """
        ...
