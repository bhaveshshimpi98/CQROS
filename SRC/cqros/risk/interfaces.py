"""CQROS Risk Management public interfaces.

Purpose:
    Define structural contracts for risk managers so every risk-policy
    implementation shares one public surface.

Responsibilities:
    - Expose ``RiskManager`` as the shared risk-evaluation contract
    - Provide shared structural validation helpers for manager inputs
    - Remain free of policy logic, calculations, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.risk.exceptions``.

Public API:
    ``RiskManager``, ``validate_portfolio_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.risk.exceptions import RiskValidationError

__all__ = [
    "RiskManager",
    "validate_portfolio_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "RISK_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "RISK_FRAME_EMPTY"


@runtime_checkable
class RiskManager(Protocol):
    """Structural contract for converting portfolio frames into risk frames.

    Implementations own risk semantics (approval, resizing, rejection, and
    related policy strategies). Pipeline orchestration delegates evaluation
    exclusively through this contract. Implementations must return a new
    DataFrame and must not mutate the input portfolio frame.
    """

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        """Convert a canonical portfolio DataFrame into a risk DataFrame.

        Args:
            portfolios: Canonical portfolio dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by the Risk
            Decision schema contract.
        """
        ...


def validate_portfolio_frame(portfolios: object) -> pl.DataFrame:
    """Validate that ``portfolios`` is a non-empty Polars DataFrame.

    Args:
        portfolios: Candidate portfolio dataset passed to a risk manager.

    Returns:
        ``portfolios`` as a DataFrame after structural checks.

    Raises:
        RiskValidationError: If ``portfolios`` is not a Polars DataFrame
            or contains no rows.
    """
    if not isinstance(portfolios, pl.DataFrame):
        raise RiskValidationError(
            "portfolios must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(portfolios).__name__},
        )
    if portfolios.height == 0:
        raise RiskValidationError(
            "portfolios must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": portfolios.height},
        )
    return portfolios
