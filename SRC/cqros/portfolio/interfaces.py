"""CQROS Portfolio public interfaces.

Purpose:
    Define structural contracts for portfolio optimizers so every
    optimization algorithm shares one public surface.

Responsibilities:
    - Expose ``PortfolioOptimizer`` as the shared optimization contract
    - Provide shared structural validation helpers for optimizer inputs
    - Remain free of allocation algorithms, persistence, and trading logic

Dependencies:
    ``polars`` and ``cqros.portfolio.exceptions``.

Public API:
    ``PortfolioOptimizer``, ``validate_signals_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.portfolio.exceptions import PortfolioValidationError

__all__ = [
    "PortfolioOptimizer",
    "validate_signals_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "PORTFOLIO_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PORTFOLIO_FRAME_EMPTY"


@runtime_checkable
class PortfolioOptimizer(Protocol):
    """Structural contract for converting signal frames into portfolio frames.

    Implementations own optimization semantics (equal weight, fixed weight,
    and related allocation strategies). Pipeline orchestration delegates
    optimization exclusively through this contract. Implementations must
    return a new DataFrame and must not mutate the input signal frame.
    """

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        """Convert a canonical signal DataFrame into a portfolio DataFrame.

        Args:
            signals: Canonical signal dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by the Portfolio
            schema contract.
        """
        ...


def validate_signals_frame(signals: object) -> pl.DataFrame:
    """Validate that ``signals`` is a non-empty Polars DataFrame.

    Args:
        signals: Candidate signal dataset passed to an optimizer.

    Returns:
        ``signals`` as a DataFrame after structural checks.

    Raises:
        PortfolioValidationError: If ``signals`` is not a Polars DataFrame
            or contains no rows.
    """
    if not isinstance(signals, pl.DataFrame):
        raise PortfolioValidationError(
            "signals must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(signals).__name__},
        )
    if signals.height == 0:
        raise PortfolioValidationError(
            "signals must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": signals.height},
        )
    return signals
