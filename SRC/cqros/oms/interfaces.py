"""CQROS Order Management System public interfaces.

Purpose:
    Define structural contracts for order managers so every OMS
    implementation shares one public surface.

Responsibilities:
    - Expose ``OrderManager`` as the shared order-creation contract
    - Provide shared structural validation helpers for manager inputs
    - Remain free of order generation, execution, persistence, and trading

Dependencies:
    ``polars`` and ``cqros.oms.exceptions``.

Public API:
    ``OrderManager``, ``validate_risk_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.oms.exceptions import OMSValidationError

__all__ = [
    "OrderManager",
    "validate_risk_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "OMS_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "OMS_FRAME_EMPTY"


@runtime_checkable
class OrderManager(Protocol):
    """Structural contract for converting risk frames into order frames.

    Implementations own order-creation semantics (simple submission, TWAP,
    VWAP, and related manager strategies). Pipeline orchestration delegates
    order creation exclusively through this contract. Implementations must
    return a new DataFrame and must not mutate the input risk frame.
    """

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        """Convert a canonical risk DataFrame into an order DataFrame.

        Args:
            risk_decisions: Canonical risk-decision dataset. Must not be
                mutated.

        Returns:
            A new DataFrame containing the columns required by the OMS Order
            schema contract.
        """
        ...


def validate_risk_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate risk-decision dataset passed to an order manager.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        OMSValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise OMSValidationError(
            "frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise OMSValidationError(
            "frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    return frame
