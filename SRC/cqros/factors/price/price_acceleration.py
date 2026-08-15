"""CQROS price acceleration research factor.

Purpose:
    Compute the change in period momentum from the close price as a pure
    price acceleration alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``PriceAccelerationFactor`` metadata
    - Append a ``price_acceleration`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``PriceAccelerationFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["PriceAccelerationFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "price_acceleration"
_FACTOR_NAME: Final[str] = "price_acceleration"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Price acceleration as the change in lookback momentum between consecutive "
    "non-overlapping lookback windows."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-PRICE-ACCELERATION-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-PRICE-ACCELERATION-002"


@dataclass(frozen=True, slots=True)
class PriceAccelerationFactor(BaseFactor):
    """Price acceleration alpha factor from consecutive momentum windows.

    Computes
    ``((close / close.shift(lookback)) - 1)
    - ((close.shift(lookback) / close.shift(2 * lookback)) - 1)``
    and appends the result as ``price_acceleration``. The first
    ``2 * lookback`` rows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``price_acceleration``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rows in each momentum leg (must be > 0). Warm-up is
            ``2 * lookback`` rows.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require a strictly positive lookback.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback <= 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 1:
            raise ValidationError(
                "lookback must be an integer greater than 0",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append price acceleration without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``price_acceleration``. The first ``2 * lookback`` rows of
            ``price_acceleration`` are null.

        Raises:
            FactorError: If ``close`` is not present in ``frame``.
        """
        if _CLOSE_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_CLOSE_COLUMN}",
                error_code=_ERROR_MISSING_CLOSE,
                details={
                    "factor": self.name,
                    "required_column": _CLOSE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        close = pl.col(_CLOSE_COLUMN)
        current_momentum = (close / close.shift(self.lookback)) - 1
        prior_momentum = (close.shift(self.lookback) / close.shift(2 * self.lookback)) - 1
        acceleration_expr = (
            current_momentum - prior_momentum
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            acceleration_expr.alias(_OUTPUT_COLUMN)
        )
