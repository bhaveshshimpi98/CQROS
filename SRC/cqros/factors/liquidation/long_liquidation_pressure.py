"""CQROS long liquidation pressure research factor.

Purpose:
    Compute the rolling mean of long liquidation volume as a pure
    liquidation alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``LongLiquidationPressureFactor`` metadata
    - Append a ``long_liquidation_pressure`` column using Polars expressions
      only
    - Fail fast on invalid lookback and missing ``long_liquidation_volume``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``LongLiquidationPressureFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["LongLiquidationPressureFactor"]

_LONG_LIQ_COLUMN: Final[str] = "long_liquidation_volume"
_OUTPUT_COLUMN: Final[str] = "long_liquidation_pressure"
_FACTOR_NAME: Final[str] = "long_liquidation_pressure"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "liquidation"
_FACTOR_DESCRIPTION: Final[str] = (
    "Long liquidation pressure as the rolling mean of long liquidation volume."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-LONG-LIQUIDATION-PRESSURE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-LONG-LIQUIDATION-PRESSURE-002"


@dataclass(frozen=True, slots=True)
class LongLiquidationPressureFactor(BaseFactor):
    """Long liquidation pressure alpha factor from rolling mean volume.

    Computes ``rolling_mean(long_liquidation_volume)`` over ``lookback`` and
    appends the result as ``long_liquidation_pressure``. Incomplete windows
    are null. Missing values are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``long_liquidation_pressure``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``liquidation``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling mean window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_LONG_LIQ_COLUMN,)
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
        """Append long liquidation pressure without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a
                ``long_liquidation_volume`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``long_liquidation_pressure``. Incomplete windows are null.

        Raises:
            FactorError: If ``long_liquidation_volume`` is not present in
                ``frame``.
        """
        if _LONG_LIQ_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_LONG_LIQ_COLUMN}",
                error_code=_ERROR_MISSING_COLUMN,
                details={
                    "factor": self.name,
                    "required_column": _LONG_LIQ_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        pressure_expr = pl.col(
            _LONG_LIQ_COLUMN
        ).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pressure_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
