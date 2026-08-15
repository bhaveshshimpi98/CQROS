"""CQROS Chaikin Money Flow research factor.

Purpose:
    Compute Chaikin Money Flow from OHLC and volume as a pure volume alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``ChaikinMoneyFlowFactor`` metadata
    - Append a ``chaikin_money_flow`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``ChaikinMoneyFlowFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["ChaikinMoneyFlowFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "chaikin_money_flow"
_FACTOR_NAME: Final[str] = "chaikin_money_flow"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "volume"
_FACTOR_DESCRIPTION: Final[str] = (
    "Chaikin Money Flow as rolling money-flow volume over rolling volume."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-CHAIKIN-MONEY-FLOW-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-CHAIKIN-MONEY-FLOW-002"


@dataclass(frozen=True, slots=True)
class ChaikinMoneyFlowFactor(BaseFactor):
    """Chaikin Money Flow alpha factor from OHLC and volume.

    Computes the money-flow multiplier
    ``((close - low) - (high - close)) / (high - low)``, forms money-flow
    volume as multiplier times volume, then appends
    ``sum(MFV, lookback) / sum(volume, lookback)`` as ``chaikin_money_flow``.
    When ``high == low`` the multiplier is ``0``. Returns null when the
    rolling volume sum is zero. Incomplete windows are null. Missing values
    are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``chaikin_money_flow``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``volume``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: CMF window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (
        _HIGH_COLUMN,
        _LOW_COLUMN,
        _CLOSE_COLUMN,
        _VOLUME_COLUMN,
    )
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
        """Append Chaikin Money Flow without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``,
                ``close``, and ``volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``chaikin_money_flow``. Incomplete windows are null.

        Raises:
            FactorError: If a required column is not present in ``frame``.
        """
        for column in self.required_features:
            if column not in frame.columns:
                raise FactorError(
                    f"required column missing: {column}",
                    error_code=_ERROR_MISSING_COLUMN,
                    details={
                        "factor": self.name,
                        "required_column": column,
                        "available_columns": tuple(frame.columns),
                    },
                )

        high = pl.col(_HIGH_COLUMN)
        low = pl.col(_LOW_COLUMN)
        close = pl.col(_CLOSE_COLUMN)
        volume = pl.col(_VOLUME_COLUMN)
        range_hl = high - low
        money_flow_multiplier = (
            pl.when(range_hl == 0).then(0.0).otherwise(((close - low) - (high - close)) / range_hl)
        )
        money_flow_volume = money_flow_multiplier * volume
        mfv_sum = money_flow_volume.rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        volume_sum = volume.rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        cmf_expr = (
            pl.when(volume_sum != 0).then(mfv_sum / volume_sum).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            cmf_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
