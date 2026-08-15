"""CQROS relative strength versus BTC research factor.

Purpose:
    Compute rolling cumulative-return relative strength of an asset versus
    Bitcoin returns as a pure relative-value alpha factor for the Factor
    Research Engine.

Responsibilities:
    - Expose immutable ``RelativeStrengthBTCFactor`` metadata
    - Append a ``relative_strength_btc`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RelativeStrengthBTCFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RelativeStrengthBTCFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_BTC_COLUMN: Final[str] = "btc_return"
_OUTPUT_COLUMN: Final[str] = "relative_strength_btc"
_FACTOR_NAME: Final[str] = "relative_strength_btc"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Relative strength as asset cumulative return minus BTC cumulative return "
    "over a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-RELATIVE-STRENGTH-BTC-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-RELATIVE-STRENGTH-BTC-002"


def _rolling_cumulative_return(column: pl.Expr, lookback: int) -> pl.Expr:
    """Return compounded cumulative return over a trailing lookback window."""
    return (1 + column).log().rolling_sum(  # pyright: ignore[reportUnknownMemberType]
        window_size=lookback
    ).exp() - 1


@dataclass(frozen=True, slots=True)
class RelativeStrengthBTCFactor(BaseFactor):
    """Relative strength alpha factor versus Bitcoin returns.

    Computes rolling compounded cumulative returns
    ``product(1 + r) - 1`` for ``asset_return`` and ``btc_return``, then
    appends their difference as ``relative_strength_btc``. Incomplete
    windows are null. Missing values are never filled. The input DataFrame
    is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``relative_strength_btc``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``relative``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling cumulative-return window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_ASSET_COLUMN, _BTC_COLUMN)
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
        """Append BTC relative strength without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``btc_return`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``relative_strength_btc``. Incomplete windows are null.

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

        asset_cum = _rolling_cumulative_return(pl.col(_ASSET_COLUMN), self.lookback)
        btc_cum = _rolling_cumulative_return(pl.col(_BTC_COLUMN), self.lookback)
        strength_expr = asset_cum - btc_cum
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            strength_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
