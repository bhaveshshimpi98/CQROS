"""CQROS signed volume research factor.

Purpose:
    Compute rolling signed taker volume as a pure microstructure alpha factor
    for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``SignedVolumeFactor`` metadata
    - Append a ``signed_volume`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``SignedVolumeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["SignedVolumeFactor"]

_BUY_COLUMN: Final[str] = "taker_buy_volume"
_SELL_COLUMN: Final[str] = "taker_sell_volume"
_OUTPUT_COLUMN: Final[str] = "signed_volume"
_FACTOR_NAME: Final[str] = "signed_volume"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Signed volume as rolling sum of taker buy volume minus taker sell volume."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-SIGNED-VOLUME-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-SIGNED-VOLUME-002"


@dataclass(frozen=True, slots=True)
class SignedVolumeFactor(BaseFactor):
    """Signed volume alpha factor from taker buy and sell volumes.

    Computes ``sum(taker_buy_volume - taker_sell_volume, lookback)`` and
    appends the result as ``signed_volume``. Incomplete windows are null.
    Missing values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``signed_volume``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_BUY_COLUMN, _SELL_COLUMN)
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
        """Append signed volume without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``taker_buy_volume`` and
                ``taker_sell_volume`` columns.

        Returns:
            A new DataFrame with all original columns plus ``signed_volume``.
            Incomplete windows are null.

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

        signed = pl.col(_BUY_COLUMN) - pl.col(_SELL_COLUMN)
        signed_expr = signed.rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signed_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
