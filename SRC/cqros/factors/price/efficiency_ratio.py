"""CQROS Kaufman efficiency ratio research factor.

Purpose:
    Compute the Kaufman Efficiency Ratio from close prices as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``EfficiencyRatioFactor`` metadata
    - Append an ``efficiency_ratio`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``EfficiencyRatioFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["EfficiencyRatioFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "efficiency_ratio"
_FACTOR_NAME: Final[str] = "efficiency_ratio"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Kaufman Efficiency Ratio as absolute net movement over path length."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-EFFICIENCY-RATIO-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-EFFICIENCY-RATIO-002"


@dataclass(frozen=True, slots=True)
class EfficiencyRatioFactor(BaseFactor):
    """Kaufman Efficiency Ratio alpha factor from close prices.

    Computes ``|close - close.shift(lookback)| / sum(|close - close.shift(1)|)``
    over the trailing ``lookback`` one-period absolute changes and appends the
    result as ``efficiency_ratio``. Values lie in ``[0, 1]``. Returns null when
    the path-length denominator is zero. The first ``lookback`` rows are null.
    Missing values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``efficiency_ratio``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Efficiency ratio window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require lookback >= 2.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback < 2``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 2:
            raise ValidationError(
                "lookback must be an integer greater than or equal to 2",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append efficiency ratio without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``efficiency_ratio``. The first ``lookback`` rows of
            ``efficiency_ratio`` are null.

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
        net_movement = (
            close - close.shift(self.lookback)
        ).abs()  # pyright: ignore[reportUnknownMemberType]
        abs_change = (close - close.shift(1)).abs()  # pyright: ignore[reportUnknownMemberType]
        path_length = abs_change.rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        ratio_expr = (
            pl.when(path_length != 0).then(net_movement / path_length).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ratio_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
