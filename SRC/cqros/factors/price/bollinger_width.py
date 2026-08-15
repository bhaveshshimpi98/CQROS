"""CQROS Bollinger width research factor.

Purpose:
    Compute Bollinger band width from close prices as a pure price alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``BollingerWidthFactor`` metadata
    - Append a ``bollinger_width`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BollingerWidthFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BollingerWidthFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "bollinger_width"
_FACTOR_NAME: Final[str] = "bollinger_width"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Bollinger band width as upper band minus lower band."
_ERROR_LOOKBACK: Final[str] = "FACTOR-BOLLINGER-WIDTH-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-BOLLINGER-WIDTH-002"
_STD_MULTIPLIER: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class BollingerWidthFactor(BaseFactor):
    """Bollinger width alpha factor from rolling Bollinger bands.

    Computes ``upper - lower`` where ``middle = close.rolling_mean(lookback)``,
    ``upper = middle + 2 * std``, and ``lower = middle - 2 * std`` with
    population standard deviation. Equivalent to ``4 * std``. The first
    ``lookback - 1`` rows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``bollinger_width``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Bollinger window size (must be >= 2).
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
        """Append Bollinger width without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``bollinger_width``. The first ``lookback - 1`` rows of
            ``bollinger_width`` are null.

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
        std = close.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        width_expr = (2.0 * _STD_MULTIPLIER) * std
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            width_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
