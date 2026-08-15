"""CQROS Bollinger position research factor.

Purpose:
    Compute the position of close within Bollinger bands as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``BollingerPositionFactor`` metadata
    - Append a ``bollinger_position`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BollingerPositionFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BollingerPositionFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "bollinger_position"
_FACTOR_NAME: Final[str] = "bollinger_position"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Position of close within Bollinger bands over a lookback window."
_ERROR_LOOKBACK: Final[str] = "FACTOR-BOLLINGER-POSITION-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-BOLLINGER-POSITION-002"
_STD_MULTIPLIER: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class BollingerPositionFactor(BaseFactor):
    """Bollinger position alpha factor within rolling Bollinger bands.

    Computes ``(close - lower) / (upper - lower)`` where
    ``middle = close.rolling_mean(lookback)``,
    ``upper = middle + 2 * std``, and ``lower = middle - 2 * std`` with
    population standard deviation. Values lie in ``[0, 1]`` when close is
    inside the bands. Returns null when band width is zero. The first
    ``lookback - 1`` rows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``bollinger_position``).
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
        """Append Bollinger position without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``bollinger_position``. The first ``lookback - 1`` rows of
            ``bollinger_position`` are null.

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
        middle = close.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        std = close.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        upper = middle + _STD_MULTIPLIER * std
        lower = middle - _STD_MULTIPLIER * std
        width = upper - lower
        position_expr = (
            pl.when(width != 0).then((close - lower) / width).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            position_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
