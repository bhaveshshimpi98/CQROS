"""CQROS historical volatility research factor.

Purpose:
    Compute rolling historical volatility of log returns as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``HistoricalVolatilityFactor`` metadata
    - Append a ``historical_volatility`` column using Polars expressions only
    - Fail fast on invalid lookback, annualization, and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``HistoricalVolatilityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["HistoricalVolatilityFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "historical_volatility"
_FACTOR_NAME: Final[str] = "historical_volatility"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Annualized rolling standard deviation of logarithmic close returns."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-HISTORICAL-VOLATILITY-001"
_ERROR_ANNUALIZATION: Final[str] = "FACTOR-HISTORICAL-VOLATILITY-002"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-HISTORICAL-VOLATILITY-003"
_DEFAULT_ANNUALIZATION: Final[int] = 365


@dataclass(frozen=True, slots=True)
class HistoricalVolatilityFactor(BaseFactor):
    """Historical volatility alpha factor from rolling log-return std.

    Computes ``std(log(close / close.shift(1)), lookback) * sqrt(annualization)``
    with population standard deviation and appends the result as
    ``historical_volatility``. The first ``lookback`` rows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``historical_volatility``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling std window size (must be >= 2).
        annualization: Periods-per-year scale factor (must be >= 1).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20
    annualization: int = _DEFAULT_ANNUALIZATION

    def __post_init__(self) -> None:
        """Validate base metadata, lookback, and annualization.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback < 2`` or ``annualization < 1``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 2:
            raise ValidationError(
                "lookback must be an integer greater than or equal to 2",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )
        annualization = cast(object, self.annualization)
        if (
            not isinstance(annualization, int)
            or isinstance(annualization, bool)
            or annualization < 1
        ):
            raise ValidationError(
                "annualization must be an integer greater than or equal to 1",
                error_code=_ERROR_ANNUALIZATION,
                details={"parameter": "annualization", "value": self.annualization},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append historical volatility without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``historical_volatility``. The first ``lookback`` rows of
            ``historical_volatility`` are null.

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
        log_returns = (close / close.shift(1)).log()  # pyright: ignore[reportUnknownMemberType]
        rolling_std = log_returns.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        volatility_expr = rolling_std * (float(self.annualization) ** 0.5)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            volatility_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
