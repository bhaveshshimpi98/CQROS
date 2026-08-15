"""CQROS Ulcer Index research factor.

Purpose:
    Compute the Ulcer Index downside risk measure from close prices as a
    pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``UlcerIndexFactor`` metadata
    - Append an ``ulcer_index`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``UlcerIndexFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["UlcerIndexFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "ulcer_index"
_FACTOR_NAME: Final[str] = "ulcer_index"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Ulcer Index as the rolling root-mean-square of percent drawdowns from peak."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-ULCER-INDEX-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-ULCER-INDEX-002"


@dataclass(frozen=True, slots=True)
class UlcerIndexFactor(BaseFactor):
    """Ulcer Index alpha factor from rolling downside drawdowns.

    Computes percent drawdown ``100 * (close / rolling_max(close) - 1)`` and
    appends the rolling root-mean-square of squared drawdowns as
    ``ulcer_index``. Returns null when the rolling peak is zero. Incomplete
    windows are null. Missing values are never filled. The input DataFrame
    is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``ulcer_index``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Ulcer Index window size (must be >= 2).
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
        """Append Ulcer Index without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus ``ulcer_index``.
            Incomplete windows of ``ulcer_index`` are null.

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
        rolling_max = close.rolling_max(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        percent_drawdown = (
            pl.when(rolling_max != 0).then(100.0 * (close / rolling_max - 1.0)).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        ulcer_expr = (
            percent_drawdown.pow(2)  # pyright: ignore[reportUnknownMemberType]
            .rolling_mean(window_size=self.lookback)  # pyright: ignore[reportUnknownMemberType]
            .sqrt()  # pyright: ignore[reportUnknownMemberType]
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ulcer_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
