"""CQROS leverage change research factor.

Purpose:
    Compute fractional open interest change as a pure liquidation alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``LeverageChangeFactor`` metadata
    - Append a ``leverage_change`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``open_interest``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``LeverageChangeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["LeverageChangeFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "leverage_change"
_FACTOR_NAME: Final[str] = "leverage_change"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "liquidation"
_FACTOR_DESCRIPTION: Final[str] = (
    "Leverage change as fractional open interest return over a lookback " "window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-LEVERAGE-CHANGE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-LEVERAGE-CHANGE-002"


@dataclass(frozen=True, slots=True)
class LeverageChangeFactor(BaseFactor):
    """Leverage change alpha factor from fractional open interest change.

    Computes ``(open_interest / open_interest.shift(lookback)) - 1`` and
    appends the result as ``leverage_change``. Returns null when the lagged
    open interest is zero. The first ``lookback`` rows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``leverage_change``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``liquidation``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Historical rows used for the change window (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_OI_COLUMN,)
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
        """Append leverage change without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing an ``open_interest``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``leverage_change``. The first ``lookback`` rows are null.

        Raises:
            FactorError: If ``open_interest`` is not present in ``frame``.
        """
        if _OI_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_OI_COLUMN}",
                error_code=_ERROR_MISSING_COLUMN,
                details={
                    "factor": self.name,
                    "required_column": _OI_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        oi = pl.col(_OI_COLUMN)
        lagged = oi.shift(self.lookback)  # pyright: ignore[reportUnknownMemberType]
        change_expr = (
            pl.when(lagged == 0).then(None).otherwise((oi / lagged) - 1)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            change_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
