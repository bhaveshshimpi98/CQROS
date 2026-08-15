"""CQROS VWAP z-score research factor.

Purpose:
    Compute the rolling z-score of close versus provided VWAP as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``VWAPZScoreFactor`` metadata
    - Append a ``vwap_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``VWAPZScoreFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.rolling_zscore import rolling_zscore_expr

__all__ = ["VWAPZScoreFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_VWAP_COLUMN: Final[str] = "vwap"
_OUTPUT_COLUMN: Final[str] = "vwap_zscore"
_FACTOR_NAME: Final[str] = "vwap_zscore"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling z-score of close-versus-VWAP basis versus its mean and std."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-VWAP-ZSCORE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-VWAP-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class VWAPZScoreFactor(BaseFactor):
    """VWAP z-score alpha factor from the close-versus-VWAP basis.

    Computes basis ``close - vwap``, then
    ``(basis - rolling_mean) / rolling_std`` using population standard
    deviation (``ddof=0``). Returns ``0.0`` when rolling standard deviation is
    zero. Incomplete windows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This factor consumes the provided ``vwap`` column and does not recompute
    VWAP from OHLC.

    Attributes:
        name: Stable factor identifier (``vwap_zscore``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN, _VWAP_COLUMN)
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
        """Append VWAP z-score without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``close`` and ``vwap``.

        Returns:
            A new DataFrame with all original columns plus ``vwap_zscore``.
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

        basis = pl.col(_CLOSE_COLUMN) - pl.col(_VWAP_COLUMN)
        zscore_expr = rolling_zscore_expr(basis, window_size=self.lookback)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
