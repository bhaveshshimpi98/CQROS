"""CQROS recovery strength research factor.

Purpose:
    Compute recovery of close from the rolling low toward the rolling high
    as a pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``RecoveryStrengthFactor`` metadata
    - Append a ``recovery_strength`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RecoveryStrengthFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RecoveryStrengthFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "recovery_strength"
_FACTOR_NAME: Final[str] = "recovery_strength"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Recovery strength as the position of close within the rolling high-low " "range."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-RECOVERY-STRENGTH-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-RECOVERY-STRENGTH-002"


@dataclass(frozen=True, slots=True)
class RecoveryStrengthFactor(BaseFactor):
    """Recovery strength alpha factor within the rolling high-low range.

    Computes ``(close - rolling_low) / (rolling_high - rolling_low)`` where
    ``rolling_high = close.rolling_max(lookback)`` and
    ``rolling_low = close.rolling_min(lookback)``. When the range is zero the
    output is null. The first ``lookback - 1`` rows are null. Missing values
    are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``recovery_strength``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling range window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
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
        """Append recovery strength without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``recovery_strength``. The first ``lookback - 1`` rows of
            ``recovery_strength`` are null.

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
        rolling_high = close.rolling_max(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        rolling_low = close.rolling_min(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        range_expr = rolling_high - rolling_low
        recovery_expr = (
            pl.when(range_expr != 0).then((close - rolling_low) / range_expr).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            recovery_expr.alias(_OUTPUT_COLUMN)
        )
