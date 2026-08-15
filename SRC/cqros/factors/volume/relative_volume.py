"""CQROS relative volume research factor.

Purpose:
    Compute relative volume versus its rolling mean as a pure volume alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``RelativeVolumeFactor`` metadata
    - Append a ``relative_volume`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``volume``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RelativeVolumeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RelativeVolumeFactor"]

_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "relative_volume"
_FACTOR_NAME: Final[str] = "relative_volume"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "volume"
_FACTOR_DESCRIPTION: Final[str] = "Relative volume as current volume divided by its rolling mean."
_ERROR_LOOKBACK: Final[str] = "FACTOR-RELATIVE-VOLUME-001"
_ERROR_MISSING_VOLUME: Final[str] = "FACTOR-RELATIVE-VOLUME-002"


@dataclass(frozen=True, slots=True)
class RelativeVolumeFactor(BaseFactor):
    """Relative volume alpha factor from traded volume.

    Computes ``volume / rolling_mean(volume, lookback)`` and appends the
    result as ``relative_volume``. Returns null when the rolling mean is
    zero. Incomplete windows are null. Missing values are never filled.
    The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``relative_volume``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``volume``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling mean window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_VOLUME_COLUMN,)
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
        """Append relative volume without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``volume`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``relative_volume``. Incomplete windows are null.

        Raises:
            FactorError: If ``volume`` is not present in ``frame``.
        """
        if _VOLUME_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_VOLUME_COLUMN}",
                error_code=_ERROR_MISSING_VOLUME,
                details={
                    "factor": self.name,
                    "required_column": _VOLUME_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        volume = pl.col(_VOLUME_COLUMN)
        mean = volume.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        rvol_expr = (
            pl.when(mean != 0).then(volume / mean).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            rvol_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
