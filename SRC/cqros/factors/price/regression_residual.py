"""CQROS regression residual research factor.

Purpose:
    Compute the residual of current log(close) versus the rolling OLS fitted
    value as a pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``RegressionResidualFactor`` metadata
    - Append a ``regression_residual`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RegressionResidualFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RegressionResidualFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "regression_residual"
_FACTOR_NAME: Final[str] = "regression_residual"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Residual of current log(close) minus the rolling OLS fitted value."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-REGRESSION-RESIDUAL-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-REGRESSION-RESIDUAL-002"


@dataclass(frozen=True, slots=True)
class RegressionResidualFactor(BaseFactor):
    """Regression residual alpha factor from rolling OLS on log(close).

    Fits ``log(close) ~ a + b * x`` over each trailing ``lookback`` window
    where ``x`` is the relative index ``0 .. lookback-1``, and appends
    ``log(close) - fitted`` at the window end as ``regression_residual``.
    The first ``lookback - 1`` rows are null. Missing values are never
    filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``regression_residual``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling OLS window size (must be >= 2).
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
        """Append regression residual without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``regression_residual``. The first ``lookback - 1`` rows of
            ``regression_residual`` are null.

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

        window = self.lookback
        y = pl.col(_CLOSE_COLUMN).log()  # pyright: ignore[reportUnknownMemberType]
        time_index = pl.int_range(0, pl.len())
        sum_y = y.rolling_sum(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        mean_y = y.rolling_mean(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        sum_ty = (time_index * y).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        t_start = time_index - window + 1
        sum_xy = sum_ty - t_start * sum_y
        sum_x = (window - 1) * window / 2.0
        sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
        denom = window * sum_x2 - sum_x * sum_x
        slope = (window * sum_xy - sum_x * sum_y) / denom
        # Fitted value at relative x = lookback - 1.
        half_span = (window - 1) / 2.0
        residual_expr = y - mean_y - slope * half_span
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            residual_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
