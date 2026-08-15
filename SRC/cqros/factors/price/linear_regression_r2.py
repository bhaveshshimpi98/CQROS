"""CQROS linear regression R² research factor.

Purpose:
    Compute the rolling coefficient of determination between a relative time
    index and log(close) as a pure price alpha factor for the Factor Research
    Engine.

Responsibilities:
    - Expose immutable ``LinearRegressionR2Factor`` metadata
    - Append a ``linear_regression_r2`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``LinearRegressionR2Factor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["LinearRegressionR2Factor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "linear_regression_r2"
_FACTOR_NAME: Final[str] = "linear_regression_r2"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling R-squared of log(close) regressed on a relative time index."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-LINEAR-REGRESSION-R2-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-LINEAR-REGRESSION-R2-002"


@dataclass(frozen=True, slots=True)
class LinearRegressionR2Factor(BaseFactor):
    """Linear regression R² alpha factor from rolling OLS on log(close).

    Fits ``log(close) ~ a + b * x`` over each trailing ``lookback`` window
    where ``x`` is the relative index ``0 .. lookback-1``, and appends the
    coefficient of determination as ``linear_regression_r2``. Returns null
    when total sum of squares is zero. The first ``lookback - 1`` rows are
    null. Missing values are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``linear_regression_r2``).
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
        """Append linear regression R² without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``linear_regression_r2``. The first ``lookback - 1`` rows of
            ``linear_regression_r2`` are null.

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
        sum_y2 = (y * y).rolling_sum(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        sum_ty = (time_index * y).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        t_start = time_index - window + 1
        sum_xy = sum_ty - t_start * sum_y
        sum_x = (window - 1) * window / 2.0
        sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
        denom = window * sum_x2 - sum_x * sum_x
        slope = (window * sum_xy - sum_x * sum_y) / denom
        ss_xx = denom / window
        ss_yy = sum_y2 - (sum_y * sum_y) / window
        # Degenerate windows (constant log-price) have undefined R².
        y_range = y.rolling_max(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        ) - y.rolling_min(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        r2_expr = (
            pl.when(y_range > 0).then((slope * slope) * ss_xx / ss_yy).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            r2_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
