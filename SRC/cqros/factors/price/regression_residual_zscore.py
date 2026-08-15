"""CQROS regression residual z-score research factor.

Purpose:
    Compute the z-score of the rolling OLS residual of log(close) as a pure
    price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``RegressionResidualZScoreFactor`` metadata
    - Append a ``regression_residual_zscore`` column using Polars expressions
      only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RegressionResidualZScoreFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RegressionResidualZScoreFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "regression_residual_zscore"
_FACTOR_NAME: Final[str] = "regression_residual_zscore"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Z-score of the residual from rolling OLS of log(close) on time."
_ERROR_LOOKBACK: Final[str] = "FACTOR-REGRESSION-RESIDUAL-ZSCORE-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-REGRESSION-RESIDUAL-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class RegressionResidualZScoreFactor(BaseFactor):
    """Regression residual z-score alpha factor from rolling OLS on log(close).

    Fits ``log(close) ~ a + b * x`` over each trailing ``lookback`` window
    where ``x`` is the relative index ``0 .. lookback-1``, computes the end-
    of-window residual, and divides by the sample residual standard deviation
    within the same window. Returns ``0.0`` when residual variance is zero.
    The first ``lookback - 1`` rows are null. Missing values are never filled.
    The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``regression_residual_zscore``).
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
        """Append regression residual z-score without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``regression_residual_zscore``. The first ``lookback - 1`` rows of
            ``regression_residual_zscore`` are null.

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
        half_span = (window - 1) / 2.0
        residual = y - mean_y - slope * half_span
        ss_xx = denom / window
        ss_yy = sum_y2 - (sum_y * sum_y) / window
        # Clip tiny negative FP noise; degenerate windows remain undefined.
        ss_res = pl.max_horizontal(
            ss_yy - (slope * slope) * ss_xx,
            pl.lit(0.0),
        )
        sigma = (ss_res / (window - 1)).sqrt()  # pyright: ignore[reportUnknownMemberType]
        y_range = y.rolling_max(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        ) - y.rolling_min(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        # Warmup: y_range is null. Zero variance / degenerate fit: return 0.0.
        # max_horizontal(ss_res, 0) can coerce warmup nulls to 0.0 sigma, so
        # completeness is gated on y_range rather than sigma alone.
        zscore_expr = (
            pl.when(y_range.is_null())
            .then(None)
            .when((y_range > 0) & (sigma > 0))
            .then(residual / sigma)
            .otherwise(0.0)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )
