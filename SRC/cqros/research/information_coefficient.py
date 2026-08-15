"""CQROS Information Coefficient (IC) engine.

Purpose:
    Measure the predictive relationship between a factor column and a
    forward-return target column using Pearson or Spearman correlation.

Responsibilities:
    - Define immutable ``InformationCoefficientResult`` value objects
    - Compute IC and two-sided p-values via ``InformationCoefficient``
    - Fail fast on missing columns, unknown methods, and insufficient data
    - Remain free of trading, signals, backtests, execution, and ML logic

Dependencies:
    ``polars``, ``scipy``, and ``cqros.core.exceptions.ResearchError``.

Public API:
    ``InformationCoefficientResult``, ``InformationCoefficient``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import polars as pl
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

from cqros.core.exceptions import ResearchError

__all__ = [
    "InformationCoefficientResult",
    "InformationCoefficient",
]

CorrelationMethod = Literal["pearson", "spearman"]

_DEFAULT_METHOD: Final[CorrelationMethod] = "spearman"
_SUPPORTED_METHODS: Final[frozenset[str]] = frozenset({"pearson", "spearman"})
_MINIMUM_OBSERVATIONS: Final[int] = 2

_ERROR_UNKNOWN_METHOD: Final[str] = "RESEARCH-IC-001"
_ERROR_MISSING_FACTOR: Final[str] = "RESEARCH-IC-002"
_ERROR_MISSING_TARGET: Final[str] = "RESEARCH-IC-003"
_ERROR_INSUFFICIENT_OBS: Final[str] = "RESEARCH-IC-004"


@dataclass(frozen=True, slots=True)
class InformationCoefficientResult:
    """Immutable result of an Information Coefficient calculation.

    Attributes:
        factor_column: Factor column name used in the calculation.
        target_column: Target column name used in the calculation.
        method: Correlation method (``pearson`` or ``spearman``).
        observations: Number of paired non-null observations used.
        coefficient: Estimated correlation coefficient.
        p_value: Two-sided p-value for the coefficient.
    """

    factor_column: str
    target_column: str
    method: str
    observations: int
    coefficient: float
    p_value: float


class InformationCoefficient:
    """Statistical Information Coefficient calculator.

    Computes Pearson or Spearman correlation between a factor column and a
    forward-return target column, together with a two-sided p-value. Null
    pairs are dropped. The input DataFrame is never mutated.
    """

    __slots__ = ("_method",)

    def __init__(self, method: str = _DEFAULT_METHOD) -> None:
        """Initialize the IC calculator.

        Args:
            method: Correlation method. Supported values are ``pearson`` and
                ``spearman``. Defaults to ``spearman``.

        Raises:
            ResearchError: If ``method`` is not a supported correlation method.
        """
        if method not in _SUPPORTED_METHODS:
            raise ResearchError(
                f"unknown correlation method: {method}",
                error_code=_ERROR_UNKNOWN_METHOD,
                details={
                    "method": method,
                    "supported_methods": tuple(sorted(_SUPPORTED_METHODS)),
                },
            )
        self._method: CorrelationMethod = cast(CorrelationMethod, method)

    @property
    def method(self) -> CorrelationMethod:
        """Return the configured correlation method."""
        return self._method

    def compute(
        self,
        frame: pl.DataFrame,
        factor_column: str,
        target_column: str,
    ) -> InformationCoefficientResult:
        """Compute the Information Coefficient for two columns.

        Rows with a null in either column are dropped before estimation.
        At least two paired observations are required.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_column: Name of the factor column.
            target_column: Name of the forward-return target column.

        Returns:
            An immutable ``InformationCoefficientResult``.

        Raises:
            ResearchError: If either column is missing or fewer than two
                paired non-null observations remain.
        """
        if factor_column not in frame.columns:
            raise ResearchError(
                f"required column missing: {factor_column}",
                error_code=_ERROR_MISSING_FACTOR,
                details={
                    "required_column": factor_column,
                    "role": "factor",
                    "available_columns": tuple(frame.columns),
                },
            )
        if target_column not in frame.columns:
            raise ResearchError(
                f"required column missing: {target_column}",
                error_code=_ERROR_MISSING_TARGET,
                details={
                    "required_column": target_column,
                    "role": "target",
                    "available_columns": tuple(frame.columns),
                },
            )

        clean = frame.select(factor_column, target_column).drop_nulls()
        observations = clean.height
        if observations < _MINIMUM_OBSERVATIONS:
            raise ResearchError(
                "insufficient observations for information coefficient",
                error_code=_ERROR_INSUFFICIENT_OBS,
                details={
                    "factor_column": factor_column,
                    "target_column": target_column,
                    "observations": observations,
                    "minimum_observations": _MINIMUM_OBSERVATIONS,
                },
            )

        factor_values = clean.get_column(factor_column).to_numpy()
        target_values = clean.get_column(target_column).to_numpy()
        coefficient, p_value = _correlate(self._method, factor_values, target_values)

        return InformationCoefficientResult(
            factor_column=factor_column,
            target_column=target_column,
            method=self._method,
            observations=observations,
            coefficient=float(coefficient),
            p_value=float(p_value),
        )


def _correlate(
    method: CorrelationMethod,
    factor_values: object,
    target_values: object,
) -> tuple[float, float]:
    """Compute correlation coefficient and two-sided p-value via SciPy."""
    if method == "pearson":
        result = stats.pearsonr(  # pyright: ignore[reportUnknownMemberType]
            factor_values,
            target_values,
        )
    else:
        result = stats.spearmanr(  # pyright: ignore[reportUnknownMemberType]
            factor_values,
            target_values,
        )
    statistic = cast(float, result[0])  # pyright: ignore[reportUnknownMemberType]
    p_value = cast(float, result[1])  # pyright: ignore[reportUnknownMemberType]
    return float(statistic), float(p_value)
