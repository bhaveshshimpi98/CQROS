"""CQROS cross-factor correlation analysis.

Purpose:
    Measure redundancy between multiple research factors by estimating an
    NxN pairwise correlation matrix.

Responsibilities:
    - Define immutable ``FactorCorrelationResult`` value objects
    - Estimate Pearson or Spearman factor correlation matrices
    - Identify highly correlated unique factor pairs
    - Remain free of machine learning, portfolio construction, and execution

Dependencies:
    ``numpy``, ``polars``, ``scipy``, and ``cqros.core.exceptions.ResearchError``.

Public API:
    ``FactorCorrelationResult``, ``FactorCorrelationAnalyzer``,
    ``find_highly_correlated``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
import polars as pl
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

from cqros.core.exceptions import ResearchError

__all__ = [
    "FactorCorrelationResult",
    "FactorCorrelationAnalyzer",
    "find_highly_correlated",
]

CorrelationMethod = Literal["pearson", "spearman"]

_DEFAULT_METHOD: Final[CorrelationMethod] = "spearman"
_SUPPORTED_METHODS: Final[frozenset[str]] = frozenset({"pearson", "spearman"})
_MINIMUM_FACTORS: Final[int] = 2
_MINIMUM_OBSERVATIONS: Final[int] = 2
_DEFAULT_THRESHOLD: Final[float] = 0.90

_ERROR_UNKNOWN_METHOD: Final[str] = "RESEARCH-CORR-001"
_ERROR_TOO_FEW_FACTORS: Final[str] = "RESEARCH-CORR-002"
_ERROR_MISSING_COLUMN: Final[str] = "RESEARCH-CORR-003"
_ERROR_INSUFFICIENT_OBS: Final[str] = "RESEARCH-CORR-004"
_ERROR_THRESHOLD_INVALID: Final[str] = "RESEARCH-CORR-005"


@dataclass(frozen=True, slots=True)
class FactorCorrelationResult:
    """Immutable cross-factor correlation analysis result.

    Attributes:
        factor_names: Ordered factor column names corresponding to matrix axes.
        method: Correlation method used (``pearson`` or ``spearman``).
        matrix: Symmetric NxN correlation matrix as nested immutable tuples.
    """

    factor_names: tuple[str, ...]
    method: str
    matrix: tuple[tuple[float, ...], ...]


class FactorCorrelationAnalyzer:
    """Cross-factor correlation matrix calculator.

    Estimates a symmetric Pearson or Spearman correlation matrix over the
    selected factor columns after dropping any row with a null in those
    columns. The input DataFrame is never mutated.
    """

    __slots__ = ("_method",)

    def __init__(self, method: str = _DEFAULT_METHOD) -> None:
        """Initialize the correlation analyzer.

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

    def analyze(
        self,
        frame: pl.DataFrame,
        factor_columns: Sequence[str],
    ) -> FactorCorrelationResult:
        """Compute the pairwise factor correlation matrix.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_columns: Factor column names to correlate. At least two
                names are required.

        Returns:
            An immutable ``FactorCorrelationResult`` with a symmetric matrix
            whose diagonal entries are exactly ``1.0``.

        Raises:
            ResearchError: If fewer than two factors are supplied, any column
                is missing, or fewer than two complete rows remain after
                null dropping.
        """
        names = _validate_factor_columns(frame, factor_columns)
        clean = frame.select(names).drop_nulls()
        if clean.height < _MINIMUM_OBSERVATIONS:
            raise ResearchError(
                "insufficient observations for factor correlation analysis",
                error_code=_ERROR_INSUFFICIENT_OBS,
                details={
                    "observations": clean.height,
                    "minimum_observations": _MINIMUM_OBSERVATIONS,
                    "factor_columns": names,
                },
            )

        values = clean.to_numpy()
        raw_matrix = _correlation_matrix(self._method, values)
        matrix = _freeze_correlation_matrix(raw_matrix)
        return FactorCorrelationResult(
            factor_names=names,
            method=self._method,
            matrix=matrix,
        )


def find_highly_correlated(
    result: FactorCorrelationResult,
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[tuple[str, str, float], ...]:
    """Return unique factor pairs at or above an absolute correlation threshold.

    Args:
        result: Correlation result produced by ``FactorCorrelationAnalyzer``.
        threshold: Absolute correlation cutoff in ``[0, 1]``. Defaults to
            ``0.90``.

    Returns:
        A tuple of ``(factor_a, factor_b, correlation)`` triples for all unique
        pairs ``i < j`` whose absolute correlation is ``>= threshold``.
        Pairs are ordered by ``(factor_a, factor_b)``.

    Raises:
        ResearchError: If ``threshold`` is outside ``[0, 1]``.
    """
    validated_threshold = _validate_threshold(threshold)
    pairs: list[tuple[str, str, float]] = []
    names = result.factor_names
    size = len(names)
    for left in range(size):
        for right in range(left + 1, size):
            coefficient = result.matrix[left][right]
            if abs(coefficient) >= validated_threshold:
                pairs.append((names[left], names[right], float(coefficient)))
    return tuple(pairs)


def _validate_factor_columns(
    frame: pl.DataFrame,
    factor_columns: Sequence[str],
) -> tuple[str, ...]:
    """Validate and freeze the requested factor column names."""
    names = tuple(factor_columns)
    if len(names) < _MINIMUM_FACTORS:
        raise ResearchError(
            "factor_columns must contain at least two factor names",
            error_code=_ERROR_TOO_FEW_FACTORS,
            details={"factor_columns": names, "count": len(names)},
        )
    missing = tuple(name for name in names if name not in frame.columns)
    if missing:
        raise ResearchError(
            f"required column missing: {missing[0]}",
            error_code=_ERROR_MISSING_COLUMN,
            details={
                "required_column": missing[0],
                "missing_columns": missing,
                "available_columns": tuple(frame.columns),
            },
        )
    return names


def _validate_threshold(threshold: float) -> float:
    """Validate that ``threshold`` lies in ``[0, 1]``."""
    if (
        not isinstance(cast(object, threshold), (int, float))
        or isinstance(threshold, bool)
        or threshold < 0.0
        or threshold > 1.0
    ):
        raise ResearchError(
            "threshold must be a number in the closed interval [0, 1]",
            error_code=_ERROR_THRESHOLD_INVALID,
            details={"parameter": "threshold", "value": threshold},
        )
    return float(threshold)


def _correlation_matrix(method: CorrelationMethod, values: object) -> object:
    """Estimate an NxN correlation matrix via NumPy/SciPy."""
    array = np.asarray(values, dtype=float)
    if method == "pearson":
        return np.corrcoef(array, rowvar=False)
    result = stats.spearmanr(array)  # pyright: ignore[reportUnknownMemberType]
    statistic = cast(object, result[0])  # pyright: ignore[reportUnknownMemberType]
    # SciPy returns a scalar for exactly two columns.
    if isinstance(statistic, (float, int, np.floating)):
        coefficient = float(cast(float | int, statistic))
        return np.array(
            [[1.0, coefficient], [coefficient, 1.0]],
            dtype=float,
        )
    return statistic


def _freeze_correlation_matrix(raw_matrix: object) -> tuple[tuple[float, ...], ...]:
    """Convert a numeric matrix into nested tuples with unit diagonal."""
    array = np.asarray(raw_matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ResearchError(
            "correlation matrix must be square",
            error_code=_ERROR_INSUFFICIENT_OBS,
            details={"shape": tuple(int(dimension) for dimension in array.shape)},
        )
    size = int(array.shape[0])
    np.fill_diagonal(array, 1.0)
    return tuple(tuple(float(array[row, column]) for column in range(size)) for row in range(size))
