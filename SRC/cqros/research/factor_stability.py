"""CQROS rolling factor stability analysis.

Purpose:
    Measure whether a factor's predictive power remains consistent through
    time by estimating Information Coefficient in consecutive windows.

Responsibilities:
    - Define immutable ``StabilityWindow`` and ``FactorStabilityResult``
    - Split a research frame into non-overlapping windows
    - Estimate IC per window via ``InformationCoefficient``
    - Summarize mean/std/min/max IC and a clamped stability score
    - Remain free of execution, backtesting, and machine-learning logic

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.core.exceptions.ResearchError``, and
    ``cqros.research.information_coefficient``.

Public API:
    ``StabilityWindow``, ``FactorStabilityResult``, ``FactorStabilityAnalyzer``
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError
from cqros.research.information_coefficient import InformationCoefficient

__all__ = [
    "StabilityWindow",
    "FactorStabilityResult",
    "FactorStabilityAnalyzer",
]

_DEFAULT_METHOD: Final[str] = "spearman"
_DEFAULT_WINDOW_SIZE: Final[int] = 500

_ERROR_WINDOW_SIZE_INVALID: Final[str] = "RESEARCH-STABILITY-001"
_ERROR_MISSING_FACTOR: Final[str] = "RESEARCH-STABILITY-002"
_ERROR_MISSING_TARGET: Final[str] = "RESEARCH-STABILITY-003"
_ERROR_INSUFFICIENT_OBS: Final[str] = "RESEARCH-STABILITY-004"


@dataclass(frozen=True, slots=True)
class StabilityWindow:
    """Immutable IC estimate for one consecutive stability window.

    Attributes:
        window_index: Zero-based window position in temporal order.
        start_row: Inclusive start row index in the input frame.
        end_row: Exclusive end row index in the input frame.
        observations: Number of paired non-null observations used for IC.
        coefficient: Information coefficient in the window.
        p_value: Two-sided p-value for the coefficient.
    """

    window_index: int
    start_row: int
    end_row: int
    observations: int
    coefficient: float
    p_value: float


@dataclass(frozen=True, slots=True)
class FactorStabilityResult:
    """Immutable rolling factor stability analysis result.

    Attributes:
        factor_column: Factor column analyzed.
        target_column: Forward-return target column analyzed.
        method: Correlation method used for IC estimation.
        window_size: Non-overlapping window length in rows.
        windows: Per-window IC estimates in temporal order.
        mean_ic: Mean window Information Coefficient.
        std_ic: Sample standard deviation of window IC values.
        min_ic: Minimum window Information Coefficient.
        max_ic: Maximum window Information Coefficient.
        stability_score: Clamped score ``1 - std_ic / abs(mean_ic)``, or
            ``0`` when ``mean_ic`` is zero.
    """

    factor_column: str
    target_column: str
    method: str
    window_size: int
    windows: tuple[StabilityWindow, ...]
    mean_ic: float
    std_ic: float
    min_ic: float
    max_ic: float
    stability_score: float


class FactorStabilityAnalyzer:
    """Rolling non-overlapping factor stability analyzer.

    The analyzer splits the input frame into consecutive windows of fixed
    length, estimates Information Coefficient in each window, and summarizes
    IC consistency through time. The input DataFrame is never mutated.
    """

    __slots__ = ("_calculator",)

    def __init__(self, method: str = _DEFAULT_METHOD) -> None:
        """Initialize the stability analyzer.

        Args:
            method: Correlation method forwarded to ``InformationCoefficient``.
                Defaults to ``spearman``.

        Raises:
            ResearchError: If ``method`` is not supported by
                ``InformationCoefficient``.
        """
        self._calculator = InformationCoefficient(method=method)

    @property
    def method(self) -> str:
        """Return the configured IC correlation method."""
        return self._calculator.method

    def analyze(
        self,
        frame: pl.DataFrame,
        factor_column: str,
        target_column: str,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> FactorStabilityResult:
        """Analyze factor IC stability across consecutive windows.

        Only complete non-overlapping windows of ``window_size`` rows are
        evaluated. Trailing rows that do not fill a full window are ignored.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_column: Factor column to evaluate.
            target_column: Forward-return target column.
            window_size: Number of rows per window. Must be ``>= 2``.
                Defaults to ``500``.

        Returns:
            An immutable ``FactorStabilityResult``.

        Raises:
            ResearchError: If ``window_size`` is invalid, a required column is
                missing, or fewer than ``window_size`` rows are available.
        """
        validated_window_size = _validate_window_size(window_size)
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
        if frame.height < validated_window_size:
            raise ResearchError(
                "insufficient observations for factor stability analysis",
                error_code=_ERROR_INSUFFICIENT_OBS,
                details={
                    "observations": frame.height,
                    "minimum_observations": validated_window_size,
                    "window_size": validated_window_size,
                },
            )

        window_count = frame.height // validated_window_size
        windows = tuple(
            _stability_window_for_slice(
                frame=frame,
                factor_column=factor_column,
                target_column=target_column,
                window_index=window_index,
                start_row=window_index * validated_window_size,
                end_row=(window_index + 1) * validated_window_size,
                calculator=self._calculator,
            )
            for window_index in range(window_count)
        )
        coefficients = tuple(window.coefficient for window in windows)
        mean_ic = float(statistics.fmean(coefficients))
        std_ic = float(statistics.stdev(coefficients)) if len(coefficients) >= 2 else 0.0
        return FactorStabilityResult(
            factor_column=factor_column,
            target_column=target_column,
            method=self._calculator.method,
            window_size=validated_window_size,
            windows=windows,
            mean_ic=mean_ic,
            std_ic=std_ic,
            min_ic=float(min(coefficients)),
            max_ic=float(max(coefficients)),
            stability_score=_stability_score(mean_ic=mean_ic, std_ic=std_ic),
        )


def _validate_window_size(window_size: int) -> int:
    """Validate and return the configured window size."""
    if (
        not isinstance(cast(object, window_size), int)
        or isinstance(window_size, bool)
        or window_size < 2
    ):
        raise ResearchError(
            "window_size must be an integer greater than or equal to 2",
            error_code=_ERROR_WINDOW_SIZE_INVALID,
            details={"parameter": "window_size", "value": window_size},
        )
    return window_size


def _stability_window_for_slice(
    *,
    frame: pl.DataFrame,
    factor_column: str,
    target_column: str,
    window_index: int,
    start_row: int,
    end_row: int,
    calculator: InformationCoefficient,
) -> StabilityWindow:
    """Estimate IC for one inclusive-start exclusive-end window slice."""
    window_frame = frame.slice(start_row, end_row - start_row)
    ic_result = calculator.compute(window_frame, factor_column, target_column)
    return StabilityWindow(
        window_index=window_index,
        start_row=start_row,
        end_row=end_row,
        observations=ic_result.observations,
        coefficient=ic_result.coefficient,
        p_value=ic_result.p_value,
    )


def _stability_score(*, mean_ic: float, std_ic: float) -> float:
    """Compute the clamped IC stability score."""
    if mean_ic == 0.0:
        return 0.0
    raw_score = 1.0 - (std_ic / abs(mean_ic))
    return max(0.0, min(1.0, float(raw_score)))
