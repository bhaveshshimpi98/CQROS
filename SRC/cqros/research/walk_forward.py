"""CQROS walk-forward validation for factor research.

Purpose:
    Evaluate research experiments across sequential train/test windows to
    measure out-of-sample factor robustness without optimization or fitting.

Responsibilities:
    - Define immutable ``WalkForwardWindow`` and ``WalkForwardResult``
    - Generate rolling train/test window bounds
    - Orchestrate repeated ``ResearchExperiment`` runs on each test slice
    - Summarize mean IC, Rank IC, quantile spread, and overall stability
    - Remain free of optimization, model fitting, plotting, and storage

Dependencies:
    ``polars``, the Python standard library, ``cqros.core.exceptions``, and
    ``cqros.research.experiment``.

Public API:
    ``WalkForwardWindow``, ``WalkForwardResult``, ``WalkForwardValidator``
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError
from cqros.research.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    ResearchExperiment,
)

__all__ = [
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardValidator",
]

_ERROR_TRAIN_SIZE_INVALID: Final[str] = "RESEARCH-WF-001"
_ERROR_TEST_SIZE_INVALID: Final[str] = "RESEARCH-WF-002"
_ERROR_STEP_SIZE_INVALID: Final[str] = "RESEARCH-WF-003"
_ERROR_INSUFFICIENT_ROWS: Final[str] = "RESEARCH-WF-004"


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """Immutable result for one walk-forward train/test window.

    Attributes:
        window_index: Zero-based window position in temporal order.
        train_start: Inclusive start row index of the train period.
        train_end: Exclusive end row index of the train period.
        test_start: Inclusive start row index of the test period.
        test_end: Exclusive end row index of the test period.
        experiment_result: ``ResearchExperiment`` result on the test slice.
    """

    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    experiment_result: ExperimentResult


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable walk-forward validation summary.

    Attributes:
        definition: Experiment definition evaluated across windows.
        windows: Per-window experiment results in temporal order.
        mean_ic: Mean Information Coefficient across windows.
        mean_rank_ic: Mean Rank Information Coefficient across windows.
        mean_quantile_spread: Mean top-minus-bottom quantile spread across
            windows.
        stability: Overall IC stability score across windows, clamped to
            ``[0, 1]`` using ``1 - std_ic / abs(mean_ic)``.
    """

    definition: ExperimentDefinition
    windows: tuple[WalkForwardWindow, ...]
    mean_ic: float
    mean_rank_ic: float
    mean_quantile_spread: float
    stability: float


class WalkForwardValidator:
    """Rolling walk-forward orchestrator for CQROS factor research.

    Generates sequential train/test windows, runs the injected
    ``ResearchExperiment`` on each out-of-sample test slice, and summarizes
    robustness metrics. Performs no optimization or model fitting. The input
    DataFrame is never mutated.
    """

    __slots__ = ("_experiment",)

    def __init__(self, experiment: ResearchExperiment) -> None:
        """Initialize with an injected research experiment engine.

        Args:
            experiment: Dependency-injected ``ResearchExperiment`` used for
                each window evaluation.
        """
        self._experiment = experiment

    def run(
        self,
        frame: pl.DataFrame,
        definition: ExperimentDefinition,
        train_size: int,
        test_size: int,
        step_size: int,
    ) -> WalkForwardResult:
        """Execute walk-forward validation across rolling windows.

        Window ``i`` uses train rows
        ``[i * step_size, i * step_size + train_size)`` and test rows
        ``[i * step_size + train_size, i * step_size + train_size + test_size)``.
        Only the test slice is evaluated by ``ResearchExperiment``.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            definition: Immutable experiment definition.
            train_size: Number of rows in each train period. Must be ``> 0``.
            test_size: Number of rows in each test period. Must be ``> 0``.
            step_size: Row advance between consecutive windows. Must be
                ``> 0``.

        Returns:
            An immutable ``WalkForwardResult`` with per-window results and
            summary statistics.

        Raises:
            ResearchError: If window sizes are invalid or the frame has too
                few rows for at least one complete train/test window.
        """
        validated_train = _validate_positive_size(train_size, parameter="train_size")
        validated_test = _validate_positive_size(test_size, parameter="test_size")
        validated_step = _validate_positive_size(step_size, parameter="step_size")
        _validate_sufficient_rows(
            row_count=frame.height,
            train_size=validated_train,
            test_size=validated_test,
        )

        windows: list[WalkForwardWindow] = []
        window_index = 0
        train_start = 0
        while True:
            train_end = train_start + validated_train
            test_start = train_end
            test_end = test_start + validated_test
            if test_end > frame.height:
                break
            test_frame = frame.slice(test_start, validated_test)
            experiment_result = self._experiment.run(test_frame, definition)
            windows.append(
                WalkForwardWindow(
                    window_index=window_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    experiment_result=experiment_result,
                )
            )
            window_index += 1
            train_start += validated_step

        window_tuple = tuple(windows)
        ic_values = tuple(_window_mean_ic(window) for window in window_tuple)
        rank_ic_values = tuple(_window_mean_rank_ic(window) for window in window_tuple)
        spread_values = tuple(_window_mean_quantile_spread(window) for window in window_tuple)
        mean_ic = float(statistics.fmean(ic_values))
        return WalkForwardResult(
            definition=definition,
            windows=window_tuple,
            mean_ic=mean_ic,
            mean_rank_ic=float(statistics.fmean(rank_ic_values)),
            mean_quantile_spread=float(statistics.fmean(spread_values)),
            stability=_stability_score(ic_values=ic_values, mean_ic=mean_ic),
        )


def _validate_positive_size(value: int, *, parameter: str) -> int:
    """Validate that a window size parameter is a positive integer."""
    if not isinstance(cast(object, value), int) or isinstance(value, bool) or value <= 0:
        error_code = {
            "train_size": _ERROR_TRAIN_SIZE_INVALID,
            "test_size": _ERROR_TEST_SIZE_INVALID,
            "step_size": _ERROR_STEP_SIZE_INVALID,
        }[parameter]
        raise ResearchError(
            f"{parameter} must be an integer greater than 0",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
    return value


def _validate_sufficient_rows(
    *,
    row_count: int,
    train_size: int,
    test_size: int,
) -> None:
    """Require enough rows for at least one complete train/test window."""
    minimum_rows = train_size + test_size
    if row_count < minimum_rows:
        raise ResearchError(
            "insufficient rows for walk-forward validation",
            error_code=_ERROR_INSUFFICIENT_ROWS,
            details={
                "row_count": row_count,
                "minimum_rows": minimum_rows,
                "train_size": train_size,
                "test_size": test_size,
            },
        )


def _window_mean_ic(window: WalkForwardWindow) -> float:
    """Average Information Coefficient across factors in one window."""
    coefficients = tuple(
        result.coefficient for result in window.experiment_result.information_coefficients
    )
    return float(statistics.fmean(coefficients))


def _window_mean_rank_ic(window: WalkForwardWindow) -> float:
    """Average Rank IC across factors in one window."""
    coefficients = tuple(
        result.coefficient for result in window.experiment_result.rank_information_coefficients
    )
    return float(statistics.fmean(coefficients))


def _window_mean_quantile_spread(window: WalkForwardWindow) -> float:
    """Average top-minus-bottom quantile spread across factors in one window."""
    spreads = tuple(result.top_minus_bottom for result in window.experiment_result.quantile_results)
    return float(statistics.fmean(spreads))


def _stability_score(*, ic_values: tuple[float, ...], mean_ic: float) -> float:
    """Compute overall walk-forward IC stability, clamped to ``[0, 1]``."""
    if mean_ic == 0.0:
        return 0.0
    std_ic = float(statistics.stdev(ic_values)) if len(ic_values) >= 2 else 0.0
    raw_score = 1.0 - (std_ic / abs(mean_ic))
    return max(0.0, min(1.0, float(raw_score)))
