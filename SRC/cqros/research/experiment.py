"""CQROS Research Experiment Engine.

Purpose:
    Orchestrate complete evaluation of one or more research factors by
    composing existing research analyzers into a single experiment run.

Responsibilities:
    - Define immutable ``ExperimentDefinition`` and ``ExperimentResult``
    - Coordinate target generation, IC, Rank IC, quantile, decay, stability,
      and correlation analyses through injected dependencies
    - Remain free of statistical calculations, logging, storage, reporting,
      plotting, machine learning, and optimization

Dependencies:
    ``polars``, the Python standard library, ``cqros.core.exceptions``,
    ``cqros.core.types``, and the composed CQROS research modules.

Public API:
    ``ExperimentDefinition``, ``ExperimentResult``, ``ResearchExperiment``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError
from cqros.core.types import Timestamp
from cqros.research.factor_correlation import (
    FactorCorrelationAnalyzer,
    FactorCorrelationResult,
    find_highly_correlated,
)
from cqros.research.factor_decay import FactorDecayAnalyzer, FactorDecayResult
from cqros.research.factor_stability import FactorStabilityAnalyzer, FactorStabilityResult
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)
from cqros.research.quantile_analysis import QuantileAnalysisResult, QuantileAnalyzer
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient
from cqros.research.target import ForwardReturnTarget, TargetDefinition

__all__ = [
    "ExperimentDefinition",
    "ExperimentResult",
    "ResearchExperiment",
]

_DEFAULT_PRICE_COLUMN: Final[str] = "close"
_DEFAULT_TARGET_HORIZON: Final[int] = 1
_DEFAULT_IC_METHOD: Final[str] = "spearman"
_DEFAULT_QUANTILES: Final[int] = 5
_DEFAULT_DECAY_HORIZONS: Final[tuple[int, ...]] = (1, 2, 4, 8, 12, 24)
_DEFAULT_STABILITY_WINDOW: Final[int] = 500
_DEFAULT_CORRELATION_THRESHOLD: Final[float] = 0.90

_ERROR_NAME_BLANK: Final[str] = "RESEARCH-EXPERIMENT-001"
_ERROR_DESCRIPTION_BLANK: Final[str] = "RESEARCH-EXPERIMENT-002"
_ERROR_NO_FACTORS: Final[str] = "RESEARCH-EXPERIMENT-003"
_ERROR_MISSING_FACTOR: Final[str] = "RESEARCH-EXPERIMENT-004"
_ERROR_MISSING_PRICE: Final[str] = "RESEARCH-EXPERIMENT-005"
_ERROR_TARGET_MISMATCH: Final[str] = "RESEARCH-EXPERIMENT-006"


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Immutable definition of a CQROS factor research experiment.

    Attributes:
        name: Stable experiment identifier.
        description: Human-readable experiment summary.
        factor_columns: Factor columns to evaluate.
        price_column: Price column used for forward-return generation.
        target_horizon: Forward-return horizon in rows for the primary target.
        ic_method: Intended IC correlation method metadata.
        quantiles: Intended quantile count metadata.
        decay_horizons: Forward horizons evaluated by decay analysis.
        stability_window: Non-overlapping window size for stability analysis.
        correlation_threshold: Absolute correlation cutoff for pair detection.
    """

    name: str
    description: str
    factor_columns: tuple[str, ...]
    price_column: str = _DEFAULT_PRICE_COLUMN
    target_horizon: int = _DEFAULT_TARGET_HORIZON
    ic_method: str = _DEFAULT_IC_METHOD
    quantiles: int = _DEFAULT_QUANTILES
    decay_horizons: tuple[int, ...] = _DEFAULT_DECAY_HORIZONS
    stability_window: int = _DEFAULT_STABILITY_WINDOW
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD

    def __post_init__(self) -> None:
        """Normalize collections and validate definition invariants."""
        if not isinstance(cast(object, self.name), str) or self.name.strip() == "":
            raise ResearchError(
                "name must be a non-blank string",
                error_code=_ERROR_NAME_BLANK,
                details={"parameter": "name", "value": self.name},
            )
        if not isinstance(cast(object, self.description), str) or self.description.strip() == "":
            raise ResearchError(
                "description must be a non-blank string",
                error_code=_ERROR_DESCRIPTION_BLANK,
                details={"parameter": "description", "value": self.description},
            )
        factors = tuple(self.factor_columns)
        if len(factors) == 0:
            raise ResearchError(
                "factor_columns must contain at least one factor",
                error_code=_ERROR_NO_FACTORS,
                details={"factor_columns": factors},
            )
        object.__setattr__(self, "factor_columns", factors)
        object.__setattr__(self, "decay_horizons", tuple(self.decay_horizons))


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Immutable result of a complete CQROS factor research experiment.

    Attributes:
        definition: Experiment definition that produced this result.
        target: Forward-return target definition used for primary evaluation.
        information_coefficients: Per-factor IC results in definition order.
        rank_information_coefficients: Per-factor Rank IC results.
        quantile_results: Per-factor quantile analysis results.
        decay_results: Per-factor decay analysis results.
        stability_results: Per-factor stability analysis results.
        correlation_result: Cross-factor correlation matrix, or ``None`` when
            fewer than two factors were evaluated.
        highly_correlated_pairs: Unique pairs at or above the correlation
            threshold.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution finished.
        duration_seconds: Wall-clock duration in seconds.
    """

    definition: ExperimentDefinition
    target: TargetDefinition
    information_coefficients: tuple[InformationCoefficientResult, ...]
    rank_information_coefficients: tuple[RankICResult, ...]
    quantile_results: tuple[QuantileAnalysisResult, ...]
    decay_results: tuple[FactorDecayResult, ...]
    stability_results: tuple[FactorStabilityResult, ...]
    correlation_result: FactorCorrelationResult | None
    highly_correlated_pairs: tuple[tuple[str, str, float], ...]
    started_at: Timestamp
    completed_at: Timestamp
    duration_seconds: float


class ResearchExperiment:
    """Dependency-injected orchestrator for institutional factor research.

    ``ResearchExperiment`` performs no statistical calculations itself. It
    delegates target generation and all analyses to injected collaborators and
    assembles an immutable ``ExperimentResult``.
    """

    __slots__ = (
        "_forward_return_target",
        "_information_coefficient",
        "_rank_information_coefficient",
        "_quantile_analyzer",
        "_factor_decay_analyzer",
        "_factor_stability_analyzer",
        "_factor_correlation_analyzer",
    )

    def __init__(
        self,
        forward_return_target: ForwardReturnTarget,
        information_coefficient: InformationCoefficient,
        rank_information_coefficient: RankInformationCoefficient,
        quantile_analyzer: QuantileAnalyzer,
        factor_decay_analyzer: FactorDecayAnalyzer,
        factor_stability_analyzer: FactorStabilityAnalyzer,
        factor_correlation_analyzer: FactorCorrelationAnalyzer,
    ) -> None:
        """Initialize with injected research collaborators.

        Args:
            forward_return_target: Primary forward-return target generator.
            information_coefficient: IC calculator.
            rank_information_coefficient: Rank IC calculator.
            quantile_analyzer: Quantile analyzer.
            factor_decay_analyzer: Factor decay analyzer.
            factor_stability_analyzer: Factor stability analyzer.
            factor_correlation_analyzer: Cross-factor correlation analyzer.
        """
        self._forward_return_target = forward_return_target
        self._information_coefficient = information_coefficient
        self._rank_information_coefficient = rank_information_coefficient
        self._quantile_analyzer = quantile_analyzer
        self._factor_decay_analyzer = factor_decay_analyzer
        self._factor_stability_analyzer = factor_stability_analyzer
        self._factor_correlation_analyzer = factor_correlation_analyzer

    def run(
        self,
        frame: pl.DataFrame,
        definition: ExperimentDefinition,
    ) -> ExperimentResult:
        """Execute the full factor research experiment.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            definition: Immutable experiment definition.

        Returns:
            An immutable ``ExperimentResult`` containing all composed analyses.

        Raises:
            ResearchError: If factor columns are missing, the price column is
                missing, or the injected forward-return target does not match
                the definition horizon/price column.
        """
        started_at = datetime.now(UTC)
        _validate_frame_columns(frame, definition)
        _validate_target_alignment(self._forward_return_target, definition)

        evaluated = self._forward_return_target.transform(frame)
        target_column = self._forward_return_target.output_column
        target = self._forward_return_target.definition

        information_coefficients: list[InformationCoefficientResult] = []
        rank_information_coefficients: list[RankICResult] = []
        quantile_results: list[QuantileAnalysisResult] = []
        decay_results: list[FactorDecayResult] = []
        stability_results: list[FactorStabilityResult] = []

        for factor_column in definition.factor_columns:
            information_coefficients.append(
                self._information_coefficient.compute(
                    evaluated,
                    factor_column,
                    target_column,
                )
            )
            rank_information_coefficients.append(
                self._rank_information_coefficient.compute(
                    evaluated,
                    factor_column,
                    target_column,
                )
            )
            quantile_results.append(
                self._quantile_analyzer.analyze(
                    evaluated,
                    factor_column,
                    target_column,
                )
            )
            decay_results.append(
                self._factor_decay_analyzer.analyze(
                    frame,
                    factor_column,
                    price_column=definition.price_column,
                    horizons=definition.decay_horizons,
                )
            )
            stability_results.append(
                self._factor_stability_analyzer.analyze(
                    evaluated,
                    factor_column,
                    target_column,
                    window_size=definition.stability_window,
                )
            )

        correlation_result: FactorCorrelationResult | None = None
        highly_correlated_pairs: tuple[tuple[str, str, float], ...] = ()
        if len(definition.factor_columns) >= 2:
            correlation_result = self._factor_correlation_analyzer.analyze(
                evaluated,
                definition.factor_columns,
            )
            highly_correlated_pairs = find_highly_correlated(
                correlation_result,
                threshold=definition.correlation_threshold,
            )

        completed_at = datetime.now(UTC)
        return ExperimentResult(
            definition=definition,
            target=target,
            information_coefficients=tuple(information_coefficients),
            rank_information_coefficients=tuple(rank_information_coefficients),
            quantile_results=tuple(quantile_results),
            decay_results=tuple(decay_results),
            stability_results=tuple(stability_results),
            correlation_result=correlation_result,
            highly_correlated_pairs=highly_correlated_pairs,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )


def _validate_frame_columns(frame: pl.DataFrame, definition: ExperimentDefinition) -> None:
    """Validate that required factor and price columns exist on ``frame``."""
    if len(definition.factor_columns) == 0:
        raise ResearchError(
            "factor_columns must contain at least one factor",
            error_code=_ERROR_NO_FACTORS,
            details={"factor_columns": definition.factor_columns},
        )
    missing_factors = tuple(name for name in definition.factor_columns if name not in frame.columns)
    if missing_factors:
        raise ResearchError(
            f"required column missing: {missing_factors[0]}",
            error_code=_ERROR_MISSING_FACTOR,
            details={
                "required_column": missing_factors[0],
                "missing_columns": missing_factors,
                "available_columns": tuple(frame.columns),
            },
        )
    if definition.price_column not in frame.columns:
        raise ResearchError(
            f"required column missing: {definition.price_column}",
            error_code=_ERROR_MISSING_PRICE,
            details={
                "required_column": definition.price_column,
                "role": "price",
                "available_columns": tuple(frame.columns),
            },
        )


def _validate_target_alignment(
    forward_return_target: ForwardReturnTarget,
    definition: ExperimentDefinition,
) -> None:
    """Ensure the injected target matches the experiment definition."""
    if (
        forward_return_target.horizon != definition.target_horizon
        or forward_return_target.price_column != definition.price_column
    ):
        raise ResearchError(
            "injected forward-return target does not match experiment definition",
            error_code=_ERROR_TARGET_MISMATCH,
            details={
                "target_horizon": definition.target_horizon,
                "price_column": definition.price_column,
                "injected_horizon": forward_return_target.horizon,
                "injected_price_column": forward_return_target.price_column,
            },
        )
