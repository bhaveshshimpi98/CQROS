"""CQROS research utility for regression signal threshold calibration.

Purpose:
    Analyze empirical prediction distributions and recommend BUY/SELL
    thresholds for ``RegressionSignalPolicy`` from historical percentiles.

Responsibilities:
    - Validate the canonical ``prediction`` column
    - Compute distribution statistics and percentiles
    - Recommend Conservative, Balanced, and Active threshold profiles
    - Estimate expected BUY / SELL / HOLD rates from historical values
    - Aggregate per-symbol/timeframe and global calibration results
    - Remain free of signal generation, persistence, and policy mutation

Dependencies:
    ``math``, ``polars``, and ``cqros.core.exceptions.ResearchError``.

Public API:
    ``PredictionDistributionStatistics``, ``ThresholdRecommendation``,
    ``SymbolTimeframeCalibration``, ``ThresholdCalibrationResult``, and
    ``SignalThresholdCalibrator``.

Notes:
    This module is read-only research analysis. It never writes thresholds,
    generates signals, or modifies repositories or policies.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError
from cqros.core.types import Symbol, Timeframe

__all__ = [
    "PredictionDistributionStatistics",
    "SignalThresholdCalibrator",
    "SymbolTimeframeCalibration",
    "ThresholdCalibrationResult",
    "ThresholdRecommendation",
]

_logger = logging.getLogger(__name__)

_PREDICTION_COLUMN: Final[str] = "prediction"

_ERROR_FRAME_TYPE: Final[str] = "RESEARCH-STC-001"
_ERROR_MISSING_COLUMN: Final[str] = "RESEARCH-STC-002"
_ERROR_EMPTY: Final[str] = "RESEARCH-STC-003"
_ERROR_NON_FINITE: Final[str] = "RESEARCH-STC-004"
_ERROR_THRESHOLD_ORDER: Final[str] = "RESEARCH-STC-005"
_ERROR_NO_GROUPS: Final[str] = "RESEARCH-STC-006"

_PROFILE_CONSERVATIVE: Final[str] = "Conservative"
_PROFILE_BALANCED: Final[str] = "Balanced"
_PROFILE_ACTIVE: Final[str] = "Active"

# (profile_name, buy_percentile, sell_percentile)
_THRESHOLD_PROFILES: Final[tuple[tuple[str, float, float], ...]] = (
    (_PROFILE_CONSERVATIVE, 0.99, 0.01),
    (_PROFILE_BALANCED, 0.95, 0.05),
    (_PROFILE_ACTIVE, 0.90, 0.10),
)


@dataclass(frozen=True, slots=True)
class PredictionDistributionStatistics:
    """Immutable summary of a prediction value distribution.

    Attributes:
        count: Number of finite prediction observations.
        minimum: Minimum prediction value.
        maximum: Maximum prediction value.
        mean: Arithmetic mean.
        std: Sample standard deviation (``NaN`` when undefined).
        median: Median prediction value.
        percentile_01: 1st percentile.
        percentile_025: 2.5th percentile.
        percentile_05: 5th percentile.
        percentile_10: 10th percentile.
        percentile_90: 90th percentile.
        percentile_95: 95th percentile.
        percentile_975: 97.5th percentile.
        percentile_99: 99th percentile.
        positive_ratio: Fraction of values strictly greater than zero.
        negative_ratio: Fraction of values strictly less than zero.
    """

    count: int
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    percentile_01: float
    percentile_025: float
    percentile_05: float
    percentile_10: float
    percentile_90: float
    percentile_95: float
    percentile_975: float
    percentile_99: float
    positive_ratio: float
    negative_ratio: float


@dataclass(frozen=True, slots=True)
class ThresholdRecommendation:
    """Immutable BUY/SELL threshold recommendation for one profile.

    Attributes:
        profile: Profile name (``Conservative``, ``Balanced``, or ``Active``).
        buy_threshold: Inclusive lower bound recommended for ``BUY``.
        sell_threshold: Inclusive upper bound recommended for ``SELL``.
        expected_buy_ratio: Historical fraction with ``prediction >= buy``.
        expected_sell_ratio: Historical fraction with ``prediction <= sell``.
        expected_hold_ratio: Historical fraction between the thresholds.
    """

    profile: str
    buy_threshold: float
    sell_threshold: float
    expected_buy_ratio: float
    expected_sell_ratio: float
    expected_hold_ratio: float


@dataclass(frozen=True, slots=True)
class SymbolTimeframeCalibration:
    """Immutable calibration result for one symbol/timeframe group.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        statistics: Prediction distribution statistics for the group.
        recommendations: Ordered Conservative / Balanced / Active profiles.
    """

    symbol: Symbol
    timeframe: Timeframe
    statistics: PredictionDistributionStatistics
    recommendations: tuple[ThresholdRecommendation, ...]


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    """Immutable aggregate calibration across prediction partitions.

    Attributes:
        symbols_analyzed: Deterministically ordered unique symbols.
        datasets_analyzed: Number of symbol/timeframe groups analyzed.
        rows_analyzed: Total finite prediction rows analyzed.
        global_statistics: Pooled distribution across all groups.
        recommendations: Global Conservative / Balanced / Active profiles.
        symbol_timeframe_results: Per-symbol/timeframe calibration results.
    """

    symbols_analyzed: tuple[Symbol, ...]
    datasets_analyzed: int
    rows_analyzed: int
    global_statistics: PredictionDistributionStatistics
    recommendations: tuple[ThresholdRecommendation, ...]
    symbol_timeframe_results: tuple[SymbolTimeframeCalibration, ...]


class SignalThresholdCalibrator:
    """Recommend regression signal thresholds from prediction distributions.

    Analyzes the canonical ``prediction`` column only. Input DataFrames are
    never mutated. Thresholds are recommended from empirical percentiles and
    are not written to policies, configuration, or storage.
    """

    __slots__ = ("_logger",)

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the calibrator.

        Args:
            logger: Optional logger. Defaults to the module logger.
        """
        self._logger = logger if logger is not None else _logger

    def analyze(self, frame: pl.DataFrame) -> PredictionDistributionStatistics:
        """Compute prediction distribution statistics for ``frame``.

        Args:
            frame: Prediction DataFrame containing a ``prediction`` column.
                Must not be mutated.

        Returns:
            Immutable distribution statistics.

        Raises:
            ResearchError: If the frame is invalid, missing ``prediction``,
                empty after cleaning, or contains non-finite predictions.
        """
        values = self._extract_prediction_values(frame)
        return self._compute_statistics(values)

    def recommend(
        self,
        frame: pl.DataFrame,
        *,
        statistics: PredictionDistributionStatistics | None = None,
    ) -> tuple[ThresholdRecommendation, ...]:
        """Recommend Conservative / Balanced / Active thresholds.

        Args:
            frame: Prediction DataFrame containing a ``prediction`` column.
            statistics: Optional precomputed statistics. When ``None``,
                statistics are derived from ``frame``.

        Returns:
            Ordered recommendations for Conservative, Balanced, and Active.

        Raises:
            ResearchError: If analysis fails or a profile cannot produce
                ``buy_threshold > sell_threshold``.
        """
        values = self._extract_prediction_values(frame)
        resolved = statistics if statistics is not None else self._compute_statistics(values)
        return self._recommend_from_values(values, resolved)

    def calibrate_group(
        self,
        frame: pl.DataFrame,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> SymbolTimeframeCalibration:
        """Calibrate thresholds for one symbol/timeframe prediction group.

        Args:
            frame: Concatenated prediction rows for the group.
            symbol: Tradeable symbol label for the result.
            timeframe: Bar interval label for the result.

        Returns:
            Immutable per-group calibration result.

        Raises:
            ResearchError: If analysis or recommendation fails.
        """
        statistics = self.analyze(frame)
        recommendations = self.recommend(frame, statistics=statistics)
        self._logger.info(
            "Calibrated symbol/timeframe prediction thresholds",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": statistics.count,
                "minimum": statistics.minimum,
                "maximum": statistics.maximum,
            },
        )
        return SymbolTimeframeCalibration(
            symbol=symbol,
            timeframe=timeframe,
            statistics=statistics,
            recommendations=recommendations,
        )

    def calibrate(
        self,
        groups: (
            Mapping[tuple[Symbol, Timeframe], pl.DataFrame]
            | Sequence[tuple[Symbol, Timeframe, pl.DataFrame]]
        ),
    ) -> ThresholdCalibrationResult:
        """Calibrate thresholds across symbol/timeframe prediction groups.

        Global statistics and recommendations are computed from the pooled
        prediction distribution across all groups.

        Args:
            groups: Mapping of ``(symbol, timeframe)`` to prediction frames,
                or a sequence of ``(symbol, timeframe, frame)`` triples.

        Returns:
            Immutable aggregate calibration result.

        Raises:
            ResearchError: If no groups are supplied or analysis fails.
        """
        items = _normalize_groups(groups)
        if len(items) == 0:
            raise ResearchError(
                "no prediction groups supplied for threshold calibration",
                error_code=_ERROR_NO_GROUPS,
                details={"group_count": 0},
            )

        symbol_results: list[SymbolTimeframeCalibration] = []
        series_parts: list[pl.Series] = []
        for symbol, timeframe, frame in items:
            values = self._extract_prediction_values(frame)
            statistics = self._compute_statistics(values)
            recommendations = self._recommend_from_values(values, statistics)
            symbol_results.append(
                SymbolTimeframeCalibration(
                    symbol=symbol,
                    timeframe=timeframe,
                    statistics=statistics,
                    recommendations=recommendations,
                )
            )
            series_parts.append(values)

        pooled = pl.concat(series_parts, how="vertical")
        global_statistics = self._compute_statistics(pooled)
        global_recommendations = self._recommend_from_values(pooled, global_statistics)

        symbols = tuple(sorted({item.symbol for item in symbol_results}))
        ordered_results = tuple(
            sorted(
                symbol_results,
                key=lambda item: (item.symbol, item.timeframe),
            )
        )

        self._logger.info(
            "Completed regression threshold calibration",
            extra={
                "symbols_analyzed": len(symbols),
                "datasets_analyzed": len(ordered_results),
                "rows_analyzed": global_statistics.count,
            },
        )

        return ThresholdCalibrationResult(
            symbols_analyzed=symbols,
            datasets_analyzed=len(ordered_results),
            rows_analyzed=global_statistics.count,
            global_statistics=global_statistics,
            recommendations=global_recommendations,
            symbol_timeframe_results=ordered_results,
        )

    def _extract_prediction_values(self, frame: pl.DataFrame) -> pl.Series:
        """Validate ``frame`` and return finite cleaned prediction values."""
        if not isinstance(cast(object, frame), pl.DataFrame):
            raise ResearchError(
                "frame must be a polars DataFrame",
                error_code=_ERROR_FRAME_TYPE,
                details={"actual_type": type(frame).__name__},
            )
        if _PREDICTION_COLUMN not in frame.columns:
            raise ResearchError(
                f"required column missing: {_PREDICTION_COLUMN}",
                error_code=_ERROR_MISSING_COLUMN,
                details={
                    "required_column": _PREDICTION_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        series = frame.get_column(_PREDICTION_COLUMN)
        clean = series.drop_nulls()
        if clean.len() == 0:
            raise ResearchError(
                "prediction column contains no non-null observations",
                error_code=_ERROR_EMPTY,
                details={"column": _PREDICTION_COLUMN, "row_count": frame.height},
            )

        values = clean.cast(pl.Float64, strict=False)
        if values.null_count() > 0:
            raise ResearchError(
                "prediction column contains non-numeric values",
                error_code=_ERROR_NON_FINITE,
                details={
                    "column": _PREDICTION_COLUMN,
                    "non_numeric_count": values.null_count(),
                },
            )

        infinite_mask = values.is_infinite()
        if bool(infinite_mask.any()):
            raise ResearchError(
                "prediction column contains non-finite values",
                error_code=_ERROR_NON_FINITE,
                details={
                    "column": _PREDICTION_COLUMN,
                    "infinite_count": int(infinite_mask.sum()),
                },
            )
        return values

    def _compute_statistics(self, values: pl.Series) -> PredictionDistributionStatistics:
        """Compute distribution statistics from finite prediction values."""
        count = values.len()
        if count == 0:
            raise ResearchError(
                "prediction column contains no non-null observations",
                error_code=_ERROR_EMPTY,
                details={"column": _PREDICTION_COLUMN, "count": 0},
            )

        std_value = values.std()
        positive_count = int((values > 0.0).sum())
        negative_count = int((values < 0.0).sum())
        return PredictionDistributionStatistics(
            count=count,
            minimum=float(values.min()),  # type: ignore[arg-type]
            maximum=float(values.max()),  # type: ignore[arg-type]
            mean=float(values.mean()),  # type: ignore[arg-type]
            std=float("nan") if std_value is None else float(std_value),
            median=float(values.median()),  # type: ignore[arg-type]
            percentile_01=_quantile(values, 0.01),
            percentile_025=_quantile(values, 0.025),
            percentile_05=_quantile(values, 0.05),
            percentile_10=_quantile(values, 0.10),
            percentile_90=_quantile(values, 0.90),
            percentile_95=_quantile(values, 0.95),
            percentile_975=_quantile(values, 0.975),
            percentile_99=_quantile(values, 0.99),
            positive_ratio=positive_count / count,
            negative_ratio=negative_count / count,
        )

    def _recommend_from_values(
        self,
        values: pl.Series,
        statistics: PredictionDistributionStatistics,
    ) -> tuple[ThresholdRecommendation, ...]:
        """Build ordered threshold recommendations from values and statistics."""
        percentile_lookup = {
            0.01: statistics.percentile_01,
            0.025: statistics.percentile_025,
            0.05: statistics.percentile_05,
            0.10: statistics.percentile_10,
            0.90: statistics.percentile_90,
            0.95: statistics.percentile_95,
            0.975: statistics.percentile_975,
            0.99: statistics.percentile_99,
        }
        recommendations: list[ThresholdRecommendation] = []
        for profile, buy_percentile, sell_percentile in _THRESHOLD_PROFILES:
            buy_threshold = percentile_lookup[buy_percentile]
            sell_threshold = percentile_lookup[sell_percentile]
            if not (buy_threshold > sell_threshold):
                raise ResearchError(
                    "prediction distribution does not support distinct "
                    "buy/sell thresholds for profile",
                    error_code=_ERROR_THRESHOLD_ORDER,
                    details={
                        "profile": profile,
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                        "buy_percentile": buy_percentile,
                        "sell_percentile": sell_percentile,
                    },
                )
            expected_buy, expected_sell, expected_hold = _expected_signal_ratios(
                values,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
            recommendations.append(
                ThresholdRecommendation(
                    profile=profile,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    expected_buy_ratio=expected_buy,
                    expected_sell_ratio=expected_sell,
                    expected_hold_ratio=expected_hold,
                )
            )
        return tuple(recommendations)


def _quantile(values: pl.Series, probability: float) -> float:
    """Return a finite quantile from ``values``."""
    result = values.quantile(probability, interpolation="linear")
    if result is None or not math.isfinite(float(result)):
        raise ResearchError(
            "unable to compute finite prediction quantile",
            error_code=_ERROR_NON_FINITE,
            details={"probability": probability, "value": result},
        )
    return float(result)


def _expected_signal_ratios(
    values: pl.Series,
    *,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[float, float, float]:
    """Estimate BUY / SELL / HOLD ratios under inclusive regression rules."""
    count = values.len()
    buy_count = int((values >= buy_threshold).sum())
    sell_count = int((values <= sell_threshold).sum())
    hold_count = count - buy_count - sell_count
    return (
        buy_count / count,
        sell_count / count,
        hold_count / count,
    )


def _normalize_groups(
    groups: (
        Mapping[tuple[Symbol, Timeframe], pl.DataFrame]
        | Sequence[tuple[Symbol, Timeframe, pl.DataFrame]]
    ),
) -> tuple[tuple[Symbol, Timeframe, pl.DataFrame], ...]:
    """Normalize mapping or sequence group inputs into ordered triples."""
    if isinstance(groups, Mapping):
        items = tuple((symbol, timeframe, frame) for (symbol, timeframe), frame in groups.items())
    else:
        items = tuple(groups)
    return tuple(
        sorted(items, key=lambda item: (item[0], item[1])),
    )
