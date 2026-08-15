"""CQROS Factor Validation Engine contracts and statistical implementation.

Purpose:
    Convert a canonical Factors dataset into a deterministic factor
    validation DataFrame conforming to ``FACTOR_VALIDATION_SCHEMA``.

Responsibilities:
    - Define ``FactorValidationEngine`` as the shared validation contract
    - Provide ``SimpleFactorValidationEngine`` for Phase-1 IC metrics
    - Validate factor DataFrame structure
    - Compute cross-sectional Information Coefficient, Rank IC, ICIR,
      IC t-statistic, IC p-value, IC Decay, quantile return spread,
      monotonicity score, factor turnover, and observation counts
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``datetime``, ``polars``, ``cqros.factor_validation.exceptions``, and
    ``cqros.factor_validation.schema``.

Public API:
    ``FactorValidationEngine``, ``SimpleFactorValidationEngine``,
    ``FACTOR_INPUT_COLUMNS``, ``validate_factor_frame``
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.factor_validation.exceptions import FactorValidationError
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_VALIDATION_SCHEMA,
    FactorValidationStatus,
)

__all__ = [
    "FACTOR_INPUT_COLUMNS",
    "FactorValidationEngine",
    "SimpleFactorValidationEngine",
    "validate_factor_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "FVAL_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "FVAL_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FVAL_MISSING_COLUMNS"
_ERROR_OPEN_TIME_ORDER: Final[str] = "FVAL_OPEN_TIME_ORDER"

_MINIMUM_OBSERVATIONS: Final[int] = 2
_MINIMUM_IC_DECAY_HORIZONS: Final[int] = 2
_QUANTILE_COUNT: Final[int] = 5
_MINIMUM_QUANTILE_OBSERVATIONS: Final[int] = 5
_MINIMUM_TURNOVER_COMMON_ASSETS: Final[int] = 5
_TOP_PORTFOLIO_DENOMINATOR: Final[int] = 5
_BETA_CF_MAX_ITERATIONS: Final[int] = 200
_BETA_CF_EPSILON: Final[float] = 3.0e-14
_BETA_CF_TINY: Final[float] = 1.0e-30
_DEFAULT_VERSION_PLACEHOLDER: Final[str] = "default"

_FACTOR_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
)

_CROSS_SECTION_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    *_FACTOR_IDENTITY_COLUMNS,
    "open_time",
)

# Optional forward-return columns used for IC Decay, in lag order.
_IC_DECAY_HORIZON_COLUMNS: Final[tuple[str, ...]] = (
    "future_return_1",
    "future_return_2",
    "future_return_3",
    "future_return_5",
    "future_return_10",
    "future_return_20",
)

# Factor columns required to assemble a validation-metrics row.
FACTOR_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "timeframe",
    "open_time",
    "symbol",
    "factor_value",
    "future_return_1",
)


@runtime_checkable
class FactorValidationEngine(Protocol):
    """Structural contract for converting Factors into validation metrics.

    Implementations own factor-validation semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, factors: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factors dataset into a factor-validation DataFrame.

        Args:
            factors: Canonical Factors dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``FACTOR_VALIDATION_SCHEMA``.
        """
        ...


class SimpleFactorValidationEngine:
    """Generate deterministic Phase-1 validation rows from Factors.

    Rules:
        - One output row per unique factor identity
          (``factor_name``, ``factor_version``, ``timeframe``)
        - Identity and metadata columns (``factor_name``, ``factor_version``,
          ``factor_category``, ``timeframe``) are preserved from the factor
        - ``validation_time`` is the latest ``open_time`` for that factor
        - ``information_coefficient`` is the mean of the chronological
          cross-sectional Pearson IC series (one IC per ``open_time``)
        - ``ic_information_ratio`` is mean(IC_series) / sample_std(IC_series)
        - ``ic_t_stat`` is mean(IC_series) / (sample_std(IC_series) / sqrt(N))
          where ``N`` is the number of valid IC timestamps
        - ``ic_p_value`` is the two-sided Student's-t p-value for
          ``H0: mean(IC) = 0`` with ``df = N - 1``
        - ``quantile_spread`` is the mean of per-``open_time`` Q5−Q1
          future-return spreads from five equal-sized factor quantiles
        - ``monotonicity_score`` is the mean fraction of eligible timestamps
          whose quantile mean returns are non-decreasing or non-increasing
          across Q1–Q5
        - ``turnover`` is the mean Top-20% portfolio transition turnover
          (``1 - overlap``) across consecutive ``open_time`` values
        - ``rank_information_coefficient`` is the pooled Spearman rank
          correlation of ``factor_value`` and ``future_return_1``
        - ``ic_decay`` is MeanIC(last available horizon) /
          MeanIC(first available horizon) across present
          ``future_return_{h}`` columns (``h`` in 1, 2, 3, 5, 10, 20)
        - ``ic_std`` is the sample standard deviation of the IC series
        - ``ic_observations`` counts valid IC timestamps used for the IC
          series (distinct from ``observations``)
        - ``validation_start_time`` / ``validation_end_time`` are the first
          and final ``open_time`` values in the validation window
        - ``dataset_version`` / ``label_version`` use the stable placeholder
          ``"default"`` when pipeline version metadata is unavailable
        - ``observations`` counts valid non-null pairs
        - Fewer than two valid observations yields null IC / Rank IC / ICIR /
          IC t-stat / IC p-value / IC std metrics and ``FAIL`` status;
          otherwise status is ``PASS``
        - ICIR, IC t-stat, and IC p-value are also null when fewer than two
          valid IC timestamps exist or when the IC-series sample standard
          deviation is zero
        - ``quantile_spread`` and ``monotonicity_score`` are null when no
          timestamp has at least five valid cross-sectional observations
        - ``turnover`` is null when fewer than two timestamps exist or no
          transition has at least five common assets
        - ``ic_decay`` is null when fewer than two horizon columns are
          present or when the first-horizon mean IC is zero

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Correlation, quantile, turnover, and IC Decay metrics use Polars
        only. The t-distribution p-value uses a pure Python regularized
        incomplete-beta implementation (no scipy/numpy).
    """

    __slots__ = ()

    def build(self, factors: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factors dataset into finalized validation metrics.

        Args:
            factors: Canonical Factors dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``FACTOR_VALIDATION_SCHEMA``.

        Raises:
            FactorValidationError: If the input fails structural validation,
                required columns are missing, or timestamps are unsorted.
        """
        frame = validate_factor_frame(factors)
        _require_columns(frame, FACTOR_INPUT_COLUMNS, "factors")
        normalized = _normalize_open_times(frame)
        ordered = normalized.sort("open_time", maintain_order=True)
        _require_sorted_open_times(ordered)
        return _build_factor_validation_rows(ordered)


def validate_factor_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factors dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        FactorValidationError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorValidationError(
            "factors frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"dataset": "factors", "actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorValidationError(
            "factors frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factors", "rows": frame.height},
        )
    return frame


def _build_factor_validation_rows(factors: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical validation rows with Phase-1 IC statistics."""
    valid_pair = pl.col("factor_value").is_not_null() & pl.col("future_return_1").is_not_null()
    insufficient = pl.col("observations") < _MINIMUM_OBSERVATIONS
    identity = list(_FACTOR_IDENTITY_COLUMNS)

    ic_stats = _compute_cross_sectional_ic_stats(factors, valid_pair)
    ic_decay_stats = _compute_ic_decay_stats(factors)
    quantile_stats = _compute_quantile_stats(factors, valid_pair)
    turnover_stats = _compute_turnover_stats(factors)

    aggregated = (
        factors.group_by(identity, maintain_order=True)
        .agg(
            pl.col("factor_category").last(),
            pl.col("open_time").min().alias("validation_start_time"),
            pl.col("open_time").max().alias("validation_end_time"),
            pl.col("open_time").last().alias("validation_time"),
            valid_pair.sum().cast(pl.Int64).alias("observations"),
            pl.corr("factor_value", "future_return_1", method="spearman").alias(
                "rank_information_coefficient"
            ),
        )
        .sort(identity, maintain_order=True)
        .join(ic_stats, on=identity, how="left")
        .join(ic_decay_stats, on=identity, how="left")
        .join(quantile_stats, on=identity, how="left")
        .join(turnover_stats, on=identity, how="left")
        .with_columns(
            pl.lit(_DEFAULT_VERSION_PLACEHOLDER).alias("dataset_version"),
            pl.lit(_DEFAULT_VERSION_PLACEHOLDER).alias("label_version"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("information_coefficient"))
            .alias("information_coefficient"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("ic_information_ratio"))
            .alias("ic_information_ratio"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("ic_std"))
            .alias("ic_std"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("ic_t_stat"))
            .alias("ic_t_stat"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("ic_p_value"))
            .alias("ic_p_value"),
            pl.when(insufficient)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("rank_information_coefficient"))
            .alias("rank_information_coefficient"),
            pl.when(insufficient)
            .then(pl.lit(FactorValidationStatus.FAIL.value))
            .otherwise(pl.lit(FactorValidationStatus.PASS.value))
            .alias("status"),
        )
    )
    return aggregated.select(list(CANONICAL_COLUMN_ORDER)).cast(FACTOR_VALIDATION_SCHEMA)


def _compute_turnover_stats(factors: pl.DataFrame) -> pl.DataFrame:
    """Compute mean Top-20% cross-sectional factor turnover per identity.

    For each consecutive ``open_time`` pair, intersect symbols with non-null
    ``factor_value``, require at least five common assets, form the Top-20%
    portfolios by factor rank, and record ``1 - |intersection| / |Top_t|``.
    The reported metric is the mean of that transition series.
    """
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    valid = factors.filter(pl.col("factor_value").is_not_null()).select(
        *identity,
        "open_time",
        "symbol",
        "factor_value",
    )
    transitions = (
        valid.select(*identity, "open_time")
        .sort([*identity, "open_time"], maintain_order=True)
        .unique(subset=[*identity, "open_time"], maintain_order=True)
        .with_columns(pl.col("open_time").shift(-1).over(identity).alias("open_time_next"))
        .filter(pl.col("open_time_next").is_not_null())
    )
    current = valid.rename({"factor_value": "factor_value_t"})
    following = valid.rename(
        {
            "open_time": "open_time_next",
            "factor_value": "factor_value_next",
        }
    )
    joined = transitions.join(current, on=[*identity, "open_time"], how="inner").join(
        following,
        on=[*identity, "open_time_next", "symbol"],
        how="inner",
    )
    per_transition = (
        joined.group_by([*identity, "open_time", "open_time_next"], maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias("_common_n"),
            pl.col("symbol")
            .sort_by(["factor_value_t", "symbol"], descending=[True, False])
            .alias("_symbols_t"),
            pl.col("symbol")
            .sort_by(["factor_value_next", "symbol"], descending=[True, False])
            .alias("_symbols_next"),
        )
        .filter(pl.col("_common_n") >= _MINIMUM_TURNOVER_COMMON_ASSETS)
        .with_columns(
            pl.max_horizontal(
                pl.lit(1),
                pl.col("_common_n") // _TOP_PORTFOLIO_DENOMINATOR,
            ).alias("_top_k")
        )
        .with_columns(
            pl.col("_symbols_t").list.head(pl.col("_top_k")).alias("_top_t"),
            pl.col("_symbols_next").list.head(pl.col("_top_k")).alias("_top_next"),
        )
        .with_columns(
            (
                1.0
                - pl.col("_top_t").list.set_intersection("_top_next").list.len()
                / pl.col("_top_k").cast(pl.Float64)
            ).alias("_turnover")
        )
        .sort([*identity, "open_time"], maintain_order=True)
    )
    return per_transition.group_by(identity, maintain_order=True).agg(
        pl.col("_turnover").mean().alias("turnover")
    )


def _compute_quantile_stats(
    factors: pl.DataFrame,
    valid_pair: pl.Expr,
) -> pl.DataFrame:
    """Compute mean Q5−Q1 spread and monotonicity score per factor identity.

    At each ``open_time`` with at least five valid observations, assets are
    sorted by ``factor_value`` into five equal-sized quantiles. The timestamp
    spread is mean(Q5 ``future_return_1``) − mean(Q1 ``future_return_1``).
    Monotonicity is ``1`` when quantile mean returns are non-decreasing or
    non-increasing across Q1–Q5, otherwise ``0``. Reported metrics are the
    means of those chronological series.
    """
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    cross_section_keys = list(_CROSS_SECTION_GROUP_COLUMNS)
    prepared = (
        factors.filter(valid_pair)
        .sort([*cross_section_keys, "factor_value"], maintain_order=True)
        .with_columns(
            pl.len().over(cross_section_keys).alias("_cs_n"),
            pl.col("factor_value")
            .rank(method="ordinal")
            .over(cross_section_keys)
            .alias("_factor_rank"),
        )
        .filter(pl.col("_cs_n") >= _MINIMUM_QUANTILE_OBSERVATIONS)
        .with_columns(
            ((pl.col("_factor_rank") - 1) * _QUANTILE_COUNT // pl.col("_cs_n"))
            .clip(upper_bound=_QUANTILE_COUNT - 1)
            .alias("_quantile")
        )
    )
    quantile_means = [
        pl.col("future_return_1")
        .filter(pl.col("_quantile") == quantile_index)
        .mean()
        .alias(f"_q{quantile_index + 1}_mean_return")
        for quantile_index in range(_QUANTILE_COUNT)
    ]
    increasing = (
        (pl.col("_q1_mean_return") <= pl.col("_q2_mean_return"))
        & (pl.col("_q2_mean_return") <= pl.col("_q3_mean_return"))
        & (pl.col("_q3_mean_return") <= pl.col("_q4_mean_return"))
        & (pl.col("_q4_mean_return") <= pl.col("_q5_mean_return"))
    )
    decreasing = (
        (pl.col("_q1_mean_return") >= pl.col("_q2_mean_return"))
        & (pl.col("_q2_mean_return") >= pl.col("_q3_mean_return"))
        & (pl.col("_q3_mean_return") >= pl.col("_q4_mean_return"))
        & (pl.col("_q4_mean_return") >= pl.col("_q5_mean_return"))
    )
    per_timestamp = (
        prepared.group_by(cross_section_keys, maintain_order=True)
        .agg(quantile_means)
        .with_columns(
            (pl.col("_q5_mean_return") - pl.col("_q1_mean_return")).alias("_spread"),
            pl.when(increasing | decreasing)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("_monotonic"),
        )
        .filter(
            pl.col("_q1_mean_return").is_not_null()
            & pl.col("_q2_mean_return").is_not_null()
            & pl.col("_q3_mean_return").is_not_null()
            & pl.col("_q4_mean_return").is_not_null()
            & pl.col("_q5_mean_return").is_not_null()
        )
        .sort([*identity, "open_time"], maintain_order=True)
    )
    return per_timestamp.group_by(identity, maintain_order=True).agg(
        pl.col("_spread").mean().alias("quantile_spread"),
        pl.col("_monotonic").mean().alias("monotonicity_score"),
    )


def _available_ic_decay_horizons(factors: pl.DataFrame) -> tuple[str, ...]:
    """Return IC Decay horizon columns present in ``factors``, in lag order."""
    return tuple(column for column in _IC_DECAY_HORIZON_COLUMNS if column in factors.columns)


def _null_ic_decay_frame(factors: pl.DataFrame) -> pl.DataFrame:
    """Return one null ``ic_decay`` row per factor identity."""
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    return (
        factors.select(identity)
        .unique(subset=identity, maintain_order=True)
        .with_columns(pl.lit(None, dtype=pl.Float64).alias("ic_decay"))
    )


def _mean_cross_sectional_ic_for_horizon(
    factors: pl.DataFrame,
    return_column: str,
) -> pl.DataFrame:
    """Compute mean cross-sectional Pearson IC for one forward-return horizon.

    For every ``open_time``, drops null ``factor_value`` / return pairs, requires
    at least two valid assets, and records the Pearson correlation. The
    reported value is the mean of that chronological IC series.
    """
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    valid_pair = pl.col("factor_value").is_not_null() & pl.col(return_column).is_not_null()
    per_timestamp = factors.group_by(
        list(_CROSS_SECTION_GROUP_COLUMNS),
        maintain_order=True,
    ).agg(
        valid_pair.sum().cast(pl.Int64).alias("_cs_observations"),
        pl.corr("factor_value", return_column).alias("_cs_ic"),
    )
    ic_series = per_timestamp.filter(
        (pl.col("_cs_observations") >= _MINIMUM_OBSERVATIONS)
        & pl.col("_cs_ic").is_not_null()
        & pl.col("_cs_ic").is_not_nan()
    ).sort([*identity, "open_time"], maintain_order=True)
    return ic_series.group_by(identity, maintain_order=True).agg(
        pl.col("_cs_ic").mean().alias(f"_mean_ic_{return_column}")
    )


def _compute_ic_decay_stats(factors: pl.DataFrame) -> pl.DataFrame:
    """Compute IC Decay as last-horizon mean IC over first-horizon mean IC.

    Horizons are the subset of ``future_return_{1,2,3,5,10,20}`` present in
    the input, evaluated in that order. Returns null when fewer than two
    horizons are available or when the first-horizon mean IC is zero.
    """
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    horizons = _available_ic_decay_horizons(factors)
    if len(horizons) < _MINIMUM_IC_DECAY_HORIZONS:
        return _null_ic_decay_frame(factors)

    first_column = horizons[0]
    last_column = horizons[-1]
    decay_stats = factors.select(identity).unique(subset=identity, maintain_order=True)
    for horizon_column in horizons:
        horizon_means = _mean_cross_sectional_ic_for_horizon(factors, horizon_column)
        decay_stats = decay_stats.join(horizon_means, on=identity, how="left")

    first_mean = pl.col(f"_mean_ic_{first_column}")
    last_mean = pl.col(f"_mean_ic_{last_column}")
    undefined_decay = first_mean.is_null() | last_mean.is_null() | (first_mean == 0.0)
    return decay_stats.with_columns(
        pl.when(undefined_decay)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(last_mean / first_mean)
        .alias("ic_decay")
    ).select(*identity, "ic_decay")


def _compute_cross_sectional_ic_stats(
    factors: pl.DataFrame,
    valid_pair: pl.Expr,
) -> pl.DataFrame:
    """Compute mean cross-sectional IC, ICIR, std, t-stat, and p-value.

    Builds a chronological Pearson IC series (one value per ``open_time`` with
    at least two valid observations), then returns mean IC, IC std, IC
    observation count, ICIR, IC t-statistic, and two-sided IC p-value.
    """
    identity = list(_FACTOR_IDENTITY_COLUMNS)
    per_timestamp = factors.group_by(
        list(_CROSS_SECTION_GROUP_COLUMNS),
        maintain_order=True,
    ).agg(
        valid_pair.sum().cast(pl.Int64).alias("_cs_observations"),
        pl.corr("factor_value", "future_return_1").alias("_cs_ic"),
    )
    ic_series = per_timestamp.filter(
        (pl.col("_cs_observations") >= _MINIMUM_OBSERVATIONS)
        & pl.col("_cs_ic").is_not_null()
        & pl.col("_cs_ic").is_not_nan()
    ).sort([*identity, "open_time"], maintain_order=True)
    insufficient_ic_history = pl.col("ic_observations") < _MINIMUM_OBSERVATIONS
    zero_or_missing_std = pl.col("ic_std").is_null() | (pl.col("ic_std") == 0.0)
    undefined_ratio = insufficient_ic_history | zero_or_missing_std
    stats = (
        ic_series.group_by(identity, maintain_order=True)
        .agg(
            pl.col("_cs_ic").mean().alias("information_coefficient"),
            pl.col("_cs_ic").std(ddof=1).alias("ic_std"),
            pl.len().cast(pl.Int64).alias("ic_observations"),
        )
        .with_columns(
            pl.when(undefined_ratio)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("information_coefficient") / pl.col("ic_std"))
            .alias("ic_information_ratio"),
            pl.when(undefined_ratio)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("information_coefficient")
                / (pl.col("ic_std") / pl.col("ic_observations").cast(pl.Float64).sqrt())
            )
            .alias("ic_t_stat"),
        )
    )
    p_values = [
        _two_sided_student_t_p_value(t_stat, ic_count)
        for t_stat, ic_count in zip(
            stats["ic_t_stat"].to_list(),
            stats["ic_observations"].to_list(),
            strict=True,
        )
    ]
    return stats.with_columns(pl.Series("ic_p_value", p_values, dtype=pl.Float64)).select(
        *identity,
        "information_coefficient",
        "ic_information_ratio",
        "ic_std",
        "ic_t_stat",
        "ic_p_value",
        "ic_observations",
    )


def _two_sided_student_t_p_value(t_stat: object, ic_count: object) -> float | None:
    """Return a two-sided Student's-t p-value for ``H0: mean(IC) = 0``.

    Args:
        t_stat: IC t-statistic. ``None`` yields ``None``.
        ic_count: Number of valid IC observations ``N``.

    Returns:
        ``2 * (1 - CDF(|t|; df=N-1))``, or ``None`` when ``N < 2`` or
        ``t_stat`` is missing/non-finite.
    """
    if t_stat is None or ic_count is None:
        return None
    if isinstance(t_stat, bool) or not isinstance(t_stat, int | float):
        return None
    if isinstance(ic_count, bool) or not isinstance(ic_count, int | float):
        return None
    t_value = float(t_stat)
    observations = int(ic_count)
    if math.isnan(t_value) or math.isinf(t_value) or observations < _MINIMUM_OBSERVATIONS:
        return None
    degrees_of_freedom = float(observations - 1)
    x = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    p_value = _regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)
    if p_value < 0.0:
        return 0.0
    if p_value > 1.0:
        return 1.0
    return p_value


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Evaluate the regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _incomplete_beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _incomplete_beta_continued_fraction(b, a, 1.0 - x) / b


def _incomplete_beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    coefficient_b = 1.0 - qab * x / qap
    if abs(coefficient_b) < _BETA_CF_TINY:
        coefficient_b = _BETA_CF_TINY
    coefficient_c = 1.0
    coefficient_d = 1.0 / coefficient_b
    result = coefficient_d
    for iteration in range(1, _BETA_CF_MAX_ITERATIONS + 1):
        em = float(iteration)
        twice_em = em + em
        numerator = em * (b - em) * x / ((qam + twice_em) * (a + twice_em))
        coefficient_d = 1.0 + numerator * coefficient_d
        if abs(coefficient_d) < _BETA_CF_TINY:
            coefficient_d = _BETA_CF_TINY
        coefficient_c = 1.0 + numerator / coefficient_c
        if abs(coefficient_c) < _BETA_CF_TINY:
            coefficient_c = _BETA_CF_TINY
        coefficient_d = 1.0 / coefficient_d
        result *= coefficient_d * coefficient_c

        numerator = -(a + em) * (qab + em) * x / ((a + twice_em) * (qap + twice_em))
        coefficient_d = 1.0 + numerator * coefficient_d
        if abs(coefficient_d) < _BETA_CF_TINY:
            coefficient_d = _BETA_CF_TINY
        coefficient_c = 1.0 + numerator / coefficient_c
        if abs(coefficient_c) < _BETA_CF_TINY:
            coefficient_c = _BETA_CF_TINY
        coefficient_d = 1.0 / coefficient_d
        delta = coefficient_d * coefficient_c
        result *= delta
        if abs(delta - 1.0) < _BETA_CF_EPSILON:
            return result
    return result


def _normalize_open_times(factors: pl.DataFrame) -> pl.DataFrame:
    """Return a copy with ``open_time`` normalized to epoch milliseconds."""
    open_times = [_to_epoch_ms(value) for value in factors["open_time"].to_list()]
    return factors.with_columns(pl.Series("open_time", open_times, dtype=pl.Int64))


def _to_epoch_ms(value: object) -> int:
    """Convert a Factors ``open_time`` value to epoch milliseconds."""
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000.0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise FactorValidationError(
        "open_time must be datetime or integer epoch milliseconds",
        error_code=_ERROR_FRAME_TYPE,
        details={"actual_type": type(value).__name__, "value": repr(value)},
    )


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_sorted_open_times(frame: pl.DataFrame) -> None:
    """Raise when ``open_time`` is not non-decreasing after sorting."""
    open_times = frame["open_time"].to_list()
    for index in range(1, len(open_times)):
        if open_times[index] < open_times[index - 1]:
            raise FactorValidationError(
                "open_time must be sorted in non-decreasing order",
                error_code=_ERROR_OPEN_TIME_ORDER,
                details={
                    "index": index,
                    "open_time": open_times[index],
                    "previous_open_time": open_times[index - 1],
                },
            )
