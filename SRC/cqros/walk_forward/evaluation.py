"""CQROS Walk-Forward evaluation engine.

Purpose:
    Persist diagnostic OOS evaluation results from the already-enriched
    Walk-Forward evaluation input (Factors + Labels ``future_return_1`` +
    Factor Selection) without mutating the Walk-Forward ledger or upstream
    research/trading layers.

Responsibilities:
    - Replay ``SimpleWalkForwardEngine`` fold windows on evaluation input
    - Emit observation-level OOS rows with null prediction fields
    - Compute fold-level and selected-factor OOS return diagnostics
    - Compute OOS IC via ``cqros.research.information_coefficient``
    - Leave annualized Sharpe and equity drawdown null (documented)
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``numpy``, ``polars``, ``cqros.walk_forward.evaluation_input``,
    ``cqros.walk_forward.evaluation_schema``, and
    ``cqros.walk_forward.exceptions``.

Public API:
    ``WalkForwardEvaluationArtifacts``, ``WalkForwardEvaluator``,
    ``TARGET_COLUMN``, ``evaluate_walk_forward_panel``.

Notes:
    ``future_return_1`` is used only for retrospective OOS diagnostics.
    Training observations never enter OOS return or IC metrics.
    Prediction/residual/correct remain null because the current engine does
    not produce forecasts. Observation artifacts persist OOS rows only;
    overlapping TRAIN windows are counted in fold/factor metrics without
    being materialized in the lake artifact. ``raw_oos_ic`` uses Polars Spearman
    ``pl.corr`` on the unchanged raw ``factor_value``. ``oriented_oos_ic`` /
    ``oos_ic`` apply Factor Selection ``selected_direction`` inherited from
    selection-time signed IC and never recompute direction from OOS rows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from cqros.walk_forward.evaluation_input import TARGET_COLUMN
from cqros.walk_forward.evaluation_schema import (
    EVALUATION_FACTOR_METRIC_COLUMNS,
    EVALUATION_FACTOR_METRIC_SCHEMA,
    EVALUATION_FOLD_METRIC_COLUMNS,
    EVALUATION_FOLD_METRIC_SCHEMA,
    EVALUATION_OBSERVATION_COLUMNS,
    EVALUATION_OBSERVATION_SCHEMA,
    EVALUATION_SUMMARY_COLUMNS,
    EVALUATION_SUMMARY_SCHEMA,
    OBSERVATION_PRIMARY_KEY_COLUMNS,
    UNAVAILABLE_METRIC_NOTES,
    WalkForwardEvaluationPartition,
    WalkForwardEvaluationStatus,
)
from cqros.walk_forward.exceptions import WalkForwardError

__all__ = [
    "TARGET_COLUMN",
    "UNAVAILABLE_METRIC_NOTES",
    "WalkForwardEvaluationArtifacts",
    "WalkForwardEvaluator",
    "evaluate_walk_forward_panel",
]

_logger = logging.getLogger(__name__)

# Match SimpleWalkForwardEngine defaults without importing private constants.
_DEFAULT_TRAIN_WINDOW: Final[int] = 252
_DEFAULT_TEST_WINDOW: Final[int] = 63
_DEFAULT_STEP_SIZE: Final[int] = 63

_ERROR_FRAME_TYPE: Final[str] = "WF_EVAL_RESULT_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "WF_EVAL_RESULT_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "WF_EVAL_RESULT_MISSING_COLUMNS"
_ERROR_INVALID_CONFIG: Final[str] = "WF_EVAL_RESULT_INVALID_CONFIG"
_ERROR_DUPLICATE_KEYS: Final[str] = "WF_EVAL_RESULT_DUPLICATE_KEYS"

_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "factor_name",
    "factor_version",
    "factor_value",
    "selected",
    "selection_time",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
    TARGET_COLUMN,
)

_FACTOR_VALUE_COLUMN: Final[str] = "factor_value"
_ORIENTED_FACTOR_ALIAS: Final[str] = "_oriented_factor_value"
_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_time",
    "symbol",
    "factor_name",
    "factor_version",
)
_OBSERVATION_SORT: Final[tuple[str, ...]] = (
    "fold_id",
    "partition",
    "observation_time",
    "symbol",
    "factor_name",
    "factor_version",
)
_CROSS_SECTION_SYMBOL: Final[str] = ""


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationArtifacts:
    """Immutable evaluation outputs for one manager/timeframe/year panel.

    Attributes:
        observations: OOS observation-level evaluation rows (or TRAIN+OOS
            when produced by ``evaluate_with_train``).
        fold_metrics: One row per fold.
        factor_metrics: One row per selected factor per fold.
        summary: Single-row panel summary.
    """

    observations: pl.DataFrame
    fold_metrics: pl.DataFrame
    factor_metrics: pl.DataFrame
    summary: pl.DataFrame


class WalkForwardEvaluator:
    """Build Walk-Forward evaluation artifacts from enriched evaluation input.

    Fold windows match ``SimpleWalkForwardEngine`` rolling semantics.
    ``future_return_1`` never influences fold construction.
    """

    __slots__ = ("_logger", "_step_size", "_test_window", "_train_window")

    def __init__(
        self,
        *,
        train_window: int = _DEFAULT_TRAIN_WINDOW,
        test_window: int = _DEFAULT_TEST_WINDOW,
        step_size: int = _DEFAULT_STEP_SIZE,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize fold-window configuration.

        Args:
            train_window: Training window length in evaluation-input rows.
            test_window: OOS window length in evaluation-input rows.
            step_size: Row advance between successive folds.
            logger: Optional logger instance.
        """
        self._train_window = _require_positive_int(train_window, "train_window")
        self._test_window = _require_positive_int(test_window, "test_window")
        self._step_size = _require_positive_int(step_size, "step_size")
        self._logger = logger if logger is not None else _logger

    def evaluate(
        self,
        evaluation_input: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        year: int,
    ) -> WalkForwardEvaluationArtifacts:
        """Evaluate one enriched Walk-Forward panel (OOS observations only)."""
        return self._evaluate(
            evaluation_input,
            manager=manager,
            engine=engine,
            year=year,
            include_train_observations=False,
        )

    def evaluate_with_train(
        self,
        evaluation_input: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        year: int,
    ) -> WalkForwardEvaluationArtifacts:
        """Evaluate a panel and include TRAIN observation rows for tests."""
        return self._evaluate(
            evaluation_input,
            manager=manager,
            engine=engine,
            year=year,
            include_train_observations=True,
        )

    def _evaluate(
        self,
        evaluation_input: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        year: int,
        include_train_observations: bool,
    ) -> WalkForwardEvaluationArtifacts:
        """Shared evaluation path for production and TRAIN-inclusive tests."""
        frame = _validate_evaluation_input(evaluation_input)
        sorted_frame = frame.sort(list(_SORT_COLUMNS))
        if sorted_frame["timeframe"].n_unique() != 1:
            raise WalkForwardError(
                "evaluation input must contain exactly one timeframe",
                error_code=_ERROR_MISSING_COLUMNS,
                details={
                    "timeframes": tuple(sorted_frame["timeframe"].unique().to_list()),
                },
            )
        timeframe = str(sorted_frame["timeframe"][0])
        folds = _fold_windows(
            row_count=sorted_frame.height,
            train_window=self._train_window,
            test_window=self._test_window,
            step_size=self._step_size,
        )
        self._logger.info(
            "Evaluating walk-forward panel",
            extra={
                "manager": manager,
                "engine": engine,
                "timeframe": timeframe,
                "year": year,
                "input_rows": sorted_frame.height,
                "folds": len(folds),
            },
        )

        indexed = sorted_frame.with_row_index("_row")
        oos_joined = _build_oos_joined(
            sorted_frame=sorted_frame,
            indexed=indexed,
            folds=folds,
            test_window=self._test_window,
            step_size=self._step_size,
        )
        fold_metrics = _build_fold_metrics(
            oos_joined,
            folds=folds,
            manager=manager,
            engine=engine,
            timeframe=timeframe,
            year=year,
        )
        factor_train_counts = _factor_train_counts(sorted_frame, folds)
        factor_metrics = _build_factor_metrics(
            oos_joined,
            factor_train_counts=factor_train_counts,
            manager=manager,
            engine=engine,
            timeframe=timeframe,
            year=year,
        )
        del factor_train_counts
        observations = _materialize_observations(
            oos_joined,
            fold_metrics=fold_metrics,
            manager=manager,
            engine=engine,
            timeframe=timeframe,
            year=year,
        )
        del oos_joined
        if include_train_observations:
            train_map = _partition_assignment_frame(
                folds,
                partition=WalkForwardEvaluationPartition.TRAIN.value,
                use_test=False,
            )
            train_joined = indexed.join(train_map, on="_row", how="inner")
            del train_map
            train_observations = _materialize_observations(
                train_joined,
                fold_metrics=fold_metrics,
                manager=manager,
                engine=engine,
                timeframe=timeframe,
                year=year,
            )
            del train_joined
            observations = pl.concat(
                [train_observations, observations],
                how="vertical",
            )
            del train_observations
        del indexed
        del sorted_frame

        observations = observations.sort(list(_OBSERVATION_SORT))
        _require_unique_observation_keys(observations)

        summary = _panel_summary(
            observations=observations,
            fold_metrics=fold_metrics,
            factor_metrics=factor_metrics,
            manager=manager,
            engine=engine,
            timeframe=timeframe,
            year=year,
        )
        return WalkForwardEvaluationArtifacts(
            observations=observations,
            fold_metrics=fold_metrics,
            factor_metrics=factor_metrics,
            summary=summary,
        )


def evaluate_walk_forward_panel(
    evaluation_input: pl.DataFrame,
    *,
    manager: str,
    engine: str,
    year: int,
    train_window: int = _DEFAULT_TRAIN_WINDOW,
    test_window: int = _DEFAULT_TEST_WINDOW,
    step_size: int = _DEFAULT_STEP_SIZE,
) -> WalkForwardEvaluationArtifacts:
    """Convenience wrapper around ``WalkForwardEvaluator.evaluate``."""
    return WalkForwardEvaluator(
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    ).evaluate(
        evaluation_input,
        manager=manager,
        engine=engine,
        year=year,
    )


@dataclass(frozen=True, slots=True)
class _FoldWindow:
    """Index bounds for one rolling fold."""

    fold_id: int
    train_start: int
    train_rows: int
    test_start: int
    test_rows: int


def _fold_windows(
    *,
    row_count: int,
    train_window: int,
    test_window: int,
    step_size: int,
) -> tuple[_FoldWindow, ...]:
    """Build rolling fold windows matching ``SimpleWalkForwardEngine``."""
    required = train_window + test_window
    folds: list[_FoldWindow] = []
    fold_id = 1
    start = 0
    while start + required <= row_count:
        train_end = start + train_window
        folds.append(
            _FoldWindow(
                fold_id=fold_id,
                train_start=start,
                train_rows=train_window,
                test_start=train_end,
                test_rows=test_window,
            )
        )
        fold_id += 1
        start += step_size

    if folds:
        return tuple(folds)

    if row_count == 1:
        train_end_index = 1
        test_start_index = 0
        test_end_index = 1
    elif row_count <= train_window:
        train_end_index = max(row_count - 1, 0)
        test_start_index = max(row_count - 1, 0)
        test_end_index = row_count
    else:
        train_end_index = train_window
        test_start_index = train_window
        test_end_index = row_count
    return (
        _FoldWindow(
            fold_id=1,
            train_start=0,
            train_rows=max(train_end_index, 0),
            test_start=test_start_index,
            test_rows=max(test_end_index - test_start_index, 0),
        ),
    )


def _build_oos_joined(
    *,
    sorted_frame: pl.DataFrame,
    indexed: pl.DataFrame,
    folds: tuple[_FoldWindow, ...],
    test_window: int,
    step_size: int,
) -> pl.DataFrame:
    """Assemble OOS rows with fold_id/partition using a memory-aware path."""
    if not folds:
        return sorted_frame.clear().with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("fold_id"),
                pl.lit(None, dtype=pl.String).alias("partition"),
            ]
        )

    contiguous = (
        step_size == test_window
        and all(fold.test_rows == test_window for fold in folds)
        and folds[0].test_start + len(folds) * test_window
        == folds[-1].test_start + folds[-1].test_rows
    )
    if contiguous:
        oos_start = folds[0].test_start
        oos_height = len(folds) * test_window
        return (
            sorted_frame.slice(oos_start, oos_height)
            .with_row_index("_local")
            .with_columns(
                [
                    ((pl.col("_local") // test_window) + 1).cast(pl.Int32).alias("fold_id"),
                    pl.lit(WalkForwardEvaluationPartition.OOS.value).alias("partition"),
                ]
            )
            .drop("_local")
        )

    oos_map = _partition_assignment_frame(
        folds,
        partition=WalkForwardEvaluationPartition.OOS.value,
        use_test=True,
    )
    return indexed.join(oos_map, on="_row", how="inner")


def _partition_assignment_frame(
    folds: tuple[_FoldWindow, ...],
    *,
    partition: str,
    use_test: bool,
) -> pl.DataFrame:
    """Build ``(_row, fold_id, partition)`` assignments for train or OOS."""
    total = 0
    for fold in folds:
        total += fold.test_rows if use_test else fold.train_rows
    if total == 0:
        return pl.DataFrame(
            schema={
                "_row": pl.UInt32,
                "fold_id": pl.Int32,
                "partition": pl.String,
            }
        )

    row_ids = np.empty(total, dtype=np.uint32)
    fold_ids = np.empty(total, dtype=np.int32)
    offset = 0
    for fold in folds:
        if use_test:
            start = fold.test_start
            length = fold.test_rows
        else:
            start = fold.train_start
            length = fold.train_rows
        if length <= 0:
            continue
        end = start + length
        row_ids[offset : offset + length] = np.arange(start, end, dtype=np.uint32)
        fold_ids[offset : offset + length] = fold.fold_id
        offset += length

    return pl.DataFrame(
        {
            "_row": row_ids[:offset],
            "fold_id": fold_ids[:offset],
        }
    ).with_columns(pl.lit(partition).alias("partition"))


def _materialize_observations(
    joined: pl.DataFrame,
    *,
    fold_metrics: pl.DataFrame,
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Cast joined evaluation rows into the observation schema."""
    if joined.height == 0:
        return pl.DataFrame(schema=dict(EVALUATION_OBSERVATION_SCHEMA))

    status_lookup = fold_metrics.select(["fold_id", "status"])
    with_status = joined.join(status_lookup, on="fold_id", how="left").with_columns(
        pl.col("status").fill_null(WalkForwardEvaluationStatus.FAIL.value)
    )
    return (
        with_status.select(
            [
                pl.lit(manager).alias("manager"),
                pl.lit(engine).alias("engine"),
                pl.col("symbol"),
                pl.lit(timeframe).alias("timeframe"),
                pl.lit(year, dtype=pl.Int32).alias("year"),
                pl.col("fold_id").cast(pl.Int32),
                pl.col("open_time").alias("observation_time"),
                pl.col("factor_name"),
                pl.col("factor_version"),
                pl.col("selected"),
                pl.col("partition"),
                pl.col(TARGET_COLUMN).alias("future_return_1"),
                pl.col(_FACTOR_VALUE_COLUMN),
                pl.col("selection_ic"),
                pl.col("selected_direction"),
                pl.col("orientation_policy"),
                pl.lit(None, dtype=pl.Float64).alias("prediction"),
                pl.lit(None, dtype=pl.Float64).alias("residual"),
                pl.lit(None, dtype=pl.Boolean).alias("correct"),
                pl.col("status"),
            ]
        )
        .select(list(EVALUATION_OBSERVATION_COLUMNS))
        .cast(EVALUATION_OBSERVATION_SCHEMA)
    )


def _build_fold_metrics(
    oos_joined: pl.DataFrame,
    *,
    folds: tuple[_FoldWindow, ...],
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Aggregate fold-level OOS return diagnostics."""
    if not folds:
        return pl.DataFrame(schema=dict(EVALUATION_FOLD_METRIC_SCHEMA))

    train_rows_by_fold = pl.DataFrame(
        {
            "fold_id": [fold.fold_id for fold in folds],
            "train_rows": [fold.train_rows for fold in folds],
            "oos_rows": [fold.test_rows for fold in folds],
        }
    ).with_columns(pl.col("fold_id").cast(pl.Int32))

    selected = oos_joined.filter(pl.col("selected"))
    if selected.height == 0:
        stats = train_rows_by_fold.with_columns(
            [
                pl.lit(0, dtype=pl.Int64).alias("oos_non_null_returns"),
                pl.lit(None, dtype=pl.Float64).alias("oos_return_mean"),
                pl.lit(None, dtype=pl.Float64).alias("oos_return_std"),
                pl.lit(None, dtype=pl.Float64).alias("oos_return_min"),
                pl.lit(None, dtype=pl.Float64).alias("oos_return_max"),
                pl.lit(None, dtype=pl.Float64).alias("oos_positive_rate"),
                pl.lit(None, dtype=pl.Float64).alias("oos_cumulative_return"),
                pl.lit(WalkForwardEvaluationStatus.FAIL.value).alias("status"),
            ]
        )
    else:
        stats = (
            selected.group_by("fold_id")
            .agg(
                [
                    pl.col(TARGET_COLUMN).drop_nulls().len().alias("oos_non_null_returns"),
                    pl.col(TARGET_COLUMN).drop_nulls().mean().alias("oos_return_mean"),
                    pl.col(TARGET_COLUMN).drop_nulls().std(ddof=1).alias("oos_return_std"),
                    pl.col(TARGET_COLUMN).drop_nulls().min().alias("oos_return_min"),
                    pl.col(TARGET_COLUMN).drop_nulls().max().alias("oos_return_max"),
                    (pl.col(TARGET_COLUMN).drop_nulls() > 0.0).mean().alias("oos_positive_rate"),
                    ((pl.col(TARGET_COLUMN).drop_nulls() + 1.0).product() - 1.0).alias(
                        "oos_cumulative_return"
                    ),
                ]
            )
            .with_columns(
                pl.when(pl.col("oos_non_null_returns") > 0)
                .then(pl.lit(WalkForwardEvaluationStatus.PASS.value))
                .otherwise(pl.lit(WalkForwardEvaluationStatus.FAIL.value))
                .alias("status")
            )
        )
        stats = train_rows_by_fold.join(stats, on="fold_id", how="left").with_columns(
            [
                pl.col("oos_non_null_returns").fill_null(0),
                pl.col("status").fill_null(WalkForwardEvaluationStatus.FAIL.value),
            ]
        )

    return (
        stats.with_columns(
            [
                pl.lit(manager).alias("manager"),
                pl.lit(engine).alias("engine"),
                pl.lit(_CROSS_SECTION_SYMBOL).alias("symbol"),
                pl.lit(timeframe).alias("timeframe"),
                pl.lit(year, dtype=pl.Int32).alias("year"),
                pl.lit(None, dtype=pl.Float64).alias("oos_sharpe"),
                pl.lit(None, dtype=pl.Float64).alias("oos_max_drawdown"),
            ]
        )
        .select(list(EVALUATION_FOLD_METRIC_COLUMNS))
        .cast(EVALUATION_FOLD_METRIC_SCHEMA)
        .sort(["fold_id", "symbol"])
    )


def _factor_train_counts(
    sorted_frame: pl.DataFrame,
    folds: tuple[_FoldWindow, ...],
) -> pl.DataFrame:
    """Count train-window rows per factor identity per fold without explosion.

    Uses integer factor codes and ``numpy.bincount`` over each train window so
    production panels with hundreds of thousands of overlapping folds stay
    within memory budgets.
    """
    if not folds:
        return pl.DataFrame(
            schema={
                "fold_id": pl.Int32,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "train_rows": pl.Int64,
            }
        )

    coded = sorted_frame.select(
        [
            pl.col("factor_name"),
            pl.col("factor_version"),
        ]
    ).with_columns(
        pl.struct(["factor_name", "factor_version"]).rank("dense").cast(pl.Int32).alias("_code")
    )
    lookup = coded.select(["_code", "factor_name", "factor_version"]).unique().sort("_code")
    codes = coded["_code"].to_numpy().astype(np.int32, copy=False)
    n_codes = int(lookup.height)
    if n_codes == 0:
        return pl.DataFrame(
            schema={
                "fold_id": pl.Int32,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "train_rows": pl.Int64,
            }
        )

    fold_ids: list[int] = []
    code_ids: list[int] = []
    counts: list[int] = []
    for fold in folds:
        if fold.train_rows <= 0:
            continue
        window = codes[fold.train_start : fold.train_start + fold.train_rows]
        bincounts = np.bincount(window, minlength=n_codes + 1)
        nonzero = np.nonzero(bincounts)[0]
        for code in nonzero.tolist():
            if code == 0:
                # dense rank is 1-based in Polars; skip unused zero bucket.
                if bincounts[code] == 0:
                    continue
            fold_ids.append(fold.fold_id)
            code_ids.append(int(code))
            counts.append(int(bincounts[code]))

    if not fold_ids:
        return pl.DataFrame(
            schema={
                "fold_id": pl.Int32,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "train_rows": pl.Int64,
            }
        )

    return (
        pl.DataFrame(
            {
                "fold_id": fold_ids,
                "_code": code_ids,
                "train_rows": counts,
            }
        )
        .with_columns(
            [
                pl.col("fold_id").cast(pl.Int32),
                pl.col("_code").cast(pl.Int32),
                pl.col("train_rows").cast(pl.Int64),
            ]
        )
        .join(lookup, on="_code", how="left")
        .select(["fold_id", "factor_name", "factor_version", "train_rows"])
    )


def _build_factor_metrics(
    oos_joined: pl.DataFrame,
    *,
    factor_train_counts: pl.DataFrame,
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Aggregate selected-factor OOS diagnostics including Spearman IC.

    ``raw_oos_ic`` uses Polars Spearman ``pl.corr`` on raw ``factor_value``.
    ``oriented_oos_ic`` multiplies by inherited ``selected_direction`` before
    correlation. ``oos_ic`` equals ``oriented_oos_ic`` so primary evaluation
    consumes the leakage-safe orientation without discarding the raw metric.
    """
    selected = oos_joined.filter(pl.col("selected"))
    if selected.height == 0:
        return pl.DataFrame(schema=dict(EVALUATION_FACTOR_METRIC_SCHEMA))

    oriented = selected.with_columns(
        (pl.col(_FACTOR_VALUE_COLUMN) * pl.col("selected_direction").cast(pl.Float64)).alias(
            _ORIENTED_FACTOR_ALIAS
        )
    )
    grouped = oriented.group_by(["fold_id", "factor_name", "factor_version"]).agg(
        [
            pl.len().alias("oos_rows"),
            pl.col(TARGET_COLUMN).drop_nulls().mean().alias("oos_return_mean"),
            pl.col(TARGET_COLUMN).drop_nulls().std(ddof=1).alias("oos_return_std"),
            (pl.col(TARGET_COLUMN).drop_nulls() > 0.0).mean().alias("oos_positive_rate"),
            pl.corr(_FACTOR_VALUE_COLUMN, TARGET_COLUMN, method="spearman").alias("raw_oos_ic"),
            pl.corr(_ORIENTED_FACTOR_ALIAS, TARGET_COLUMN, method="spearman").alias(
                "oriented_oos_ic"
            ),
            ((pl.col(TARGET_COLUMN).drop_nulls() + 1.0).product() - 1.0).alias(
                "oos_cumulative_return"
            ),
            pl.col("selection_ic").first().alias("selection_ic"),
            pl.col("selected_direction").first().alias("selected_direction"),
            pl.col("orientation_policy").first().alias("orientation_policy"),
        ]
    )
    return (
        grouped.join(
            factor_train_counts,
            on=["fold_id", "factor_name", "factor_version"],
            how="left",
        )
        .with_columns(pl.col("train_rows").fill_null(0))
        .with_columns(
            [
                pl.col("oriented_oos_ic").alias("oos_ic"),
                pl.lit(manager).alias("manager"),
                pl.lit(engine).alias("engine"),
                pl.lit(_CROSS_SECTION_SYMBOL).alias("symbol"),
                pl.lit(timeframe).alias("timeframe"),
                pl.lit(year, dtype=pl.Int32).alias("year"),
            ]
        )
        .select(list(EVALUATION_FACTOR_METRIC_COLUMNS))
        .cast(EVALUATION_FACTOR_METRIC_SCHEMA)
        .sort(["fold_id", "symbol", "factor_name", "factor_version"])
    )


def _as_float(value: object) -> float | None:
    """Convert a scalar to float, or ``None`` when undefined."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _cumulative_return_from_list(values: list[object] | None) -> float | None:
    """Compound ``(1 + r)`` for a list of non-null returns."""
    if not values:
        return None
    growth = 1.0
    for value in values:
        number = _as_float(value)
        if number is None:
            return None
        growth *= 1.0 + number
    return growth - 1.0


def _panel_summary(
    *,
    observations: pl.DataFrame,
    fold_metrics: pl.DataFrame,
    factor_metrics: pl.DataFrame,
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Build the single-row panel summary from OOS selected returns."""
    oos = observations.filter(pl.col("partition") == WalkForwardEvaluationPartition.OOS.value)
    selected_returns = oos.filter(pl.col("selected")).get_column("future_return_1")
    clean = selected_returns.drop_nulls()
    non_null = clean.len()
    if non_null == 0:
        mean = std = positive_rate = cumulative = None
        status = WalkForwardEvaluationStatus.FAIL.value
    else:
        values = clean.to_list()
        mean = _as_float(clean.mean())
        std = _as_float(clean.std(ddof=1)) if non_null >= 2 else None
        positive_count = int((clean > 0.0).sum())
        positive_rate = float(positive_count) / float(non_null)
        cumulative = _cumulative_return_from_list(values)
        status = WalkForwardEvaluationStatus.PASS.value

    unique_factors = (
        factor_metrics.select(["factor_name", "factor_version"]).n_unique()
        if factor_metrics.height > 0
        else 0
    )
    row = {
        "manager": manager,
        "engine": engine,
        "symbol": _CROSS_SECTION_SYMBOL,
        "timeframe": timeframe,
        "year": year,
        "folds": fold_metrics.height,
        "train_rows": (int(fold_metrics["train_rows"].sum()) if fold_metrics.height > 0 else 0),
        "oos_rows": int(oos.height),
        "oos_non_null_returns": non_null,
        "oos_return_mean": mean,
        "oos_return_std": std,
        "oos_positive_rate": positive_rate,
        "oos_cumulative_return": cumulative,
        "oos_sharpe": None,
        "oos_max_drawdown": None,
        "unique_selected_factors": unique_factors,
        "status": status,
        "error": None,
    }
    return (
        pl.DataFrame([row]).select(list(EVALUATION_SUMMARY_COLUMNS)).cast(EVALUATION_SUMMARY_SCHEMA)
    )


def _validate_evaluation_input(frame: object) -> pl.DataFrame:
    """Validate evaluation-input structure."""
    if not isinstance(frame, pl.DataFrame):
        raise WalkForwardError(
            "evaluation input must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise WalkForwardError(
            "evaluation input must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise WalkForwardError(
            "evaluation input is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _REQUIRED_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    return frame


def _require_unique_observation_keys(frame: pl.DataFrame) -> None:
    """Reject duplicate evaluation observation primary keys."""
    key_columns = list(OBSERVATION_PRIMARY_KEY_COLUMNS)
    unique_keys = frame.select(key_columns).n_unique()
    if unique_keys != frame.height:
        raise WalkForwardError(
            "evaluation observations contain duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "row_count": frame.height,
                "unique_key_count": unique_keys,
                "primary_key": OBSERVATION_PRIMARY_KEY_COLUMNS,
            },
        )


def _require_positive_int(value: object, name: str) -> int:
    """Validate positive integer configuration."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WalkForwardError(
            f"{name} must be a positive integer",
            error_code=_ERROR_INVALID_CONFIG,
            details={"parameter": name, "actual_value": value},
        )
    return value
