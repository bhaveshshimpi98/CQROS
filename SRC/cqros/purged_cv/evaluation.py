"""CQROS Purged-CV evaluation engine.

Purpose:
    Persist diagnostic OOS evaluation results for already-generated
    purged-CV ledgers without mutating ``data/purged_cv`` or inventing
    predictive / performance metrics that the engine does not produce.

Responsibilities:
    - Evaluate one purged-CV panel into observation / fold / factor / summary
      artifacts
    - Reconstruct purge/embargo membership from the Walk-Forward ledger
    - Accept non-contiguous training windows (train may exist on both sides
      of the test block)
    - Order folds by ``fold_id`` (never by ``train_start_time``)
    - Compute Labels ``future_return_1`` OOS diagnostics only from evaluation
      input mapped through purged TEST Walk-Forward windows
    - Leave prediction / Sharpe / drawdown / PnL null
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``polars``, ``cqros.purged_cv.engine``, ``cqros.purged_cv.exceptions``,
    ``cqros.purged_cv.evaluation_schema``, ``cqros.purged_cv.schema``, and
    ``cqros.walk_forward.evaluation_input.TARGET_COLUMN``.

Public API:
    ``PurgedCVEvaluationArtifacts``, ``PurgedCVEvaluator``,
    ``TARGET_COLUMN``, ``evaluate_purged_cv_panel``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.purged_cv.engine import (
    WALK_FORWARD_INPUT_COLUMNS,
    SimplePurgedCVEngine,
    validate_walk_forward_frame,
)
from cqros.purged_cv.evaluation_schema import (
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
    PurgedCVEvaluationPartition,
    PurgedCVEvaluationStatus,
)
from cqros.purged_cv.exceptions import PurgedCVError
from cqros.purged_cv.schema import (
    CANONICAL_COLUMN_ORDER,
    PRIMARY_KEY_COLUMNS,
    PURGED_CV_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.walk_forward.evaluation_input import TARGET_COLUMN

__all__ = [
    "TARGET_COLUMN",
    "UNAVAILABLE_METRIC_NOTES",
    "PurgedCVEvaluationArtifacts",
    "PurgedCVEvaluator",
    "evaluate_purged_cv_panel",
]

_logger = logging.getLogger(__name__)

_ERROR_FRAME_TYPE: Final[str] = "PCV_EVAL_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PCV_EVAL_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PCV_EVAL_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PCV_EVAL_DUPLICATE_KEYS"
_ERROR_INVALID_CONFIG: Final[str] = "PCV_EVAL_INVALID_CONFIG"
_ERROR_TIMEFRAME: Final[str] = "PCV_EVAL_TIMEFRAME"

_CHRONOLOGICAL_COLUMN: Final[str] = "test_start"
_FACTOR_VALUE_COLUMN: Final[str] = "factor_value"
_ORIENTED_FACTOR_ALIAS: Final[str] = "_oriented_factor_value"
_CROSS_SECTION_SYMBOL: Final[str] = ""

_REQUIRED_EVAL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
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

_OBSERVATION_SORT: Final[tuple[str, ...]] = (
    "fold_id",
    "partition",
    "observation_time",
    "symbol",
    "factor_name",
    "factor_version",
)


@dataclass(frozen=True, slots=True)
class PurgedCVEvaluationArtifacts:
    """Immutable evaluation outputs for one manager/timeframe/year panel."""

    observations: pl.DataFrame
    fold_metrics: pl.DataFrame
    factor_metrics: pl.DataFrame
    summary: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _FoldMembership:
    """Reconstructed index membership for one purged fold."""

    fold_id: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purge_before_indices: tuple[int, ...]
    purge_after_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]


class PurgedCVEvaluator:
    """Build Purged-CV evaluation artifacts from ledger + optional Labels input.

    Fold ordering is canonical by ``fold_id``. Training may be non-contiguous
    around the test block; ``train_end_time < test_start_time`` is never used
    as a validity rule.
    """

    __slots__ = ("_logger",)

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the evaluator.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def evaluate(
        self,
        purged_cv: pl.DataFrame,
        walk_forward: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        exchange: str,
        market: str,
        year: int,
        evaluation_input: pl.DataFrame | None = None,
    ) -> PurgedCVEvaluationArtifacts:
        """Evaluate one purged-CV panel (OOS observations only when available)."""
        return self._evaluate(
            purged_cv,
            walk_forward,
            manager=manager,
            engine=engine,
            exchange=exchange,
            market=market,
            year=year,
            evaluation_input=evaluation_input,
            include_train_observations=False,
        )

    def evaluate_with_train(
        self,
        purged_cv: pl.DataFrame,
        walk_forward: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        exchange: str,
        market: str,
        year: int,
        evaluation_input: pl.DataFrame | None = None,
    ) -> PurgedCVEvaluationArtifacts:
        """Evaluate a panel and include TRAIN observation rows for tests."""
        return self._evaluate(
            purged_cv,
            walk_forward,
            manager=manager,
            engine=engine,
            exchange=exchange,
            market=market,
            year=year,
            evaluation_input=evaluation_input,
            include_train_observations=True,
        )

    def _evaluate(
        self,
        purged_cv: pl.DataFrame,
        walk_forward: pl.DataFrame,
        *,
        manager: str,
        engine: str,
        exchange: str,
        market: str,
        year: int,
        evaluation_input: pl.DataFrame | None,
        include_train_observations: bool,
    ) -> PurgedCVEvaluationArtifacts:
        ledger = _validate_purged_cv(purged_cv)
        wf = _validate_walk_forward(walk_forward)
        sorted_ledger = ledger.sort(["fold_id", *PRIMARY_KEY_COLUMNS])
        if sorted_ledger["timeframe"].n_unique() != 1:
            raise PurgedCVError(
                "purged-CV evaluation input must contain exactly one timeframe",
                error_code=_ERROR_TIMEFRAME,
                details={
                    "timeframes": tuple(sorted(sorted_ledger["timeframe"].unique().to_list())),
                },
            )
        timeframe = str(sorted_ledger["timeframe"][0])
        memberships = _reconstruct_memberships(wf, sorted_ledger)
        self._logger.info(
            "Evaluating purged-CV panel",
            extra={
                "manager": manager,
                "engine": engine,
                "timeframe": timeframe,
                "year": year,
                "folds": sorted_ledger.height,
            },
        )

        eval_frame: pl.DataFrame | None = None
        label_stats: dict[int, dict[str, object]] = {}
        factor_parts: list[pl.DataFrame] = []
        observation_parts: list[pl.DataFrame] = []
        if evaluation_input is not None:
            eval_frame = _validate_evaluation_input(evaluation_input)
            label_stats, factor_parts, observation_parts = _aggregate_oos_by_fold(
                eval_frame,
                walk_forward=wf,
                memberships=memberships,
                manager=manager,
                engine=engine,
                timeframe=timeframe,
                year=year,
                materialize_observations=(
                    include_train_observations or eval_frame.height < 100_000
                ),
            )

        fold_metrics = _build_fold_metrics(
            sorted_ledger,
            memberships=memberships,
            walk_forward=wf,
            label_stats=label_stats,
            manager=manager,
            engine=engine,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        if factor_parts:
            factor_metrics = (
                pl.concat(factor_parts, how="vertical")
                .select(list(EVALUATION_FACTOR_METRIC_COLUMNS))
                .cast(EVALUATION_FACTOR_METRIC_SCHEMA)
                .sort(["fold_id", "symbol", "factor_name", "factor_version"])
            )
        else:
            factor_metrics = pl.DataFrame(schema=dict(EVALUATION_FACTOR_METRIC_SCHEMA))

        if include_train_observations and eval_frame is not None:
            train_joined = _map_partition_observations(
                eval_frame,
                walk_forward=wf,
                memberships=memberships,
                partition=PurgedCVEvaluationPartition.TRAIN.value,
                use_train=True,
            )
            observation_parts.append(
                _materialize_observations(
                    train_joined,
                    fold_metrics=fold_metrics,
                    manager=manager,
                    engine=engine,
                    timeframe=timeframe,
                    year=year,
                )
            )
            # Attach train row counts onto factor metrics for tests.
            train_counts = _factor_train_counts(
                eval_frame,
                walk_forward=wf,
                memberships=memberships,
            )
            if factor_metrics.height > 0 and train_counts.height > 0:
                factor_metrics = (
                    factor_metrics.drop("train_rows")
                    .join(
                        train_counts,
                        on=["fold_id", "factor_name", "factor_version"],
                        how="left",
                    )
                    .with_columns(pl.col("train_rows").fill_null(0))
                    .select(list(EVALUATION_FACTOR_METRIC_COLUMNS))
                    .cast(EVALUATION_FACTOR_METRIC_SCHEMA)
                    .sort(["fold_id", "symbol", "factor_name", "factor_version"])
                )

        if observation_parts:
            observations = pl.concat(observation_parts, how="vertical").sort(
                list(_OBSERVATION_SORT)
            )
        else:
            observations = pl.DataFrame(schema=dict(EVALUATION_OBSERVATION_SCHEMA))
        _require_unique_observation_keys(observations)
        summary = _panel_summary(
            fold_metrics=fold_metrics,
            factor_metrics=factor_metrics,
            label_stats=label_stats,
            manager=manager,
            engine=engine,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        return PurgedCVEvaluationArtifacts(
            observations=observations,
            fold_metrics=fold_metrics,
            factor_metrics=factor_metrics,
            summary=summary,
        )


def evaluate_purged_cv_panel(
    purged_cv: pl.DataFrame,
    walk_forward: pl.DataFrame,
    *,
    manager: str,
    engine: str,
    exchange: str,
    market: str,
    year: int,
    evaluation_input: pl.DataFrame | None = None,
) -> PurgedCVEvaluationArtifacts:
    """Convenience wrapper around ``PurgedCVEvaluator.evaluate``."""
    return PurgedCVEvaluator().evaluate(
        purged_cv,
        walk_forward,
        manager=manager,
        engine=engine,
        exchange=exchange,
        market=market,
        year=year,
        evaluation_input=evaluation_input,
    )


def _validate_purged_cv(frame: object) -> pl.DataFrame:
    """Validate and cast a purged-CV ledger frame."""
    if not isinstance(frame, pl.DataFrame):
        raise PurgedCVError(
            "purged_cv frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PurgedCVError(
            "purged_cv frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PurgedCVError(
            "purged_cv frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
            },
        )
    duplicates = frame.group_by(list(PRIMARY_KEY_COLUMNS)).len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise PurgedCVError(
            "purged_cv frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={"duplicate_rows": duplicates.height},
        )
    return frame.select(list(CANONICAL_COLUMN_ORDER)).cast(PURGED_CV_SCHEMA)


def _validate_walk_forward(frame: object) -> pl.DataFrame:
    """Validate the Walk-Forward ledger used for membership reconstruction."""
    validated = validate_walk_forward_frame(frame)
    missing = [column for column in WALK_FORWARD_INPUT_COLUMNS if column not in validated.columns]
    if missing:
        raise PurgedCVError(
            "walk_forward frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": "walk_forward",
                "missing_columns": tuple(missing),
                "required_columns": WALK_FORWARD_INPUT_COLUMNS,
            },
        )
    if "test_end" not in validated.columns:
        raise PurgedCVError(
            "walk_forward frame is missing test_end for OOS window mapping",
            error_code=_ERROR_MISSING_COLUMNS,
            details={"missing_columns": ("test_end",)},
        )
    return validated.sort(_CHRONOLOGICAL_COLUMN, descending=False)


def _validate_evaluation_input(frame: object) -> pl.DataFrame:
    """Validate Labels/Factors evaluation input assembled upstream."""
    if not isinstance(frame, pl.DataFrame):
        raise PurgedCVError(
            "evaluation_input must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PurgedCVError(
            "evaluation_input must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )
    missing = [column for column in _REQUIRED_EVAL_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise PurgedCVError(
            "evaluation_input is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _REQUIRED_EVAL_INPUT_COLUMNS,
            },
        )
    return frame.sort(["timeframe", "selection_time", "symbol", "factor_name", "factor_version"])


def _reconstruct_memberships(
    walk_forward: pl.DataFrame,
    purged_cv: pl.DataFrame,
) -> tuple[_FoldMembership, ...]:
    """Reconstruct purged fold index membership from Walk-Forward chronology."""
    n_folds = int(purged_cv.height)
    if n_folds <= 0:
        raise PurgedCVError(
            "purged_cv must contain at least one fold",
            error_code=_ERROR_INVALID_CONFIG,
            details={"folds": n_folds},
        )
    purge_size = int(purged_cv["purge_size"][0])
    embargo_size = int(purged_cv["embargo_size"][0])
    if purge_size < 0 or embargo_size < 0:
        raise PurgedCVError(
            "purge_size and embargo_size must be non-negative",
            error_code=_ERROR_INVALID_CONFIG,
            details={"purge_size": purge_size, "embargo_size": embargo_size},
        )
    row_count = walk_forward.height
    boundaries = _fold_boundaries(row_count, n_folds)
    memberships: list[_FoldMembership] = []
    for fold_id, (test_start_index, test_end_index) in enumerate(boundaries, start=1):
        purge_before_start = max(0, test_start_index - purge_size)
        purge_after_end = min(row_count, test_end_index + purge_size)
        embargo_end = min(row_count, test_end_index + purge_size + embargo_size)
        purge_before = tuple(range(purge_before_start, test_start_index))
        purge_after = tuple(range(test_end_index, purge_after_end))
        embargo = tuple(range(purge_after_end, embargo_end))
        train_indices = tuple(
            _train_indices(
                row_count,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                purge_size=purge_size,
                embargo_size=embargo_size,
            )
        )
        test_indices = tuple(range(test_start_index, test_end_index))
        memberships.append(
            _FoldMembership(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_before_indices=purge_before,
                purge_after_indices=purge_after,
                embargo_indices=embargo,
            )
        )
    return tuple(memberships)


def _fold_boundaries(row_count: int, n_folds: int) -> list[tuple[int, int]]:
    """Mirror ``SimplePurgedCVEngine`` contiguous fold boundaries."""
    base_size = row_count // n_folds
    remainder = row_count % n_folds
    boundaries: list[tuple[int, int]] = []
    start = 0
    for fold_index in range(n_folds):
        fold_size = base_size + (1 if fold_index < remainder else 0)
        end = start + fold_size
        boundaries.append((start, end))
        start = end
    return boundaries


def _train_indices(
    row_count: int,
    *,
    test_start_index: int,
    test_end_index: int,
    purge_size: int,
    embargo_size: int,
) -> list[int]:
    """Mirror engine training indices after purge and embargo."""
    purge_before_start = max(0, test_start_index - purge_size)
    post_test_exclusion_end = min(
        row_count,
        test_end_index + purge_size + embargo_size,
    )
    indices: list[int] = []
    for index in range(row_count):
        if test_start_index <= index < test_end_index:
            continue
        if purge_before_start <= index < test_start_index:
            continue
        if test_end_index <= index < post_test_exclusion_end:
            continue
        indices.append(index)
    return indices


def _aggregate_oos_by_fold(
    evaluation_input: pl.DataFrame,
    *,
    walk_forward: pl.DataFrame,
    memberships: tuple[_FoldMembership, ...],
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
    materialize_observations: bool,
) -> tuple[dict[int, dict[str, object]], list[pl.DataFrame], list[pl.DataFrame]]:
    """Aggregate OOS Labels/factor diagnostics one fold at a time.

    Avoids concatenating the full OOS observation frame for large panels.
    """
    label_stats: dict[int, dict[str, object]] = {}
    factor_parts: list[pl.DataFrame] = []
    observation_parts: list[pl.DataFrame] = []
    for membership in memberships:
        matched = _filter_evaluation_window(
            evaluation_input,
            walk_forward=walk_forward,
            indices=membership.test_indices,
        )
        if matched.height == 0:
            continue
        selected = matched.filter(pl.col("selected"))
        if selected.height > 0:
            oriented = selected.with_columns(
                (
                    pl.col(_FACTOR_VALUE_COLUMN) * pl.col("selected_direction").cast(pl.Float64)
                ).alias(_ORIENTED_FACTOR_ALIAS)
            )
            returns = oriented.get_column(TARGET_COLUMN).drop_nulls()
            raw_ic_value = _as_float(
                oriented.select(
                    pl.corr(_FACTOR_VALUE_COLUMN, TARGET_COLUMN, method="spearman")
                ).item()
            )
            oriented_ic_value = _as_float(
                oriented.select(
                    pl.corr(_ORIENTED_FACTOR_ALIAS, TARGET_COLUMN, method="spearman")
                ).item()
            )
            label_stats[membership.fold_id] = {
                "oos_non_null_returns": int(returns.len()),
                "oos_return_mean": _as_float(returns.mean()) if returns.len() else None,
                "oos_return_median": _as_float(returns.median()) if returns.len() else None,
                "oos_return_std": (_as_float(returns.std(ddof=1)) if returns.len() > 1 else None),
                "oos_positive_rate": (_as_float((returns > 0.0).mean()) if returns.len() else None),
                "raw_oos_ic": raw_ic_value,
                "oriented_oos_ic": oriented_ic_value,
                "oos_ic": oriented_ic_value,
            }
            factor_part = (
                oriented.group_by(["factor_name", "factor_version"])
                .agg(
                    [
                        pl.len().alias("oos_rows"),
                        pl.col(TARGET_COLUMN).drop_nulls().mean().alias("oos_return_mean"),
                        pl.col(TARGET_COLUMN).drop_nulls().std(ddof=1).alias("oos_return_std"),
                        (pl.col(TARGET_COLUMN).drop_nulls() > 0.0)
                        .mean()
                        .alias("oos_positive_rate"),
                        pl.corr(_FACTOR_VALUE_COLUMN, TARGET_COLUMN, method="spearman").alias(
                            "raw_oos_ic"
                        ),
                        pl.corr(_ORIENTED_FACTOR_ALIAS, TARGET_COLUMN, method="spearman").alias(
                            "oriented_oos_ic"
                        ),
                        pl.col("selection_ic").first().alias("selection_ic"),
                        pl.col("selected_direction").first().alias("selected_direction"),
                        pl.col("orientation_policy").first().alias("orientation_policy"),
                    ]
                )
                .with_columns(
                    [
                        pl.col("oriented_oos_ic").alias("oos_ic"),
                        pl.lit(membership.fold_id, dtype=pl.Int32).alias("fold_id"),
                        pl.lit(0, dtype=pl.Int64).alias("train_rows"),
                        pl.when(pl.col("oos_rows") > 0)
                        .then(pl.lit(PurgedCVEvaluationStatus.PASS.value))
                        .otherwise(pl.lit(PurgedCVEvaluationStatus.FAIL.value))
                        .alias("status"),
                        pl.lit(manager).alias("manager"),
                        pl.lit(engine).alias("engine"),
                        pl.lit(_CROSS_SECTION_SYMBOL).alias("symbol"),
                        pl.lit(timeframe).alias("timeframe"),
                        pl.lit(year, dtype=pl.Int32).alias("year"),
                    ]
                )
                .select(list(EVALUATION_FACTOR_METRIC_COLUMNS))
            )
            factor_parts.append(factor_part)
        if materialize_observations:
            with_fold = matched.with_columns(
                [
                    pl.lit(membership.fold_id, dtype=pl.Int32).alias("fold_id"),
                    pl.lit(PurgedCVEvaluationPartition.OOS.value).alias("partition"),
                ]
            )
            observation_parts.append(
                _materialize_observations(
                    with_fold,
                    fold_metrics=None,
                    manager=manager,
                    engine=engine,
                    timeframe=timeframe,
                    year=year,
                    default_status=PurgedCVEvaluationStatus.PASS.value,
                )
            )
        del matched
    return label_stats, factor_parts, observation_parts


def _filter_evaluation_window(
    evaluation_input: pl.DataFrame,
    *,
    walk_forward: pl.DataFrame,
    indices: tuple[int, ...],
) -> pl.DataFrame:
    """Return evaluation rows whose selection_time falls in WF index runs."""
    if not indices:
        return evaluation_input.clear()
    parts: list[pl.DataFrame] = []
    for run_start, run_end in _contiguous_runs(indices):
        subset = walk_forward.slice(run_start, run_end - run_start + 1)
        start_time = _require_int(subset["test_start"].min())
        end_time = _require_int(subset["test_end"].max())
        matched = evaluation_input.filter(
            (pl.col("selection_time") >= start_time) & (pl.col("selection_time") <= end_time)
        )
        if matched.height > 0:
            parts.append(matched)
    if not parts:
        return evaluation_input.clear()
    return pl.concat(parts, how="vertical")


def _map_train_observations(
    evaluation_input: pl.DataFrame,
    *,
    walk_forward: pl.DataFrame,
    memberships: tuple[_FoldMembership, ...],
) -> pl.DataFrame:
    """Map evaluation-input bars onto purged TRAIN Walk-Forward windows."""
    return _map_partition_observations(
        evaluation_input,
        walk_forward=walk_forward,
        memberships=memberships,
        partition=PurgedCVEvaluationPartition.TRAIN.value,
        use_train=True,
    )


def _map_partition_observations(
    evaluation_input: pl.DataFrame,
    *,
    walk_forward: pl.DataFrame,
    memberships: tuple[_FoldMembership, ...],
    partition: str,
    use_train: bool,
) -> pl.DataFrame:
    """Join evaluation bars whose selection_time falls in fold window unions.

    Training indices are split into contiguous runs so the gap containing the
    test/purge/embargo block is never silently included via a single min/max.
    """
    parts: list[pl.DataFrame] = []
    for membership in memberships:
        indices = membership.train_indices if use_train else membership.test_indices
        if not indices:
            continue
        for run_start, run_end in _contiguous_runs(indices):
            subset = walk_forward.slice(run_start, run_end - run_start + 1)
            start_time = _require_int(subset["test_start"].min())
            end_time = _require_int(subset["test_end"].max())
            matched = evaluation_input.filter(
                (pl.col("selection_time") >= start_time) & (pl.col("selection_time") <= end_time)
            ).with_columns(
                [
                    pl.lit(membership.fold_id, dtype=pl.Int32).alias("fold_id"),
                    pl.lit(partition).alias("partition"),
                ]
            )
            if matched.height > 0:
                parts.append(matched)
    if not parts:
        return evaluation_input.clear().with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("fold_id"),
                pl.lit(None, dtype=pl.String).alias("partition"),
            ]
        )
    return pl.concat(parts, how="vertical")


def _contiguous_runs(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Split sorted indices into inclusive ``(start, end)`` contiguous runs."""
    if not indices:
        return []
    ordered = tuple(sorted(indices))
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous))
        start = previous = index
    runs.append((start, previous))
    return runs


def _build_fold_metrics(
    purged_cv: pl.DataFrame,
    *,
    memberships: tuple[_FoldMembership, ...],
    walk_forward: pl.DataFrame,
    label_stats: dict[int, dict[str, object]],
    manager: str,
    engine: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Assemble fold-level diagnostics and purge/embargo audit flags."""
    membership_by_id = {item.fold_id: item for item in memberships}
    rebuilt = SimplePurgedCVEngine(
        n_folds=purged_cv.height,
        purge_size=int(purged_cv["purge_size"][0]),
        embargo_size=int(purged_cv["embargo_size"][0]),
    ).build(walk_forward.select(list(WALK_FORWARD_INPUT_COLUMNS)))
    rebuilt_by_fold = {int(row["fold_id"]): row for row in rebuilt.iter_rows(named=True)}
    fold_order_ok = _fold_order_valid(purged_cv)

    rows: list[dict[str, object]] = []
    for row in purged_cv.iter_rows(named=True):
        fold_id = int(row["fold_id"])
        membership = membership_by_id[fold_id]
        rebuilt_row = rebuilt_by_fold[fold_id]
        train_set = set(membership.train_indices)
        test_set = set(membership.test_indices)
        purge_set = set(membership.purge_before_indices) | set(membership.purge_after_indices)
        embargo_set = set(membership.embargo_indices)

        train_test_disjoint = train_set.isdisjoint(test_set)
        purge_valid = train_set.isdisjoint(purge_set) and test_set.isdisjoint(purge_set)
        embargo_valid = train_set.isdisjoint(embargo_set) and test_set.isdisjoint(embargo_set)
        timestamp_valid = _timestamp_valid(row) and _row_matches_rebuilt(row, rebuilt_row)
        counts_match = int(row["train_rows"]) == len(membership.train_indices) and int(
            row["test_rows"]
        ) == len(membership.test_indices)
        audit_pass = (
            train_test_disjoint
            and purge_valid
            and embargo_valid
            and fold_order_ok
            and timestamp_valid
            and counts_match
        )
        stats = label_stats.get(fold_id, {})
        status = (
            PurgedCVEvaluationStatus.PASS.value
            if audit_pass and str(row["status"]) == "PASS"
            else PurgedCVEvaluationStatus.FAIL.value
        )
        rows.append(
            {
                "manager": manager,
                "engine": engine,
                "exchange": exchange,
                "market": market,
                "symbol": _CROSS_SECTION_SYMBOL,
                "timeframe": timeframe,
                "year": year,
                "fold_id": fold_id,
                "strategy_name": str(row["strategy_name"]),
                "strategy_version": str(row["strategy_version"]),
                "train_rows": int(row["train_rows"]),
                "oos_rows": int(row["test_rows"]),
                "oos_non_null_returns": _require_int(stats.get("oos_non_null_returns", 0)),
                "oos_return_mean": stats.get("oos_return_mean"),
                "oos_return_median": stats.get("oos_return_median"),
                "oos_return_std": stats.get("oos_return_std"),
                "oos_positive_rate": stats.get("oos_positive_rate"),
                "raw_oos_ic": stats.get("raw_oos_ic"),
                "oriented_oos_ic": stats.get("oriented_oos_ic"),
                "oos_ic": stats.get("oos_ic"),
                "train_score": _as_float(row["train_score"]),
                "oos_score": _as_float(row["test_score"]),
                "overfit_gap": _as_float(row["overfit_gap"]),
                "oos_sharpe": None,
                "oos_max_drawdown": None,
                "purge_size": int(row["purge_size"]),
                "embargo_size": int(row["embargo_size"]),
                "purge_valid": purge_valid and counts_match,
                "embargo_valid": embargo_valid and counts_match,
                "train_test_disjoint": train_test_disjoint,
                "fold_order_valid": fold_order_ok,
                "timestamp_valid": timestamp_valid,
                "status": status,
            }
        )
    return (
        pl.DataFrame(rows)
        .select(list(EVALUATION_FOLD_METRIC_COLUMNS))
        .cast(EVALUATION_FOLD_METRIC_SCHEMA)
        .sort(["fold_id", "symbol"])
    )


def _factor_train_counts(
    evaluation_input: pl.DataFrame | None,
    *,
    walk_forward: pl.DataFrame,
    memberships: tuple[_FoldMembership, ...],
) -> pl.DataFrame:
    """Count TRAIN-window evaluation rows per factor identity per fold."""
    empty = pl.DataFrame(
        schema={
            "fold_id": pl.Int32,
            "factor_name": pl.String,
            "factor_version": pl.String,
            "train_rows": pl.Int64,
        }
    )
    if evaluation_input is None or evaluation_input.height == 0:
        return empty
    train_joined = _map_train_observations(
        evaluation_input,
        walk_forward=walk_forward,
        memberships=memberships,
    )
    if train_joined.height == 0:
        return empty
    return (
        train_joined.group_by(["fold_id", "factor_name", "factor_version"])
        .agg(pl.len().alias("train_rows"))
        .with_columns(
            [
                pl.col("fold_id").cast(pl.Int32),
                pl.col("train_rows").cast(pl.Int64),
            ]
        )
    )


def _materialize_observations(
    joined: pl.DataFrame | None,
    *,
    fold_metrics: pl.DataFrame | None,
    manager: str,
    engine: str,
    timeframe: str,
    year: int,
    default_status: str = PurgedCVEvaluationStatus.PASS.value,
) -> pl.DataFrame:
    """Cast joined evaluation rows into the observation schema."""
    if joined is None or joined.height == 0:
        return pl.DataFrame(schema=dict(EVALUATION_OBSERVATION_SCHEMA))
    if fold_metrics is not None and fold_metrics.height > 0:
        status_lookup = fold_metrics.select(["fold_id", "status"])
        with_status = joined.join(status_lookup, on="fold_id", how="left").with_columns(
            pl.col("status").fill_null(PurgedCVEvaluationStatus.FAIL.value)
        )
    else:
        with_status = joined.with_columns(pl.lit(default_status).alias("status"))
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


def _panel_summary(
    *,
    fold_metrics: pl.DataFrame,
    factor_metrics: pl.DataFrame,
    label_stats: dict[int, dict[str, object]],
    manager: str,
    engine: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Build the single-row panel summary from fold/factor aggregates."""
    non_null = sum(
        _require_int(stats.get("oos_non_null_returns", 0)) for stats in label_stats.values()
    )
    means = [
        _as_float(stats.get("oos_return_mean"))
        for stats in label_stats.values()
        if stats.get("oos_return_mean") is not None
    ]
    positives = [
        _as_float(stats.get("oos_positive_rate"))
        for stats in label_stats.values()
        if stats.get("oos_positive_rate") is not None
    ]
    ics = [
        _as_float(stats.get("oos_ic"))
        for stats in label_stats.values()
        if stats.get("oos_ic") is not None
    ]
    raw_ics = [
        _as_float(stats.get("raw_oos_ic"))
        for stats in label_stats.values()
        if stats.get("raw_oos_ic") is not None
    ]
    oriented_ics = [
        _as_float(stats.get("oriented_oos_ic"))
        for stats in label_stats.values()
        if stats.get("oriented_oos_ic") is not None
    ]
    unique_factors = (
        int(factor_metrics.select(["factor_name", "factor_version"]).unique().height)
        if factor_metrics.height > 0
        else 0
    )
    status = (
        PurgedCVEvaluationStatus.PASS.value
        if fold_metrics.height > 0
        and all(item == PurgedCVEvaluationStatus.PASS.value for item in fold_metrics["status"])
        else PurgedCVEvaluationStatus.FAIL.value
    )
    row = {
        "manager": manager,
        "engine": engine,
        "exchange": exchange,
        "market": market,
        "symbol": _CROSS_SECTION_SYMBOL,
        "timeframe": timeframe,
        "year": year,
        "folds": int(fold_metrics.height),
        "train_rows": int(fold_metrics["train_rows"].sum()) if fold_metrics.height else 0,
        "oos_rows": int(fold_metrics["oos_rows"].sum()) if fold_metrics.height else 0,
        "oos_non_null_returns": non_null,
        "oos_return_mean": (
            sum(value for value in means if value is not None) / len(means) if means else None
        ),
        "oos_return_std": None,
        "oos_positive_rate": (
            sum(value for value in positives if value is not None) / len(positives)
            if positives
            else None
        ),
        "raw_oos_ic": (
            sum(value for value in raw_ics if value is not None) / len(raw_ics) if raw_ics else None
        ),
        "oriented_oos_ic": (
            sum(value for value in oriented_ics if value is not None) / len(oriented_ics)
            if oriented_ics
            else None
        ),
        "oos_ic": (sum(value for value in ics if value is not None) / len(ics) if ics else None),
        "oos_sharpe": None,
        "oos_max_drawdown": None,
        "unique_selected_factors": unique_factors,
        "purge_valid_folds": int(fold_metrics["purge_valid"].sum()) if fold_metrics.height else 0,
        "embargo_valid_folds": (
            int(fold_metrics["embargo_valid"].sum()) if fold_metrics.height else 0
        ),
        "train_test_disjoint_folds": (
            int(fold_metrics["train_test_disjoint"].sum()) if fold_metrics.height else 0
        ),
        "fold_order_valid_folds": (
            int(fold_metrics["fold_order_valid"].sum()) if fold_metrics.height else 0
        ),
        "timestamp_valid_folds": (
            int(fold_metrics["timestamp_valid"].sum()) if fold_metrics.height else 0
        ),
        "status": status,
        "error": None,
    }
    return (
        pl.DataFrame([row]).select(list(EVALUATION_SUMMARY_COLUMNS)).cast(EVALUATION_SUMMARY_SCHEMA)
    )


def _fold_order_valid(purged_cv: pl.DataFrame) -> bool:
    """Return whether fold_id values are unique and sorted ascending."""
    fold_ids = purged_cv["fold_id"].to_list()
    if len(fold_ids) != len(set(fold_ids)):
        return False
    return fold_ids == sorted(fold_ids)


def _timestamp_valid(row: dict[str, object]) -> bool:
    """Validate intra-window timestamp ordering (not train-before-test)."""
    train_start = _as_int(row["train_start_time"])
    train_end = _as_int(row["train_end_time"])
    test_start = _as_int(row["test_start_time"])
    test_end = _as_int(row["test_end_time"])
    if None in (train_start, train_end, test_start, test_end):
        return False
    assert train_start is not None and train_end is not None
    assert test_start is not None and test_end is not None
    if train_start > train_end:
        return False
    if test_start > test_end:
        return False
    return True


def _row_matches_rebuilt(row: dict[str, object], rebuilt: dict[str, object]) -> bool:
    """Compare persisted fold metadata against engine reconstruction."""
    for column in (
        "train_rows",
        "test_rows",
        "train_start_time",
        "train_end_time",
        "test_start_time",
        "test_end_time",
        "purge_size",
        "embargo_size",
        "status",
    ):
        if row[column] != rebuilt[column]:
            return False
    return True


def _require_unique_observation_keys(observations: pl.DataFrame) -> None:
    """Reject duplicated observation primary keys."""
    if observations.height == 0:
        return
    duplicates = (
        observations.group_by(list(OBSERVATION_PRIMARY_KEY_COLUMNS)).len().filter(pl.col("len") > 1)
    )
    if duplicates.height > 0:
        raise PurgedCVError(
            "purged-CV evaluation observations contain duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={"duplicate_groups": duplicates.height},
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


def _require_int(value: object) -> int:
    """Convert a scalar to int, raising when undefined."""
    as_int = _as_int(value)
    if as_int is not None:
        return as_int
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value)!r}")


def _as_int(value: object) -> int | None:
    """Convert a scalar to int, or ``None`` when undefined."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)
