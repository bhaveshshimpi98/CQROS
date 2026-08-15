"""CQROS Walk-Forward Engine contracts and rolling-fold implementation.

Purpose:
    Convert a canonical Factor Selection dataset into a deterministic
    walk-forward DataFrame conforming to ``WALK_FORWARD_SCHEMA``.

Responsibilities:
    - Define ``WalkForwardEngine`` as the shared walk-forward contract
    - Provide ``SimpleWalkForwardEngine`` for rolling train/test folds
    - Validate Factor Selection DataFrame structure
    - Evaluate selected-factor out-of-sample returns on each test window
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.walk_forward.exceptions``, and
    ``cqros.walk_forward.schema``.

Public API:
    ``WalkForwardEngine``, ``SimpleWalkForwardEngine``,
    ``FACTOR_SELECTION_INPUT_COLUMNS``, ``validate_factor_selection_frame``
"""

from __future__ import annotations

from typing import Final, NamedTuple, Protocol, runtime_checkable

import polars as pl

from cqros.walk_forward.exceptions import WalkForwardError
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER,
    WALK_FORWARD_SCHEMA,
    WalkForwardStatus,
)

__all__ = [
    "FACTOR_SELECTION_INPUT_COLUMNS",
    "SimpleWalkForwardEngine",
    "WalkForwardEngine",
    "apply_walk_forward_aggregate_metrics",
    "build_walk_forward_fold",
    "validate_factor_selection_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "WF_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "WF_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "WF_MISSING_COLUMNS"
_ERROR_INVALID_CONFIG: Final[str] = "WF_INVALID_CONFIG"

_DEFAULT_STRATEGY_NAME: Final[str] = "default_strategy"
_DEFAULT_STRATEGY_VERSION: Final[str] = "v1"
_DEFAULT_MODEL_VERSION: Final[str] = "v1"

_DEFAULT_TRAIN_WINDOW: Final[int] = 252
_DEFAULT_TEST_WINDOW: Final[int] = 63
_DEFAULT_STEP_SIZE: Final[int] = 63

_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
)

# Factor Selection columns required to assemble a walk-forward row.
FACTOR_SELECTION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_time",
    "selected",
)

# Evaluation column required for out-of-sample fold metrics.
_EVALUATION_INPUT_COLUMNS: Final[tuple[str, ...]] = ("future_return_1",)

_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    *FACTOR_SELECTION_INPUT_COLUMNS,
    *_EVALUATION_INPUT_COLUMNS,
)


@runtime_checkable
class WalkForwardEngine(Protocol):
    """Structural contract for converting selection decisions into folds.

    Implementations own walk-forward semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, factor_selection: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Selection dataset into a walk-forward DataFrame.

        Args:
            factor_selection: Canonical Factor Selection dataset.
                Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``WALK_FORWARD_SCHEMA``.
        """
        ...


class SimpleWalkForwardEngine:
    """Generate deterministic rolling walk-forward folds from selection.

    Rules:
        - Rows are grouped by ``strategy_name``, ``strategy_version``, and
          ``timeframe`` (strategy identity defaults to ``default_strategy`` /
          ``v1`` because Factor Selection does not carry those columns)
        - Each group is sorted ascending by ``selection_time``
        - Rolling folds advance by ``step_size`` with fixed ``train_window``
          and ``test_window`` lengths
        - Fold ``i`` uses train indices
          ``[start, start + train_window)`` and test indices
          ``[start + train_window, start + train_window + test_window)``
          where ``start = (i - 1) * step_size``
        - Generation stops when remaining rows cannot fill both windows
        - Groups with insufficient history emit one sentinel fold
        - Training windows are not fitted; evaluation uses the test window only
        - Selected-factor ``future_return_1`` values in the test window yield
          mean return, sample volatility, Sharpe ratio, and win rate
        - Per-fold evaluation maps temporarily to
          ``train_score`` / ``test_score`` / ``overfit_gap`` before
          aggregation
        - After all folds in a group are evaluated, PASS-fold aggregates
          overwrite every emitted fold:
          ``train_score`` = mean train score,
          ``test_score`` = mean test score,
          ``overfit_gap`` = walk-forward stability
          (``mean_test_score / test_score_std`` when at least two PASS
          folds exist and ``test_score_std > 0``, otherwise ``null``)
        - ``status`` is ``PASS`` when observations ``> 0``, otherwise ``FAIL``
        - ``model_version`` is always ``v1``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Complete folds keep train indices strictly before test indices with
        no overlap. Insufficient-history sentinel folds may reuse the only
        available row so both ``train_rows`` and ``test_rows`` stay positive.
        Training-window returns never enter evaluation metrics.
        FAIL folds are excluded from aggregate statistics but still receive
        the group aggregate values.
    """

    __slots__ = ("_step_size", "_test_window", "_train_window")

    _train_window: int
    _test_window: int
    _step_size: int

    def __init__(
        self,
        train_window: int = _DEFAULT_TRAIN_WINDOW,
        test_window: int = _DEFAULT_TEST_WINDOW,
        step_size: int = _DEFAULT_STEP_SIZE,
    ) -> None:
        """Initialize rolling-window fold configuration.

        Args:
            train_window: Number of chronologically ordered rows in each
                training window. Defaults to ``252``.
            test_window: Number of chronologically ordered rows in each
                testing window. Defaults to ``63``.
            step_size: Number of rows to advance between successive folds.
                Defaults to ``63``.

        Raises:
            WalkForwardError: If any window parameter is not a positive
                integer.
        """
        self._train_window = _require_positive_int(train_window, "train_window")
        self._test_window = _require_positive_int(test_window, "test_window")
        self._step_size = _require_positive_int(step_size, "step_size")

    def build(self, factor_selection: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Selection dataset into finalized walk-forward rows.

        Args:
            factor_selection: Canonical Factor Selection dataset.
                Must not be mutated. Requires ``future_return_1`` for
                out-of-sample evaluation.

        Returns:
            A new DataFrame matching ``WALK_FORWARD_SCHEMA``.

        Raises:
            WalkForwardError: If the input fails structural validation
                or required columns are missing.
        """
        frame = validate_factor_selection_frame(factor_selection)
        _require_columns(frame, _REQUIRED_INPUT_COLUMNS, "factor_selection")
        return _build_walk_forward_rows(
            frame,
            train_window=self._train_window,
            test_window=self._test_window,
            step_size=self._step_size,
        )

    @property
    def train_window(self) -> int:
        """Return the configured train-window row count."""
        return self._train_window

    @property
    def test_window(self) -> int:
        """Return the configured test-window row count."""
        return self._test_window

    @property
    def step_size(self) -> int:
        """Return the configured fold step in rows."""
        return self._step_size


def build_walk_forward_fold(
    group: pl.DataFrame,
    *,
    fold_id: int,
    train_start_index: int,
    train_end_index: int,
    test_start_index: int,
    test_end_index: int,
) -> dict[str, object]:
    """Build one raw fold using the canonical engine formulas.

    This public primitive lets bounded-memory executors evaluate an exact row
    window without duplicating score, status, or boundary semantics.
    """
    _require_columns(group, _REQUIRED_INPUT_COLUMNS, "factor_selection")
    if group.height == 0:
        raise WalkForwardError(
            "factor_selection frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_selection", "rows": group.height},
        )
    return _fold_row(
        group=group,
        strategy_name=_DEFAULT_STRATEGY_NAME,
        strategy_version=_DEFAULT_STRATEGY_VERSION,
        timeframe=str(group["timeframe"][0]),
        fold_id=fold_id,
        train_start_index=train_start_index,
        train_end_index=train_end_index,
        test_start_index=test_start_index,
        test_end_index=test_end_index,
    )


def apply_walk_forward_aggregate_metrics(raw_folds: pl.DataFrame) -> pl.DataFrame:
    """Apply canonical PASS-only aggregate metrics to raw folds.

    The reduction uses the same ordered Polars columns and formulas as the
    full-panel engine. The returned frame preserves input row order.
    """
    aggregates = _compute_aggregate_metrics_frame(
        raw_folds.select(["train_score", "test_score", "overfit_gap", "status"])
    )
    return raw_folds.with_columns(
        pl.lit(aggregates.mean_train_score, dtype=pl.Float64).alias("train_score"),
        pl.lit(aggregates.mean_test_score, dtype=pl.Float64).alias("test_score"),
        pl.lit(aggregates.walk_forward_stability, dtype=pl.Float64).alias("overfit_gap"),
    )


def validate_factor_selection_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Selection dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        WalkForwardError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise WalkForwardError(
            "factor_selection frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_selection",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise WalkForwardError(
            "factor_selection frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_selection", "rows": frame.height},
        )
    return frame


def _build_walk_forward_rows(
    factor_selection: pl.DataFrame,
    *,
    train_window: int,
    test_window: int,
    step_size: int,
) -> pl.DataFrame:
    """Assemble canonical rolling walk-forward rows from Factor Selection."""
    annotated = factor_selection.with_columns(
        pl.lit(_DEFAULT_STRATEGY_NAME).alias("strategy_name"),
        pl.lit(_DEFAULT_STRATEGY_VERSION).alias("strategy_version"),
    )
    partitions = annotated.partition_by(
        list(_GROUP_COLUMNS),
        maintain_order=True,
        include_key=True,
    )
    fold_rows: list[dict[str, object]] = []
    for partition in partitions:
        sorted_partition = partition.sort("selection_time", descending=False)
        fold_rows.extend(
            _folds_for_group(
                sorted_partition,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
            )
        )
    return pl.DataFrame(fold_rows).select(list(CANONICAL_COLUMN_ORDER)).cast(WALK_FORWARD_SCHEMA)


def _folds_for_group(
    group: pl.DataFrame,
    *,
    train_window: int,
    test_window: int,
    step_size: int,
) -> list[dict[str, object]]:
    """Generate rolling folds for one strategy/timeframe group."""
    strategy_name = str(group["strategy_name"][0])
    strategy_version = str(group["strategy_version"][0])
    timeframe = str(group["timeframe"][0])
    row_count = group.height
    required = train_window + test_window

    folds: list[dict[str, object]] = []
    fold_id = 1
    start = 0
    while start + required <= row_count:
        train_end_index = start + train_window
        test_end_index = train_end_index + test_window
        folds.append(
            _fold_row(
                group=group,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                timeframe=timeframe,
                fold_id=fold_id,
                train_start_index=start,
                train_end_index=train_end_index,
                test_start_index=train_end_index,
                test_end_index=test_end_index,
            )
        )
        fold_id += 1
        start += step_size

    if not folds:
        folds.append(
            _insufficient_history_fold(
                group=group,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                timeframe=timeframe,
                train_window=train_window,
            )
        )
    return _apply_aggregate_metrics(folds)


def _insufficient_history_fold(
    *,
    group: pl.DataFrame,
    strategy_name: str,
    strategy_version: str,
    timeframe: str,
    train_window: int,
) -> dict[str, object]:
    """Build a single sentinel fold when a group cannot fill both windows."""
    row_count = group.height
    if row_count == 1:
        # Single-row groups cannot form disjoint windows; reuse the row.
        train_end_index = 1
        test_start_index = 0
        test_end_index = 1
    elif row_count <= train_window:
        # Keep both windows non-empty and chronologically ordered.
        train_end_index = row_count - 1
        test_start_index = row_count - 1
        test_end_index = row_count
    else:
        train_end_index = train_window
        test_start_index = train_window
        test_end_index = row_count

    return _fold_row(
        group=group,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        timeframe=timeframe,
        fold_id=1,
        train_start_index=0,
        train_end_index=train_end_index,
        test_start_index=test_start_index,
        test_end_index=test_end_index,
    )


def _fold_row(
    *,
    group: pl.DataFrame,
    strategy_name: str,
    strategy_version: str,
    timeframe: str,
    fold_id: int,
    train_start_index: int,
    train_end_index: int,
    test_start_index: int,
    test_end_index: int,
) -> dict[str, object]:
    """Assemble one canonical walk-forward fold dictionary."""
    train_times = group["selection_time"].slice(
        train_start_index,
        train_end_index - train_start_index,
    )
    test_times = group["selection_time"].slice(
        test_start_index,
        test_end_index - test_start_index,
    )
    selected_factors = int(
        group["selected"].slice(train_start_index, train_end_index - train_start_index).sum()
    )
    mean_return, sharpe_ratio, win_rate, status = _evaluate_test_window(
        group,
        test_start_index=test_start_index,
        test_end_index=test_end_index,
    )
    return {
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "timeframe": timeframe,
        "fold_id": fold_id,
        "train_start": int(train_times[0]),
        "train_end": int(train_times[-1]),
        "test_start": int(test_times[0]),
        "test_end": int(test_times[-1]),
        "train_rows": train_times.len(),
        "test_rows": test_times.len(),
        "selected_factors": selected_factors,
        "model_version": _DEFAULT_MODEL_VERSION,
        "train_score": mean_return,
        "test_score": sharpe_ratio,
        "overfit_gap": win_rate,
        "status": status,
    }


def _evaluate_test_window(
    group: pl.DataFrame,
    *,
    test_start_index: int,
    test_end_index: int,
) -> tuple[float | None, float | None, float | None, str]:
    """Evaluate selected-factor returns on the test window only.

    Returns:
        ``(mean_return, sharpe_ratio, win_rate, status)``. Sharpe is
        ``null`` when sample volatility is missing or not strictly
        positive. Status is ``PASS`` when observations ``> 0``.
    """
    test_height = test_end_index - test_start_index
    test_window = group.slice(test_start_index, test_height)
    returns = test_window.filter(pl.col("selected")).get_column("future_return_1").drop_nulls()
    observations = returns.len()
    if observations == 0:
        return (
            None,
            None,
            None,
            WalkForwardStatus.FAIL.value,
        )

    mean_return = _as_optional_float(returns.mean())
    if mean_return is None:
        return (
            None,
            None,
            None,
            WalkForwardStatus.FAIL.value,
        )
    volatility = _as_optional_float(returns.std(ddof=1))
    sharpe_ratio = mean_return / volatility if volatility is not None and volatility > 0.0 else None
    wins = int((returns > 0.0).sum())
    win_rate = float(wins) / float(observations)
    return (
        mean_return,
        sharpe_ratio,
        win_rate,
        WalkForwardStatus.PASS.value,
    )


def _apply_aggregate_metrics(
    folds: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Overwrite score fields with PASS-fold aggregate walk-forward metrics.

    Fold generation and per-fold evaluation remain unchanged. Only PASS folds
    contribute to means, sample test-score volatility, and stability. Every
    emitted fold in the group receives the same aggregate values.

    Schema mapping (no dedicated aggregate columns):
        ``train_score`` ← mean train score
        ``test_score`` ← mean test score
        ``overfit_gap`` ← walk-forward stability
    """
    aggregates = _compute_aggregate_metrics(folds)
    aggregated: list[dict[str, object]] = []
    for fold in folds:
        updated = dict(fold)
        updated["train_score"] = aggregates.mean_train_score
        updated["test_score"] = aggregates.mean_test_score
        updated["overfit_gap"] = aggregates.walk_forward_stability
        aggregated.append(updated)
    return aggregated


class _AggregateMetrics(NamedTuple):
    """Aggregate walk-forward statistics across PASS folds."""

    mean_train_score: float | None
    mean_test_score: float | None
    mean_overfit_gap: float | None
    test_score_std: float | None
    walk_forward_stability: float | None


def _compute_aggregate_metrics(folds: list[dict[str, object]]) -> _AggregateMetrics:
    """Compute PASS-only aggregate walk-forward statistics for one group."""
    pass_frame = pl.DataFrame(
        {
            "train_score": [fold["train_score"] for fold in folds],
            "test_score": [fold["test_score"] for fold in folds],
            "overfit_gap": [fold["overfit_gap"] for fold in folds],
            "status": [fold["status"] for fold in folds],
        }
    )
    return _compute_aggregate_metrics_frame(pass_frame)


def _compute_aggregate_metrics_frame(folds: pl.DataFrame) -> _AggregateMetrics:
    """Compute canonical aggregate statistics from ordered raw-fold columns."""
    pass_status = WalkForwardStatus.PASS.value
    pass_frame = folds.filter(pl.col("status") == pass_status)

    pass_count = pass_frame.height
    if pass_count == 0:
        return _AggregateMetrics(
            mean_train_score=None,
            mean_test_score=None,
            mean_overfit_gap=None,
            test_score_std=None,
            walk_forward_stability=None,
        )

    mean_train_score = _as_optional_float(pass_frame["train_score"].mean())
    mean_test_score = _as_optional_float(pass_frame["test_score"].mean())
    mean_overfit_gap = _as_optional_float(pass_frame["overfit_gap"].mean())
    test_score_std = (
        _as_optional_float(pass_frame["test_score"].std(ddof=1)) if pass_count >= 2 else None
    )
    walk_forward_stability = (
        mean_test_score / test_score_std
        if (
            pass_count >= 2
            and mean_test_score is not None
            and test_score_std is not None
            and test_score_std > 0.0
        )
        else None
    )
    return _AggregateMetrics(
        mean_train_score=mean_train_score,
        mean_test_score=mean_test_score,
        mean_overfit_gap=mean_overfit_gap,
        test_score_std=test_score_std,
        walk_forward_stability=walk_forward_stability,
    )


def _as_optional_float(value: object) -> float | None:
    """Convert a Polars scalar to ``float``, or ``None`` when undefined."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value != value:  # NaN
        return None
    return float(value)


def _require_positive_int(value: object, name: str) -> int:
    """Validate that ``value`` is a positive integer configuration parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WalkForwardError(
            f"{name} must be a positive integer",
            error_code=_ERROR_INVALID_CONFIG,
            details={"parameter": name, "actual_value": value},
        )
    return value


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise WalkForwardError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )
