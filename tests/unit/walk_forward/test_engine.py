"""Unit tests for CQROS ``SimpleWalkForwardEngine``."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.walk_forward import (
    WALK_FORWARD_SCHEMA,
    SimpleWalkForwardEngine,
    WalkForwardError,
    WalkForwardStatus,
)
from cqros.walk_forward.engine import (
    FACTOR_SELECTION_INPUT_COLUMNS,
    validate_factor_selection_frame,
)
from cqros.walk_forward.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_SELECTION_TIME = 1_704_067_200_000
_SELECTION_SCORE = 0.12
_HOUR_MS = 3_600_000
_DEFAULT_RETURN = 0.01


def _selection_times(count: int, *, start: int = _SELECTION_TIME) -> list[int]:
    """Build ``count`` ascending selection timestamps spaced one hour apart."""
    return [start + (index * _HOUR_MS) for index in range(count)]


def _sample_std(values: list[float]) -> float:
    """Return sample standard deviation for expected-value assertions."""
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(variance)


def _fold_sharpe(test_returns: list[float]) -> float:
    """Sharpe ratio for a single fold's test-window returns."""
    mean_return = sum(test_returns) / float(len(test_returns))
    return mean_return / _sample_std(test_returns)


def _factor_selection_frame(
    *,
    row_count: int | None = None,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    timeframes: list[str] | None = None,
    selection_times: list[int] | None = None,
    selected: list[bool] | None = None,
    selection_scores: list[float] | None = None,
    selection_ranks: list[int] | None = None,
    selection_reasons: list[str] | None = None,
    statuses: list[str] | None = None,
    future_returns: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a Factor Selection frame with evaluation returns for engine tests."""
    if selection_times is not None:
        resolved_count = len(selection_times)
    elif factor_names is not None:
        resolved_count = len(factor_names)
    elif future_returns is not None:
        resolved_count = len(future_returns)
    elif row_count is not None:
        resolved_count = row_count
    else:
        resolved_count = 1

    factor_names = (
        factor_names
        if factor_names is not None
        else [f"{_FACTOR_NAME}_{index}" for index in range(resolved_count)]
    )
    factor_versions = (
        factor_versions if factor_versions is not None else [_FACTOR_VERSION] * resolved_count
    )
    factor_categories = (
        factor_categories if factor_categories is not None else [_FACTOR_CATEGORY] * resolved_count
    )
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * resolved_count
    selection_times = (
        selection_times if selection_times is not None else _selection_times(resolved_count)
    )
    selected = selected if selected is not None else [True] * resolved_count
    selection_scores = (
        selection_scores if selection_scores is not None else [_SELECTION_SCORE] * resolved_count
    )
    selection_ranks = selection_ranks if selection_ranks is not None else [1] * resolved_count
    selection_reasons = (
        selection_reasons
        if selection_reasons is not None
        else ["v1_default_selection"] * resolved_count
    )
    statuses = statuses if statuses is not None else ["SELECTED"] * resolved_count
    future_returns = (
        future_returns if future_returns is not None else [_DEFAULT_RETURN] * resolved_count
    )
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "timeframe": timeframes,
            "selection_time": selection_times,
            "factor_category": factor_categories,
            "selected": selected,
            "selection_score": selection_scores,
            "selection_rank": selection_ranks,
            "selection_reason": selection_reasons,
            "status": statuses,
            "future_return_1": future_returns,
        }
    )


def _engine(
    *,
    train_window: int = 3,
    test_window: int = 2,
    step_size: int = 2,
) -> SimpleWalkForwardEngine:
    """Build an engine with compact windows for deterministic unit tests."""
    return SimpleWalkForwardEngine(
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    )


def _build(
    engine: SimpleWalkForwardEngine,
    *,
    factor_selection: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build walk-forward rows with a default Factor Selection frame."""
    return engine.build(
        factor_selection if factor_selection is not None else _factor_selection_frame()
    )


# ---------------------------------------------------------------------------
# Input / validation
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """FACTOR_SELECTION_INPUT_COLUMNS enumerates structural selection columns."""
    for column in ("timeframe", "selection_time", "selected"):
        assert column in FACTOR_SELECTION_INPUT_COLUMNS


def test_validate_factor_selection_frame_rejects_non_dataframe() -> None:
    """validate_factor_selection_frame rejects non-DataFrame with WF_FRAME_TYPE."""
    with pytest.raises(WalkForwardError) as exc_info:
        validate_factor_selection_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "WF_FRAME_TYPE"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty Factor Selection frames."""
    empty = pl.DataFrame(schema={column: pl.String for column in ("timeframe",)}).clear()
    with pytest.raises(WalkForwardError) as exc_info:
        _engine().build(empty)
    assert exc_info.value.error_code == "WF_FRAME_EMPTY"


def test_build_rejects_missing_future_return_1() -> None:
    """Missing future_return_1 raises WF_MISSING_COLUMNS."""
    with pytest.raises(WalkForwardError) as exc_info:
        _build(
            _engine(),
            factor_selection=_factor_selection_frame().drop("future_return_1"),
        )
    assert exc_info.value.error_code == "WF_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Aggregate walk-forward statistics
# ---------------------------------------------------------------------------


def test_single_pass_fold_means_equal_fold_scores_and_null_stability() -> None:
    """One PASS fold keeps mean scores and leaves stability null."""
    returns = [-0.50, -0.40, -0.30, 0.02, 0.04]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    test_returns = [0.02, 0.04]
    mean_return = sum(test_returns) / 2.0
    sharpe = _fold_sharpe(test_returns)
    assert result.height == 1
    assert result["train_score"].to_list()[0] == pytest.approx(mean_return)
    assert result["test_score"].to_list()[0] == pytest.approx(sharpe)
    assert result["overfit_gap"].to_list() == [None]
    assert result["status"].to_list() == [WalkForwardStatus.PASS.value]


def test_multiple_folds_share_identical_aggregate_metrics() -> None:
    """Every fold receives the same PASS-fold aggregate score fields."""
    # Fold 1 test [0.02, 0.04]; fold 2 test [-0.02, -0.04].
    returns = [0.0, 0.0, 0.0, 0.02, 0.04, -0.02, -0.04]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    fold1_mean = (0.02 + 0.04) / 2.0
    fold2_mean = (-0.02 + -0.04) / 2.0
    fold1_sharpe = _fold_sharpe([0.02, 0.04])
    fold2_sharpe = _fold_sharpe([-0.02, -0.04])
    mean_train = (fold1_mean + fold2_mean) / 2.0
    mean_test = (fold1_sharpe + fold2_sharpe) / 2.0
    stability = mean_test / _sample_std([fold1_sharpe, fold2_sharpe])

    assert result.height == 2
    assert result["train_score"].to_list() == [
        pytest.approx(mean_train),
        pytest.approx(mean_train),
    ]
    assert result["test_score"].to_list() == [
        pytest.approx(mean_test),
        pytest.approx(mean_test),
    ]
    assert result["overfit_gap"].to_list() == [
        pytest.approx(stability),
        pytest.approx(stability),
    ]


def test_walk_forward_stability_is_mean_test_over_std() -> None:
    """Stability equals mean test score divided by sample test-score std."""
    # Fold 1: mild positive; fold 2: stronger mixed → distinct Sharpes.
    returns = [0.0, 0.0, 0.0, 0.01, 0.02, 0.10, -0.02]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    sharpes = [
        _fold_sharpe([0.01, 0.02]),
        _fold_sharpe([0.10, -0.02]),
    ]
    expected = (sum(sharpes) / 2.0) / _sample_std(sharpes)
    assert sharpes[0] != pytest.approx(sharpes[1])
    assert result["overfit_gap"].to_list()[0] == pytest.approx(expected)


def test_zero_test_score_variance_yields_null_stability() -> None:
    """Identical PASS-fold test scores yield zero std and null stability."""
    # Both folds use identical positive test returns → identical Sharpe.
    returns = [0.0, 0.0, 0.0, 0.02, 0.04, 0.02, 0.04]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    assert result.height == 2
    assert result["test_score"].to_list()[0] == pytest.approx(_fold_sharpe([0.02, 0.04]))
    assert result["overfit_gap"].to_list() == [None, None]


def test_failed_folds_are_excluded_from_aggregates() -> None:
    """FAIL folds do not enter means/std but still receive aggregate values."""
    # Fold 1 PASS on [0.02, 0.04]; fold 2 FAIL on null test returns.
    returns: list[float | None] = [0.0, 0.0, 0.0, 0.02, 0.04, None, None]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    fold1_mean = (0.02 + 0.04) / 2.0
    fold1_sharpe = _fold_sharpe([0.02, 0.04])
    assert result.height == 2
    assert result["status"].to_list() == [
        WalkForwardStatus.PASS.value,
        WalkForwardStatus.FAIL.value,
    ]
    # Only one PASS fold → means equal that fold; stability null.
    assert result["train_score"].to_list() == [
        pytest.approx(fold1_mean),
        pytest.approx(fold1_mean),
    ]
    assert result["test_score"].to_list() == [
        pytest.approx(fold1_sharpe),
        pytest.approx(fold1_sharpe),
    ]
    assert result["overfit_gap"].to_list() == [None, None]


def test_all_failed_folds_yield_null_aggregates() -> None:
    """When no PASS folds exist, aggregate score fields are null."""
    returns: list[float | None] = [0.01, 0.02, 0.03, None, None]
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(future_returns=returns),
    )
    assert result["train_score"].to_list() == [None]
    assert result["test_score"].to_list() == [None]
    assert result["overfit_gap"].to_list() == [None]
    assert result["status"].to_list() == [WalkForwardStatus.FAIL.value]


# ---------------------------------------------------------------------------
# Fold generation mechanics
# ---------------------------------------------------------------------------


def test_one_fold_when_history_exactly_fills_windows() -> None:
    """Exactly train_window + test_window rows produce one fold."""
    times = _selection_times(5)
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(selection_times=times),
    )
    assert result.height == 1
    assert result["fold_id"].to_list() == [1]
    assert result["train_start"].to_list() == [times[0]]
    assert result["test_end"].to_list() == [times[4]]
    assert result["train_rows"].to_list() == [3]
    assert result["test_rows"].to_list() == [2]


def test_train_and_test_windows_do_not_overlap() -> None:
    """Train indices always end before test indices begin for complete folds."""
    times = _selection_times(9)
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(selection_times=times),
    )
    assert result.height == 3
    for row in result.iter_rows(named=True):
        assert row["train_end"] < row["test_start"]
        assert row["train_rows"] == 3
        assert row["test_rows"] == 2


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and schema dtypes."""
    result = _build(
        _engine(train_window=3, test_window=2, step_size=2),
        factor_selection=_factor_selection_frame(selection_times=_selection_times(5)),
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == WALK_FORWARD_SCHEMA
    assert result.schema["fold_id"] == pl.Int32
    assert result.schema["train_score"] == pl.Float64
    assert result.schema["status"] == pl.String


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied Factor Selection frame."""
    factor_selection = _factor_selection_frame(selection_times=_selection_times(5))
    before = factor_selection.clone()
    _engine(train_window=3, test_window=2, step_size=2).build(factor_selection)
    assert_frame_equal(factor_selection, before)


def test_output_is_deterministic() -> None:
    """Identical Factor Selection inputs produce identical walk-forward outputs."""
    factor_selection = _factor_selection_frame(
        selection_times=_selection_times(7),
        future_returns=[0.01, 0.02, -0.01, 0.03, -0.02, 0.04, -0.03],
    )
    engine = _engine(train_window=3, test_window=2, step_size=2)
    first = engine.build(factor_selection)
    second = engine.build(factor_selection)
    assert_frame_equal(first, second)
