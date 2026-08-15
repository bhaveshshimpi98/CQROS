"""Unit tests for CQROS ``SimplePurgedCVEngine``."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.purged_cv import (
    PURGED_CV_SCHEMA,
    PurgedCVError,
    PurgedCVStatus,
    SimplePurgedCVEngine,
)
from cqros.purged_cv.engine import (
    WALK_FORWARD_INPUT_COLUMNS,
    validate_walk_forward_frame,
)
from cqros.purged_cv.schema import CANONICAL_COLUMN_ORDER

_STRATEGY_NAME = "default_strategy"
_STRATEGY_VERSION = "v1"
_TIMEFRAME = "1h"
_HOUR_MS = 3_600_000
_BASE_TIME = 1_704_067_200_000


def _test_starts(count: int, *, start: int = _BASE_TIME) -> list[int]:
    """Build ``count`` ascending walk-forward ``test_start`` timestamps."""
    return [start + (index * _HOUR_MS) for index in range(count)]


def _walk_forward_frame(
    *,
    row_count: int | None = None,
    strategy_names: list[str] | None = None,
    strategy_versions: list[str] | None = None,
    timeframes: list[str] | None = None,
    test_starts: list[int] | None = None,
    train_scores: list[float | None] | None = None,
    test_scores: list[float | None] | None = None,
) -> pl.DataFrame:
    """Build a minimal Walk-Forward frame for purged-CV engine tests."""
    if test_starts is not None:
        resolved_count = len(test_starts)
    elif train_scores is not None:
        resolved_count = len(train_scores)
    elif row_count is not None:
        resolved_count = row_count
    else:
        resolved_count = 1

    strategy_names = (
        strategy_names if strategy_names is not None else [_STRATEGY_NAME] * resolved_count
    )
    strategy_versions = (
        strategy_versions if strategy_versions is not None else [_STRATEGY_VERSION] * resolved_count
    )
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * resolved_count
    test_starts = test_starts if test_starts is not None else _test_starts(resolved_count)
    train_scores = (
        train_scores
        if train_scores is not None
        else [0.10 + (0.01 * index) for index in range(resolved_count)]
    )
    test_scores = (
        test_scores
        if test_scores is not None
        else [0.05 + (0.01 * index) for index in range(resolved_count)]
    )
    return pl.DataFrame(
        {
            "strategy_name": strategy_names,
            "strategy_version": strategy_versions,
            "timeframe": timeframes,
            "fold_id": list(range(1, resolved_count + 1)),
            "train_start": test_starts,
            "train_end": test_starts,
            "test_start": test_starts,
            "test_end": test_starts,
            "train_rows": [10] * resolved_count,
            "test_rows": [5] * resolved_count,
            "selected_factors": [1] * resolved_count,
            "model_version": ["v1"] * resolved_count,
            "train_score": train_scores,
            "test_score": test_scores,
            "overfit_gap": [None] * resolved_count,
            "status": ["PASS"] * resolved_count,
        }
    )


def _engine(
    *,
    n_folds: int = 5,
    purge_size: int = 5,
    embargo_size: int = 5,
) -> SimplePurgedCVEngine:
    """Build a purged-CV engine with explicit fold configuration."""
    return SimplePurgedCVEngine(
        n_folds=n_folds,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )


def _build(
    engine: SimplePurgedCVEngine,
    *,
    walk_forward: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build purged-CV rows with a default Walk-Forward frame."""
    return engine.build(walk_forward if walk_forward is not None else _walk_forward_frame())


# ---------------------------------------------------------------------------
# Input / validation
# ---------------------------------------------------------------------------


def test_default_constructor_preserves_simple_engine() -> None:
    """SimplePurgedCVEngine() accepts zero arguments with institutional defaults."""
    engine = SimplePurgedCVEngine()
    assert isinstance(engine, SimplePurgedCVEngine)


def test_input_columns_contract() -> None:
    """WALK_FORWARD_INPUT_COLUMNS enumerates required walk-forward columns."""
    for column in (
        "strategy_name",
        "strategy_version",
        "timeframe",
        "test_start",
        "train_score",
        "test_score",
    ):
        assert column in WALK_FORWARD_INPUT_COLUMNS


def test_validate_walk_forward_frame_rejects_non_dataframe() -> None:
    """validate_walk_forward_frame rejects non-DataFrame with PCV_FRAME_TYPE."""
    with pytest.raises(PurgedCVError) as exc_info:
        validate_walk_forward_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PCV_FRAME_TYPE"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty Walk-Forward frames."""
    empty = pl.DataFrame(schema={column: pl.String for column in ("strategy_name",)}).clear()
    with pytest.raises(PurgedCVError) as exc_info:
        _engine().build(empty)
    assert exc_info.value.error_code == "PCV_FRAME_EMPTY"


def test_build_rejects_missing_required_columns() -> None:
    """Missing test_start raises PCV_MISSING_COLUMNS."""
    with pytest.raises(PurgedCVError) as exc_info:
        _build(
            _engine(n_folds=1, purge_size=0, embargo_size=0),
            walk_forward=_walk_forward_frame(row_count=3).drop("test_start"),
        )
    assert exc_info.value.error_code == "PCV_MISSING_COLUMNS"


def test_invalid_n_folds_raises() -> None:
    """n_folds must be a positive integer."""
    with pytest.raises(PurgedCVError) as exc_info:
        SimplePurgedCVEngine(n_folds=0)
    assert exc_info.value.error_code == "PCV_INVALID_CONFIG"


def test_invalid_purge_size_raises() -> None:
    """purge_size must be a non-negative integer."""
    with pytest.raises(PurgedCVError) as exc_info:
        SimplePurgedCVEngine(purge_size=-1)
    assert exc_info.value.error_code == "PCV_INVALID_CONFIG"


def test_invalid_embargo_size_raises() -> None:
    """embargo_size must be a non-negative integer."""
    with pytest.raises(PurgedCVError) as exc_info:
        SimplePurgedCVEngine(embargo_size=-1)
    assert exc_info.value.error_code == "PCV_INVALID_CONFIG"


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def test_one_fold_uses_all_observations_as_test() -> None:
    """A single fold treats every observation as the test set."""
    times = _test_starts(6)
    train_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    test_scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    result = _build(
        _engine(n_folds=1, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(
            test_starts=times,
            train_scores=train_scores,
            test_scores=test_scores,
        ),
    )
    assert result.height == 1
    assert result["fold_id"].to_list() == [1]
    assert result["test_rows"].to_list() == [6]
    assert result["train_rows"].to_list() == [0]
    assert result["test_start_time"].to_list() == [times[0]]
    assert result["test_end_time"].to_list() == [times[-1]]
    assert result["status"].to_list() == [PurgedCVStatus.FAIL.value]


def test_multiple_folds_are_contiguous_and_chronological() -> None:
    """Multiple folds cover observations contiguously in chronological order."""
    times = _test_starts(10)
    result = _build(
        _engine(n_folds=5, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    assert result.height == 5
    assert result["fold_id"].to_list() == [1, 2, 3, 4, 5]
    assert result["test_rows"].to_list() == [2, 2, 2, 2, 2]
    assert result["test_start_time"].to_list() == [
        times[0],
        times[2],
        times[4],
        times[6],
        times[8],
    ]
    assert result["test_end_time"].to_list() == [
        times[1],
        times[3],
        times[5],
        times[7],
        times[9],
    ]
    for index in range(1, result.height):
        assert result["test_start_time"][index] > result["test_end_time"][index - 1]


def test_remainder_rows_go_to_earliest_folds() -> None:
    """Remainder observations are distributed to the earliest folds."""
    times = _test_starts(11)
    result = _build(
        _engine(n_folds=5, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    assert result["test_rows"].to_list() == [3, 2, 2, 2, 2]


def test_purge_removes_adjacent_observations() -> None:
    """Purge excludes nearest observations before and after the test fold."""
    # 12 rows, 3 folds of 4. Fold 2 test indices [4, 8).
    # purge_size=1 removes index 3 before and index 8 after from training.
    times = _test_starts(12)
    train_scores = [float(index) for index in range(12)]
    test_scores = [float(index) for index in range(12)]
    result = _build(
        _engine(n_folds=3, purge_size=1, embargo_size=0),
        walk_forward=_walk_forward_frame(
            test_starts=times,
            train_scores=train_scores,
            test_scores=test_scores,
        ),
    )
    fold_two = result.filter(pl.col("fold_id") == 2).row(0, named=True)
    # Candidate train without purge: [0,1,2,3,8,9,10,11] (8 rows).
    # After purge before+after: drop 3 and 8 → [0,1,2,9,10,11] (6 rows).
    assert fold_two["test_rows"] == 4
    assert fold_two["train_rows"] == 6
    assert fold_two["purge_size"] == 1
    expected_train_mean = (0.0 + 1.0 + 2.0 + 9.0 + 10.0 + 11.0) / 6.0
    assert fold_two["train_score"] == pytest.approx(expected_train_mean)
    expected_test_mean = (4.0 + 5.0 + 6.0 + 7.0) / 4.0
    assert fold_two["test_score"] == pytest.approx(expected_test_mean)


def test_embargo_removes_future_observations() -> None:
    """Embargo additionally removes observations after the post-test purge."""
    # 12 rows, 3 folds of 4. Fold 1 test [0, 4).
    # purge_size=1 removes index 4; embargo_size=2 also removes 5 and 6.
    times = _test_starts(12)
    train_scores = [float(index) for index in range(12)]
    test_scores = [float(index) for index in range(12)]
    result = _build(
        _engine(n_folds=3, purge_size=1, embargo_size=2),
        walk_forward=_walk_forward_frame(
            test_starts=times,
            train_scores=train_scores,
            test_scores=test_scores,
        ),
    )
    fold_one = result.filter(pl.col("fold_id") == 1).row(0, named=True)
    # Train without exclusions: [4..11]. After purge+embargo drop [4,5,6] → [7..11].
    assert fold_one["test_rows"] == 4
    assert fold_one["train_rows"] == 5
    assert fold_one["embargo_size"] == 2
    expected_train_mean = (7.0 + 8.0 + 9.0 + 10.0 + 11.0) / 5.0
    assert fold_one["train_score"] == pytest.approx(expected_train_mean)


def test_purge_and_embargo_are_additive_after_test() -> None:
    """Post-test purge and embargo together exclude purge_size + embargo_size rows."""
    times = _test_starts(15)
    result = _build(
        _engine(n_folds=3, purge_size=2, embargo_size=3),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    fold_one = result.filter(pl.col("fold_id") == 1).row(0, named=True)
    # Fold 1 test [0, 5). Exclude [5, 5+2+3) = [5, 10). Train = [10..14] → 5 rows.
    assert fold_one["test_rows"] == 5
    assert fold_one["train_rows"] == 5


def test_no_overlap_between_train_and_test_windows() -> None:
    """Reported train and test time spans never share purged-CV observations."""
    times = _test_starts(20)
    result = _build(
        _engine(n_folds=5, purge_size=2, embargo_size=1),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    for row in result.iter_rows(named=True):
        assert row["train_rows"] > 0
        assert row["test_rows"] > 0
        # Contiguous fold test block is disjoint from retained train by construction.
        assert row["status"] == PurgedCVStatus.PASS.value


def test_train_and_test_counts_match_purge_embargo_math() -> None:
    """Train/test row counts match contiguous fold plus purge/embargo exclusions."""
    times = _test_starts(20)
    n_folds = 4
    purge_size = 2
    embargo_size = 1
    result = _build(
        _engine(n_folds=n_folds, purge_size=purge_size, embargo_size=embargo_size),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    # Equal folds of 5.
    expected_test_rows = [5, 5, 5, 5]
    # Fold 1: exclude [5, 5+2+1)=[5,8) → train [8..19] = 12
    # Fold 2: test [5,10); purge before [3,5); after [10,13) → train [0,1,2]+[13..19]=10
    # Fold 3: test [10,15); purge before [8,10); after [15,18) → train [0..7]+[18,19]=10
    # Fold 4: test [15,20); purge before [13,15); no after → train [0..12]=13
    expected_train_rows = [12, 10, 10, 13]
    assert result["test_rows"].to_list() == expected_test_rows
    assert result["train_rows"].to_list() == expected_train_rows


def test_chronological_ordering_of_input_is_enforced() -> None:
    """Unsorted walk-forward rows are sorted before fold construction."""
    times = [_BASE_TIME + (index * _HOUR_MS) for index in (4, 1, 3, 0, 2)]
    ordered = sorted(times)
    result = _build(
        _engine(n_folds=5, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(
            test_starts=times,
            train_scores=[1.0, 2.0, 3.0, 4.0, 5.0],
            test_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        ),
    )
    assert result["test_start_time"].to_list() == ordered
    assert result["test_end_time"].to_list() == ordered


# ---------------------------------------------------------------------------
# Scores and status
# ---------------------------------------------------------------------------


def test_scores_are_means_and_overfit_gap_is_difference() -> None:
    """train_score/test_score are means; overfit_gap is train − test."""
    times = _test_starts(6)
    train_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    test_scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    result = _build(
        _engine(n_folds=2, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(
            test_starts=times,
            train_scores=train_scores,
            test_scores=test_scores,
        ),
    )
    # Fold 1 test [0,3), train [3,6)
    fold_one = result.filter(pl.col("fold_id") == 1).row(0, named=True)
    expected_train = (0.40 + 0.50 + 0.60) / 3.0
    expected_test = (0.01 + 0.02 + 0.03) / 3.0
    assert fold_one["train_score"] == pytest.approx(expected_train)
    assert fold_one["test_score"] == pytest.approx(expected_test)
    assert fold_one["overfit_gap"] == pytest.approx(expected_train - expected_test)
    assert fold_one["status"] == PurgedCVStatus.PASS.value


def test_pass_status_requires_non_empty_train_and_test() -> None:
    """PASS requires both train_rows and test_rows to be strictly positive."""
    times = _test_starts(10)
    result = _build(
        _engine(n_folds=2, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    assert result["status"].to_list() == [
        PurgedCVStatus.PASS.value,
        PurgedCVStatus.PASS.value,
    ]
    assert all(rows > 0 for rows in result["train_rows"].to_list())
    assert all(rows > 0 for rows in result["test_rows"].to_list())


def test_fail_status_when_purge_empties_training() -> None:
    """Large purge/embargo that empties training yields FAIL."""
    times = _test_starts(6)
    result = _build(
        _engine(n_folds=2, purge_size=3, embargo_size=3),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    # Fold 1 test [0,3); post exclusion [3, 3+3+3)=[3,6) → train empty.
    fold_one = result.filter(pl.col("fold_id") == 1).row(0, named=True)
    assert fold_one["test_rows"] == 3
    assert fold_one["train_rows"] == 0
    assert fold_one["status"] == PurgedCVStatus.FAIL.value
    assert fold_one["train_score"] is None
    assert fold_one["overfit_gap"] is None


def test_fail_status_for_empty_test_fold() -> None:
    """Fewer observations than folds produces empty later folds with FAIL."""
    times = _test_starts(3)
    result = _build(
        _engine(n_folds=5, purge_size=0, embargo_size=0),
        walk_forward=_walk_forward_frame(test_starts=times),
    )
    assert result.height == 5
    assert result["test_rows"].to_list() == [1, 1, 1, 0, 0]
    assert result["status"].to_list()[-2:] == [
        PurgedCVStatus.FAIL.value,
        PurgedCVStatus.FAIL.value,
    ]


# ---------------------------------------------------------------------------
# Output schema, determinism, immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output enforces canonical column order and schema dtypes."""
    result = _build(
        _engine(n_folds=2, purge_size=1, embargo_size=1),
        walk_forward=_walk_forward_frame(test_starts=_test_starts(10)),
    )
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == PURGED_CV_SCHEMA
    assert result.schema["fold_id"] == pl.Int32
    assert result.schema["purge_size"] == pl.Int64
    assert result.schema["embargo_size"] == pl.Int64
    assert result.schema["train_score"] == pl.Float64
    assert result.schema["status"] == pl.String


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied Walk-Forward frame."""
    walk_forward = _walk_forward_frame(test_starts=_test_starts(10))
    before = walk_forward.clone()
    _engine(n_folds=2, purge_size=1, embargo_size=1).build(walk_forward)
    assert_frame_equal(walk_forward, before)


def test_output_is_deterministic() -> None:
    """Identical Walk-Forward inputs produce identical purged-CV outputs."""
    walk_forward = _walk_forward_frame(
        test_starts=_test_starts(12),
        train_scores=[0.1 * index for index in range(12)],
        test_scores=[0.05 * index for index in range(12)],
    )
    engine = _engine(n_folds=3, purge_size=1, embargo_size=1)
    first = engine.build(walk_forward)
    second = engine.build(walk_forward)
    assert_frame_equal(first, second)


def test_multiple_strategy_groups_are_processed_independently() -> None:
    """Distinct strategy groups emit independent purged-CV fold sets."""
    first = _walk_forward_frame(test_starts=_test_starts(6), row_count=6)
    second = _walk_forward_frame(
        strategy_names=["alt_strategy"] * 6,
        test_starts=_test_starts(6, start=_BASE_TIME + 100_000),
        train_scores=[1.0] * 6,
        test_scores=[0.5] * 6,
    )
    result = _build(
        _engine(n_folds=2, purge_size=0, embargo_size=0),
        walk_forward=pl.concat([first, second], how="vertical_relaxed"),
    )
    assert result.height == 4
    assert set(result["strategy_name"].to_list()) == {_STRATEGY_NAME, "alt_strategy"}
    assert result.filter(pl.col("strategy_name") == _STRATEGY_NAME).height == 2
    assert result.filter(pl.col("strategy_name") == "alt_strategy").height == 2
