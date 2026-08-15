"""Unit tests for CQROS Factor Timeframe Analysis engine and schema contracts."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_timeframe_analysis import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_SELECTION_INPUT_COLUMNS,
    FACTOR_TIMEFRAME_ANALYSIS_COLUMNS,
    TIMEFRAME_ANALYSIS_SCHEMA,
    FactorTimeframeAnalysisEngineRegistry,
    FactorTimeframeAnalysisError,
    FactorTimeframeAnalysisPipeline,
    SimpleFactorTimeframeAnalysisEngine,
    TimeframeAnalysisStatus,
    validate_factor_selection_frame,
)
from cqros.factor_timeframe_analysis.schema import PRIMARY_KEY_COLUMNS

_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_SELECTION_TIME = 1_704_067_200_000


def _selection_frame(
    *,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    timeframes: list[str] | None = None,
    selection_scores: list[float | None] | None = None,
    selection_ranks: list[int] | None = None,
    selected: list[bool] | None = None,
    selection_times: list[int] | None = None,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a Factor Selection frame with selectable defaults."""
    row_count = max(
        len(values)
        for values in (
            factor_names or [_FACTOR_NAME],
            factor_versions or [],
            factor_categories or [],
            timeframes or [],
            selection_scores or [],
            selection_ranks or [],
            selected or [],
            selection_times or [],
            statuses or [],
            [_FACTOR_NAME],
        )
    )
    factor_names = factor_names if factor_names is not None else [_FACTOR_NAME] * row_count
    factor_versions = (
        factor_versions if factor_versions is not None else [_FACTOR_VERSION] * row_count
    )
    factor_categories = (
        factor_categories if factor_categories is not None else [_FACTOR_CATEGORY] * row_count
    )
    timeframes = timeframes if timeframes is not None else ["1h"] * row_count
    selection_scores = selection_scores if selection_scores is not None else [0.80] * row_count
    selection_ranks = selection_ranks if selection_ranks is not None else [1] * row_count
    selected = selected if selected is not None else [True] * row_count
    selection_times = (
        selection_times if selection_times is not None else [_SELECTION_TIME] * row_count
    )
    statuses = statuses if statuses is not None else ["SELECTED"] * row_count
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "factor_category": factor_categories,
            "timeframe": timeframes,
            "selection_score": selection_scores,
            "selection_rank": selection_ranks,
            "selected": selected,
            "selection_time": selection_times,
            "status": statuses,
        }
    )


def _build(frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run ``SimpleFactorTimeframeAnalysisEngine`` on a selection frame."""
    engine = SimpleFactorTimeframeAnalysisEngine()
    return engine.build(frame if frame is not None else _selection_frame())


def _expected_stability(scores: list[float]) -> float:
    """Mirror engine stability: ``1 - clamp(std / gap, 0, 1)``."""
    if len(scores) < 2:
        return 1.0
    gap = max(scores) - min(scores)
    if gap == 0.0:
        return 1.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance)
    return max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, std / gap))))


def _expected_confidence(scores: list[float]) -> float:
    """Mirror engine confidence from absolute margin and stability."""
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores, reverse=True)
    stability = _expected_stability(scores)
    margin = ordered[0] - ordered[1]
    margin_component = max(0.0, min(1.0, margin))
    return max(0.0, min(1.0, 0.5 * margin_component + 0.5 * stability))


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


def test_canonical_column_order_matches_specification() -> None:
    """Output columns follow the declared timeframe analysis contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "factor_name",
        "factor_version",
        "factor_category",
        "analysis_time",
        "best_timeframe",
        "best_selection_score",
        "timeframe_rank",
        "timeframe_stability",
        "winner_margin",
        "score_gap",
        "timeframe_confidence",
        "selected",
        "source_selection_version",
        "status",
    )
    assert FACTOR_TIMEFRAME_ANALYSIS_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == ("factor_name", "factor_version", "analysis_time")


def test_input_columns_contract() -> None:
    """FACTOR_SELECTION_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "factor_name",
        "factor_version",
        "factor_category",
        "timeframe",
        "selection_score",
        "selection_rank",
        "selected",
        "selection_time",
        "status",
    ):
        assert column in FACTOR_SELECTION_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Frame validator / invalid schema / empty
# ---------------------------------------------------------------------------


def test_validate_factor_selection_frame_rejects_non_dataframe() -> None:
    """validate_factor_selection_frame rejects non-DataFrame with FTA_FRAME_TYPE."""
    with pytest.raises(FactorTimeframeAnalysisError) as exc_info:
        validate_factor_selection_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FTA_FRAME_TYPE"


def test_validate_factor_selection_frame_rejects_empty_dataframe() -> None:
    """validate_factor_selection_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"factor_name": []})
    with pytest.raises(FactorTimeframeAnalysisError) as exc_info:
        validate_factor_selection_frame(empty)
    assert exc_info.value.error_code == "FTA_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty Factor Selection frames."""
    empty = pl.DataFrame(schema={"factor_name": pl.String}).clear()
    with pytest.raises(FactorTimeframeAnalysisError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "FTA_FRAME_EMPTY"


def test_build_rejects_invalid_schema() -> None:
    """build rejects frames missing required Factor Selection columns."""
    frame = pl.DataFrame({"factor_name": [_FACTOR_NAME], "selected": [True]})
    with pytest.raises(FactorTimeframeAnalysisError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "FTA_MISSING_COLUMNS"


def test_build_rejects_when_no_selected_rows() -> None:
    """build rejects frames where every row has selected == False."""
    frame = _selection_frame(selected=[False, False], timeframes=["1h", "4h"])
    with pytest.raises(FactorTimeframeAnalysisError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "FTA_NO_SELECTED"


# ---------------------------------------------------------------------------
# Single / two / three timeframes
# ---------------------------------------------------------------------------


def test_single_timeframe() -> None:
    """Single timeframe yields null margin/gap, stability 1.0, confidence 1.0."""
    result = _build(
        _selection_frame(
            timeframes=["1h"],
            selection_scores=[0.72],
        )
    )
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "1h"
    assert row["best_selection_score"] == pytest.approx(0.72)
    assert row["timeframe_rank"] == 1
    assert row["winner_margin"] is None
    assert row["score_gap"] is None
    assert row["timeframe_stability"] == pytest.approx(1.0)
    assert row["timeframe_confidence"] == pytest.approx(1.0)
    assert row["status"] == TimeframeAnalysisStatus.PASS.value


def test_two_timeframes_winner_margin_and_score_gap() -> None:
    """Two timeframes compute winner, margin, and score gap correctly."""
    scores = [0.40, 0.90]
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=scores,
            selection_ranks=[2, 1],
        )
    )
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "4h"
    assert row["best_selection_score"] == pytest.approx(0.90)
    assert row["winner_margin"] == pytest.approx(0.50)
    assert row["score_gap"] == pytest.approx(0.50)
    assert row["timeframe_stability"] == pytest.approx(_expected_stability(scores))
    assert row["timeframe_confidence"] == pytest.approx(_expected_confidence(scores))
    assert row["status"] == TimeframeAnalysisStatus.PASS.value


def test_three_timeframes_winner_selection() -> None:
    """Three timeframes select the highest selection_score as best_timeframe."""
    scores = [0.55, 0.80, 0.30]
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h", "1d"],
            selection_scores=scores,
            selection_ranks=[2, 1, 3],
        )
    )
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "4h"
    assert row["best_selection_score"] == pytest.approx(0.80)
    assert row["timeframe_rank"] == 1
    assert row["winner_margin"] == pytest.approx(0.25)
    assert row["score_gap"] == pytest.approx(0.50)
    assert row["timeframe_stability"] == pytest.approx(_expected_stability(scores))
    assert row["timeframe_confidence"] == pytest.approx(_expected_confidence(scores))


# ---------------------------------------------------------------------------
# Stability / confidence / ranking
# ---------------------------------------------------------------------------


def test_stability_calculation_equal_scores() -> None:
    """Equal scores across timeframes produce stability 1.0 and zero gap."""
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h", "1d"],
            selection_scores=[0.60, 0.60, 0.60],
        )
    )
    row = result.row(0, named=True)
    assert row["winner_margin"] == pytest.approx(0.0)
    assert row["score_gap"] == pytest.approx(0.0)
    assert row["timeframe_stability"] == pytest.approx(1.0)
    assert row["timeframe_confidence"] == pytest.approx(0.5)


def test_confidence_increases_with_clear_winner() -> None:
    """Larger winner margin relative to gap increases timeframe confidence."""
    tight = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=[0.50, 0.51],
        )
    )
    clear = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=[0.10, 0.90],
        )
    )
    assert clear["timeframe_confidence"][0] > tight["timeframe_confidence"][0]


def test_deterministic_ranking_tie_break_by_timeframe() -> None:
    """Equal scores break ties by ascending timeframe for deterministic winners."""
    first = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["4h", "1h"],
            selection_scores=[0.70, 0.70],
        )
    )
    second = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=[0.70, 0.70],
        )
    )
    assert_frame_equal(first, second)
    assert first["best_timeframe"].to_list() == ["1h"]


def test_output_is_deterministic_across_repeated_builds() -> None:
    """Repeated builds with identical inputs produce identical DataFrames."""
    frame = _selection_frame(
        factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
        timeframes=["1d", "1h", "4h"],
        selection_scores=[0.20, 0.90, 0.55],
    )
    engine = SimpleFactorTimeframeAnalysisEngine()
    assert_frame_equal(engine.build(frame), engine.build(frame))


# ---------------------------------------------------------------------------
# Missing scores / duplicates / immutability / PASS-FAIL
# ---------------------------------------------------------------------------


def test_missing_scores_produce_fail_status() -> None:
    """All-null selection scores yield FAIL because best score is null."""
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=[None, None],
        )
    )
    row = result.row(0, named=True)
    assert row["best_selection_score"] is None
    assert row["status"] == TimeframeAnalysisStatus.FAIL.value


def test_missing_scores_prefer_finite_winner() -> None:
    """Finite scores win over null scores within a factor group."""
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h", "1d"],
            selection_scores=[None, 0.65, None],
        )
    )
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "4h"
    assert row["best_selection_score"] == pytest.approx(0.65)
    assert row["status"] == TimeframeAnalysisStatus.PASS.value


def test_duplicate_factor_timeframe_keeps_highest_score() -> None:
    """Duplicate (factor, version, timeframe) rows keep the highest score."""
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "1h", "4h"],
            selection_scores=[0.40, 0.85, 0.70],
            selection_times=[_SELECTION_TIME, _SELECTION_TIME + 1, _SELECTION_TIME],
        )
    )
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "1h"
    assert row["best_selection_score"] == pytest.approx(0.85)
    assert row["winner_margin"] == pytest.approx(0.15)


def test_multiple_factors_produce_one_row_each() -> None:
    """Distinct factor identities each produce exactly one analysis row."""
    result = _build(
        _selection_frame(
            factor_names=["alpha", "alpha", "beta", "beta"],
            factor_versions=["1.0.0", "1.0.0", "2.0.0", "2.0.0"],
            timeframes=["1h", "4h", "1h", "1d"],
            selection_scores=[0.30, 0.90, 0.80, 0.40],
        )
    )
    assert result.height == 2
    assert result["factor_name"].to_list() == ["alpha", "beta"]
    assert result["best_timeframe"].to_list() == ["4h", "1h"]
    assert result["best_selection_score"].to_list() == pytest.approx([0.90, 0.80])


def test_unselected_rows_are_ignored() -> None:
    """Only selected==True rows participate in timeframe analysis."""
    result = _build(
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h", "1d"],
            selection_scores=[0.99, 0.50, 0.40],
            selected=[False, True, True],
            statuses=["REJECTED", "SELECTED", "SELECTED"],
        )
    )
    row = result.row(0, named=True)
    assert row["best_timeframe"] == "4h"
    assert row["best_selection_score"] == pytest.approx(0.50)
    assert row["score_gap"] == pytest.approx(0.10)


def test_input_frame_is_not_mutated() -> None:
    """Engine returns a new frame and never mutates the caller input."""
    frame = _selection_frame(
        factor_names=[_FACTOR_NAME, _FACTOR_NAME],
        timeframes=["1h", "4h"],
        selection_scores=[0.40, 0.80],
    )
    before = frame.clone()
    result = _build(frame)
    assert_frame_equal(frame, before)
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == TIMEFRAME_ANALYSIS_SCHEMA


def test_pass_status_requires_winner_and_score() -> None:
    """PASS requires non-empty best_timeframe and non-null best_selection_score."""
    result = _build(
        _selection_frame(
            timeframes=["1d"],
            selection_scores=[0.55],
        )
    )
    assert result["status"].to_list() == [TimeframeAnalysisStatus.PASS.value]


def test_fail_status_when_best_score_missing() -> None:
    """FAIL is emitted when the winning score is null."""
    result = _build(
        _selection_frame(
            timeframes=["1h"],
            selection_scores=[None],
        )
    )
    assert result["status"].to_list() == [TimeframeAnalysisStatus.FAIL.value]


# ---------------------------------------------------------------------------
# Pipeline / registry smoke
# ---------------------------------------------------------------------------


def test_pipeline_runs_registered_engine() -> None:
    """Pipeline resolves a registered engine and finalizes schema output."""
    registry = FactorTimeframeAnalysisEngineRegistry()
    registry.register("simple", SimpleFactorTimeframeAnalysisEngine())
    pipeline = FactorTimeframeAnalysisPipeline(registry)
    result = pipeline.run(
        "simple",
        _selection_frame(
            factor_names=[_FACTOR_NAME, _FACTOR_NAME],
            timeframes=["1h", "4h"],
            selection_scores=[0.20, 0.75],
        ),
    )
    assert result.schema == TIMEFRAME_ANALYSIS_SCHEMA
    assert result.height == 1
    assert result["best_timeframe"].to_list() == ["4h"]
