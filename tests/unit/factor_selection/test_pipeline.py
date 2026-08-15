"""Unit tests for CQROS ``FactorSelectionPipeline``."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_selection import (
    FACTOR_SELECTION_SCHEMA,
    FactorSelectionEngineRegistry,
    FactorSelectionError,
    FactorSelectionPipeline,
    FactorSelectionStatus,
    SimpleFactorSelectionEngine,
)
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER, COLUMN_DTYPES
from cqros.factor_validation.schema import (
    FACTOR_VALIDATION_SCHEMA,
    FactorValidationStatus,
)

_TIMEFRAME = "1h"
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_DATASET_VERSION = "dataset-v1"
_LABEL_VERSION = "label-v1"
_VALIDATION_TIME = 1_704_067_200_000
_VALIDATION_START_TIME = 1_703_980_800_000
_VALIDATION_END_TIME = 1_704_067_200_000

_STRONG_IC = 0.12
_STRONG_RANK_IC = 0.10
_STRONG_ICIR = 0.80
_STRONG_IC_STD = 0.15
_STRONG_P_VALUE = 0.01
_STRONG_IC_DECAY = 0.70
_STRONG_TURNOVER = 0.20
_STRONG_MONOTONICITY_SCORE = 0.80
_STRONG_QUANTILE_SPREAD = 0.05
_STRONG_OBSERVATIONS = 200
_STRONG_IC_OBSERVATIONS = 150

_REASON_TOP_N = "top_n"
_REASON_OUTSIDE_TOP_N = "outside_top_n"


def _factor_validation_frame(
    *,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    timeframes: list[str] | None = None,
    validation_times: list[int] | None = None,
    dataset_versions: list[str] | None = None,
    label_versions: list[str] | None = None,
    validation_start_times: list[int] | None = None,
    validation_end_times: list[int] | None = None,
    information_coefficients: list[float] | None = None,
    rank_information_coefficients: list[float] | None = None,
    ic_information_ratios: list[float] | None = None,
    ic_stds: list[float] | None = None,
    ic_p_values: list[float] | None = None,
    ic_t_stats: list[float] | None = None,
    ic_decays: list[float] | None = None,
    turnovers: list[float] | None = None,
    monotonicity_scores: list[float] | None = None,
    quantile_spreads: list[float] | None = None,
    observations: list[int] | None = None,
    ic_observations: list[int] | None = None,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical Factor Validation ledger for pipeline tests."""
    factor_names = factor_names if factor_names is not None else [_FACTOR_NAME]
    row_count = len(factor_names)
    factor_versions = (
        factor_versions if factor_versions is not None else [_FACTOR_VERSION] * row_count
    )
    factor_categories = (
        factor_categories if factor_categories is not None else [_FACTOR_CATEGORY] * row_count
    )
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    validation_times = (
        validation_times if validation_times is not None else [_VALIDATION_TIME] * row_count
    )
    dataset_versions = (
        dataset_versions if dataset_versions is not None else [_DATASET_VERSION] * row_count
    )
    label_versions = label_versions if label_versions is not None else [_LABEL_VERSION] * row_count
    validation_start_times = (
        validation_start_times
        if validation_start_times is not None
        else [_VALIDATION_START_TIME] * row_count
    )
    validation_end_times = (
        validation_end_times
        if validation_end_times is not None
        else [_VALIDATION_END_TIME] * row_count
    )
    information_coefficients = (
        information_coefficients
        if information_coefficients is not None
        else [_STRONG_IC] * row_count
    )
    rank_information_coefficients = (
        rank_information_coefficients
        if rank_information_coefficients is not None
        else [_STRONG_RANK_IC] * row_count
    )
    ic_information_ratios = (
        ic_information_ratios if ic_information_ratios is not None else [_STRONG_ICIR] * row_count
    )
    ic_stds = ic_stds if ic_stds is not None else [_STRONG_IC_STD] * row_count
    ic_p_values = ic_p_values if ic_p_values is not None else [_STRONG_P_VALUE] * row_count
    ic_t_stats = ic_t_stats if ic_t_stats is not None else [3.0] * row_count
    ic_decays = ic_decays if ic_decays is not None else [_STRONG_IC_DECAY] * row_count
    turnovers = turnovers if turnovers is not None else [_STRONG_TURNOVER] * row_count
    monotonicity_scores = (
        monotonicity_scores
        if monotonicity_scores is not None
        else [_STRONG_MONOTONICITY_SCORE] * row_count
    )
    quantile_spreads = (
        quantile_spreads if quantile_spreads is not None else [_STRONG_QUANTILE_SPREAD] * row_count
    )
    observations = observations if observations is not None else [_STRONG_OBSERVATIONS] * row_count
    ic_observations = (
        ic_observations if ic_observations is not None else [_STRONG_IC_OBSERVATIONS] * row_count
    )
    statuses = statuses if statuses is not None else [FactorValidationStatus.PASS.value] * row_count
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "timeframe": timeframes,
            "validation_time": validation_times,
            "factor_category": factor_categories,
            "dataset_version": dataset_versions,
            "label_version": label_versions,
            "validation_start_time": validation_start_times,
            "validation_end_time": validation_end_times,
            "information_coefficient": information_coefficients,
            "rank_information_coefficient": rank_information_coefficients,
            "ic_information_ratio": ic_information_ratios,
            "ic_std": ic_stds,
            "ic_p_value": ic_p_values,
            "ic_t_stat": ic_t_stats,
            "ic_decay": ic_decays,
            "turnover": turnovers,
            "monotonicity_score": monotonicity_scores,
            "quantile_spread": quantile_spreads,
            "observations": observations,
            "ic_observations": ic_observations,
            "status": statuses,
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    )


def _canonical_selection_row(
    *,
    status: str = FactorSelectionStatus.SELECTED.value,
    selection_reason: str = _REASON_TOP_N,
) -> pl.DataFrame:
    """Build one canonical factor-selection row for synthetic engine outputs."""
    from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY

    return pl.DataFrame(
        {
            "factor_name": [_FACTOR_NAME],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "selection_time": [_VALIDATION_TIME],
            "factor_category": [_FACTOR_CATEGORY],
            "selected": [True],
            "selection_score": [_STRONG_IC],
            "selection_rank": [1],
            "selection_reason": [selection_reason],
            "selection_ic": [_STRONG_IC],
            "selected_direction": [1],
            "orientation_policy": [FACTOR_ORIENTATION_POLICY],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build_registry(engine_name: str = "simple") -> FactorSelectionEngineRegistry:
    """Build a registry with SimpleFactorSelectionEngine under engine_name."""
    registry = FactorSelectionEngineRegistry()
    registry.register(engine_name, SimpleFactorSelectionEngine())
    return registry


def _run(
    pipeline: FactorSelectionPipeline,
    *,
    engine_name: str = "simple",
    factor_validation: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the pipeline with a default Factor Validation frame."""
    return pipeline.run(
        engine_name,
        (factor_validation if factor_validation is not None else _factor_validation_frame()),
    )


# ---------------------------------------------------------------------------
# Engine-name validation
# ---------------------------------------------------------------------------


def test_run_rejects_blank_engine_name() -> None:
    """Blank or whitespace engine names raise FSEL_PIPE_NAME_BLANK."""
    pipeline = FactorSelectionPipeline(_build_registry())
    for blank in ("", "   "):
        with pytest.raises(FactorSelectionError) as exc_info:
            pipeline.run(blank, _factor_validation_frame())
        assert exc_info.value.error_code == "FSEL_PIPE_NAME_BLANK"


def test_run_rejects_unknown_engine_name() -> None:
    """Unregistered engine name causes FSEL_REG_UNKNOWN from registry lookup."""
    pipeline = FactorSelectionPipeline(_build_registry())
    with pytest.raises(FactorSelectionError) as exc_info:
        _run(pipeline, engine_name="unknown-engine")
    assert exc_info.value.error_code == "FSEL_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# Duplicate / missing primary key rejection
# ---------------------------------------------------------------------------


def test_run_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys in engine output raise FSEL_PIPE_DUPLICATE_KEYS."""

    class _DuplicatingEngine:
        """Test engine that returns two rows with identical primary keys."""

        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_selection_row()
            return pl.concat([row, row])

    registry = FactorSelectionEngineRegistry()
    registry.register("duplicating", _DuplicatingEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("duplicating", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_DUPLICATE_KEYS"


def test_run_rejects_missing_primary_key_columns() -> None:
    """Engine output missing primary-key columns raises FSEL_PIPE_MISSING_PRIMARY_KEYS."""

    class _MissingPrimaryKeyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return _canonical_selection_row().drop("selection_time")

    registry = FactorSelectionEngineRegistry()
    registry.register("missing-pk", _MissingPrimaryKeyEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("missing-pk", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_MISSING_PRIMARY_KEYS"


# ---------------------------------------------------------------------------
# Invalid engine output types
# ---------------------------------------------------------------------------


def test_run_rejects_non_dataframe_engine_output() -> None:
    """Engine returning a non-DataFrame raises FSEL_PIPE_INVALID_OUTPUT."""

    class _BadEngine:
        def build(self, *args: object, **kwargs: object) -> object:
            return "not-a-dataframe"

    registry = FactorSelectionEngineRegistry()
    registry.register("bad", _BadEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("bad", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_INVALID_OUTPUT"


def test_run_rejects_empty_engine_output() -> None:
    """Engine returning an empty DataFrame raises FSEL_PIPE_OUTPUT_EMPTY."""

    class _EmptyEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame({"factor_name": []})

    registry = FactorSelectionEngineRegistry()
    registry.register("empty", _EmptyEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("empty", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_OUTPUT_EMPTY"


def test_run_rejects_missing_schema_columns_in_engine_output() -> None:
    """Engine returning a DataFrame missing required columns raises FSEL_PIPE_MISSING_COLUMNS."""

    class _IncompleteEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            # Include primary keys so the failure is specifically missing required fields.
            return pl.DataFrame(
                {
                    "factor_name": [_FACTOR_NAME],
                    "factor_version": [_FACTOR_VERSION],
                    "timeframe": [_TIMEFRAME],
                    "selection_time": [_VALIDATION_TIME],
                    "selection_score": [_STRONG_IC],
                }
            )

    registry = FactorSelectionEngineRegistry()
    registry.register("incomplete", _IncompleteEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("incomplete", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_MISSING_COLUMNS"


def test_run_rejects_schema_cast_failure() -> None:
    """Engine output that cannot cast to FACTOR_SELECTION_SCHEMA raises FSEL_PIPE_SCHEMA_CAST."""

    class _UncastableEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_selection_row()
            return row.with_columns(pl.lit("not-a-float").alias("selection_score"))

    registry = FactorSelectionEngineRegistry()
    registry.register("uncastable", _UncastableEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)

    with pytest.raises(FactorSelectionError) as exc_info:
        pipeline.run("uncastable", _factor_validation_frame())
    assert exc_info.value.error_code == "FSEL_PIPE_SCHEMA_CAST"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_produces_canonical_selection_row() -> None:
    """Pipeline with default inputs produces one canonical selection output row."""
    from cqros.factor_selection.schema import ELIGIBILITY_COLUMNS

    pipeline = FactorSelectionPipeline(_build_registry())
    result = _run(pipeline)
    assert result.height == 1
    # Canonical columns come first.
    assert tuple(result.columns[: len(CANONICAL_COLUMN_ORDER)]) == CANONICAL_COLUMN_ORDER
    # Canonical dtypes intact.
    for col, dtype in FACTOR_SELECTION_SCHEMA.items():
        assert result.schema[col] == dtype
    # Eligibility extension present.
    for col in ELIGIBILITY_COLUMNS:
        assert col in result.columns
    assert result["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert result["selection_time"].to_list() == [_VALIDATION_TIME]
    assert result["selected"].to_list() == [True]
    assert result["selection_score"].to_list()[0] == pytest.approx(1.0)
    assert result["selection_rank"].to_list() == [1]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]


def test_run_resolves_engine_from_registry() -> None:
    """Pipeline resolves and executes the engine registered under engine_name."""
    registry = FactorSelectionEngineRegistry()
    engine = SimpleFactorSelectionEngine()
    registry.register("custom", engine)
    pipeline = FactorSelectionPipeline(registry)
    result = _run(pipeline, engine_name="custom")
    # Canonical schema dtypes intact.
    for col, dtype in FACTOR_SELECTION_SCHEMA.items():
        assert result.schema[col] == dtype
    assert result.height == 1


def test_run_reorders_columns_to_canonical_order() -> None:
    """Pipeline reorders engine output columns to CANONICAL_COLUMN_ORDER."""

    class _ShuffledEngine:
        def build(self, *args: object, **kwargs: object) -> pl.DataFrame:
            row = _canonical_selection_row()
            shuffled = list(reversed(row.columns))
            return row.select(shuffled)

    registry = FactorSelectionEngineRegistry()
    registry.register("shuffled", _ShuffledEngine())  # type: ignore[arg-type]
    pipeline = FactorSelectionPipeline(registry)
    result = pipeline.run("shuffled", _factor_validation_frame())
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == FACTOR_SELECTION_SCHEMA


def test_run_does_not_mutate_input_frame() -> None:
    """Pipeline run must not mutate the caller-supplied Factor Validation frame."""
    pipeline = FactorSelectionPipeline(_build_registry())
    factor_validation = _factor_validation_frame()
    before = factor_validation.clone()
    pipeline.run("simple", factor_validation)
    assert_frame_equal(factor_validation, before)


def test_run_applies_top_n_ranking_selection() -> None:
    """Pipeline selects factors by rank within DEFAULT_TOP_N, not thresholds."""
    pipeline = FactorSelectionPipeline(_build_registry())
    result = pipeline.run(
        "simple",
        _factor_validation_frame(
            factor_names=["weak", "strong"],
            information_coefficients=[0.01, 0.20],
            monotonicity_scores=[0.10, 0.90],
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["strong"]["selected"] is True
    assert by_name["weak"]["selected"] is True
    assert by_name["strong"]["selection_rank"] == 1
    assert by_name["weak"]["selection_rank"] == 2
    assert by_name["strong"]["selection_reason"] == _REASON_TOP_N
    assert by_name["weak"]["selection_reason"] == _REASON_TOP_N


# ---------------------------------------------------------------------------
# Ranking selection through the pipeline
# ---------------------------------------------------------------------------


def test_pipeline_scores_and_selects_pass_rows() -> None:
    """PASS validation rows are scored, ranked, and selected when inside top N."""
    pipeline = FactorSelectionPipeline(_build_registry())
    result = pipeline.run(
        "simple",
        _factor_validation_frame(statuses=[FactorValidationStatus.PASS.value]),
    )
    assert result.height == 1
    assert result["selected"].to_list() == [True]
    assert result["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]


def test_pipeline_scores_fail_rows_by_rank() -> None:
    """FAIL validation rows are scored and selected when inside top N."""
    pipeline = FactorSelectionPipeline(_build_registry())
    result = pipeline.run(
        "simple",
        _factor_validation_frame(statuses=[FactorValidationStatus.FAIL.value]),
    )
    assert result.height == 1
    assert result["selected"].to_list() == [True]
    assert result["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]


def test_pipeline_scores_skipped_rows_by_rank() -> None:
    """SKIPPED validation rows are scored and selected when inside top N."""
    pipeline = FactorSelectionPipeline(_build_registry())
    result = pipeline.run(
        "simple",
        _factor_validation_frame(statuses=[FactorValidationStatus.SKIPPED.value]),
    )
    assert result.height == 1
    assert result["selected"].to_list() == [True]
    assert result["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]


def test_pipeline_preserves_engine_ranking_decisions() -> None:
    """Pipeline finalization preserves engine ranking decisions without alteration."""
    mixed = _factor_validation_frame(
        factor_names=["pass_factor", "fail_factor", "skipped_factor"],
        factor_versions=["1.0.0", "1.0.0", "1.0.0"],
        information_coefficients=[0.10, 0.30, 0.20],
        statuses=[
            FactorValidationStatus.PASS.value,
            FactorValidationStatus.FAIL.value,
            FactorValidationStatus.SKIPPED.value,
        ],
    )
    engine = SimpleFactorSelectionEngine()
    registry = FactorSelectionEngineRegistry()
    registry.register("simple", engine)
    pipeline = FactorSelectionPipeline(registry)

    engine_output = engine.build(mixed)
    pipeline_output = pipeline.run("simple", mixed)

    assert_frame_equal(pipeline_output, engine_output)
    by_name = {row["factor_name"]: row for row in pipeline_output.to_dicts()}
    assert by_name["fail_factor"]["selection_rank"] == 1
    assert by_name["skipped_factor"]["selection_rank"] == 2
    assert by_name["pass_factor"]["selection_rank"] == 3
    assert pipeline_output["selected"].to_list() == [True, True, True]
    assert pipeline_output["status"].to_list() == [
        FactorSelectionStatus.SELECTED.value,
        FactorSelectionStatus.SELECTED.value,
        FactorSelectionStatus.SELECTED.value,
    ]
    assert _REASON_OUTSIDE_TOP_N not in pipeline_output["selection_reason"].to_list()


def test_validation_fixture_matches_factor_validation_schema() -> None:
    """Pipeline fixtures are canonical Factor Validation ledgers with status."""
    frame = _factor_validation_frame()
    assert frame.schema == FACTOR_VALIDATION_SCHEMA
    assert "monotonicity_score" in frame.columns
    assert "status" in frame.columns
    assert frame["status"].to_list() == [FactorValidationStatus.PASS.value]
    assert frame["dataset_version"].to_list() == [_DATASET_VERSION]
    assert frame["label_version"].to_list() == [_LABEL_VERSION]
    assert frame["validation_start_time"].to_list() == [_VALIDATION_START_TIME]
    assert frame["validation_end_time"].to_list() == [_VALIDATION_END_TIME]
