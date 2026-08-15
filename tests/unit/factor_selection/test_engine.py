"""Unit tests for CQROS ``SimpleFactorSelectionEngine`` ranking selection."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_selection import (
    FACTOR_SELECTION_SCHEMA,
    FactorSelectionError,
    FactorSelectionStatus,
    SimpleFactorSelectionEngine,
)
from cqros.factor_selection.engine import (
    DEFAULT_TOP_N,
    FACTOR_VALIDATION_INPUT_COLUMNS,
    validate_factor_validation_frame,
)
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER

_TIMEFRAME = "1h"
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_VALIDATION_TIME = 1_704_067_200_000

_STRONG_IC = 0.12
_STRONG_RANK_IC = 0.10
_STRONG_ICIR = 0.80
_STRONG_P_VALUE = 0.01
_STRONG_IC_DECAY = 0.70
_STRONG_TURNOVER = 0.20
_STRONG_MONOTONICITY_SCORE = 0.80
_STRONG_QUANTILE_SPREAD = 0.05
_STRONG_OBSERVATIONS = 200

_REASON_TOP_N = "top_n"
_REASON_OUTSIDE_TOP_N = "outside_top_n"

# Fixed composite-score weights used by the ranking engine.
_WEIGHT_ABS_IC = 0.30
_WEIGHT_ABS_RANK_IC = 0.20
_WEIGHT_ICIR = 0.20
_WEIGHT_QUANTILE_SPREAD = 0.10
_WEIGHT_MONOTONICITY = 0.10
_WEIGHT_IC_DECAY = 0.05
_WEIGHT_INVERSE_TURNOVER = 0.05


def _factor_validation_frame(
    *,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    timeframes: list[str] | None = None,
    validation_times: list[int] | None = None,
    information_coefficients: list[float] | None = None,
    rank_information_coefficients: list[float] | None = None,
    ic_information_ratios: list[float] | None = None,
    ic_p_values: list[float] | None = None,
    ic_decays: list[float] | None = None,
    turnovers: list[float] | None = None,
    monotonicity_scores: list[float] | None = None,
    quantile_spreads: list[float] | None = None,
    observations: list[int] | None = None,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a Factor Validation frame with strong defaults unless overridden."""
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
    ic_p_values = ic_p_values if ic_p_values is not None else [_STRONG_P_VALUE] * row_count
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
    statuses = statuses if statuses is not None else ["PASS"] * row_count
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "timeframe": timeframes,
            "validation_time": validation_times,
            "factor_category": factor_categories,
            "information_coefficient": information_coefficients,
            "rank_information_coefficient": rank_information_coefficients,
            "ic_information_ratio": ic_information_ratios,
            "ic_p_value": ic_p_values,
            "ic_t_stat": [3.0] * row_count,
            "ic_decay": ic_decays,
            "turnover": turnovers,
            "monotonicity_score": monotonicity_scores,
            "quantile_spread": quantile_spreads,
            "observations": observations,
            "status": statuses,
        }
    )


def _build(
    engine: SimpleFactorSelectionEngine,
    *,
    factor_validation: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build factor selection rows with a default strong Factor Validation frame."""
    return engine.build(
        factor_validation if factor_validation is not None else _factor_validation_frame()
    )


def _minmax(values: list[float], value: float) -> float:
    """Min-max normalize ``value`` within ``values`` using engine semantics."""
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return 1.0
    return (value - minimum) / (maximum - minimum)


def _expected_score(
    *,
    information_coefficients: list[float],
    rank_information_coefficients: list[float],
    ic_information_ratios: list[float],
    quantile_spreads: list[float],
    monotonicity_scores: list[float],
    ic_decays: list[float],
    turnovers: list[float],
    index: int,
) -> float:
    """Compute the expected composite selection score for row ``index``."""
    abs_ics = [abs(value) for value in information_coefficients]
    abs_rank_ics = [abs(value) for value in rank_information_coefficients]
    inverse_turnovers = [-value for value in turnovers]
    return (
        _WEIGHT_ABS_IC * _minmax(abs_ics, abs_ics[index])
        + _WEIGHT_ABS_RANK_IC * _minmax(abs_rank_ics, abs_rank_ics[index])
        + _WEIGHT_ICIR * _minmax(ic_information_ratios, ic_information_ratios[index])
        + _WEIGHT_QUANTILE_SPREAD * _minmax(quantile_spreads, quantile_spreads[index])
        + _WEIGHT_MONOTONICITY * _minmax(monotonicity_scores, monotonicity_scores[index])
        + _WEIGHT_IC_DECAY * _minmax(ic_decays, ic_decays[index])
        + _WEIGHT_INVERSE_TURNOVER * _minmax(inverse_turnovers, inverse_turnovers[index])
    )


# ---------------------------------------------------------------------------
# Input column contracts
# ---------------------------------------------------------------------------


def test_input_columns_contract() -> None:
    """FACTOR_VALIDATION_INPUT_COLUMNS enumerates every column the engine consumes."""
    for column in (
        "factor_name",
        "factor_version",
        "factor_category",
        "timeframe",
        "validation_time",
        "information_coefficient",
        "rank_information_coefficient",
        "ic_information_ratio",
        "ic_p_value",
        "ic_decay",
        "turnover",
        "monotonicity_score",
        "quantile_spread",
        "observations",
        "status",
    ):
        assert column in FACTOR_VALIDATION_INPUT_COLUMNS


def test_default_top_n_constant() -> None:
    """DEFAULT_TOP_N is fixed at 20 and used when top_n is omitted."""
    assert DEFAULT_TOP_N == 20
    engine = SimpleFactorSelectionEngine()
    assert engine.top_n == DEFAULT_TOP_N


def test_omitted_top_n_defaults_to_twenty() -> None:
    """Omitting top_n selects at most 20 factors per timeframe."""
    names = [f"factor_{index:02d}" for index in range(25)]
    ics = [1.0 - (index * 0.01) for index in range(25)]
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=names,
            information_coefficients=ics,
        ),
    )
    assert result.filter(pl.col("selected")).height == 20


# ---------------------------------------------------------------------------
# Frame validator tests
# ---------------------------------------------------------------------------


def test_validate_factor_validation_frame_rejects_non_dataframe() -> None:
    """validate_factor_validation_frame rejects non-DataFrame with FSEL_FRAME_TYPE."""
    with pytest.raises(FactorSelectionError) as exc_info:
        validate_factor_validation_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FSEL_FRAME_TYPE"


def test_validate_factor_validation_frame_rejects_empty_dataframe() -> None:
    """validate_factor_validation_frame rejects DataFrames with zero rows."""
    empty = pl.DataFrame({"factor_name": []})
    with pytest.raises(FactorSelectionError) as exc_info:
        validate_factor_validation_frame(empty)
    assert exc_info.value.error_code == "FSEL_FRAME_EMPTY"


def test_build_rejects_empty_dataframe() -> None:
    """build rejects empty Factor Validation frames."""
    empty = pl.DataFrame(schema={column: pl.String for column in ("factor_name",)}).clear()
    with pytest.raises(FactorSelectionError) as exc_info:
        SimpleFactorSelectionEngine().build(empty)
    assert exc_info.value.error_code == "FSEL_FRAME_EMPTY"


# ---------------------------------------------------------------------------
# Missing column validation
# ---------------------------------------------------------------------------


def test_build_rejects_missing_factor_validation_columns() -> None:
    """Missing required Factor Validation columns raise FSEL_MISSING_COLUMNS."""
    engine = SimpleFactorSelectionEngine()
    with pytest.raises(FactorSelectionError) as exc_info:
        _build(engine, factor_validation=_factor_validation_frame().drop("factor_category"))
    assert exc_info.value.error_code == "FSEL_MISSING_COLUMNS"


def test_build_rejects_missing_ic_information_ratio() -> None:
    """Missing ic_information_ratio raises FSEL_MISSING_COLUMNS."""
    with pytest.raises(FactorSelectionError) as exc_info:
        _build(
            SimpleFactorSelectionEngine(),
            factor_validation=_factor_validation_frame().drop("ic_information_ratio"),
        )
    assert exc_info.value.error_code == "FSEL_MISSING_COLUMNS"


def test_build_rejects_missing_status() -> None:
    """Missing validation status raises FSEL_MISSING_COLUMNS."""
    with pytest.raises(FactorSelectionError) as exc_info:
        _build(
            SimpleFactorSelectionEngine(),
            factor_validation=_factor_validation_frame().drop("status"),
        )
    assert exc_info.value.error_code == "FSEL_MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_single_factor_receives_full_component_score() -> None:
    """A lone factor within a timeframe receives selection_score 1.0."""
    result = _build(SimpleFactorSelectionEngine())
    assert result.height == 1
    assert result["selection_score"].to_list()[0] == pytest.approx(1.0)
    assert result["selected"].to_list() == [True]
    assert result["status"].to_list() == [FactorSelectionStatus.SELECTED.value]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]
    assert result["selection_rank"].to_list() == [1]


def test_selection_score_uses_fixed_weighted_components() -> None:
    """selection_score matches the fixed weighted min-max composite formula."""
    information_coefficients = [0.04, 0.20]
    rank_information_coefficients = [0.05, 0.15]
    ic_information_ratios = [0.40, 0.90]
    quantile_spreads = [0.02, 0.08]
    monotonicity_scores = [0.50, 0.90]
    ic_decays = [0.40, 0.80]
    turnovers = [0.40, 0.10]
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["weak", "strong"],
            information_coefficients=information_coefficients,
            rank_information_coefficients=rank_information_coefficients,
            ic_information_ratios=ic_information_ratios,
            quantile_spreads=quantile_spreads,
            monotonicity_scores=monotonicity_scores,
            ic_decays=ic_decays,
            turnovers=turnovers,
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    expected_weak = _expected_score(
        information_coefficients=information_coefficients,
        rank_information_coefficients=rank_information_coefficients,
        ic_information_ratios=ic_information_ratios,
        quantile_spreads=quantile_spreads,
        monotonicity_scores=monotonicity_scores,
        ic_decays=ic_decays,
        turnovers=turnovers,
        index=0,
    )
    expected_strong = _expected_score(
        information_coefficients=information_coefficients,
        rank_information_coefficients=rank_information_coefficients,
        ic_information_ratios=ic_information_ratios,
        quantile_spreads=quantile_spreads,
        monotonicity_scores=monotonicity_scores,
        ic_decays=ic_decays,
        turnovers=turnovers,
        index=1,
    )
    assert by_name["weak"]["selection_score"] == pytest.approx(expected_weak)
    assert by_name["strong"]["selection_score"] == pytest.approx(expected_strong)
    assert by_name["strong"]["selection_score"] > by_name["weak"]["selection_score"]


def test_score_normalization_is_per_timeframe() -> None:
    """Min-max normalization runs independently inside each timeframe."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["a", "b", "c"],
            timeframes=["1h", "1h", "4h"],
            information_coefficients=[0.04, 0.20, 0.04],
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["b"]["selection_score"] > by_name["a"]["selection_score"]
    assert by_name["c"]["selection_score"] == pytest.approx(1.0)


def test_weak_metrics_are_still_scored_and_ranked() -> None:
    """Threshold-weak metrics no longer reject; factors remain scored and ranked."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            information_coefficients=[0.01],
            rank_information_coefficients=[0.01],
            ic_information_ratios=[0.10],
            ic_p_values=[0.90],
            quantile_spreads=[0.0],
            monotonicity_scores=[0.10],
            turnovers=[0.90],
            ic_decays=[0.10],
            observations=[1],
        ),
    )
    assert result.height == 1
    assert result["selection_score"].to_list()[0] == pytest.approx(1.0)
    assert result["selection_rank"].to_list() == [1]
    assert result["selected"].to_list() == [True]
    assert result["selection_reason"].to_list() == [_REASON_TOP_N]


# ---------------------------------------------------------------------------
# Ranking and top-N selection
# ---------------------------------------------------------------------------


def test_ranking_order_within_timeframe() -> None:
    """Within a timeframe, higher selection_score receives a better (lower) rank."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["weak_ic", "strong_ic"],
            information_coefficients=[0.04, 0.20],
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["strong_ic"]["selection_rank"] == 1
    assert by_name["weak_ic"]["selection_rank"] == 2
    assert by_name["strong_ic"]["selection_score"] > by_name["weak_ic"]["selection_score"]


def test_ranking_is_independent_across_timeframes() -> None:
    """Each timeframe receives its own 1..N selection_rank sequence."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["a", "b", "c"],
            timeframes=["1h", "1h", "4h"],
            information_coefficients=[0.04, 0.20, 0.10],
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["b"]["selection_rank"] == 1
    assert by_name["a"]["selection_rank"] == 2
    assert by_name["c"]["selection_rank"] == 1


def test_ranking_tie_break_is_deterministic() -> None:
    """Equal scores break ties by factor_name then factor_version."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["zeta", "alpha", "alpha"],
            factor_versions=["1.0.0", "2.0.0", "1.0.0"],
        ),
    )
    ordered = result.sort("selection_rank")
    assert ordered["factor_name"].to_list() == ["alpha", "alpha", "zeta"]
    assert ordered["factor_version"].to_list() == ["1.0.0", "2.0.0", "1.0.0"]
    assert ordered["selection_rank"].to_list() == [1, 2, 3]


def test_top_n_selection_marks_rank_within_default_top_n() -> None:
    """Factors with selection_rank <= DEFAULT_TOP_N are SELECTED with reason top_n."""
    names = [f"factor_{index:02d}" for index in range(DEFAULT_TOP_N + 5)]
    ics = [1.0 - (index * 0.01) for index in range(DEFAULT_TOP_N + 5)]
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=names,
            information_coefficients=ics,
        ),
    )
    selected = result.filter(pl.col("selected")).sort("selection_rank")
    rejected = result.filter(~pl.col("selected")).sort("selection_rank")
    assert selected.height == DEFAULT_TOP_N
    assert rejected.height == 5
    assert selected["selection_rank"].to_list() == list(range(1, DEFAULT_TOP_N + 1))
    assert rejected["selection_rank"].to_list() == list(range(DEFAULT_TOP_N + 1, DEFAULT_TOP_N + 6))
    assert selected["selection_reason"].to_list() == [_REASON_TOP_N] * DEFAULT_TOP_N
    assert rejected["selection_reason"].to_list() == [_REASON_OUTSIDE_TOP_N] * 5
    assert selected["status"].to_list() == [FactorSelectionStatus.SELECTED.value] * DEFAULT_TOP_N
    assert rejected["status"].to_list() == [FactorSelectionStatus.REJECTED.value] * 5


@pytest.mark.parametrize("top_n", [10, 20, 30])
def test_configurable_top_n_selects_exact_count(top_n: int) -> None:
    """Configured top_n selects exactly that many factors when enough exist."""
    count = top_n + 5
    names = [f"factor_{index:02d}" for index in range(count)]
    ics = [1.0 - (index * 0.01) for index in range(count)]
    result = _build(
        SimpleFactorSelectionEngine(top_n=top_n),
        factor_validation=_factor_validation_frame(
            factor_names=names,
            information_coefficients=ics,
        ),
    )
    selected = result.filter(pl.col("selected"))
    assert selected.height == top_n
    assert selected["selection_rank"].max() == top_n
    assert result.filter(~pl.col("selected")).height == 5


def test_top_n_selects_all_when_fewer_factors_than_limit() -> None:
    """When fewer factors exist than top_n, every factor is selected."""
    names = [f"factor_{index:02d}" for index in range(17)]
    ics = [1.0 - (index * 0.01) for index in range(17)]
    result = _build(
        SimpleFactorSelectionEngine(top_n=20),
        factor_validation=_factor_validation_frame(
            factor_names=names,
            information_coefficients=ics,
        ),
    )
    assert result.height == 17
    assert result.filter(pl.col("selected")).height == 17
    assert result["selection_reason"].to_list() == [_REASON_TOP_N] * 17


@pytest.mark.parametrize(
    "invalid_top_n",
    [0, -1, -10, 1.5, None, True, False],
)
def test_invalid_top_n_raises(invalid_top_n: object) -> None:
    """Non-positive or non-integer top_n values raise FSEL_TOP_N_INVALID."""
    with pytest.raises(FactorSelectionError) as exc_info:
        SimpleFactorSelectionEngine(top_n=invalid_top_n)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FSEL_TOP_N_INVALID"


def test_top_n_is_applied_independently_per_timeframe() -> None:
    """Each timeframe independently retains its own Top-N set."""
    names_1h = [f"h_{index:02d}" for index in range(15)]
    names_4h = [f"d_{index:02d}" for index in range(12)]
    factor_names = names_1h + names_4h
    timeframes = (["1h"] * 15) + (["4h"] * 12)
    ics = [1.0 - (index * 0.01) for index in range(15)] + [
        1.0 - (index * 0.01) for index in range(12)
    ]
    result = _build(
        SimpleFactorSelectionEngine(top_n=10),
        factor_validation=_factor_validation_frame(
            factor_names=factor_names,
            timeframes=timeframes,
            information_coefficients=ics,
        ),
    )
    selected_1h = result.filter((pl.col("timeframe") == "1h") & pl.col("selected"))
    selected_4h = result.filter((pl.col("timeframe") == "4h") & pl.col("selected"))
    assert selected_1h.height == 10
    assert selected_4h.height == 10
    assert result.filter(pl.col("timeframe") == "1h").height == 15
    assert result.filter(pl.col("timeframe") == "4h").height == 12


def test_all_factors_scored_including_fail_and_skipped() -> None:
    """FAIL and SKIPPED validation rows are scored and ranked with every other factor."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["pass_weak", "failed", "skipped", "pass_strong"],
            information_coefficients=[0.04, 0.99, 0.50, 0.20],
            statuses=["PASS", "FAIL", "SKIPPED", "PASS"],
        ),
    )
    by_name = {row["factor_name"]: row for row in result.to_dicts()}
    assert by_name["failed"]["selection_rank"] == 1
    assert by_name["skipped"]["selection_rank"] == 2
    assert by_name["pass_strong"]["selection_rank"] == 3
    assert by_name["pass_weak"]["selection_rank"] == 4
    assert by_name["failed"]["selected"] is True
    assert by_name["failed"]["selection_reason"] == _REASON_TOP_N
    assert all(row["selection_score"] is not None for row in result.to_dicts())


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_preserves_factor_identity_and_metadata() -> None:
    """factor_name, factor_version, factor_category, and timeframe are preserved."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["breakout"],
            factor_versions=["2.1.0"],
            factor_categories=["composite"],
            timeframes=["4h"],
        ),
    )
    assert result["factor_name"].to_list() == ["breakout"]
    assert result["factor_version"].to_list() == ["2.1.0"]
    assert result["factor_category"].to_list() == ["composite"]
    assert result["timeframe"].to_list() == ["4h"]


def test_selection_time_equals_validation_time() -> None:
    """selection_time copies validation_time from each validated factor."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=["momentum", "rsi"],
            validation_times=[1_700_000_000_000, 1_700_000_003_600],
        ),
    )
    assert result["selection_time"].to_list() == [1_700_000_000_000, 1_700_000_003_600]


def test_one_selection_row_per_validated_factor() -> None:
    """Engine emits exactly one selection row for each validated factor."""
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(factor_names=["momentum", "rsi"]),
    )
    assert result.height == 2
    assert result["factor_name"].to_list() == ["momentum", "rsi"]


# ---------------------------------------------------------------------------
# Output schema, invariants, and immutability
# ---------------------------------------------------------------------------


def test_output_canonical_ordering_and_dtype_schema() -> None:
    """Engine output starts with canonical columns and extends with eligibility metadata."""
    from cqros.factor_selection.schema import ELIGIBILITY_COLUMNS

    result = _build(SimpleFactorSelectionEngine())
    # Canonical columns come first in the correct order.
    assert tuple(result.columns[: len(CANONICAL_COLUMN_ORDER)]) == CANONICAL_COLUMN_ORDER
    # Canonical schema dtypes are preserved.
    for col, dtype in FACTOR_SELECTION_SCHEMA.items():
        assert result.schema[col] == dtype, f"dtype mismatch for {col}"
    # Eligibility columns are present after canonical columns.
    for col in ELIGIBILITY_COLUMNS:
        assert col in result.columns
    assert result.schema["selection_time"] == pl.Int64
    assert result.schema["selected"] == pl.Boolean
    assert result.schema["selection_score"] == pl.Float64
    assert result.schema["selection_rank"] == pl.Int32
    assert result.schema["status"] == pl.String


def test_schema_preserved_for_outside_top_n_rows() -> None:
    """Outside-top-N factors still emit the full canonical selection schema."""
    from cqros.factor_selection.schema import ELIGIBILITY_COLUMNS

    names = [f"factor_{index:02d}" for index in range(DEFAULT_TOP_N + 1)]
    ics = [1.0 - (index * 0.01) for index in range(DEFAULT_TOP_N + 1)]
    result = _build(
        SimpleFactorSelectionEngine(),
        factor_validation=_factor_validation_frame(
            factor_names=names,
            information_coefficients=ics,
        ),
    )
    outside = result.filter(pl.col("selection_rank") == DEFAULT_TOP_N + 1)
    assert outside.height == 1
    # Canonical columns come first in the correct order.
    assert tuple(result.columns[: len(CANONICAL_COLUMN_ORDER)]) == CANONICAL_COLUMN_ORDER
    # Eligibility columns follow.
    for col in ELIGIBILITY_COLUMNS:
        assert col in result.columns
    assert outside["selected"].to_list() == [False]
    assert outside["selection_reason"].to_list() == [_REASON_OUTSIDE_TOP_N]


def test_inputs_are_immutable() -> None:
    """build must not mutate the caller-supplied Factor Validation frame."""
    factor_validation = _factor_validation_frame()
    before = factor_validation.clone()
    SimpleFactorSelectionEngine().build(factor_validation)
    assert_frame_equal(factor_validation, before)


def test_output_is_deterministic() -> None:
    """Identical Factor Validation inputs produce identical selection outputs."""
    factor_validation = _factor_validation_frame(
        factor_names=["momentum", "rsi"],
        information_coefficients=[0.08, 0.15],
    )
    engine = SimpleFactorSelectionEngine()
    first = engine.build(factor_validation)
    second = engine.build(factor_validation)
    assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Eligibility policy injection — constructor and selection behaviour
# ---------------------------------------------------------------------------


def test_eligibility_policy_is_stored_on_engine() -> None:
    """Engine exposes the injected eligibility policy via property."""
    from cqros.factor_selection import FactorEligibilityPolicy

    policy = FactorEligibilityPolicy()
    engine = SimpleFactorSelectionEngine(eligibility_policy=policy)
    assert engine.eligibility_policy is policy


def test_no_eligibility_policy_returns_none() -> None:
    """Engine without an explicit policy returns None from the property."""
    engine = SimpleFactorSelectionEngine()
    assert engine.eligibility_policy is None


def test_zero_observation_factor_is_not_selected_when_policy_active() -> None:
    """A factor with zero observations must not be selected when the eligibility policy is set."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=FactorEligibilityPolicy())
    factor_validation = _factor_validation_frame(
        factor_names=["eligible_a", "eligible_b", "zero_obs_factor"],
        information_coefficients=[0.15, 0.12, 0.20],
        observations=[200, 180, 0],
    )
    result = engine.build(factor_validation)
    selected = result.filter(pl.col("selected"))
    assert "zero_obs_factor" not in selected["factor_name"].to_list()


def test_zero_observation_factor_does_not_consume_selection_slot() -> None:
    """Ineligible zero-observation factors must not fill top-N slots."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = SimpleFactorSelectionEngine(top_n=2, eligibility_policy=FactorEligibilityPolicy())
    factor_validation = _factor_validation_frame(
        factor_names=["eligible_a", "eligible_b", "zero_obs"],
        information_coefficients=[0.15, 0.12, 0.99],
        observations=[200, 180, 0],
    )
    result = engine.build(factor_validation)
    selected = result.filter(pl.col("selected"))
    assert selected.height == 2
    assert set(selected["factor_name"].to_list()) == {"eligible_a", "eligible_b"}


def test_fewer_than_target_selected_when_all_remaining_ineligible() -> None:
    """Selected count may be less than top_n when eligible candidates are exhausted."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=FactorEligibilityPolicy())
    factor_validation = _factor_validation_frame(
        factor_names=["good_a", "zero_b", "zero_c"],
        information_coefficients=[0.10, 0.30, 0.25],
        observations=[100, 0, 0],
    )
    result = engine.build(factor_validation)
    selected = result.filter(pl.col("selected"))
    assert selected.height == 1
    assert selected["factor_name"].to_list() == ["good_a"]


def test_ranking_of_eligible_factors_uses_abs_ic() -> None:
    """Eligible factors are ranked by abs(IC); eligibility policy must not alter this ordering."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = SimpleFactorSelectionEngine(top_n=3, eligibility_policy=FactorEligibilityPolicy())
    # Provide ICs where the sign varies so we confirm abs(IC) is used.
    factor_validation = _factor_validation_frame(
        factor_names=["neg_high", "pos_low", "pos_mid"],
        information_coefficients=[-0.20, 0.08, 0.15],
        observations=[200, 200, 200],
    )
    result = engine.build(factor_validation)
    selected = result.filter(pl.col("selected")).sort("selection_rank")
    assert selected["factor_name"].to_list()[0] == "neg_high"


def test_orientation_uses_signed_ic_v1_with_policy() -> None:
    """Adding an eligibility policy must not alter the signed-IC orientation of selected factors."""
    from cqros.factor_selection import FactorEligibilityPolicy

    engine = SimpleFactorSelectionEngine(top_n=2, eligibility_policy=FactorEligibilityPolicy())
    factor_validation = _factor_validation_frame(
        factor_names=["neg_ic_factor", "pos_ic_factor"],
        information_coefficients=[-0.18, 0.12],
        observations=[200, 200],
    )
    result = engine.build(factor_validation)
    row_neg = result.filter(pl.col("factor_name") == "neg_ic_factor")
    row_pos = result.filter(pl.col("factor_name") == "pos_ic_factor")
    assert row_neg["selected_direction"].to_list() == [-1]
    assert row_pos["selected_direction"].to_list() == [1]


def test_eligibility_policy_regression_all_eligible_identical_selection() -> None:
    """When all factors are eligible the policy must not change selection vs. no-policy mode."""
    from cqros.factor_selection import FactorEligibilityPolicy

    factor_validation = _factor_validation_frame(
        factor_names=["alpha", "beta"],
        information_coefficients=[0.10, 0.14],
        observations=[200, 200],
    )
    without_policy = SimpleFactorSelectionEngine(top_n=2)
    with_policy = SimpleFactorSelectionEngine(top_n=2, eligibility_policy=FactorEligibilityPolicy())
    no_pol = without_policy.build(factor_validation)
    with_pol = with_policy.build(factor_validation)
    # Same factors selected.
    assert set(no_pol.filter(pl.col("selected"))["factor_name"].to_list()) == set(
        with_pol.filter(pl.col("selected"))["factor_name"].to_list()
    )
    # Same orientations.
    assert (
        no_pol.sort("factor_name")["selected_direction"].to_list()
        == with_pol.sort("factor_name")["selected_direction"].to_list()
    )
