"""Unit tests for CQROS Factor Selection detailed CSV audit export."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_selection import (
    DETAILED_AUDIT_COLUMNS,
    NORMALIZATION_METHOD,
    SCORING_METHOD,
    WEIGHT_ABS_IC,
    WEIGHT_ABS_RANK_IC,
    WEIGHT_IC_DECAY,
    WEIGHT_ICIR,
    WEIGHT_INVERSE_TURNOVER,
    WEIGHT_MONOTONICITY,
    WEIGHT_QUANTILE_SPREAD,
    SimpleFactorSelectionEngine,
    build_detailed_audit_frame,
    combined_detailed_csv_path,
    detailed_csv_path,
    write_combined_detailed_csv,
    write_detailed_csv,
)
from cqros.factor_selection.detailed_export import contribution_sum_expression
from cqros.factor_validation.schema import FACTOR_VALIDATION_SCHEMA

_MANAGER = "default"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_TOP_N = 20
_YEAR = 2026
_FLOAT_TOLERANCE = 1e-12

_CONTRIBUTION_COLUMNS = (
    "information_coefficient_contribution",
    "rank_information_coefficient_contribution",
    "ic_information_ratio_contribution",
    "quantile_spread_contribution",
    "monotonicity_contribution",
    "ic_decay_contribution",
    "turnover_contribution",
)


def _validation_rows(
    *,
    factor_names: list[str],
    timeframes: list[str] | None = None,
    information_coefficients: list[float] | None = None,
    rank_information_coefficients: list[float] | None = None,
    ic_information_ratios: list[float] | None = None,
    quantile_spreads: list[float] | None = None,
    monotonicity_scores: list[float] | None = None,
    ic_decays: list[float] | None = None,
    turnovers: list[float] | None = None,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a full Factor Validation frame for detailed-export tests."""
    row_count = len(factor_names)
    timeframes = timeframes if timeframes is not None else ["1h"] * row_count
    information_coefficients = (
        information_coefficients
        if information_coefficients is not None
        else [0.04 + (0.01 * index) for index in range(row_count)]
    )
    rank_information_coefficients = (
        rank_information_coefficients
        if rank_information_coefficients is not None
        else [0.03 + (0.01 * index) for index in range(row_count)]
    )
    ic_information_ratios = (
        ic_information_ratios
        if ic_information_ratios is not None
        else [0.40 + (0.05 * index) for index in range(row_count)]
    )
    quantile_spreads = (
        quantile_spreads
        if quantile_spreads is not None
        else [0.02 + (0.01 * index) for index in range(row_count)]
    )
    monotonicity_scores = (
        monotonicity_scores
        if monotonicity_scores is not None
        else [0.50 + (0.05 * index) for index in range(row_count)]
    )
    ic_decays = (
        ic_decays
        if ic_decays is not None
        else [0.40 + (0.05 * index) for index in range(row_count)]
    )
    turnovers = (
        turnovers
        if turnovers is not None
        else [0.40 - (0.02 * index) for index in range(row_count)]
    )
    statuses = statuses if statuses is not None else ["PASS"] * row_count
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": ["1.0.0"] * row_count,
            "timeframe": timeframes,
            "validation_time": [1_700_000_000_000 + index for index in range(row_count)],
            "factor_category": ["price"] * row_count,
            "dataset_version": ["dataset-v1"] * row_count,
            "label_version": ["label-v1"] * row_count,
            "validation_start_time": [1_699_000_000_000] * row_count,
            "validation_end_time": [1_700_000_000_000] * row_count,
            "information_coefficient": information_coefficients,
            "rank_information_coefficient": rank_information_coefficients,
            "ic_information_ratio": ic_information_ratios,
            "ic_std": [0.15] * row_count,
            "ic_p_value": [0.01] * row_count,
            "ic_t_stat": [3.0] * row_count,
            "ic_decay": ic_decays,
            "turnover": turnovers,
            "monotonicity_score": monotonicity_scores,
            "quantile_spread": quantile_spreads,
            "observations": [200] * row_count,
            "ic_observations": [150] * row_count,
            "status": statuses,
        },
        schema=FACTOR_VALIDATION_SCHEMA,
    )


def _build_audit(
    validation: pl.DataFrame,
    *,
    top_n: int = _TOP_N,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (selection, detailed_audit) for ``validation``."""
    selection = SimpleFactorSelectionEngine(top_n=top_n).build(validation)
    audit = build_detailed_audit_frame(
        validation,
        selection,
        top_n=top_n,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return selection, audit


def test_detailed_audit_contains_every_factor_and_required_columns() -> None:
    """Audit frame retains every factor and every required audit column."""
    names = [f"factor_{index:02d}" for index in range(25)]
    validation = _validation_rows(factor_names=names)
    selection, audit = _build_audit(validation, top_n=20)

    assert audit.height == validation.height
    assert audit.height == selection.height
    assert list(audit.columns) == list(DETAILED_AUDIT_COLUMNS)
    assert set(audit["factor_name"].to_list()) == set(names)


def test_detailed_audit_contains_every_timeframe_isolated() -> None:
    """Each timeframe is ranked independently and retained in the audit CSV frame."""
    validation = _validation_rows(
        factor_names=["a", "b", "c", "d"],
        timeframes=["1h", "1h", "4h", "4h"],
        information_coefficients=[0.10, 0.20, 0.30, 0.05],
        rank_information_coefficients=[0.10, 0.20, 0.30, 0.05],
        ic_information_ratios=[0.40, 0.80, 0.90, 0.20],
        quantile_spreads=[0.02, 0.08, 0.09, 0.01],
        monotonicity_scores=[0.40, 0.80, 0.90, 0.20],
        ic_decays=[0.40, 0.80, 0.90, 0.20],
        turnovers=[0.40, 0.10, 0.10, 0.50],
    )
    _, audit = _build_audit(validation, top_n=1)

    assert set(audit["timeframe"].to_list()) == {"1h", "4h"}
    selected = audit.filter(pl.col("selected"))
    assert selected.height == 2
    assert set(selected["timeframe"].to_list()) == {"1h", "4h"}
    assert selected.filter(pl.col("timeframe") == "1h")["factor_name"].to_list() == ["b"]
    assert selected.filter(pl.col("timeframe") == "4h")["factor_name"].to_list() == ["c"]


def test_raw_metrics_match_factor_validation() -> None:
    """Raw metric columns exactly match the underlying Factor Validation values."""
    validation = _validation_rows(
        factor_names=["weak", "strong"],
        information_coefficients=[0.04, 0.20],
        rank_information_coefficients=[0.05, 0.15],
        ic_information_ratios=[0.40, 0.90],
        quantile_spreads=[0.02, 0.08],
        monotonicity_scores=[0.50, 0.90],
        ic_decays=[0.40, 0.80],
        turnovers=[0.40, 0.10],
        statuses=["FAIL", "PASS"],
    )
    _, audit = _build_audit(validation)

    joined = validation.join(
        audit.select(
            "factor_name",
            "factor_version",
            "timeframe",
            "information_coefficient",
            "rank_information_coefficient",
            "ic_information_ratio",
            "ic_p_value",
            "ic_decay",
            "quantile_spread",
            "monotonicity_score",
            "turnover",
            "observations",
            "validation_time",
            "validation_status",
            "dataset_version",
            "label_version",
            "validation_start_time",
            "validation_end_time",
        ),
        on=["factor_name", "factor_version", "timeframe"],
        suffix="_audit",
    )
    for column in (
        "information_coefficient",
        "rank_information_coefficient",
        "ic_information_ratio",
        "ic_p_value",
        "ic_decay",
        "quantile_spread",
        "monotonicity_score",
        "turnover",
        "observations",
        "validation_time",
        "dataset_version",
        "label_version",
        "validation_start_time",
        "validation_end_time",
    ):
        assert joined[column].to_list() == joined[f"{column}_audit"].to_list()
    assert joined["status"].to_list() == joined["validation_status"].to_list()


def test_normalized_weights_contributions_and_score_reconstruction() -> None:
    """Normalized values, weights, contributions, and score reconstruction match."""
    validation = _validation_rows(
        factor_names=["weak", "strong"],
        information_coefficients=[0.04, 0.20],
        rank_information_coefficients=[0.05, 0.15],
        ic_information_ratios=[0.40, 0.90],
        quantile_spreads=[0.02, 0.08],
        monotonicity_scores=[0.50, 0.90],
        ic_decays=[0.40, 0.80],
        turnovers=[0.40, 0.10],
    )
    selection, audit = _build_audit(validation)
    strong = audit.filter(pl.col("factor_name") == "strong").to_dicts()[0]

    assert strong["abs_information_coefficient"] == pytest.approx(abs(0.20))
    assert strong["abs_rank_information_coefficient"] == pytest.approx(abs(0.15))
    assert strong["inverse_turnover"] == pytest.approx(-0.10)

    assert strong["information_coefficient_normalized"] == pytest.approx(1.0)
    assert strong["information_coefficient_weight"] == WEIGHT_ABS_IC
    assert strong["rank_information_coefficient_weight"] == WEIGHT_ABS_RANK_IC
    assert strong["ic_information_ratio_weight"] == WEIGHT_ICIR
    assert strong["quantile_spread_weight"] == WEIGHT_QUANTILE_SPREAD
    assert strong["monotonicity_weight"] == WEIGHT_MONOTONICITY
    assert strong["ic_decay_weight"] == WEIGHT_IC_DECAY
    assert strong["turnover_weight"] == WEIGHT_INVERSE_TURNOVER

    assert strong["information_coefficient_contribution"] == pytest.approx(
        strong["information_coefficient_normalized"] * WEIGHT_ABS_IC
    )
    assert strong["rank_information_coefficient_contribution"] == pytest.approx(
        strong["rank_information_coefficient_normalized"] * WEIGHT_ABS_RANK_IC
    )
    assert strong["ic_information_ratio_contribution"] == pytest.approx(
        strong["ic_information_ratio_normalized"] * WEIGHT_ICIR
    )
    assert strong["quantile_spread_contribution"] == pytest.approx(
        strong["quantile_spread_normalized"] * WEIGHT_QUANTILE_SPREAD
    )
    assert strong["monotonicity_contribution"] == pytest.approx(
        strong["monotonicity_normalized"] * WEIGHT_MONOTONICITY
    )
    assert strong["ic_decay_contribution"] == pytest.approx(
        strong["ic_decay_normalized"] * WEIGHT_IC_DECAY
    )
    assert strong["turnover_contribution"] == pytest.approx(
        strong["turnover_normalized"] * WEIGHT_INVERSE_TURNOVER
    )

    contribution_sum = sum(strong[column] for column in _CONTRIBUTION_COLUMNS)
    assert contribution_sum == pytest.approx(strong["selection_score"], abs=_FLOAT_TOLERANCE)

    selection_strong = selection.filter(pl.col("factor_name") == "strong").to_dicts()[0]
    assert strong["selection_score"] == pytest.approx(selection_strong["selection_score"])
    assert strong["selection_rank"] == selection_strong["selection_rank"]


def test_selection_rank_selected_status_reason_match_engine() -> None:
    """Decision columns come from the canonical engine output."""
    names = [f"factor_{index:02d}" for index in range(25)]
    validation = _validation_rows(factor_names=names)
    selection, audit = _build_audit(validation, top_n=20)

    compared = selection.join(
        audit.select(
            "factor_name",
            "factor_version",
            "timeframe",
            "selection_score",
            "selection_rank",
            "selected",
            "status",
            "selection_reason",
        ),
        on=["factor_name", "factor_version", "timeframe"],
        suffix="_audit",
    )
    assert compared["selection_rank"].to_list() == compared["selection_rank_audit"].to_list()
    assert compared["selected"].to_list() == compared["selected_audit"].to_list()
    assert compared["status"].to_list() == compared["status_audit"].to_list()
    assert compared["selection_reason"].to_list() == compared["selection_reason_audit"].to_list()
    for left, right in zip(
        compared["selection_score"].to_list(),
        compared["selection_score_audit"].to_list(),
        strict=True,
    ):
        assert left == pytest.approx(right, abs=_FLOAT_TOLERANCE)

    assert audit.filter(pl.col("selection_rank") <= 20)["selected"].to_list() == [True] * 20
    assert audit.filter(pl.col("selection_rank") > 20)["selected"].to_list() == [False] * 5
    assert set(audit.filter(pl.col("selected"))["status"].to_list()) == {"SELECTED"}
    assert set(audit.filter(~pl.col("selected"))["status"].to_list()) == {"REJECTED"}
    assert set(audit.filter(pl.col("selected"))["selection_reason"].to_list()) == {"top_n"}
    assert set(audit.filter(~pl.col("selected"))["selection_reason"].to_list()) == {"outside_top_n"}


def test_configuration_audit_fields_recorded() -> None:
    """top_n, scoring_method, and normalization_method are recorded on every row."""
    validation = _validation_rows(factor_names=["alpha", "beta"])
    _, audit = _build_audit(validation, top_n=7)
    assert audit["top_n"].to_list() == [7, 7]
    assert audit["scoring_method"].to_list() == [SCORING_METHOD, SCORING_METHOD]
    assert audit["normalization_method"].to_list() == [
        NORMALIZATION_METHOD,
        NORMALIZATION_METHOD,
    ]
    assert SCORING_METHOD == "fixed_weighted_minmax"
    assert NORMALIZATION_METHOD == "timeframe_minmax"
    assert audit["manager"].to_list() == [_MANAGER, _MANAGER]
    assert audit["exchange"].to_list() == [_EXCHANGE, _EXCHANGE]
    assert audit["market"].to_list() == [_MARKET, _MARKET]


def test_null_metrics_preserve_raw_nulls_and_engine_normalization() -> None:
    """Null raw metrics remain null while normalized values follow engine semantics."""
    validation = _validation_rows(
        factor_names=["null_ic", "present"],
        information_coefficients=[None, 0.20],  # type: ignore[list-item]
        rank_information_coefficients=[0.05, 0.15],
    )
    _, audit = _build_audit(validation)
    null_row = audit.filter(pl.col("factor_name") == "null_ic").to_dicts()[0]
    present_row = audit.filter(pl.col("factor_name") == "present").to_dicts()[0]

    assert null_row["information_coefficient"] is None
    assert null_row["abs_information_coefficient"] is None
    assert null_row["information_coefficient_normalized"] == pytest.approx(0.0)
    assert present_row["information_coefficient_normalized"] == pytest.approx(1.0)


def test_csv_generation_is_deterministic(tmp_path: Path) -> None:
    """Writing the detailed CSV twice produces equivalent content."""
    validation = _validation_rows(
        factor_names=["a", "b", "c"],
        timeframes=["1h", "1h", "4h"],
    )
    _, audit = _build_audit(validation)
    path_one = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe="1h",
        year=_YEAR,
    )
    path_two = tmp_path / "second_detailed.csv"
    write_detailed_csv(audit, path_one)
    write_detailed_csv(audit, path_two)
    assert path_one.read_text(encoding="utf-8") == path_two.read_text(encoding="utf-8")

    reloaded_one = pl.read_csv(path_one)
    reloaded_two = pl.read_csv(path_two)
    assert_frame_equal(reloaded_one, reloaded_two)


def test_csv_preserves_names_versions_and_ordering(tmp_path: Path) -> None:
    """Factor names/versions are preserved and CSV ordering is deterministic."""
    validation = _validation_rows(
        factor_names=["zeta", "alpha", "beta"],
        timeframes=["4h", "1h", "1h"],
        information_coefficients=[0.01, 0.30, 0.20],
    )
    _, audit = _build_audit(validation, top_n=2)
    path = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe="mixed",
        year=_YEAR,
    )
    write_detailed_csv(audit, path)
    loaded = pl.read_csv(path)
    assert loaded["factor_name"].to_list() == audit["factor_name"].to_list()
    assert loaded["factor_version"].to_list() == audit["factor_version"].to_list()
    assert loaded["timeframe"].to_list() == sorted(loaded["timeframe"].to_list())
    # Within timeframe, ranks ascend.
    for timeframe in loaded["timeframe"].unique().to_list():
        ranks = loaded.filter(pl.col("timeframe") == timeframe)["selection_rank"].to_list()
        assert ranks == sorted(ranks)


def test_combined_detailed_csv_retains_timeframe(tmp_path: Path) -> None:
    """Combined detailed CSV concatenates partitions and retains timeframe."""
    validation_1h = _validation_rows(factor_names=["a", "b"], timeframes=["1h", "1h"])
    validation_4h = _validation_rows(factor_names=["c", "d"], timeframes=["4h", "4h"])
    _, audit_1h = _build_audit(validation_1h)
    _, audit_4h = _build_audit(validation_4h)

    combined_path = combined_detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    write_combined_detailed_csv([audit_1h, audit_4h], combined_path)
    loaded = pl.read_csv(combined_path)
    assert loaded.height == 4
    assert set(loaded["timeframe"].to_list()) == {"1h", "4h"}
    assert list(loaded.columns) == list(DETAILED_AUDIT_COLUMNS)


def test_contribution_sum_expression_matches_selection_score() -> None:
    """Contribution sum expression equals selection_score within float tolerance."""
    validation = _validation_rows(factor_names=["a", "b", "c"])
    _, audit = _build_audit(validation)
    checked = audit.with_columns(contribution_sum_expression())
    for row in checked.to_dicts():
        assert row["contribution_sum"] == pytest.approx(
            row["selection_score"],
            abs=_FLOAT_TOLERANCE,
        )
