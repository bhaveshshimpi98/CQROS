"""Unit tests for the 1d factor-replacement investigation reporter."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_1d_factor_replacement import (
    CANDIDATE_ELIGIBILITY_CSV_NAME,
    CANDIDATE_FACTORS_CSV_NAME,
    CANDIDATE_FOLDS_CSV_NAME,
    CANDIDATE_INVENTORY_CSV_NAME,
    CANDIDATE_SELECTION_CSV_NAME,
    CANDIDATE_SET_VERSION,
    CROSS_TIMEFRAME_CSV_NAME,
    DECISION_ELIGIBLE_AND_SELECTED,
    DECISION_INELIGIBLE_ZERO_OBSERVATIONS,
    DECISION_RETIRED_EXISTING_FACTOR,
    FLAG_1D_STATISTICAL_POWER_LIMITATION,
    HASHES_AFTER_NAME,
    HASHES_BEFORE_NAME,
    REPLACEMENT_CANDIDATES_V1,
    RETIRED_1D_FACTORS,
    SUMMARY_TXT_NAME,
    TARGET_TIMEFRAME,
    VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES,
    VERDICT_REPLACEMENT_INCONCLUSIVE,
    FactorStability1dFactorReplacementReporter,
    classify_factor_family,
    classify_replacement_decision,
    classify_verdict,
    forbidden_import_violations,
    hash_watched_production_artifacts,
    is_cumulative_level_factor,
)

_MANAGER = "default"
_YEAR = 2026
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_BASE_TS = 1_751_155_200_000


def test_classify_factor_family_taxonomy() -> None:
    assert classify_factor_family("momentum", "price") == "price momentum"
    assert classify_factor_family("trend_slope", "price") == "trend"
    assert classify_factor_family("historical_volatility", "price") == "volatility"
    assert classify_factor_family("rsi", "price") == "candle/return structure"
    assert classify_factor_family("volume_zscore", "volume") == "volume"
    assert classify_factor_family("buy_sell_imbalance", "microstructure") == "order-flow"
    assert classify_factor_family("funding_rate_momentum", "funding") == "funding"
    assert classify_factor_family("open_interest_zscore", "open_interest") == "open-interest"


def test_is_cumulative_level_factor() -> None:
    assert is_cumulative_level_factor("price_volume_trend")
    assert is_cumulative_level_factor("on_balance_volume")
    assert is_cumulative_level_factor("open_interest_level")
    assert not is_cumulative_level_factor("volume_zscore")
    assert not is_cumulative_level_factor("open_interest_momentum")


def test_classify_replacement_decision() -> None:
    assert (
        classify_replacement_decision(
            factor_name="price_volume_trend",
            eligibility_status="ELIGIBLE",
            selected=True,
        )
        == DECISION_RETIRED_EXISTING_FACTOR
    )
    assert (
        classify_replacement_decision(
            factor_name="volume_zscore",
            eligibility_status="INELIGIBLE_ZERO_OBSERVATIONS",
            selected=False,
        )
        == DECISION_INELIGIBLE_ZERO_OBSERVATIONS
    )
    assert (
        classify_replacement_decision(
            factor_name="rsi",
            eligibility_status="ELIGIBLE",
            selected=True,
        )
        == DECISION_ELIGIBLE_AND_SELECTED
    )


def test_classify_verdict_underpowered_no_entries() -> None:
    verdict = classify_verdict(
        entered_new_factors=(),
        removed_factors=list(RETIRED_1D_FACTORS),
        power_limitation=True,
        candidate_eligible_selected=0,
    )
    assert verdict == VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES


def test_classify_verdict_underpowered_inconclusive() -> None:
    verdict = classify_verdict(
        entered_new_factors=("williams_r",),
        removed_factors=list(RETIRED_1D_FACTORS),
        power_limitation=True,
        candidate_eligible_selected=1,
    )
    assert verdict == VERDICT_REPLACEMENT_INCONCLUSIVE


def test_classify_verdict_no_entries_even_with_retained_candidates() -> None:
    verdict = classify_verdict(
        entered_new_factors=(),
        removed_factors=list(RETIRED_1D_FACTORS),
        power_limitation=True,
        candidate_eligible_selected=6,
    )
    assert verdict == VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES


def test_candidate_set_excludes_retired_and_is_versioned() -> None:
    assert CANDIDATE_SET_VERSION == "1d_replacement_candidates_v1"
    for name in RETIRED_1D_FACTORS:
        assert name not in REPLACEMENT_CANDIDATES_V1
    assert "volume_zscore" in REPLACEMENT_CANDIDATES_V1
    assert "open_interest_momentum" in REPLACEMENT_CANDIDATES_V1


def test_forbidden_imports_clean_for_module() -> None:
    source = Path("src/cqros/reporting/factor_stability_1d_factor_replacement.py").read_text(
        encoding="utf-8"
    )
    assert forbidden_import_violations(source) == ()
    cli = Path("src/cqros/cli/report_factor_stability_1d_factor_replacement.py").read_text(
        encoding="utf-8"
    )
    assert forbidden_import_violations(cli) == ()


def _validation_rows() -> list[dict[str, object]]:
    specs = [
        ("stochastic_k", "price", -0.14, 492, "PASS"),
        ("williams_r", "price", -0.14, 492, "PASS"),
        ("money_flow_index", "volume", -0.11, 369, "PASS"),
        ("rsi", "price", -0.02, 369, "PASS"),
        ("price_volume_trend", "volume", 0.02, 3368, "PASS"),
        ("on_balance_volume", "volume", 0.01, 3368, "PASS"),
        ("open_interest_level", "open_interest", 0.04, 3691, "PASS"),
        ("volume_zscore", "volume", None, 0, "FAIL"),
        ("open_interest_momentum", "open_interest", None, 0, "FAIL"),
    ]
    rows: list[dict[str, object]] = []
    for name, category, ic, obs, status in specs:
        rows.append(
            {
                "factor_name": name,
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "validation_time": 1_784_073_600_000,
                "factor_category": category,
                "dataset_version": "v1",
                "label_version": "v1",
                "validation_start_time": _BASE_TS,
                "validation_end_time": _BASE_TS + 16 * 86_400_000,
                "information_coefficient": ic,
                "rank_information_coefficient": ic,
                "ic_information_ratio": 0.5 if ic is not None else None,
                "ic_std": 0.1 if ic is not None else None,
                "ic_p_value": 0.1 if ic is not None else None,
                "ic_t_stat": 1.0 if ic is not None else None,
                "ic_decay": 0.1 if ic is not None else None,
                "turnover": 0.2 if ic is not None else None,
                "monotonicity_score": 0.1 if ic is not None else None,
                "quantile_spread": 0.01 if ic is not None else None,
                "observations": obs,
                "ic_observations": max(obs // 10, 0),
                "status": status,
            }
        )
    return rows


def _selection_from_validation(validation: pl.DataFrame) -> pl.DataFrame:
    """Build a production-like selection frame for OLD baseline comparisons."""
    rows: list[dict[str, object]] = []
    rank = 0
    for row in validation.sort("factor_name").to_dicts():
        name = str(row["factor_name"])
        eligible = int(row["observations"] or 0) > 0
        selected = eligible and name != "williams_r"
        if selected:
            rank += 1
        ic = row["information_coefficient"]
        direction = 1 if ic is None or float(ic) >= 0 else -1
        rows.append(
            {
                "factor_name": name,
                "factor_version": "1.0.0",
                "timeframe": TARGET_TIMEFRAME,
                "selection_time": 1_784_073_600_000,
                "factor_category": row["factor_category"],
                "selected": selected,
                "selection_score": abs(float(ic)) if ic is not None else 0.0,
                "selection_rank": rank if selected else 100,
                "selection_reason": "top_n" if selected else "hard_ineligible",
                "selection_ic": ic,
                "selected_direction": direction,
                "orientation_policy": "signed_ic_v1",
                "status": "SELECTED" if selected else "REJECTED",
                "eligibility_status": ("ELIGIBLE" if eligible else "INELIGIBLE_ZERO_OBSERVATIONS"),
                "eligibility_reason": "test",
                "eligibility_policy": "coverage_v1",
                "usable_observations": int(row["observations"] or 0),
                "total_observations": None,
                "coverage_ratio": None,
                "null_rate": None,
                "required_lookback": 0,
                "available_history": 17,
                "warmup_sufficient": True,
                "companion_dependencies": "",
                "companion_coverage_status": None,
            }
        )
    return pl.DataFrame(rows)


def _oos_frame(selected_names: list[str]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
    for day in range(17):
        ts = _BASE_TS + day * 86_400_000
        fold_id = (day % 5) + 1
        for symbol_index, symbol in enumerate(symbols):
            factor_level = float(symbol_index + 1 + day)
            target = -0.01 * float(symbol_index) + 0.001 * day
            for name in selected_names:
                rows.append(
                    {
                        "manager": _MANAGER,
                        "engine": "simple",
                        "symbol": symbol,
                        "timeframe": TARGET_TIMEFRAME,
                        "year": _YEAR,
                        "fold_id": fold_id,
                        "observation_time": ts,
                        "factor_name": name,
                        "factor_version": "1.0.0",
                        "selected": True,
                        "partition": "OOS",
                        "future_return_1": target,
                        "factor_value": factor_level,
                        "selection_ic": 0.02,
                        "selected_direction": 1,
                        "orientation_policy": "signed_ic_v1",
                        "status": "PASS",
                    }
                )
    return pl.DataFrame(rows)


def _write_factor_partitions(root: Path, names: list[str]) -> None:
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    for symbol in symbols:
        rows: list[dict[str, object]] = []
        for day in range(20):
            ts = _BASE_TS + day * 86_400_000
            for name in names:
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": TARGET_TIMEFRAME,
                        "open_time": ts,
                        "factor_name": name,
                        "factor_version": "1.0.0",
                        "factor_value": float(day + 1),
                        "factor_category": "price",
                        "lookback": 14,
                        "status": "ACTIVE",
                    }
                )
        path = root / "factors" / _MANAGER / _EXCHANGE / _MARKET / symbol / TARGET_TIMEFRAME
        path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(path / f"{_YEAR}.parquet")


def _build_lake(root: Path) -> None:
    validation = pl.DataFrame(_validation_rows())
    selection = _selection_from_validation(validation)
    selected_names = selection.filter(pl.col("selected") == True)[  # noqa: E712
        "factor_name"
    ].to_list()
    oos = _oos_frame(selected_names)

    path = root / "factor_validation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    validation.write_parquet(path / f"{_YEAR}.parquet")

    path = root / "factor_selection" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    selection.write_parquet(path / f"{_YEAR}.parquet")

    path = root / "purged_cv_evaluation" / _MANAGER / _EXCHANGE / _MARKET / TARGET_TIMEFRAME
    path.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path / f"{_YEAR}.parquet")

    # Watched empty tiers so hash manifests are comparable.
    for tier in ("walk_forward", "purged_cv", "walk_forward_evaluation"):
        (root / tier).mkdir(parents=True, exist_ok=True)

    _write_factor_partitions(root, selected_names + ["williams_r", "volume_zscore"])


def test_reporter_writes_deliverables_and_preserves_hashes(tmp_path: Path) -> None:
    lake = tmp_path / "data"
    output = tmp_path / "reports" / "factor_stability" / "1d_factor_replacement"
    _build_lake(lake)

    reporter = FactorStability1dFactorReplacementReporter(
        storage_root=lake,
        output_root=output,
        manager=_MANAGER,
    )
    result = reporter.run(year=_YEAR)

    assert result.production_artifacts_unchanged is True
    assert result.deterministic is True
    assert result.candidate_set_version == CANDIDATE_SET_VERSION
    assert FLAG_1D_STATISTICAL_POWER_LIMITATION in result.summary_text
    assert "FACTOR_REPLACEMENT_SUCCESS=False" in result.summary_text
    assert result.verdict in {
        VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES,
        VERDICT_REPLACEMENT_INCONCLUSIVE,
    }
    for name in RETIRED_1D_FACTORS:
        assert name not in result.new_selected

    expected = [
        CANDIDATE_INVENTORY_CSV_NAME,
        CANDIDATE_ELIGIBILITY_CSV_NAME,
        CANDIDATE_SELECTION_CSV_NAME,
        CANDIDATE_FACTORS_CSV_NAME,
        CANDIDATE_FOLDS_CSV_NAME,
        CROSS_TIMEFRAME_CSV_NAME,
        SUMMARY_TXT_NAME,
        HASHES_BEFORE_NAME,
        HASHES_AFTER_NAME,
    ]
    for filename in expected:
        assert (output / filename).exists()

    eligibility = pl.read_csv(output / CANDIDATE_ELIGIBILITY_CSV_NAME)
    retired = eligibility.filter(pl.col("factor_name").is_in(list(RETIRED_1D_FACTORS)))
    assert retired.height == 3
    assert set(retired["decision"].to_list()) == {DECISION_RETIRED_EXISTING_FACTOR}


def test_reporter_is_deterministic(tmp_path: Path) -> None:
    lake = tmp_path / "data"
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    _build_lake(lake)
    first = FactorStability1dFactorReplacementReporter(
        storage_root=lake,
        output_root=output_a,
        manager=_MANAGER,
    ).run(year=_YEAR)
    second = FactorStability1dFactorReplacementReporter(
        storage_root=lake,
        output_root=output_b,
        manager=_MANAGER,
    ).run(year=_YEAR)
    assert first.verdict == second.verdict
    assert first.old_selected == second.old_selected
    assert first.new_selected == second.new_selected
    assert (
        first.summary_text.split("generated_at_utc=")[0]
        == second.summary_text.split("generated_at_utc=")[0]
    )


def test_missing_1d_partition_fails(tmp_path: Path) -> None:
    with pytest.raises(ReportingValidationError, match="1d factor selection"):
        FactorStability1dFactorReplacementReporter(
            storage_root=tmp_path,
            output_root=tmp_path / "out",
            manager=_MANAGER,
        ).run(year=_YEAR)


def test_hash_helper_reads_watched_tiers(tmp_path: Path) -> None:
    path = tmp_path / "factor_selection" / "x.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"abc")
    hashes = hash_watched_production_artifacts(tmp_path)
    assert "factor_selection/x.parquet" in hashes
