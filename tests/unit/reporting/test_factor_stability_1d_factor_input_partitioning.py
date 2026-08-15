"""Unit tests for the 1d factor-input partitioning architecture investigation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cqros.reporting.factor_stability_1d_factor_input_partitioning import (
    ARCHITECTURE_TRACE_CSV_NAME,
    CLASS_FUNDING_DEPENDENT,
    CLASS_OHLCV_ONLY,
    CLASS_OHLCV_PLUS_VOLUME,
    CLASS_OI_DEPENDENT,
    CLASS_TAKER_DEPENDENT,
    CLASS_UNKNOWN,
    COMPANION_TRUNCATION_CSV_NAME,
    COVERAGE_CSV_NAME,
    DEPENDENCY_CSV_NAME,
    HASHES_AFTER_NAME,
    HASHES_BEFORE_NAME,
    INPUT_BOUNDARY_CSV_NAME,
    SUMMARY_TXT_NAME,
    VERDICT_PARTITIONING_NOT_SAFE,
    VERDICT_PARTITIONING_SAFE_AND_BENEFICIAL,
    VERDICT_PARTITIONING_SAFE_BUT_LOW_VALUE,
    WARMUP_CSV_NAME,
    classify_architecture_verdict,
    classify_dependency_class,
    forbidden_import_violations,
    hash_watched_production_artifacts,
    simulate_factor_specific_start_index,
)


def test_classify_dependency_class_from_required_features() -> None:
    assert classify_dependency_class(("close",)) == CLASS_OHLCV_ONLY
    assert classify_dependency_class(("close", "volume")) == CLASS_OHLCV_PLUS_VOLUME
    assert classify_dependency_class(("open_interest",)) == CLASS_OI_DEPENDENT
    assert (
        classify_dependency_class(("taker_buy_volume", "taker_sell_volume"))
        == CLASS_TAKER_DEPENDENT
    )
    assert classify_dependency_class(("funding_rate",)) == CLASS_FUNDING_DEPENDENT
    assert classify_dependency_class(("asset_return", "btc_return")) == CLASS_UNKNOWN


def test_classify_architecture_verdict_beneficial() -> None:
    assert (
        classify_architecture_verdict(
            ohlcv_only_truncated=True,
            max_bars_recovered=178,
            recovered_factor_count=50,
            leakage_safe=True,
            oi_boundary_preserved=True,
        )
        == VERDICT_PARTITIONING_SAFE_AND_BENEFICIAL
    )


def test_classify_architecture_verdict_low_value_and_unsafe() -> None:
    assert (
        classify_architecture_verdict(
            ohlcv_only_truncated=False,
            max_bars_recovered=0,
            recovered_factor_count=0,
            leakage_safe=True,
            oi_boundary_preserved=True,
        )
        == VERDICT_PARTITIONING_SAFE_BUT_LOW_VALUE
    )
    assert (
        classify_architecture_verdict(
            ohlcv_only_truncated=True,
            max_bars_recovered=10,
            recovered_factor_count=1,
            leakage_safe=True,
            oi_boundary_preserved=False,
        )
        == VERDICT_PARTITIONING_NOT_SAFE
    )


def test_simulate_ohlcv_only_uses_full_timeline() -> None:
    ohlcv = pl.DataFrame(
        {
            "open_time": [1_000, 2_000, 3_000, 4_000],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    companions = {
        "open_interest": pl.DataFrame({"timestamp": [3_000, 4_000], "open_interest": [1.0, 2.0]}),
        "taker_volume": pl.DataFrame(
            {
                "timestamp": [3_000, 4_000],
                "taker_buy_volume": [1.0, 1.0],
                "taker_sell_volume": [1.0, 1.0],
            }
        ),
        "global_long_short_account_ratio": pl.DataFrame(
            {"timestamp": [3_000, 4_000], "long_short_ratio": [1.0, 1.1]}
        ),
        "funding": pl.DataFrame(
            {
                "funding_time": [1_000, 2_000, 3_000, 4_000],
                "funding_rate": [0.0, 0.0, 0.0, 0.0],
                "mark_price": [1.0, 1.0, 1.0, 1.0],
            }
        ),
    }
    start, first_ts, bars = simulate_factor_specific_start_index(
        ohlcv=ohlcv,
        companions=companions,
        required_companion_columns=(),
    )
    assert start == 0
    assert first_ts == 1_000
    assert bars == 4


def test_simulate_oi_factor_keeps_oi_boundary() -> None:
    ohlcv = pl.DataFrame(
        {
            "open_time": [1_000, 2_000, 3_000, 4_000],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    companions = {
        "open_interest": pl.DataFrame({"timestamp": [3_000, 4_000], "open_interest": [9.0, 10.0]}),
        "taker_volume": pl.DataFrame(
            {
                "timestamp": [2_000, 3_000, 4_000],
                "taker_buy_volume": [1.0, 1.0, 1.0],
                "taker_sell_volume": [1.0, 1.0, 1.0],
            }
        ),
        "global_long_short_account_ratio": pl.DataFrame(
            {"timestamp": [2_000, 3_000, 4_000], "long_short_ratio": [1.0, 1.0, 1.0]}
        ),
        "funding": pl.DataFrame(
            {
                "funding_time": [1_000, 2_000, 3_000, 4_000],
                "funding_rate": [0.0, 0.0, 0.0, 0.0],
                "mark_price": [1.0, 1.0, 1.0, 1.0],
            }
        ),
    }
    start, first_ts, bars = simulate_factor_specific_start_index(
        ohlcv=ohlcv,
        companions=companions,
        required_companion_columns=("open_interest",),
    )
    assert start == 2
    assert first_ts == 3_000
    assert bars == 2


def test_simulate_taker_factor_ignores_later_oi() -> None:
    ohlcv = pl.DataFrame(
        {
            "open_time": [1_000, 2_000, 3_000, 4_000],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    companions = {
        "open_interest": pl.DataFrame({"timestamp": [4_000], "open_interest": [1.0]}),
        "taker_volume": pl.DataFrame(
            {
                "timestamp": [2_000, 3_000, 4_000],
                "taker_buy_volume": [1.0, 1.0, 1.0],
                "taker_sell_volume": [1.0, 1.0, 1.0],
            }
        ),
        "global_long_short_account_ratio": pl.DataFrame(
            {"timestamp": [4_000], "long_short_ratio": [1.0]}
        ),
        "funding": pl.DataFrame(
            {
                "funding_time": [1_000, 2_000, 3_000, 4_000],
                "funding_rate": [0.0, 0.0, 0.0, 0.0],
                "mark_price": [1.0, 1.0, 1.0, 1.0],
            }
        ),
    }
    start, first_ts, bars = simulate_factor_specific_start_index(
        ohlcv=ohlcv,
        companions=companions,
        required_companion_columns=("taker_buy_volume", "taker_sell_volume"),
    )
    assert start == 1
    assert first_ts == 2_000
    assert bars == 3


def test_forbidden_import_violations_detects_banned_modules() -> None:
    source = "import cqros.alpha\nfrom cqros.ml import x\n"
    assert forbidden_import_violations(source) == ("cqros.alpha", "cqros.ml")


def test_hash_watched_production_artifacts_empty_root(tmp_path: Path) -> None:
    assert hash_watched_production_artifacts(tmp_path) == {}


def test_module_source_has_no_forbidden_imports() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "cqros"
        / "reporting"
        / "factor_stability_1d_factor_input_partitioning.py"
    )
    assert forbidden_import_violations(path.read_text(encoding="utf-8")) == ()


def test_report_artifact_names_are_stable() -> None:
    assert SUMMARY_TXT_NAME == "summary.txt"
    assert DEPENDENCY_CSV_NAME == "factor_dependencies.csv"
    assert INPUT_BOUNDARY_CSV_NAME == "input_boundaries.csv"
    assert COVERAGE_CSV_NAME == "current_vs_partitioned_coverage.csv"
    assert WARMUP_CSV_NAME == "warmup_analysis.csv"
    assert COMPANION_TRUNCATION_CSV_NAME == "companion_truncation.csv"
    assert ARCHITECTURE_TRACE_CSV_NAME == "architecture_trace.csv"
    assert HASHES_BEFORE_NAME == "hashes_before.txt"
    assert HASHES_AFTER_NAME == "hashes_after.txt"
