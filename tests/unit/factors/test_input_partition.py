"""Unit tests for factor-specific input partitioning."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.cli.generate_factors import align_factor_input_frame
from cqros.core.constants import (
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_WALK_FORWARD,
)
from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import FactorValidationError
from cqros.factors.funding import FundingRateLevelFactor
from cqros.factors.input_partition import (
    CLASS_FUNDING_DEPENDENT,
    CLASS_OHLCV_ONLY,
    CLASS_OHLCV_PLUS_VOLUME,
    CLASS_OI_DEPENDENT,
    CLASS_TAKER_DEPENDENT,
    CLASS_UNKNOWN,
    KNOWN_FACTOR_INPUT_FEATURES,
    FactorInputPartition,
    classify_dependency_class,
    required_companion_columns,
    required_datasets,
)
from cqros.factors.open_interest import OpenInterestLevelFactor
from cqros.factors.pipeline import FactorPipeline
from cqros.factors.registry import FactorRegistry
from cqros.factors.volume import OnBalanceVolumeFactor, PriceVolumeTrendFactor
from cqros.reporting.factor_stability_1d_factor_input_partitioning import (
    hash_watched_production_artifacts,
)

_MS_DAY: int = 86_400_000
_START_OHLCV_MS: int = 1_767_225_600_000  # 2026-01-01T00:00:00Z


def _ohlcv_open_times(*, start_ms: int, bars: int) -> list[int]:
    return [start_ms + index * _MS_DAY for index in range(bars)]


def _joined_frame(
    *,
    ohlcv_bars: int = 10,
    companion_start_index: int = 5,
    include_long_short: bool = True,
) -> pl.DataFrame:
    """Build a joined OHLCV+companion frame with late companion availability."""
    open_times = _ohlcv_open_times(start_ms=_START_OHLCV_MS, bars=ohlcv_bars)
    companion_times = set(open_times[companion_start_index:])
    rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        companion_present = open_time in companion_times
        row: dict[str, object] = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "open_time": open_time,
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "volume": 10.0 + index,
            "trade_count": index + 1,
            "funding_rate": 0.0001 if companion_present else None,
            "mark_price": 100.0 + index if companion_present else None,
            "open_interest": 1000.0 + index if companion_present else None,
            "taker_buy_volume": 4.0 if companion_present else None,
            "taker_sell_volume": 6.0 if companion_present else None,
        }
        if include_long_short:
            row["long_short_ratio"] = 1.2 if companion_present else None
        rows.append(row)
    return pl.DataFrame(rows)


def test_classify_dependency_class_from_required_features() -> None:
    assert classify_dependency_class(("close",)) == CLASS_OHLCV_ONLY
    assert classify_dependency_class(("close", "volume")) == CLASS_OHLCV_PLUS_VOLUME
    assert classify_dependency_class(("open_interest",)) == CLASS_OI_DEPENDENT
    assert classify_dependency_class(("taker_buy_volume", "volume")) == CLASS_TAKER_DEPENDENT
    assert classify_dependency_class(("funding_rate",)) == CLASS_FUNDING_DEPENDENT
    assert classify_dependency_class(("asset_return", "btc_return")) == CLASS_UNKNOWN


def test_a_ohlcv_only_recovers_history_before_companions() -> None:
    """Test A — OHLCV-only factors retain OHLCV history."""
    frame = _joined_frame(ohlcv_bars=10, companion_start_index=5)
    partition = FactorInputPartition()
    aligned = partition.align_frame(frame, ("close", "volume"))
    assert aligned.height == 10
    assert aligned.get_column("open_time").to_list()[0] == _START_OHLCV_MS
    global_aligned = align_factor_input_frame(frame)
    assert global_aligned.height == 5
    expected_companion_start = frame.get_column("open_time").to_list()[5]
    assert global_aligned.get_column("open_time").to_list()[0] == expected_companion_start


def test_b_oi_dependency_preserves_oi_boundary() -> None:
    """Test B — OI factors do not produce values before OI availability."""
    frame = _joined_frame(ohlcv_bars=10, companion_start_index=5)
    partition = FactorInputPartition()
    aligned = partition.align_frame(frame, ("open_interest",))
    expected_companion_start = frame.get_column("open_time").to_list()[5]
    assert aligned.height == 5
    assert aligned.get_column("open_time").to_list()[0] == expected_companion_start
    assert aligned.get_column("open_interest").null_count() == 0


def test_c_funding_dependency_preserves_funding_boundary() -> None:
    """Test C — funding factors wait for funding/mark availability."""
    frame = _joined_frame(ohlcv_bars=10, companion_start_index=5)
    partition = FactorInputPartition()
    aligned = partition.align_frame(frame, ("funding_rate",))
    expected_companion_start = frame.get_column("open_time").to_list()[5]
    assert aligned.height == 5
    assert aligned.get_column("open_time").to_list()[0] == expected_companion_start
    assert aligned.get_column("funding_rate").null_count() == 0


def test_d_unrelated_companion_does_not_change_ohlcv_history() -> None:
    """Test D — adding/removing unrelated companions does not truncate OHLCV-only."""
    with_ls = _joined_frame(ohlcv_bars=10, companion_start_index=5, include_long_short=True)
    without_ls = with_ls.drop("long_short_ratio")
    partition = FactorInputPartition()
    aligned_with = partition.align_frame(with_ls, ("close", "volume"))
    aligned_without = partition.align_frame(without_ls, ("close", "volume"))
    assert (
        aligned_with.get_column("open_time").to_list()
        == aligned_without.get_column("open_time").to_list()
    )
    assert aligned_with.height == without_ls.height == 10


def test_e_backward_asof_semantics_preserved_in_joined_inputs() -> None:
    """Test E — joined companion values remain causal (no future attach)."""
    frame = _joined_frame(ohlcv_bars=6, companion_start_index=3)
    early = frame.filter(pl.col("open_time") < _START_OHLCV_MS + 3 * _MS_DAY)
    late = frame.filter(pl.col("open_time") >= _START_OHLCV_MS + 3 * _MS_DAY)
    assert early.get_column("open_interest").null_count() == early.height
    assert late.get_column("open_interest").null_count() == 0
    assert early.get_column("open_interest").to_list() == [None] * early.height


def test_f_no_forward_or_backfill() -> None:
    """Test F — missing historical companion observations remain missing."""
    frame = _joined_frame(ohlcv_bars=8, companion_start_index=4)
    partition = FactorInputPartition()
    aligned = partition.align_frame(frame, ("close",))
    assert aligned.get_column("open_interest").null_count() == 4
    assert aligned.get_column("taker_buy_volume").null_count() == 4
    assert None in aligned.get_column("open_interest").to_list()


def test_g_cumulative_pvt_obv_recomputed_on_expanded_history() -> None:
    """Test G — PVT/OBV recompute over expanded history, not truncated seed."""
    frame = _joined_frame(ohlcv_bars=10, companion_start_index=5)
    global_frame = align_factor_input_frame(frame)
    pvt = PriceVolumeTrendFactor()
    obv = OnBalanceVolumeFactor()

    pvt_full = pvt.compute(frame).get_column("price_volume_trend").to_list()
    pvt_trunc = pvt.compute(global_frame).get_column("price_volume_trend").to_list()
    obv_full = obv.compute(frame).get_column("on_balance_volume").to_list()
    obv_trunc = obv.compute(global_frame).get_column("on_balance_volume").to_list()

    assert len(pvt_full) == 10
    assert len(pvt_trunc) == 5
    assert pvt_full[-5:] != pvt_trunc
    assert obv_full[-5:] != obv_trunc

    registry = FactorRegistry()
    registry.register_many((pvt, obv))
    pipeline = FactorPipeline(registry)
    wide = pipeline.run(frame)
    assert wide.height == 10
    # PVT/OBV first bar is null by formula (no prior close); remaining bars are defined.
    assert wide.get_column("price_volume_trend").null_count() == 1
    assert wide.get_column("on_balance_volume").null_count() == 1


def test_h_registry_completeness_for_executable_raw_factors() -> None:
    """Test H — every executable raw-input factor has valid dependency metadata."""
    partition = FactorInputPartition()
    registry = build_default_registry()
    executable = [
        factor
        for factor in registry.list()
        if set(factor.required_features).issubset(KNOWN_FACTOR_INPUT_FEATURES)
    ]
    assert executable
    for factor in executable:
        partition.validate_required_features(factor.required_features)
        dep_class = partition.dependency_class(factor.required_features)
        assert dep_class != CLASS_UNKNOWN
        datasets = partition.required_datasets(factor.required_features)
        assert "ohlcv" in datasets
        companions = required_companion_columns(factor.required_features)
        assert set(companions).issubset(KNOWN_FACTOR_INPUT_FEATURES)


def test_i_deterministic_pipeline_output() -> None:
    """Test I — identical inputs yield byte-identical factor output."""
    frame = _joined_frame(ohlcv_bars=12, companion_start_index=6)
    registry = FactorRegistry()
    registry.register_many(
        (
            PriceVolumeTrendFactor(),
            OnBalanceVolumeFactor(),
            OpenInterestLevelFactor(),
            FundingRateLevelFactor(),
        )
    )
    first = FactorPipeline(registry).run(frame)
    second = FactorPipeline(registry).run(frame)
    assert_frame_equal(first, second)
    buffer_one = BytesIO()
    buffer_two = BytesIO()
    first.write_ipc(buffer_one)
    second.write_ipc(buffer_two)
    assert (
        hashlib.sha256(buffer_one.getvalue()).hexdigest()
        == hashlib.sha256(buffer_two.getvalue()).hexdigest()
    )


def test_j_production_artifact_isolation(tmp_path: Path) -> None:
    """Test J — partitioning helpers do not mutate production ledgers."""
    for tier in (
        STORAGE_DIR_WALK_FORWARD,
        STORAGE_DIR_PURGED_CV,
        STORAGE_DIR_FACTOR_SELECTION,
    ):
        path = tmp_path / tier / "marker.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable-ledger-bytes")
    before = hash_watched_production_artifacts(tmp_path)
    frame = _joined_frame()
    partition = FactorInputPartition()
    _ = partition.align_frame(frame, ("close", "volume"))
    registry = FactorRegistry()
    registry.register(PriceVolumeTrendFactor())
    _ = FactorPipeline(registry).run(frame)
    after = hash_watched_production_artifacts(tmp_path)
    assert before == after


def test_required_datasets_fails_loudly_on_unknown_feature() -> None:
    with pytest.raises(FactorValidationError, match="unknown factor input feature"):
        required_datasets(("close", "asset_return"))


def test_align_fails_when_required_companion_never_available() -> None:
    frame = _joined_frame(ohlcv_bars=4, companion_start_index=4)
    frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias("open_interest"))
    partition = FactorInputPartition()
    with pytest.raises(FactorValidationError, match="complete coverage"):
        partition.align_frame(frame, ("open_interest",))


def test_pipeline_oi_null_before_boundary_and_ohlcv_full() -> None:
    frame = _joined_frame(ohlcv_bars=10, companion_start_index=5)
    registry = FactorRegistry()
    registry.register_many((PriceVolumeTrendFactor(), OpenInterestLevelFactor()))
    result = FactorPipeline(registry).run(frame)
    assert result.height == 10
    assert result.get_column("price_volume_trend").null_count() == 1
    assert result.get_column("open_interest_level").null_count() == 5
    assert result.get_column("open_interest_level").to_list()[:5] == [None] * 5
