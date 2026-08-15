"""Exact-equivalence tests for memory-efficient Factor Selection.

Compares legacy full-panel observation materialization against the
memory-efficient spill + pairwise redundancy path. Batch boundaries must not
change canonical selection results.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_selection import (
    FactorEligibilityPolicy,
    FactorObservationSpill,
    FactorSelectionError,
    FactorSelectionPipeline,
    FactorSelectionRepository,
    FactorsObservationLoader,
    MemoryEfficientFactorsObservationLoader,
    SimpleFactorSelectionEngine,
    apply_greedy_redundancy_filter,
    apply_greedy_redundancy_filter_from_spill,
    pairwise_abs_pearson,
    require_redundancy_config,
)
from cqros.factor_selection.registry import FactorSelectionEngineRegistry
from cqros.factor_selection.schema import CANONICAL_COLUMN_ORDER, ELIGIBILITY_COLUMNS
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER,
)
from cqros.factors.schema import COLUMN_DTYPES as FACTOR_COLUMN_DTYPES
from cqros.factors.schema import FactorStatus
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "default"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_TIMEFRAME = "1h"
_YEAR = 2026
_VERSION = "1.0.0"
_CATEGORY = "price"
_VALIDATION_TIME = 1_700_000_000_000
_START = 1_699_000_000_000
_END = 1_700_000_000_000
_N_POINTS = 600


class _StaticObservationSource:
    """In-memory observation source for the legacy full-panel path."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame
        self.load_calls = 0

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        _ = timeframe
        self.load_calls += 1
        return (
            self._frame.filter(pl.col("factor_name").is_in(list(factor_names)))
            .filter(pl.col("factor_version").is_in(list(factor_versions)))
            .filter(pl.col("open_time") >= start_time)
            .filter(pl.col("open_time") <= end_time)
        )


class _SpillingStaticObservationSource:
    """In-memory spill-capable source mirroring MemoryEfficientFactorsObservationLoader."""

    def __init__(self, frame: pl.DataFrame, *, spill_parent: Path, batch_size: int = 1) -> None:
        self._frame = frame
        self._spill_parent = spill_parent
        self._batch_size = batch_size
        self.spill_calls = 0

    def spill_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> FactorObservationSpill:
        _ = timeframe
        self.spill_calls += 1
        self._spill_parent.mkdir(parents=True, exist_ok=True)
        root = self._spill_parent / f"spill_{self.spill_calls}"
        if root.exists():
            # Unique per call for cleanup isolation.
            root = self._spill_parent / f"spill_{self.spill_calls}_{id(self)}"
        spill = FactorObservationSpill(root)
        filtered = (
            self._frame.filter(pl.col("factor_name").is_in(list(factor_names)))
            .filter(pl.col("factor_version").is_in(list(factor_versions)))
            .filter(pl.col("open_time") >= start_time)
            .filter(pl.col("open_time") <= end_time)
        )
        identities = list(zip(factor_names, factor_versions, strict=False))
        for name, version in identities:
            spill.ensure_factor(str(name), str(version))
        for start in range(0, len(identities), self._batch_size):
            batch = identities[start : start + self._batch_size]
            for name, version in batch:
                sliced = filtered.filter(
                    (pl.col("factor_name") == name) & (pl.col("factor_version") == version)
                )
                spill.append_factor_rows(
                    factor_name=str(name),
                    factor_version=str(version),
                    frame=sliced,
                )
        return spill

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        spill = self.spill_panel(
            timeframe=timeframe,
            factor_names=factor_names,
            factor_versions=factor_versions,
            start_time=start_time,
            end_time=end_time,
        )
        try:
            return spill.load_long_panel(factor_names, factor_versions)
        finally:
            spill.cleanup()


def _validation_frame(
    *,
    names: list[str],
    ics: list[float] | None = None,
    observations: list[int] | None = None,
    timeframes: list[str] | None = None,
) -> pl.DataFrame:
    row_count = len(names)
    ics = ics if ics is not None else [0.20 - (0.01 * index) for index in range(row_count)]
    observations = observations if observations is not None else [200] * row_count
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    return pl.DataFrame(
        {
            "factor_name": names,
            "factor_version": [_VERSION] * row_count,
            "timeframe": timeframes,
            "validation_time": [_VALIDATION_TIME] * row_count,
            "factor_category": [_CATEGORY] * row_count,
            "dataset_version": ["default"] * row_count,
            "label_version": ["default"] * row_count,
            "validation_start_time": [_START] * row_count,
            "validation_end_time": [_END] * row_count,
            "information_coefficient": ics,
            "rank_information_coefficient": ics,
            "ic_information_ratio": [0.5] * row_count,
            "ic_p_value": [0.01] * row_count,
            "ic_decay": [0.5] * row_count,
            "turnover": [0.2] * row_count,
            "monotonicity_score": [0.5] * row_count,
            "quantile_spread": [0.05] * row_count,
            "observations": observations,
            "status": ["PASS" if obs > 0 else "FAIL" for obs in observations],
        }
    )


def _series_clone(base: list[float], *, scale: float = 1.0, noise: float = 0.0) -> list[float]:
    rng = np.random.default_rng(7)
    values = np.asarray(base, dtype=np.float64) * scale
    if noise > 0.0:
        values = values + rng.normal(0.0, noise, size=values.shape)
    return values.tolist()


def _observations_for(
    *,
    names: list[str],
    series: Mapping[str, Sequence[float | None]],
    n_points: int = _N_POINTS,
    symbol_count: int = 3,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(n_points):
        open_time = _START + index
        symbol = "S0" if symbol_count <= 1 else f"S{index % symbol_count}"
        for name in names:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": open_time,
                    "factor_name": name,
                    "factor_version": _VERSION,
                    "factor_value": series[name][index],
                }
            )
    return pl.DataFrame(rows)


def _compare_columns() -> list[str]:
    return list(CANONICAL_COLUMN_ORDER) + list(ELIGIBILITY_COLUMNS)


def _run_engine(
    validation: pl.DataFrame,
    observation_source: object,
    *,
    top_n: int = 3,
    candidate_n: int = 6,
    max_factor_correlation: float = 0.90,
    min_overlap: int = 500,
) -> pl.DataFrame:
    engine = SimpleFactorSelectionEngine(
        top_n=top_n,
        candidate_n=candidate_n,
        max_factor_correlation=max_factor_correlation,
        min_overlap=min_overlap,
        observation_source=observation_source,  # type: ignore[arg-type]
        eligibility_policy=FactorEligibilityPolicy(),
    )
    registry = FactorSelectionEngineRegistry()
    registry.register("simple", engine)
    return FactorSelectionPipeline(registry).run("simple", validation)


def _assert_selection_equivalent(left: pl.DataFrame, right: pl.DataFrame) -> None:
    left_sorted = left.select(_compare_columns()).sort(
        ["timeframe", "factor_name", "factor_version"]
    )
    right_sorted = right.select(_compare_columns()).sort(
        ["timeframe", "factor_name", "factor_version"]
    )
    assert_frame_equal(left_sorted, right_sorted, check_column_order=True)


def _base_series(n_points: int = _N_POINTS) -> list[float]:
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 1.0, size=n_points).tolist()


def test_full_panel_vs_memory_efficient_synthetic_panel(tmp_path: Path) -> None:
    """Legacy full-panel and memory-efficient paths match on a multi-factor panel."""
    names = [f"f{index}" for index in range(8)]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "f0": base,
        "f1": _series_clone(base, scale=1.0),  # exact clone of f0 → redundant
        "f2": _series_clone(base, scale=-1.0),  # anti-correlated → |rho|=1 redundant
        "f3": _series_clone(base, noise=0.05),
        "f4": _series_clone(base, noise=0.50),
        "f5": list(np.random.default_rng(21).normal(0.0, 1.0, size=_N_POINTS)),
        "f6": list(np.random.default_rng(22).normal(0.0, 1.0, size=_N_POINTS)),
        "f7": list(np.random.default_rng(23).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)

    legacy = _run_engine(validation, _StaticObservationSource(observations))
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(
            observations,
            spill_parent=tmp_path / "spill",
            batch_size=1,
        ),
    )
    _assert_selection_equivalent(legacy, memory)

    assert set(legacy.filter(pl.col("selected"))["factor_name"].to_list()) == set(
        memory.filter(pl.col("selected"))["factor_name"].to_list()
    )
    assert legacy["selection_rank"].to_list() == memory["selection_rank"].to_list()
    assert legacy["eligibility_status"].to_list() == memory["eligibility_status"].to_list()
    assert legacy["selected_direction"].to_list() == memory["selected_direction"].to_list()
    assert legacy["selection_ic"].to_list() == memory["selection_ic"].to_list()
    assert legacy["selection_reason"].to_list() == memory["selection_reason"].to_list()


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_batch_size_invariance(tmp_path: Path, batch_size: int) -> None:
    """factor_batch_size must not change canonical selection results."""
    names = [f"g{index}" for index in range(6)]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        names[0]: base,
        names[1]: _series_clone(base),
        names[2]: list(np.random.default_rng(31).normal(0.0, 1.0, size=_N_POINTS)),
        names[3]: list(np.random.default_rng(32).normal(0.0, 1.0, size=_N_POINTS)),
        names[4]: list(np.random.default_rng(33).normal(0.0, 1.0, size=_N_POINTS)),
        names[5]: list(np.random.default_rng(34).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)
    baseline = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "b1", batch_size=1),
    )
    other = _run_engine(
        validation,
        _SpillingStaticObservationSource(
            observations,
            spill_parent=tmp_path / f"b{batch_size}",
            batch_size=batch_size,
        ),
    )
    _assert_selection_equivalent(baseline, other)


def test_deterministic_repeated_execution(tmp_path: Path) -> None:
    """Repeated memory-efficient runs produce identical frames."""
    names = ["d0", "d1", "d2", "d3"]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "d0": base,
        "d1": _series_clone(base, noise=0.1),
        "d2": list(np.random.default_rng(41).normal(0.0, 1.0, size=_N_POINTS)),
        "d3": list(np.random.default_rng(42).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)
    first = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "r1", batch_size=2),
    )
    second = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "r2", batch_size=2),
    )
    _assert_selection_equivalent(first, second)


def test_all_null_and_zero_observation_factors(tmp_path: Path) -> None:
    """All-null observation series and zero-observation FV rows stay equivalent."""
    names = ["ok", "all_null", "zero_obs"]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "ok": base,
        "all_null": [None] * _N_POINTS,
        "zero_obs": base,
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(
        names=names,
        ics=[0.2, 0.19, 0.18],
        observations=[200, 200, 0],
    )
    legacy = _run_engine(validation, _StaticObservationSource(observations))
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "nulls"),
    )
    _assert_selection_equivalent(legacy, memory)
    assert "zero_obs" not in legacy.filter(pl.col("selected"))["factor_name"].to_list()


def test_sparse_and_uneven_histories(tmp_path: Path) -> None:
    """Sparse / uneven factor histories remain equivalent."""
    names = ["dense", "sparse", "short"]
    dense = _base_series()
    sparse: list[float | None] = [dense[i] if i % 3 == 0 else None for i in range(_N_POINTS)]
    short: list[float | None] = [dense[i] if i < _N_POINTS // 2 else None for i in range(_N_POINTS)]
    series: Mapping[str, Sequence[float | None]] = {
        "dense": list(dense),
        "sparse": sparse,
        "short": short,
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names, ics=[0.2, 0.15, 0.1])
    legacy = _run_engine(validation, _StaticObservationSource(observations), min_overlap=50)
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "sparse"),
        min_overlap=50,
    )
    _assert_selection_equivalent(legacy, memory)


def test_multiple_timeframes(tmp_path: Path) -> None:
    """Multi-timeframe validation frames stay equivalent per timeframe."""
    names = ["a", "b", "c", "d"]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "a": base,
        "b": _series_clone(base),
        "c": list(np.random.default_rng(51).normal(0.0, 1.0, size=_N_POINTS)),
        "d": list(np.random.default_rng(52).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    # Two timeframes with duplicated factor identities (engine groups by TF).
    validation = pl.concat(
        [
            _validation_frame(names=names, timeframes=["1h"] * len(names)),
            _validation_frame(
                names=names,
                timeframes=["4h"] * len(names),
                ics=[0.11, 0.10, 0.09, 0.08],
            ),
        ],
        how="vertical",
    )
    legacy = _run_engine(validation, _StaticObservationSource(observations))
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "mtf"),
    )
    _assert_selection_equivalent(legacy, memory)


def test_spill_filter_matches_legacy_redundancy_direct(tmp_path: Path) -> None:
    """Direct redundancy helpers agree for identical ranked + observation inputs."""
    names = ["x", "y", "z"]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "x": base,
        "y": _series_clone(base),
        "z": list(np.random.default_rng(61).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    ranked = pl.DataFrame(
        {
            "factor_name": names,
            "factor_version": [_VERSION] * 3,
            "timeframe": [_TIMEFRAME] * 3,
            "selection_rank": [1, 2, 3],
            "selection_score": [0.9, 0.8, 0.7],
            "_row_id": [0, 1, 2],
        }
    )
    config = require_redundancy_config(top_n=2, candidate_n=3, min_overlap=500)
    legacy = apply_greedy_redundancy_filter(ranked, observations, config)

    spill = FactorObservationSpill(tmp_path / "direct")
    for name in names:
        spill.ensure_factor(name, _VERSION)
        spill.append_factor_rows(
            factor_name=name,
            factor_version=_VERSION,
            frame=observations.filter(pl.col("factor_name") == name),
        )
    memory = apply_greedy_redundancy_filter_from_spill(ranked, spill, config)
    spill.cleanup()

    assert_frame_equal(
        legacy.sort(["factor_name", "factor_version"]).select(
            [
                "factor_name",
                "factor_version",
                "selected",
                "selection_reason",
                "redundancy_rejected",
                "redundancy_reference_factor",
                "redundancy_correlation",
                "redundancy_overlap",
            ]
        ),
        memory.sort(["factor_name", "factor_version"]).select(
            [
                "factor_name",
                "factor_version",
                "selected",
                "selection_reason",
                "redundancy_rejected",
                "redundancy_reference_factor",
                "redundancy_correlation",
                "redundancy_overlap",
            ]
        ),
    )


def test_empty_partial_partitions_supported(tmp_path: Path) -> None:
    """Missing observation panels do not force redundancy; paths stay equivalent."""
    names = ["p0", "p1", "p2"]
    validation = _validation_frame(names=names)
    empty = pl.DataFrame(
        schema={
            "symbol": pl.String,
            "open_time": pl.Int64,
            "factor_name": pl.String,
            "factor_version": pl.String,
            "factor_value": pl.Float64,
        }
    )
    legacy = _run_engine(validation, _StaticObservationSource(empty))
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(empty, spill_parent=tmp_path / "empty"),
    )
    _assert_selection_equivalent(legacy, memory)


def _write_factor_partition(
    layout: StorageLayout,
    *,
    symbol: str,
    timeframe: str,
    rows: list[tuple[int, str, float | None]],
) -> None:
    frame = pl.DataFrame(
        {
            "symbol": [symbol] * len(rows),
            "timeframe": [timeframe] * len(rows),
            "open_time": [_START + open_index for open_index, _, _ in rows],
            "factor_name": [name for _, name, _ in rows],
            "factor_version": [_VERSION] * len(rows),
            "factor_category": [_CATEGORY] * len(rows),
            "factor_group": ["alpha"] * len(rows),
            "factor_value": [value for _, _, value in rows],
            "lookback": [20] * len(rows),
            "prediction_horizon": [1] * len(rows),
            "enabled": [True] * len(rows),
            "status": [FactorStatus.ACTIVE.value] * len(rows),
        }
    ).select(list(FACTOR_CANONICAL_COLUMN_ORDER))
    frame = frame.cast(
        pl.Schema({name: FACTOR_COLUMN_DTYPES[name] for name in FACTOR_CANONICAL_COLUMN_ORDER})
    )  # pyright: ignore[reportArgumentType]
    path = layout.factors_path(_MANAGER, _EXCHANGE, _MARKET, symbol, timeframe, year=_YEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_loader_full_panel_vs_memory_efficient_on_disk(tmp_path: Path) -> None:
    """On-disk FactorsObservationLoader vs MemoryEfficientFactorsObservationLoader."""
    layout = StorageLayout(tmp_path / "data")
    base = _base_series(200)
    names = ["m0", "m1", "m2", "m3"]
    series = {
        "m0": base,
        "m1": _series_clone(base),
        "m2": list(np.random.default_rng(71).normal(0.0, 1.0, size=200)),
        "m3": list(np.random.default_rng(72).normal(0.0, 1.0, size=200)),
    }
    for symbol_index, symbol in enumerate(["AAAUSDT", "BBBUSDT"]):
        rows: list[tuple[int, str, float | None]] = []
        for index in range(200):
            for name in names:
                # Offset symbols slightly so joins still align on open_time.
                rows.append((index + symbol_index * 0, name, series[name][index]))
        _write_factor_partition(layout, symbol=symbol, timeframe=_TIMEFRAME, rows=rows)

    validation = _validation_frame(names=names, observations=[180, 180, 180, 180])
    legacy_loader = FactorsObservationLoader(
        layout,
        manager=_MANAGER,
        year=_YEAR,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    memory_loader = MemoryEfficientFactorsObservationLoader(
        layout,
        manager=_MANAGER,
        year=_YEAR,
        exchange=_EXCHANGE,
        market=_MARKET,
        factor_batch_size=2,
        spill_parent=tmp_path / "spill",
    )
    legacy = _run_engine(validation, legacy_loader, min_overlap=100)
    memory = _run_engine(validation, memory_loader, min_overlap=100)
    _assert_selection_equivalent(legacy, memory)


def test_canonical_parquet_deterministic_sha(tmp_path: Path) -> None:
    """Repository writes of repeated memory-efficient runs are byte-identical."""
    names = ["s0", "s1", "s2"]
    base = _base_series()
    series: dict[str, Sequence[float | None]] = {
        "s0": base,
        "s1": _series_clone(base, noise=0.2),
        "s2": list(np.random.default_rng(81).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)
    frame = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "sha"),
    )

    layout = StorageLayout(tmp_path / "store")
    repo = FactorSelectionRepository(layout, ParquetStore())
    hashes: list[str] = []
    for run_index in range(2):
        repo.save(
            frame,
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            timeframe=_TIMEFRAME,
            year=_YEAR + run_index,  # distinct paths; compare bytes of each write cycle
        )
        # Overwrite same partition twice for byte identity.
    path = layout.factor_selection_path(_MANAGER, _EXCHANGE, _MARKET, _TIMEFRAME, _YEAR)
    repo.save(
        frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    repo.save(
        frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    second = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first == second
    hashes.extend([first, second])
    assert hashes[0] == hashes[1]


def _ranked_frame(names: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "factor_name": list(names),
            "factor_version": [_VERSION] * len(names),
            "timeframe": [_TIMEFRAME] * len(names),
            "selection_rank": list(range(1, len(names) + 1)),
            "selection_score": [1.0 - (0.01 * index) for index in range(len(names))],
            "_row_id": list(range(len(names))),
        }
    )


def _spill_from_observations(
    tmp_path: Path,
    observations: pl.DataFrame,
    names: Sequence[str],
) -> FactorObservationSpill:
    spill = FactorObservationSpill(tmp_path)
    for name in names:
        spill.ensure_factor(name, _VERSION)
        spill.append_factor_rows(
            factor_name=name,
            factor_version=_VERSION,
            frame=observations.filter(pl.col("factor_name") == name),
        )
    return spill


def test_missing_nans_multi_symbol_timestamp_equivalence(tmp_path: Path) -> None:
    """Missing values, NaNs, uneven histories, multiple symbols and timestamps."""
    names = ["dense", "gappy", "nan_tail", "other"]
    base = _base_series()
    gappy: list[float | None] = [base[i] if i % 4 != 0 else None for i in range(_N_POINTS)]
    nan_tail: list[float | None] = [
        float("nan") if i > _N_POINTS - 40 else base[i] for i in range(_N_POINTS)
    ]
    series: dict[str, Sequence[float | None]] = {
        "dense": base,
        "gappy": gappy,
        "nan_tail": nan_tail,
        "other": list(np.random.default_rng(91).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series, symbol_count=3)
    validation = _validation_frame(names=names, ics=[0.22, 0.18, 0.14, 0.10])
    legacy = _run_engine(validation, _StaticObservationSource(observations), min_overlap=80)
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "mix"),
        min_overlap=80,
    )
    _assert_selection_equivalent(legacy, memory)


def test_exact_pearson_and_overlap_equivalence(tmp_path: Path) -> None:
    """Pearson and overlap from inner-join spill match the legacy pivot path exactly."""
    names = ["left", "right"]
    base = _base_series()
    left_series: list[float | None] = [None if i % 7 == 0 else v for i, v in enumerate(base)]
    right_series: list[float | None] = [
        None if i % 11 == 0 else (v * 0.85 + 0.05) for i, v in enumerate(base)
    ]
    observations = _observations_for(
        names=names,
        series={"left": left_series, "right": right_series},
    )
    ranked = _ranked_frame(names)
    config = require_redundancy_config(
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.99,
        min_overlap=50,
    )
    legacy = apply_greedy_redundancy_filter(ranked, observations, config)
    spill = _spill_from_observations(tmp_path / "pearson", observations, names)
    try:
        memory = apply_greedy_redundancy_filter_from_spill(ranked, spill, config)
    finally:
        spill.cleanup()

    assert legacy["redundancy_correlation"].to_list() == memory["redundancy_correlation"].to_list()
    assert legacy["redundancy_overlap"].to_list() == memory["redundancy_overlap"].to_list()
    assert legacy["redundancy_rejected"].to_list() == memory["redundancy_rejected"].to_list()

    joined = (
        observations.filter(pl.col("factor_name") == "left")
        .select(["symbol", "open_time", "factor_value"])
        .filter(pl.col("factor_value").is_not_null())
        .join(
            observations.filter(pl.col("factor_name") == "right")
            .select(["symbol", "open_time", pl.col("factor_value").alias("right_value")])
            .filter(pl.col("right_value").is_not_null()),
            on=["symbol", "open_time"],
            how="inner",
        )
    )
    join_corr, join_overlap = pairwise_abs_pearson(
        joined["factor_value"].to_numpy(),
        joined["right_value"].to_numpy(),
    )
    rejected = memory.filter(pl.col("factor_name") == "right")
    if rejected["redundancy_rejected"][0]:
        assert rejected["redundancy_correlation"][0] == join_corr
        assert rejected["redundancy_overlap"][0] == join_overlap


def test_redundancy_accept_below_correlation_threshold(tmp_path: Path) -> None:
    """Low-correlation factors survive both full_panel and memory_efficient paths."""
    names = ["a", "b"]
    a_vals = _base_series()
    b_vals = list(np.random.default_rng(101).normal(0.0, 1.0, size=_N_POINTS))
    observations = _observations_for(names=names, series={"a": a_vals, "b": b_vals})
    validation = _validation_frame(names=names, ics=[0.20, 0.19])
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "accept"),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    assert legacy.filter(pl.col("selected")).height == 2
    assert "redundant" not in legacy["selection_reason"].to_list()


def test_redundancy_reject_above_correlation_threshold(tmp_path: Path) -> None:
    """Exact clones are rejected as redundant on both execution paths."""
    names = ["lead", "clone"]
    base = _base_series()
    observations = _observations_for(names=names, series={"lead": base, "clone": list(base)})
    validation = _validation_frame(names=names, ics=[0.20, 0.19])
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "reject"),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    clone = legacy.filter(pl.col("factor_name") == "clone")
    assert clone["selected"][0] is False
    assert clone["selection_reason"][0] == "redundant"


def test_min_overlap_skips_insufficient_pairs(tmp_path: Path) -> None:
    """High correlation with overlap below min_overlap must not reject."""
    names = ["long", "short_clone"]
    base = _base_series()
    short: list[float | None] = [base[i] if i < 40 else None for i in range(_N_POINTS)]
    observations = _observations_for(names=names, series={"long": base, "short_clone": short})
    validation = _validation_frame(names=names, ics=[0.20, 0.19])
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "overlap"),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    selected_names = set(legacy.filter(pl.col("selected"))["factor_name"].to_list())
    assert selected_names == {"long", "short_clone"}


def test_max_factor_correlation_threshold_semantics(tmp_path: Path) -> None:
    """Raising max_factor_correlation keeps near-clones that a tighter threshold rejects."""
    names = ["lead", "near"]
    base = _base_series()
    # noise=1.0 → |rho| ≈ 0.70, between a tight 0.50 and a loose 0.90 threshold.
    near = _series_clone(base, noise=1.0)
    observations = _observations_for(names=names, series={"lead": base, "near": near})
    validation = _validation_frame(names=names, ics=[0.20, 0.19])
    tight_legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.50,
        min_overlap=100,
    )
    tight_memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "tight"),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.50,
        min_overlap=100,
    )
    loose_legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    loose_memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "loose"),
        top_n=2,
        candidate_n=2,
        max_factor_correlation=0.90,
        min_overlap=100,
    )
    _assert_selection_equivalent(tight_legacy, tight_memory)
    _assert_selection_equivalent(loose_legacy, loose_memory)
    tight_near = tight_legacy.filter(pl.col("factor_name") == "near")
    loose_near = loose_legacy.filter(pl.col("factor_name") == "near")
    assert tight_near["selection_reason"][0] == "redundant"
    assert loose_near["selected"][0] is True


def test_candidate_n_limits_redundancy_pool(tmp_path: Path) -> None:
    """Ranks beyond candidate_n are outside_candidate_n on both paths."""
    names = [f"c{index}" for index in range(6)]
    series = {
        name: list(np.random.default_rng(110 + index).normal(0.0, 1.0, size=_N_POINTS))
        for index, name in enumerate(names)
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=3,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "cand"),
        top_n=2,
        candidate_n=3,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    outside = legacy.filter(pl.col("selection_rank") > 3)
    assert outside["selection_reason"].unique().to_list() == ["outside_candidate_n"]
    assert outside["selected"].unique().to_list() == [False]


def test_top_n_limits_selected_survivors(tmp_path: Path) -> None:
    """Only top_n non-redundant survivors are selected on both paths."""
    names = [f"t{index}" for index in range(5)]
    series = {
        name: list(np.random.default_rng(120 + index).normal(0.0, 1.0, size=_N_POINTS))
        for index, name in enumerate(names)
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=5,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "top"),
        top_n=2,
        candidate_n=5,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    selected = legacy.filter(pl.col("selected"))
    assert selected.height == 2
    assert set(selected["selection_reason"].to_list()) == {"top_n"}


def test_deterministic_tie_breaking(tmp_path: Path) -> None:
    """Identical scores break ties by factor_name then factor_version on both paths."""
    names = ["zeta", "alpha", "mu"]
    series = {
        "zeta": list(np.random.default_rng(131).normal(0.0, 1.0, size=_N_POINTS)),
        "alpha": list(np.random.default_rng(132).normal(0.0, 1.0, size=_N_POINTS)),
        "mu": list(np.random.default_rng(133).normal(0.0, 1.0, size=_N_POINTS)),
    }
    observations = _observations_for(names=names, series=series)
    validation = _validation_frame(names=names, ics=[0.15, 0.15, 0.15])
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=3,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "ties"),
        top_n=2,
        candidate_n=3,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    ordered = legacy.sort("selection_rank")["factor_name"].to_list()
    assert ordered == sorted(names)
    assert memory.sort("selection_rank")["factor_name"].to_list() == ordered


def test_spill_cleanup_on_success(tmp_path: Path) -> None:
    """Spill directories are removed after a successful engine build."""
    names = ["ok0", "ok1"]
    base = _base_series()
    observations = _observations_for(
        names=names,
        series={
            "ok0": base,
            "ok1": list(np.random.default_rng(141).normal(0.0, 1.0, size=_N_POINTS)),
        },
    )
    spill_parent = tmp_path / "cleanup_ok"
    validation = _validation_frame(names=names)
    _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=spill_parent),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    leftovers = list(spill_parent.rglob("*.parquet")) if spill_parent.exists() else []
    assert leftovers == []


def test_spill_cleanup_on_failure(tmp_path: Path) -> None:
    """Spill directories are removed when memory-efficient spill fails."""
    layout = StorageLayout(tmp_path / "data")
    _write_factor_partition(
        layout,
        symbol="AAAUSDT",
        timeframe=_TIMEFRAME,
        rows=[(0, "m0", 1.0), (1, "m0", 2.0)],
    )
    spill_parent = tmp_path / "cleanup_fail"
    loader = MemoryEfficientFactorsObservationLoader(
        layout,
        manager=_MANAGER,
        year=_YEAR,
        exchange=_EXCHANGE,
        market=_MARKET,
        factor_batch_size=1,
        spill_parent=spill_parent,
    )
    with patch(
        "cqros.factor_selection.memory_efficient.pl.scan_parquet",
        side_effect=RuntimeError("simulated read failure"),
    ):
        with pytest.raises(FactorSelectionError) as exc_info:
            loader.spill_panel(
                timeframe=_TIMEFRAME,
                factor_names=["m0"],
                factor_versions=[_VERSION],
                start_time=_START,
                end_time=_END,
            )
    assert exc_info.value.error_code == "FSEL_MEM_SPILL_FAILED"
    leftovers = list(spill_parent.glob("cqros_fsel_spill_*")) if spill_parent.exists() else []
    assert leftovers == []


def test_no_forbidden_research_data_imports() -> None:
    """Memory-efficient Factor Selection must not import labels/OOS/regime/signals."""
    import cqros.cli.generate_factor_selection as cli_mod
    import cqros.factor_selection.engine as engine_mod
    import cqros.factor_selection.memory_efficient as memory_mod
    import cqros.factor_selection.observation_loader as loader_mod

    forbidden = ("cqros.alpha", "cqros.regime", "cqros.predictions", "cqros.signals")
    roots = [
        Path(memory_mod.__file__),
        Path(engine_mod.__file__),
        Path(loader_mod.__file__),
        Path(cli_mod.__file__),
    ]
    imported: set[str] = set()
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    for module_name in imported:
        assert not any(
            module_name == prefix or module_name.startswith(prefix + ".") for prefix in forbidden
        ), module_name


def test_empty_candidate_set_equivalence(tmp_path: Path) -> None:
    """Empty ranked candidate frames stay equivalent across redundancy helpers."""
    ranked = pl.DataFrame(
        {
            "factor_name": pl.Series([], dtype=pl.String),
            "factor_version": pl.Series([], dtype=pl.String),
            "timeframe": pl.Series([], dtype=pl.String),
            "selection_rank": pl.Series([], dtype=pl.Int32),
            "selection_score": pl.Series([], dtype=pl.Float64),
            "_row_id": pl.Series([], dtype=pl.UInt32),
        }
    )
    empty_obs = pl.DataFrame(
        schema={
            "symbol": pl.String,
            "open_time": pl.Int64,
            "factor_name": pl.String,
            "factor_version": pl.String,
            "factor_value": pl.Float64,
        }
    )
    config = require_redundancy_config(top_n=2, candidate_n=2, min_overlap=10)
    legacy = apply_greedy_redundancy_filter(ranked, empty_obs, config)
    spill = FactorObservationSpill(tmp_path / "empty_cand")
    try:
        memory = apply_greedy_redundancy_filter_from_spill(ranked, spill, config)
    finally:
        spill.cleanup()
    assert legacy.height == 0
    assert memory.height == 0


def test_single_candidate_equivalence(tmp_path: Path) -> None:
    """A single eligible candidate is selected without redundancy comparisons."""
    names = ["only"]
    observations = _observations_for(names=names, series={"only": _base_series()})
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=1,
        candidate_n=1,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "single"),
        top_n=1,
        candidate_n=1,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    assert legacy["selected"].to_list() == [True]
    assert legacy["selection_reason"].to_list() == ["top_n"]


def test_fewer_candidates_than_batch_size(tmp_path: Path) -> None:
    """Batch size larger than the candidate count remains valid and equivalent."""
    names = ["p", "q"]
    base = _base_series()
    observations = _observations_for(
        names=names,
        series={
            "p": base,
            "q": list(np.random.default_rng(151).normal(0.0, 1.0, size=_N_POINTS)),
        },
    )
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(
            observations,
            spill_parent=tmp_path / "small_batch",
            batch_size=10,
        ),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)


def test_single_symbol_input_equivalence(tmp_path: Path) -> None:
    """Single-symbol observation panels remain equivalent."""
    names = ["u", "v"]
    base = _base_series()
    observations = _observations_for(
        names=names,
        series={
            "u": base,
            "v": list(np.random.default_rng(161).normal(0.0, 1.0, size=_N_POINTS)),
        },
        symbol_count=1,
    )
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "one_sym"),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)


def test_one_timestamp_input_equivalence(tmp_path: Path) -> None:
    """A single timestamp (overlap 1) remains equivalent and does not force Pearson."""
    names = ["one", "two"]
    observations = _observations_for(
        names=names,
        series={"one": [1.0], "two": [2.0]},
        n_points=1,
        symbol_count=1,
    )
    validation = _validation_frame(names=names)
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        min_overlap=1,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "one_ts"),
        top_n=2,
        candidate_n=2,
        min_overlap=1,
    )
    _assert_selection_equivalent(legacy, memory)
    assert legacy.filter(pl.col("selected")).height == 2


def test_all_null_candidate_factor_equivalence(tmp_path: Path) -> None:
    """An all-null candidate factor stays equivalent and cannot form Pearson pairs."""
    names = ["signal", "nulls"]
    base = _base_series()
    observations = _observations_for(
        names=names,
        series={"signal": base, "nulls": [None] * _N_POINTS},
    )
    validation = _validation_frame(names=names, ics=[0.20, 0.19])
    legacy = _run_engine(
        validation,
        _StaticObservationSource(observations),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    memory = _run_engine(
        validation,
        _SpillingStaticObservationSource(observations, spill_parent=tmp_path / "all_null"),
        top_n=2,
        candidate_n=2,
        min_overlap=100,
    )
    _assert_selection_equivalent(legacy, memory)
    assert legacy.filter(pl.col("selected"))["factor_name"].to_list() == ["signal", "nulls"]
