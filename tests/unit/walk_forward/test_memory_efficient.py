"""Equivalence and lifecycle tests for bounded-memory Walk Forward execution."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER as SELECTION_COLUMNS,
)
from cqros.factor_selection.schema import COLUMN_DTYPES as SELECTION_DTYPES
from cqros.factor_selection.schema import FactorSelectionStatus
from cqros.factors import FactorsRepository, FactorStatus
from cqros.factors.schema import CANONICAL_COLUMN_ORDER as FACTOR_COLUMNS
from cqros.factors.schema import COLUMN_DTYPES as FACTOR_DTYPES
from cqros.labels.schema import CANONICAL_COLUMN_ORDER as LABEL_COLUMNS
from cqros.labels.schema import COLUMN_DTYPES as LABEL_DTYPES
from cqros.storage import LabelRepository, ParquetStore, StorageLayout
from cqros.walk_forward import (
    MemoryEfficientExecutionConfig,
    MemoryEfficientWalkForwardExecutor,
    SimpleWalkForwardEngine,
    WalkForwardError,
    WalkForwardInputBuilder,
    assert_walk_forward_equivalent,
)

_MANAGER = "default"
_TIMEFRAME = "1h"
_YEAR = 2026
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_FACTORS = ("momentum", "value")
_VERSION = "1.0.0"
_START = 1_700_000_000_000


def _selection() -> pl.DataFrame:
    """Return selected and rejected decisions for cross-factor geometry."""
    return pl.DataFrame(
        {
            "factor_name": list(_FACTORS),
            "factor_version": [_VERSION, _VERSION],
            "timeframe": [_TIMEFRAME, _TIMEFRAME],
            "selection_time": [_START, _START],
            "factor_category": ["price", "price"],
            "selected": [True, False],
            "selection_score": [0.2, 0.1],
            "selection_rank": [1, 2],
            "selection_reason": ["fixture", "fixture"],
            "selection_ic": [0.08, -0.04],
            "selected_direction": [1, -1],
            "orientation_policy": [
                FACTOR_ORIENTATION_POLICY,
                FACTOR_ORIENTATION_POLICY,
            ],
            "status": [
                FactorSelectionStatus.SELECTED.value,
                FactorSelectionStatus.REJECTED.value,
            ],
        },
        schema=dict(SELECTION_DTYPES),
    ).select(list(SELECTION_COLUMNS))


def _factors(symbol: str, count: int = 5) -> pl.DataFrame:
    """Return repeated-timestamp rows for two factor identities."""
    times = [_START + index * 3_600_000 for index in range(count)]
    rows: list[dict[str, object]] = []
    for open_time in reversed(times):
        for factor_name in reversed(_FACTORS):
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": _TIMEFRAME,
                    "open_time": open_time,
                    "factor_name": factor_name,
                    "factor_version": _VERSION,
                    "factor_category": "price",
                    "factor_group": "alpha",
                    "factor_value": float(open_time % 17),
                    "lookback": 2,
                    "prediction_horizon": 1,
                    "enabled": True,
                    "status": FactorStatus.ACTIVE.value,
                }
            )
    return pl.DataFrame(rows, schema=dict(FACTOR_DTYPES)).select(list(FACTOR_COLUMNS))


def _labels(symbol: str, *, duplicate: bool = False, count: int = 5) -> pl.DataFrame:
    """Return aligned labels, optionally with a duplicate primary key."""
    times = [_START + index * 3_600_000 for index in range(count)]
    if duplicate:
        times.append(times[0])
    row_count = len(times)
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": times,
            "future_return_1": [((index % 3) - 1) * 0.01 for index in range(row_count)],
            "future_return_5": [0.0] * row_count,
            "future_return_10": [0.0] * row_count,
            "future_return_20": [0.0] * row_count,
            "direction_1": [0] * row_count,
            "direction_5": [0] * row_count,
            "direction_10": [0] * row_count,
            "direction_20": [0] * row_count,
        },
        schema=dict(LABEL_DTYPES),
    ).select(list(LABEL_COLUMNS))


def _seed(
    root: Path,
    *,
    symbols: tuple[str, ...] = _SYMBOLS,
    duplicate_symbol: str | None = None,
    count: int = 5,
) -> tuple[
    StorageLayout,
    FactorsRepository,
    LabelRepository,
    WalkForwardInputBuilder,
]:
    """Persist deterministic multi-symbol Factors and Labels fixtures."""
    layout = StorageLayout(root)
    store = ParquetStore()
    factors_repository = FactorsRepository(layout, store)
    label_repository = LabelRepository(layout, store)
    for symbol in symbols:
        factors_repository.save(
            _factors(symbol, count=count),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        label_repository.save(
            _labels(symbol, duplicate=symbol == duplicate_symbol, count=count),
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    return (
        layout,
        factors_repository,
        label_repository,
        WalkForwardInputBuilder(factors_repository, label_repository),
    )


def _executor(
    layout: StorageLayout,
    factors: FactorsRepository,
    labels: LabelRepository,
    spill_parent: Path,
) -> MemoryEfficientWalkForwardExecutor:
    """Compose compact-window bounded execution for tests."""
    return MemoryEfficientWalkForwardExecutor(
        layout,
        factors,
        labels,
        SimpleWalkForwardEngine(train_window=4, test_window=2, step_size=2),
        MemoryEfficientExecutionConfig(
            spill_parent=spill_parent,
            memory_budget_mb=1,
        ),
    )


def _multi_part_executor(
    layout: StorageLayout,
    factors: FactorsRepository,
    labels: LabelRepository,
    spill_parent: Path,
) -> MemoryEfficientWalkForwardExecutor:
    """Compose a bounded executor whose panel spills several raw-fold parts."""
    return MemoryEfficientWalkForwardExecutor(
        layout,
        factors,
        labels,
        SimpleWalkForwardEngine(train_window=4, test_window=2, step_size=1),
        MemoryEfficientExecutionConfig(
            spill_parent=spill_parent,
            memory_budget_mb=1,
        ),
    )


def test_memory_efficient_exactly_matches_full_panel_across_symbols_and_factors(
    tmp_path: Path,
) -> None:
    """Tie order, alignment, folds, scores, and schema match exactly."""
    layout, factors, labels, builder = _seed(tmp_path / "data")
    selection = _selection()
    canonical = SimpleWalkForwardEngine(
        train_window=4,
        test_window=2,
        step_size=2,
    ).build(
        builder.build(
            selection,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    )
    spill_parent = tmp_path / "spill"
    bounded = _executor(layout, factors, labels, spill_parent).execute(
        selection,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert_walk_forward_equivalent(canonical, bounded)
    assert not spill_parent.exists() or not any(spill_parent.iterdir())

    canonical_path = tmp_path / "canonical.parquet"
    bounded_path = tmp_path / "bounded.parquet"
    ParquetStore().write(canonical_path, canonical)
    ParquetStore().write(bounded_path, bounded)
    assert (
        hashlib.sha256(canonical_path.read_bytes()).digest()
        == hashlib.sha256(bounded_path.read_bytes()).digest()
    )


def test_spill_is_cleaned_after_failure(tmp_path: Path) -> None:
    """Duplicate-key failure leaves no run-scoped spill evidence."""
    layout, factors, labels, _ = _seed(
        tmp_path / "data",
        symbols=(_SYMBOLS[0],),
        duplicate_symbol=_SYMBOLS[0],
    )
    spill_parent = tmp_path / "spill"
    with pytest.raises(WalkForwardError) as exc_info:
        _executor(layout, factors, labels, spill_parent).execute(
            _selection(),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    assert exc_info.value.error_code == "WF_EVAL_DUPLICATE_KEYS"
    assert not spill_parent.exists() or not any(spill_parent.iterdir())


def test_symbol_count_does_not_change_configured_retention_unit(
    tmp_path: Path,
) -> None:
    """A larger symbol fixture remains equivalent under the same tiny budget."""
    layout, factors, labels, builder = _seed(tmp_path / "data")
    selection = _selection()
    engine = SimpleWalkForwardEngine(train_window=4, test_window=2, step_size=2)
    expected = engine.build(
        builder.build(
            selection,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    )
    executor = _executor(
        layout,
        factors,
        labels,
        tmp_path / "spill",
    )
    actual = executor.execute(
        selection,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_walk_forward_equivalent(expected, actual)
    repeated = executor.execute(
        selection,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_walk_forward_equivalent(actual, repeated)


def test_multi_part_spill_is_bit_exact_vs_full_panel(tmp_path: Path) -> None:
    """Multi-part raw-fold spills reproduce canonical floating-point bits."""
    layout, factors, labels, builder = _seed(
        tmp_path / "data",
        count=5500,
    )
    selection = _selection()
    engine = SimpleWalkForwardEngine(train_window=4, test_window=2, step_size=1)
    expected = engine.build(
        builder.build(
            selection,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    )
    spill_parent = tmp_path / "spill"
    executor = _multi_part_executor(layout, factors, labels, spill_parent)
    actual = executor.execute(
        selection,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert actual.height > 2 * 8192
    assert_walk_forward_equivalent(expected, actual)
    assert not spill_parent.exists() or not any(spill_parent.iterdir())


def test_bounded_replay_materializes_only_aggregate_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded tail reads aggregate columns and streams the output."""
    layout, factors, labels, _ = _seed(
        tmp_path / "data",
        count=5500,
    )
    import cqros.walk_forward.memory_efficient as memory_efficient

    aggregate_inputs: list[tuple[str, ...]] = []
    real_aggregate = memory_efficient._compute_aggregate_metrics_frame

    def spy_aggregate(folds: pl.DataFrame) -> object:
        aggregate_inputs.append(tuple(folds.columns))
        assert set(folds.columns) == set(memory_efficient._AGGREGATE_COLUMNS)
        return real_aggregate(folds)

    monkeypatch.setattr(memory_efficient, "_compute_aggregate_metrics_frame", spy_aggregate)

    multi_part_reads: list[tuple[object, ...]] = []
    real_read_parquet = pl.read_parquet

    def spy_read_parquet(*args: object, **kwargs: object) -> object:
        source = args[0] if args else kwargs.get("source")
        if isinstance(source, (list, tuple)):
            multi_part_reads.append(args)
            columns = kwargs.get("columns")
            assert columns is not None and set(columns) <= set(memory_efficient._AGGREGATE_COLUMNS)
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(memory_efficient.pl, "read_parquet", spy_read_parquet)

    collect_engines: list[object] = []
    real_collect = pl.LazyFrame.collect

    def spy_collect(
        self: pl.LazyFrame,
        *args: object,
        **kwargs: object,
    ) -> pl.DataFrame:
        collect_engines.append(kwargs.get("engine"))
        return real_collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", spy_collect)

    output = _multi_part_executor(
        layout,
        factors,
        labels,
        tmp_path / "spill",
    ).execute(
        _selection(),
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert output.height > 2 * 8192
    assert aggregate_inputs
    assert multi_part_reads
    assert "streaming" in collect_engines


def test_peak_rss_is_bounded_for_multi_part_panel(tmp_path: Path) -> None:
    """A multi-part panel executes far below the eager-materialization ceiling."""
    test_dir = Path(__file__).resolve().parent
    src_dir = Path(__file__).resolve().parents[3] / "SRC"
    script = f"""
import pathlib
import resource
import sys
import tempfile

sys.path.insert(0, {str(test_dir)!r})
sys.path.insert(0, {str(src_dir)!r})

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.walk_forward import (
    MemoryEfficientExecutionConfig,
    MemoryEfficientWalkForwardExecutor,
    SimpleWalkForwardEngine,
)

from test_memory_efficient import _MANAGER, _TIMEFRAME, _YEAR, _seed, _selection

layout, factors, labels, _ = _seed(
    pathlib.Path(tempfile.mkdtemp()) / "data",
    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    count=5500,
)
executor = MemoryEfficientWalkForwardExecutor(
    layout,
    factors,
    labels,
    SimpleWalkForwardEngine(train_window=4, test_window=2, step_size=1),
    MemoryEfficientExecutionConfig(
        spill_parent=pathlib.Path(tempfile.mkdtemp()) / "spill",
        memory_budget_mb=1,
    ),
)
output = executor.execute(
    _selection(),
    manager=_MANAGER,
    exchange=EXCHANGE_BINANCE,
    market=MARKET_USDT_PERPETUAL,
    timeframe=_TIMEFRAME,
    year=_YEAR,
)
print("rows=", output.height)
print("peak_mb=", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    peak_line = next(line for line in result.stdout.splitlines() if line.startswith("peak_mb="))
    peak_mb = float(peak_line.split("=", 1)[1])
    assert peak_mb < 1200.0
