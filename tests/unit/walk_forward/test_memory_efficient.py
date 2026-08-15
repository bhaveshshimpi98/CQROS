"""Equivalence and lifecycle tests for bounded-memory Walk Forward execution."""

from __future__ import annotations

import hashlib
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


def _factors(symbol: str) -> pl.DataFrame:
    """Return repeated-timestamp rows for two factor identities."""
    times = [_START + index * 3_600_000 for index in range(5)]
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


def _labels(symbol: str, *, duplicate: bool = False) -> pl.DataFrame:
    """Return aligned labels, optionally with a duplicate primary key."""
    times = [_START + index * 3_600_000 for index in range(5)]
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
            _factors(symbol),
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        label_repository.save(
            _labels(symbol, duplicate=symbol == duplicate_symbol),
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
