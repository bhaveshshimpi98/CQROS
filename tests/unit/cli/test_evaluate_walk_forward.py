"""Unit tests for the walk-forward evaluation CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl
import pytest

from cqros.cli import evaluate_walk_forward as cli
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.exceptions import ValidationError
from cqros.core.types import FilePath
from cqros.factor_selection.repository import FactorSelectionRepository
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_SELECTION_COLUMNS,
)
from cqros.factor_selection.schema import (
    COLUMN_DTYPES as FACTOR_SELECTION_DTYPES,
)
from cqros.factor_selection.schema import FactorSelectionStatus
from cqros.factors import FactorsRepository, FactorStatus
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER,
)
from cqros.factors.schema import COLUMN_DTYPES as FACTOR_COLUMN_DTYPES
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER as LABEL_CANONICAL_COLUMN_ORDER,
)
from cqros.labels.schema import COLUMN_DTYPES as LABEL_COLUMN_DTYPES
from cqros.storage import LabelRepository, StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward.evaluation_input import WalkForwardInputBuilder
from cqros.walk_forward.evaluation_repository import WalkForwardEvaluationRepository
from cqros.walk_forward.repository import WalkForwardRepository
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER as WALK_FORWARD_COLUMNS,
)
from cqros.walk_forward.schema import (
    COLUMN_DTYPES as WALK_FORWARD_DTYPES,
)
from cqros.walk_forward.schema import WalkForwardStatus

_MANAGER = "default"
_ENGINE = "simple"
_TIMEFRAME = "1h"
_YEAR = 2026
_SYMBOL = "BTCUSDT"
_FACTOR = "momentum"
_FACTOR_VERSION = "1.0.0"


class _InMemoryDataStore:
    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.raw: dict[Path, bytes] = {}

    def write(self, path: FilePath, frame: pl.DataFrame) -> None:
        target = Path(path)
        self.frames[target] = frame.clone()
        # Simulate parquet bytes for hash checks when path is used on disk.
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        self.raw[target] = target.read_bytes()

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        try:
            return self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        return Path(path) in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


def _seed_panel(tmp_path: Path) -> tuple[StorageLayout, _InMemoryDataStore]:
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(8)]

    selection = pl.DataFrame(
        {
            "factor_name": [_FACTOR],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [_TIMEFRAME],
            "selection_time": [open_times[0]],
            "factor_category": ["price"],
            "selected": [True],
            "selection_score": [0.1],
            "selection_rank": [1],
            "selection_reason": ["test"],
            "selection_ic": [0.08],
            "selected_direction": [1],
            "orientation_policy": ["signed_ic_v1"],
            "status": [FactorSelectionStatus.SELECTED.value],
        },
        schema=dict(FACTOR_SELECTION_DTYPES),
    ).select(list(FACTOR_SELECTION_COLUMNS))
    FactorSelectionRepository(layout, datastore).save(
        selection,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    factors = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 8,
            "timeframe": [_TIMEFRAME] * 8,
            "open_time": open_times,
            "factor_name": [_FACTOR] * 8,
            "factor_version": [_FACTOR_VERSION] * 8,
            "factor_category": ["price"] * 8,
            "factor_group": ["alpha"] * 8,
            "factor_value": [float(index + 1) for index in range(8)],
            "lookback": [20] * 8,
            "prediction_horizon": [1] * 8,
            "enabled": [True] * 8,
            "status": [FactorStatus.ACTIVE.value] * 8,
        },
        schema=dict(FACTOR_COLUMN_DTYPES),
    ).select(list(FACTOR_CANONICAL_COLUMN_ORDER))
    FactorsRepository(layout, datastore).save(
        factors,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    labels = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 8,
            "timeframe": [_TIMEFRAME] * 8,
            "open_time": open_times,
            "future_return_1": [0.01 * float(index + 1) for index in range(8)],
            "future_return_5": [0.05] * 8,
            "future_return_10": [0.10] * 8,
            "future_return_20": [0.20] * 8,
            "direction_1": [1] * 8,
            "direction_5": [1] * 8,
            "direction_10": [0] * 8,
            "direction_20": [0] * 8,
        },
        schema=dict(LABEL_COLUMN_DTYPES),
    ).select(list(LABEL_CANONICAL_COLUMN_ORDER))
    LabelRepository(layout, datastore).save(
        labels,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    ledger = pl.DataFrame(
        {
            "strategy_name": ["default_strategy"],
            "strategy_version": ["v1"],
            "timeframe": [_TIMEFRAME],
            "fold_id": [1],
            "train_start": [open_times[0]],
            "train_end": [open_times[3]],
            "test_start": [open_times[4]],
            "test_end": [open_times[5]],
            "train_rows": [4],
            "test_rows": [2],
            "selected_factors": [1],
            "model_version": ["v1"],
            "train_score": [0.01],
            "test_score": [0.5],
            "overfit_gap": [0.5],
            "status": [WalkForwardStatus.PASS.value],
        },
        schema=dict(WALK_FORWARD_DTYPES),
    ).select(list(WALK_FORWARD_COLUMNS))
    WalkForwardRepository(layout, datastore).save(
        ledger,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    return layout, datastore


def test_build_options_and_discover_work(tmp_path: Path) -> None:
    """CLI discovers existing Walk-Forward ledger partitions only."""
    layout, datastore = _seed_panel(tmp_path)
    options = cli.build_options(
        argparse_namespace(
            manager=_MANAGER,
            engine=_ENGINE,
            timeframes=None,
            years=None,
            overwrite=True,
            workers=1,
            verbose=False,
            debug=False,
            storage_root=tmp_path,
            report_output=tmp_path / "reports",
        )
    )
    work = cli.discover_work(WalkForwardRepository(layout, datastore), options)
    assert len(work) == 1
    assert work[0].timeframe == _TIMEFRAME
    assert work[0].years == (_YEAR,)


def test_run_evaluation_writes_artifact_and_preserves_ledger(tmp_path: Path) -> None:
    """Evaluation CLI persists evaluation results without mutating the ledger."""
    layout, datastore = _seed_panel(tmp_path)
    ledger_path = layout.walk_forward_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    before = ledger_path.read_bytes()
    options = cli.WalkForwardEvaluationOptions(
        storage_root=tmp_path,
        manager=_MANAGER,
        engine=_ENGINE,
        timeframes=None,
        years=None,
        overwrite=True,
        workers=1,
        verbose=False,
        debug=False,
        report_output=tmp_path / "reports",
    )
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    evaluation_repository = WalkForwardEvaluationRepository(layout, datastore)
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    builder = WalkForwardInputBuilder(
        FactorsRepository(layout, datastore),
        LabelRepository(layout, datastore),
    )
    work = cli.discover_work(walk_forward_repository, options)
    summary = asyncio.run(
        cli.run_evaluation(
            walk_forward_repository=walk_forward_repository,
            evaluation_repository=evaluation_repository,
            factor_selection_repository=factor_selection_repository,
            walk_forward_input_builder=builder,
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.ledger_hashes_unchanged is True
    assert ledger_path.read_bytes() == before
    assert evaluation_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert (tmp_path / "reports" / "walk_forward_evaluation_all.csv").is_file()


def argparse_namespace(**kwargs: object) -> object:
    """Build a simple namespace-like object for build_options."""

    class _Namespace:
        pass

    namespace = _Namespace()
    for key, value in kwargs.items():
        setattr(namespace, key, value)
    return namespace


def test_main_rejects_invalid_workers(tmp_path: Path) -> None:
    """CLI rejects non-positive worker counts."""
    with pytest.raises(ValidationError):
        cli.build_options(
            argparse_namespace(
                manager=_MANAGER,
                engine=_ENGINE,
                timeframes=None,
                years=None,
                overwrite=False,
                workers=0,
                verbose=False,
                debug=False,
                storage_root=tmp_path,
                report_output=tmp_path / "reports",
            )
        )
