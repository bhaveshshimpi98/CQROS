"""Unit tests for the purged-CV evaluation CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl
import pytest

from cqros.cli import evaluate_purged_cv as cli
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
from cqros.purged_cv.engine import SimplePurgedCVEngine
from cqros.purged_cv.evaluation_repository import PurgedCVEvaluationRepository
from cqros.purged_cv.repository import PurgedCVRepository
from cqros.storage import LabelRepository, StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward.evaluation_input import WalkForwardInputBuilder
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

    def write(self, path: FilePath, frame: pl.DataFrame) -> None:
        target = Path(path)
        self.frames[target] = frame.clone()
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)

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


def argparse_namespace(**kwargs: object) -> object:
    """Build a simple namespace-like object for build_options."""

    class _Namespace:
        pass

    namespace = _Namespace()
    for key, value in kwargs.items():
        setattr(namespace, key, value)
    return namespace


def _seed_panel(tmp_path: Path) -> tuple[StorageLayout, _InMemoryDataStore]:
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(40)]

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
            "symbol": [_SYMBOL] * 40,
            "timeframe": [_TIMEFRAME] * 40,
            "open_time": open_times,
            "factor_name": [_FACTOR] * 40,
            "factor_version": [_FACTOR_VERSION] * 40,
            "factor_category": ["price"] * 40,
            "factor_group": ["alpha"] * 40,
            "factor_value": [float(index + 1) for index in range(40)],
            "lookback": [20] * 40,
            "prediction_horizon": [1] * 40,
            "enabled": [True] * 40,
            "status": [FactorStatus.ACTIVE.value] * 40,
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
            "symbol": [_SYMBOL] * 40,
            "timeframe": [_TIMEFRAME] * 40,
            "open_time": open_times,
            "future_return_1": [0.01 * float(index + 1) for index in range(40)],
            "future_return_5": [0.05] * 40,
            "future_return_10": [0.10] * 40,
            "future_return_20": [0.20] * 40,
            "direction_1": [1] * 40,
            "direction_5": [1] * 40,
            "direction_10": [0] * 40,
            "direction_20": [0] * 40,
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

    walk_forward_rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        walk_forward_rows.append(
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "timeframe": _TIMEFRAME,
                "fold_id": index + 1,
                "train_start": open_time - 10 * 3_600_000,
                "train_end": open_time - 3_600_000,
                "test_start": open_time,
                "test_end": open_time + 2 * 3_600_000,
                "train_rows": 10,
                "test_rows": 3,
                "selected_factors": 1,
                "model_version": "v1",
                "train_score": 0.01 * float(index + 1),
                "test_score": 0.005 * float(index + 1),
                "overfit_gap": 0.005 * float(index + 1),
                "status": WalkForwardStatus.PASS.value,
            }
        )
    ledger = pl.DataFrame(walk_forward_rows, schema=dict(WALK_FORWARD_DTYPES)).select(
        list(WALK_FORWARD_COLUMNS)
    )
    WalkForwardRepository(layout, datastore).save(
        ledger,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    purged = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1).build(ledger)
    PurgedCVRepository(layout, datastore).save(
        purged,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    return layout, datastore


def test_build_options_and_discover_work(tmp_path: Path) -> None:
    """CLI discovers existing Purged-CV ledger partitions only."""
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
            output=tmp_path / "reports",
        )
    )
    work = cli.discover_work(PurgedCVRepository(layout, datastore), options)
    assert len(work) == 1
    assert work[0].timeframe == _TIMEFRAME
    assert work[0].years == (_YEAR,)


def test_cli_does_not_hardcode_timeframes() -> None:
    """Discovery uses repository artifacts rather than a fixed timeframe list."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert 'timeframes=("5m", "15m", "1h", "4h", "1d")' not in source
    assert "SUPPORTED_TIMEFRAMES" in source


def test_run_evaluation_writes_artifact_and_preserves_ledger(tmp_path: Path) -> None:
    """Evaluation CLI persists evaluation results without mutating purged_cv."""
    layout, datastore = _seed_panel(tmp_path)
    ledger_path = layout.purged_cv_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        _TIMEFRAME,
        _YEAR,
    )
    before = ledger_path.read_bytes()
    options = cli.PurgedCVEvaluationOptions(
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
    purged_cv_repository = PurgedCVRepository(layout, datastore)
    walk_forward_repository = WalkForwardRepository(layout, datastore)
    evaluation_repository = PurgedCVEvaluationRepository(layout, datastore)
    factor_selection_repository = FactorSelectionRepository(layout, datastore)
    builder = WalkForwardInputBuilder(
        FactorsRepository(layout, datastore),
        LabelRepository(layout, datastore),
    )
    work = cli.discover_work(purged_cv_repository, options)
    summary = asyncio.run(
        cli.run_evaluation(
            purged_cv_repository=purged_cv_repository,
            walk_forward_repository=walk_forward_repository,
            evaluation_repository=evaluation_repository,
            factor_selection_repository=factor_selection_repository,
            walk_forward_input_builder=builder,
            options=options,
            work=work,
        )
    )
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 0
    assert summary.ledger_hashes_unchanged is True
    assert ledger_path.read_bytes() == before
    assert evaluation_repository.exists(
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert (tmp_path / "reports" / "purged_cv_evaluation_all.csv").is_file()
    assert (tmp_path / "reports" / "purged_cv_evaluation_folds.csv").is_file()
    assert (tmp_path / "reports" / "purged_cv_evaluation_factors.csv").is_file()
    assert (tmp_path / "reports" / "purged_cv_evaluation_global.csv").is_file()


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
                output=tmp_path / "reports",
            )
        )


def test_main_rejects_unsupported_timeframe(tmp_path: Path) -> None:
    """CLI rejects unsupported timeframe filters."""
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        cli.build_options(
            argparse_namespace(
                manager=_MANAGER,
                engine=_ENGINE,
                timeframes=["99z"],
                years=None,
                overwrite=False,
                workers=1,
                verbose=False,
                debug=False,
                storage_root=tmp_path,
                output=tmp_path / "reports",
            )
        )


def test_help_exposes_expected_flags() -> None:
    """Parser help documents the required CLI surface."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    for flag in (
        "--manager",
        "--timeframes",
        "--years",
        "--workers",
        "--verbose",
        "--debug",
        "--storage-root",
        "--output",
    ):
        assert flag in help_text
