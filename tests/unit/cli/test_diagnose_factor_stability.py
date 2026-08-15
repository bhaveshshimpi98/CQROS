"""Unit tests for the factor-stability diagnostic CLI."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import polars as pl
import pytest

from cqros.cli import diagnose_factor_stability as cli
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
from cqros.factor_validation.repository import FactorValidationRepository
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_VALIDATION_COLUMNS,
)
from cqros.factor_validation.schema import (
    COLUMN_DTYPES as FACTOR_VALIDATION_DTYPES,
)
from cqros.factor_validation.schema import FactorValidationStatus
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
from cqros.purged_cv.repository import PurgedCVRepository
from cqros.reporting.factor_stability_diagnostic import (
    GLOBAL_CSV_NAME,
    STABILITY_ALL_CSV_NAME,
)
from cqros.storage import LabelRepository, StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward.repository import WalkForwardRepository
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER as WALK_FORWARD_COLUMNS,
)
from cqros.walk_forward.schema import (
    COLUMN_DTYPES as WALK_FORWARD_DTYPES,
)
from cqros.walk_forward.schema import WalkForwardStatus

_MANAGER = "default"
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


def _seed_panel(tmp_path: Path, *, timeframe: str = _TIMEFRAME) -> tuple[StorageLayout, Path]:
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    open_times = [1_700_000_000_000 + index * 3_600_000 for index in range(40)]

    selection = pl.DataFrame(
        {
            "factor_name": [_FACTOR],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [timeframe],
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
        timeframe=timeframe,
        year=_YEAR,
    )

    validation = pl.DataFrame(
        {
            "factor_name": [_FACTOR],
            "factor_version": [_FACTOR_VERSION],
            "timeframe": [timeframe],
            "validation_time": [open_times[-1]],
            "factor_category": ["price"],
            "dataset_version": ["v1"],
            "label_version": ["v1"],
            "validation_start_time": [open_times[0]],
            "validation_end_time": [open_times[-1]],
            "information_coefficient": [0.2],
            "rank_information_coefficient": [0.25],
            "ic_information_ratio": [1.0],
            "ic_std": [0.1],
            "ic_p_value": [0.01],
            "ic_t_stat": [2.0],
            "ic_decay": [0.9],
            "turnover": [0.1],
            "monotonicity_score": [0.5],
            "quantile_spread": [0.01],
            "observations": [40],
            "ic_observations": [20],
            "status": [FactorValidationStatus.PASS.value],
        },
        schema=dict(FACTOR_VALIDATION_DTYPES),
    ).select(list(FACTOR_VALIDATION_COLUMNS))
    FactorValidationRepository(layout, datastore).save(
        validation,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=_YEAR,
    )

    factors = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 40,
            "timeframe": [timeframe] * 40,
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
        timeframe=timeframe,
        year=_YEAR,
    )

    labels = pl.DataFrame(
        {
            "symbol": [_SYMBOL] * 40,
            "timeframe": [timeframe] * 40,
            "open_time": open_times,
            "future_return_1": [0.01 * ((-1) ** index) for index in range(40)],
            "future_return_5": [0.0] * 40,
            "future_return_10": [0.0] * 40,
            "future_return_20": [0.0] * 40,
            "direction_1": [1 if index % 2 == 0 else 0 for index in range(40)],
            "direction_5": [0] * 40,
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
        timeframe=timeframe,
        year=_YEAR,
    )

    walk_forward_rows: list[dict[str, object]] = []
    for index, open_time in enumerate(open_times):
        walk_forward_rows.append(
            {
                "strategy_name": "default_strategy",
                "strategy_version": "v1",
                "timeframe": timeframe,
                "fold_id": index + 1,
                "train_start": open_time - 10 * 3_600_000,
                "train_end": open_time - 3_600_000,
                "test_start": open_time,
                "test_end": open_time + 2 * 3_600_000,
                "train_rows": 10,
                "test_rows": 3,
                "selected_factors": 1,
                "model_version": "v1",
                "train_score": 0.01,
                "test_score": 0.005,
                "overfit_gap": 0.005,
                "status": WalkForwardStatus.PASS.value,
            }
        )
    walk_forward = pl.DataFrame(walk_forward_rows, schema=dict(WALK_FORWARD_DTYPES)).select(
        list(WALK_FORWARD_COLUMNS)
    )
    WalkForwardRepository(layout, datastore).save(
        walk_forward,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=_YEAR,
    )
    purged_cv = SimplePurgedCVEngine(n_folds=5, purge_size=2, embargo_size=1).build(walk_forward)
    PurgedCVRepository(layout, datastore).save(
        purged_cv,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=_YEAR,
    )
    ledger_path = layout.purged_cv_path(
        _MANAGER,
        EXCHANGE_BINANCE,
        MARKET_USDT_PERPETUAL,
        timeframe,
        _YEAR,
    )
    return layout, ledger_path


def test_build_parser_help() -> None:
    """CLI help exposes required flags."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "--manager" in help_text
    assert "--timeframes" in help_text
    assert "--years" in help_text
    assert "--workers" in help_text
    assert "--storage-root" in help_text
    assert "--output" in help_text


def test_build_options_validation() -> None:
    """Invalid workers / manager / timeframe raise ValidationError."""
    with pytest.raises(ValidationError):
        cli.build_options(
            argparse_namespace(
                manager="",
                timeframes=None,
                years=None,
                workers=1,
                verbose=False,
                debug=False,
                storage_root=None,
                output=Path("reports/purged_cv"),
            )
        )
    with pytest.raises(ValidationError):
        cli.build_options(
            argparse_namespace(
                manager=_MANAGER,
                timeframes=None,
                years=None,
                workers=0,
                verbose=False,
                debug=False,
                storage_root=None,
                output=Path("reports/purged_cv"),
            )
        )
    with pytest.raises(ValidationError):
        cli.build_options(
            argparse_namespace(
                manager=_MANAGER,
                timeframes=["2m"],
                years=None,
                workers=1,
                verbose=False,
                debug=False,
                storage_root=None,
                output=Path("reports/purged_cv"),
            )
        )


def test_cli_discovery(tmp_path: Path) -> None:
    """Discovery finds seeded purged-CV timeframes only."""
    _seed_panel(tmp_path, timeframe="1h")
    _seed_panel(tmp_path, timeframe="4h")
    options = cli.build_options(
        argparse_namespace(
            manager=_MANAGER,
            timeframes=None,
            years=None,
            workers=1,
            verbose=False,
            debug=False,
            storage_root=tmp_path,
            output=tmp_path / "reports",
        )
    )
    repository = PurgedCVRepository(StorageLayout(tmp_path), cli.ParquetStore())
    work = cli.discover_work(repository, options)
    timeframes = {item.timeframe for item in work}
    assert timeframes == {"1h", "4h"}


def test_cli_filtering(tmp_path: Path) -> None:
    """Timeframe/year filters restrict discovered work."""
    _seed_panel(tmp_path, timeframe="1h")
    _seed_panel(tmp_path, timeframe="4h")
    options = cli.build_options(
        argparse_namespace(
            manager=_MANAGER,
            timeframes=["1h"],
            years=["2026"],
            workers=1,
            verbose=False,
            debug=False,
            storage_root=tmp_path,
            output=tmp_path / "reports",
        )
    )
    repository = PurgedCVRepository(StorageLayout(tmp_path), cli.ParquetStore())
    work = cli.discover_work(repository, options)
    assert len(work) == 1
    assert work[0].timeframe == "1h"
    assert work[0].years == (2026,)


def test_cli_run_diagnostic_and_immutability(tmp_path: Path) -> None:
    """End-to-end diagnostic writes reports and leaves ledgers unchanged."""
    _, ledger_path = _seed_panel(tmp_path)
    before = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    selection_root = tmp_path / "factor_selection"
    walk_forward_root = tmp_path / "walk_forward"
    selection_before = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in selection_root.rglob("*.parquet")
    }
    walk_forward_before = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in walk_forward_root.rglob("*.parquet")
    }
    options = cli.build_options(
        argparse_namespace(
            manager=_MANAGER,
            timeframes=["1h"],
            years=["2026"],
            workers=2,
            verbose=False,
            debug=True,
            storage_root=tmp_path,
            output=tmp_path / "reports",
        )
    )
    summary = asyncio.run(cli.run_diagnostic(options))
    after = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert before == after
    assert summary.ledger_hashes_unchanged is True
    assert summary.folds == 5
    assert (tmp_path / "reports" / STABILITY_ALL_CSV_NAME).exists()
    assert (tmp_path / "reports" / GLOBAL_CSV_NAME).exists()
    selection_after = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in selection_root.rglob("*.parquet")
    }
    walk_forward_after = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in walk_forward_root.rglob("*.parquet")
    }
    assert selection_before == selection_after
    assert walk_forward_before == walk_forward_after
    global_frame = pl.read_csv(tmp_path / "reports" / GLOBAL_CSV_NAME)
    metrics = {row["metric"]: row["value"] for row in global_frame.iter_rows(named=True)}
    assert metrics["q1_future_return_1_aligned"] == "YES"
    assert metrics["q2_factor_values_preserved"] == "YES"
    assert metrics["q8_methodology_aligned"] == "YES"
