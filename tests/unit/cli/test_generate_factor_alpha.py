"""Unit tests for CQROS Alpha generation CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.alpha.schema import ALPHA_SCHEMA, AlphaStatus
from cqros.cli.generate_factor_alpha import (
    AlphaGenerationOptions,
    AlphaGenerationSummary,
    AlphaTaskResult,
    DiscoveredWorkItem,
    build_options,
    build_parser,
    discover_work,
    format_summary,
    main,
    run_generation,
)
from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_ALPHA,
)
from cqros.core.exceptions import ValidationError
from cqros.factor_orthogonalization.repository import FactorOrthogonalizationPartitionRef
from cqros.factor_orthogonalization.schema import (
    FACTOR_ORTHOGONALIZATION_SCHEMA,
    FactorOrthogonalizationStatus,
)
from cqros.factors.schema import FACTOR_SCHEMA, FactorStatus
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_SYMBOL_ETH = "ETHUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIME_1 = 1_700_000_000_000
_OPEN_TIME_2 = 1_700_000_003_600_000
_VALIDATION_START = _OPEN_TIME_1
_VALIDATION_END = _OPEN_TIME_2


def _run[T](coro: Coroutine[object, object, T]) -> T:
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _options(
    *,
    storage_root: Path,
    manager: str = _MANAGER,
    years: tuple[int, ...] | None = None,
    symbols: tuple[str, ...] | None = None,
    overwrite: bool = False,
    export_detailed_csv: bool = False,
    workers: int = 1,
    verbose: bool = False,
    debug: bool = False,
) -> AlphaGenerationOptions:
    """Build AlphaGenerationOptions against a temporary storage root."""
    return AlphaGenerationOptions(
        storage_root=storage_root,
        manager=manager,
        years=years,
        symbols=symbols,
        overwrite=overwrite,
        export_detailed_csv=export_detailed_csv,
        workers=workers,
        verbose=verbose,
        debug=debug,
    )


def _fo_partition_ref(
    *,
    manager: str = _MANAGER,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> FactorOrthogonalizationPartitionRef:
    """Build one FO partition reference."""
    return FactorOrthogonalizationPartitionRef(
        manager=manager,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=timeframe,
        year=year,
    )


def _canonical_fo_frame(
    *,
    combination_ids: Sequence[str] = ("ema_distance|sma_distance",),
    selected: Sequence[bool] = (True,),
    statuses: Sequence[str] = (FactorOrthogonalizationStatus.PASS.value,),
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    """Build a schema-conformant Factor Orthogonalization frame."""
    rows = len(combination_ids)
    frame = pl.DataFrame(
        {
            "combination_id": list(combination_ids),
            "factor_names": [["ema_distance", "sma_distance"]] * rows,
            "factor_versions": [["1.0.0", "1.0.0"]] * rows,
            "factor_categories": [["price", "price"]] * rows,
            "timeframe": [timeframe] * rows,
            "combination_size": [2] * rows,
            "combination_method": ["equal_weight"] * rows,
            "orthogonalization_method": ["greedy_correlation"] * rows,
            "orthogonalization_version": ["1.0"] * rows,
            "analysis_time": [_OPEN_TIME_1] * rows,
            "source_combination_rank": list(range(1, rows + 1)),
            "source_combination_score": [1.0] * rows,
            "source_stability_score": [1.0] * rows,
            "source_confidence_score": [1.0] * rows,
            "correlation_score": [0.1] * rows,
            "vif_score": [1.0] * rows,
            "redundancy_score": [0.1] * rows,
            "orthogonality_score": [0.9] * rows,
            "information_retained": [1.0] * rows,
            "correlation_overlap": [10] * rows,
            "correlation_threshold": [0.7] * rows,
            "min_overlap_threshold": [5] * rows,
            "redundancy_checked": [True] * rows,
            "redundancy_rejected": [False] * rows,
            "redundancy_reference_combination_id": [None] * rows,
            "selected": list(selected),
            "orthogonalization_rank": list(range(1, rows + 1)),
            "orthogonalization_reason": ["accepted"] * rows,
            "source_combination_version": [str(_YEAR)] * rows,
            "source_fta_version": [str(_YEAR)] * rows,
            "source_selection_version": [str(_YEAR)] * rows,
            "dataset_version": [str(_YEAR)] * rows,
            "validation_start_time": [_VALIDATION_START] * rows,
            "validation_end_time": [_VALIDATION_END] * rows,
            "status": list(statuses),
        }
    )
    return frame.select(FACTOR_ORTHOGONALIZATION_SCHEMA.names()).cast(
        FACTOR_ORTHOGONALIZATION_SCHEMA
    )


def _factors_frame(
    *,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
) -> pl.DataFrame:
    """Build a minimal Factors observation partition for Alpha scoring."""
    rows = [
        (_OPEN_TIME_1, "ema_distance", 1.0),
        (_OPEN_TIME_1, "sma_distance", 3.0),
        (_OPEN_TIME_2, "ema_distance", 2.0),
        (_OPEN_TIME_2, "sma_distance", 4.0),
    ]
    frame = pl.DataFrame(
        {
            "symbol": [symbol] * len(rows),
            "timeframe": [timeframe] * len(rows),
            "open_time": [row[0] for row in rows],
            "factor_name": [row[1] for row in rows],
            "factor_version": ["1.0.0"] * len(rows),
            "factor_category": ["price"] * len(rows),
            "factor_group": ["trend"] * len(rows),
            "factor_value": [row[2] for row in rows],
            "lookback": [14] * len(rows),
            "prediction_horizon": [1] * len(rows),
            "enabled": [True] * len(rows),
            "status": [FactorStatus.ACTIVE.value] * len(rows),
        }
    )
    return frame.select(FACTOR_SCHEMA.names()).cast(FACTOR_SCHEMA)


def _seed_fo_and_factors(
    storage_root: Path,
    *,
    symbols: Sequence[str] = (_SYMBOL,),
    fo_frame: pl.DataFrame | None = None,
) -> tuple[StorageLayout, ParquetStore]:
    """Seed FO and Factors partitions under ``storage_root``."""
    layout = StorageLayout(storage_root)
    datastore = ParquetStore()
    from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
    from cqros.factors import FactorsRepository

    fo_repo = FactorOrthogonalizationRepository(layout, datastore)
    factors_repo = FactorsRepository(layout, datastore)
    frame = fo_frame if fo_frame is not None else _canonical_fo_frame()
    fo_repo.save(
        frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    for symbol in symbols:
        factors_repo.save(
            _factors_frame(symbol=symbol),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
    return layout, datastore


# ---------------------------------------------------------------------------
# build_parser / build_options
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_manager_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple"])
        options = build_options(args)
        assert options.manager == "simple"
        assert options.storage_root == Path(DEFAULT_STORAGE_ROOT)
        assert options.years is None
        assert options.symbols is None
        assert options.overwrite is False
        assert options.export_detailed_csv is False
        assert options.workers == ResearchConfig().worker_count
        assert options.verbose is False
        assert options.debug is False

    def test_symbols_and_years_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--manager",
                "simple",
                "--years",
                "2026",
                "2025",
                "--symbols",
                "BTCUSDT",
                "ETHUSDT",
                "--overwrite",
                "--export-detailed-csv",
                "--workers",
                "2",
                "--verbose",
                "--debug",
                "--storage-root",
                "tmp-root",
            ]
        )
        options = build_options(args)
        assert options.years == (2025, 2026)
        assert options.symbols == ("BTCUSDT", "ETHUSDT")
        assert options.overwrite is True
        assert options.export_detailed_csv is True
        assert options.workers == 2
        assert options.verbose is True
        assert options.debug is True
        assert options.storage_root == Path("tmp-root")

    def test_build_options_rejects_invalid_workers(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--workers", "0"])
        with pytest.raises(ValidationError):
            build_options(args)

    def test_build_options_rejects_invalid_year(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--manager", "simple", "--years", "abc"])
        with pytest.raises(ValidationError):
            build_options(args)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


class TestDiscoverWork:
    def test_discovers_fo_partitions_and_symbols(self) -> None:
        fo_repo = MagicMock()
        fo_repo.discover_partitions.return_value = (
            _fo_partition_ref(timeframe="1h", year=2026),
            _fo_partition_ref(timeframe="4h", year=2026),
        )
        factors_repo = MagicMock()
        factors_repo.discover_symbols.return_value = (_SYMBOL, _SYMBOL_ETH)

        work = discover_work(fo_repo, factors_repo, _options(storage_root=Path("tmp")))

        fo_repo.discover_partitions.assert_called_once_with(
            managers=(_MANAGER,),
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        factors_repo.discover_symbols.assert_called_once_with(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
        )
        assert work == (
            DiscoveredWorkItem(_MANAGER, "1h", 2026, _SYMBOL),
            DiscoveredWorkItem(_MANAGER, "1h", 2026, _SYMBOL_ETH),
            DiscoveredWorkItem(_MANAGER, "4h", 2026, _SYMBOL),
            DiscoveredWorkItem(_MANAGER, "4h", 2026, _SYMBOL_ETH),
        )

    def test_symbols_filter_skips_discovery(self) -> None:
        fo_repo = MagicMock()
        fo_repo.discover_partitions.return_value = (_fo_partition_ref(),)
        factors_repo = MagicMock()

        work = discover_work(
            fo_repo,
            factors_repo,
            _options(storage_root=Path("tmp"), symbols=(_SYMBOL,)),
        )

        factors_repo.discover_symbols.assert_not_called()
        assert work == (DiscoveredWorkItem(_MANAGER, _TIMEFRAME, _YEAR, _SYMBOL),)

    def test_year_filter(self) -> None:
        fo_repo = MagicMock()
        fo_repo.discover_partitions.return_value = (
            _fo_partition_ref(year=2025),
            _fo_partition_ref(year=2026),
        )
        factors_repo = MagicMock()
        factors_repo.discover_symbols.return_value = (_SYMBOL,)

        work = discover_work(
            fo_repo,
            factors_repo,
            _options(storage_root=Path("tmp"), years=(2026,)),
        )

        assert work == (DiscoveredWorkItem(_MANAGER, _TIMEFRAME, 2026, _SYMBOL),)


# ---------------------------------------------------------------------------
# summary / exit behavior
# ---------------------------------------------------------------------------


def test_format_summary_includes_empty() -> None:
    summary = AlphaGenerationSummary(
        manager=_MANAGER,
        panels=1,
        rows=10,
        successful_tasks=1,
        failed_tasks=0,
        skipped_tasks=0,
        empty_tasks=1,
        duration_seconds=1.25,
        output_directory=Path("data/alpha"),
        failed_task_labels=(),
    )
    text = format_summary(summary)
    assert "CQROS Alpha Generation Summary" in text
    assert "Empty: 1" in text
    assert "Successful: 1" in text


def test_main_with_empty_storage(tmp_path: Path) -> None:
    code = _run(
        main(
            [
                "--manager",
                _MANAGER,
                "--storage-root",
                str(tmp_path),
            ]
        )
    )
    assert code == 0


def test_main_exit_failure_on_invalid_workers(tmp_path: Path) -> None:
    code = _run(
        main(
            [
                "--manager",
                _MANAGER,
                "--workers",
                "0",
                "--storage-root",
                str(tmp_path),
            ]
        )
    )
    assert code == 1


# ---------------------------------------------------------------------------
# generation behavior with mocks / seeded storage
# ---------------------------------------------------------------------------


class TestRunGeneration:
    def test_skips_existing_without_overwrite(self, tmp_path: Path) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        alpha_repo = AlphaRepository(layout, datastore)
        alpha_repo.save(
            pl.DataFrame(
                {
                    "factor_set_id": ["ema_distance|sma_distance"],
                    "alpha_model": ["placeholder"],
                    "alpha_version": ["1.0"],
                    "symbol": [_SYMBOL],
                    "timeframe": [_TIMEFRAME],
                    "prediction_time": [_OPEN_TIME_1],
                    "expected_return": [None],
                    "alpha_score": [2.0],
                    "confidence": [None],
                    "uncertainty": [None],
                    "prediction_horizon": [1],
                    "status": [AlphaStatus.PASS.value],
                }
            ).cast(ALPHA_SCHEMA),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )

        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path, overwrite=False),
                work=work,
            )
        )
        assert summary.skipped_tasks == 1
        assert summary.successful_tasks == 0

    def test_overwrite_regenerates_partition(self, tmp_path: Path) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        alpha_repo = AlphaRepository(layout, datastore)
        alpha_repo.save(
            pl.DataFrame(
                {
                    "factor_set_id": ["stale"],
                    "alpha_model": ["placeholder"],
                    "alpha_version": ["1.0"],
                    "symbol": [_SYMBOL],
                    "timeframe": [_TIMEFRAME],
                    "prediction_time": [_OPEN_TIME_1],
                    "expected_return": [None],
                    "alpha_score": [99.0],
                    "confidence": [None],
                    "uncertainty": [None],
                    "prediction_horizon": [1],
                    "status": [AlphaStatus.PASS.value],
                }
            ).cast(ALPHA_SCHEMA),
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )

        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path, overwrite=True),
                work=work,
            )
        )
        assert summary.successful_tasks == 1
        loaded = alpha_repo.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        assert "stale" not in loaded["factor_set_id"].to_list()
        assert loaded["factor_set_id"].to_list() == [
            "ema_distance|sma_distance",
            "ema_distance|sma_distance",
        ]

    def test_missing_factor_partition_is_empty(self, tmp_path: Path) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path, symbols=())
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        # Seed FO only; create symbol discovery root without factors year file
        # by saving FO and discovering an explicit symbol with no factors.
        factors_repo = FactorsRepository(layout, datastore)
        alpha_repo = AlphaRepository(layout, datastore)
        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=factors_repo,
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path),
                work=work,
            )
        )
        assert summary.empty_tasks == 1
        assert summary.successful_tasks == 0
        assert not alpha_repo.exists(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )

    def test_rejected_combinations_never_persist(self, tmp_path: Path) -> None:
        fo_frame = _canonical_fo_frame(
            combination_ids=("keep|pair", "reject|pair"),
            selected=(True, False),
            statuses=(
                FactorOrthogonalizationStatus.PASS.value,
                FactorOrthogonalizationStatus.FAIL.value,
            ),
        )
        # Both rows share same member names for schema; rejection is selected/status.
        layout, datastore = _seed_fo_and_factors(tmp_path, fo_frame=fo_frame)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        alpha_repo = AlphaRepository(layout, datastore)
        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path),
                work=work,
            )
        )
        assert summary.successful_tasks == 1
        loaded = alpha_repo.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        assert set(loaded["factor_set_id"].to_list()) == {"keep|pair"}
        assert "reject|pair" not in loaded["factor_set_id"].to_list()

    def test_factor_set_id_and_prediction_time_and_schema(self, tmp_path: Path) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        alpha_repo = AlphaRepository(layout, datastore)
        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path),
                work=work,
            )
        )
        assert summary.successful_tasks == 1
        loaded = alpha_repo.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        assert loaded.schema == ALPHA_SCHEMA
        assert set(loaded["factor_set_id"].to_list()) == {"ema_distance|sma_distance"}
        assert set(loaded["prediction_time"].to_list()) == {_OPEN_TIME_1, _OPEN_TIME_2}
        assert set(loaded["symbol"].to_list()) == {_SYMBOL}
        assert loaded["alpha_score"].to_list() == [2.0, 3.0]

        path = layout.alpha_path(_MANAGER, _EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
        assert path.as_posix().endswith(
            f"{STORAGE_DIR_ALPHA}/{_MANAGER}/{_EXCHANGE}/{_MARKET}/"
            f"{_SYMBOL}/{_TIMEFRAME}/{_YEAR}.parquet"
        )

    def test_no_accepted_combinations_is_empty(self, tmp_path: Path) -> None:
        fo_frame = _canonical_fo_frame(
            selected=(False,),
            statuses=(FactorOrthogonalizationStatus.FAIL.value,),
        )
        layout, datastore = _seed_fo_and_factors(tmp_path, fo_frame=fo_frame)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        alpha_repo = AlphaRepository(layout, datastore)
        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=alpha_repo,
                layout=layout,
                options=_options(storage_root=tmp_path),
                work=work,
            )
        )
        assert summary.empty_tasks == 1
        assert not alpha_repo.exists(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )

    def test_deterministic_generation(self, tmp_path: Path) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path)
        from cqros.alpha import AlphaRepository
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        options = _options(storage_root=tmp_path, overwrite=True)
        fo_repo = FactorOrthogonalizationRepository(layout, datastore)
        factors_repo = FactorsRepository(layout, datastore)
        alpha_repo = AlphaRepository(layout, datastore)

        _run(
            run_generation(
                orthogonalization_repository=fo_repo,
                factors_repository=factors_repo,
                alpha_repository=alpha_repo,
                layout=layout,
                options=options,
                work=work,
            )
        )
        first = alpha_repo.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        _run(
            run_generation(
                orthogonalization_repository=fo_repo,
                factors_repository=factors_repo,
                alpha_repository=alpha_repo,
                layout=layout,
                options=options,
                work=work,
            )
        )
        second = alpha_repo.load(
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        assert_frame_equal(first, second)

    def test_failure_sets_exit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        layout, datastore = _seed_fo_and_factors(tmp_path)
        from cqros.alpha import AlphaRepository
        from cqros.cli import generate_factor_alpha as cli_module
        from cqros.factor_orthogonalization import FactorOrthogonalizationRepository
        from cqros.factors import FactorsRepository

        def _boom(*_args: Any, **_kwargs: Any) -> pl.DataFrame:
            raise RuntimeError("forced failure")

        monkeypatch.setattr(cli_module.SimpleAlphaEngine, "build", _boom)

        work = (
            DiscoveredWorkItem(
                manager=_MANAGER,
                timeframe=_TIMEFRAME,
                year=_YEAR,
                symbol=_SYMBOL,
            ),
        )
        summary = _run(
            run_generation(
                orthogonalization_repository=FactorOrthogonalizationRepository(layout, datastore),
                factors_repository=FactorsRepository(layout, datastore),
                alpha_repository=AlphaRepository(layout, datastore),
                layout=layout,
                options=_options(storage_root=tmp_path),
                work=work,
            )
        )
        assert summary.failed_tasks == 1
        assert summary.failed_task_labels == (f"{_YEAR}/{_TIMEFRAME}/{_SYMBOL}",)

        code = _run(
            main(
                [
                    "--manager",
                    _MANAGER,
                    "--symbols",
                    _SYMBOL,
                    "--years",
                    str(_YEAR),
                    "--storage-root",
                    str(tmp_path),
                    "--overwrite",
                ]
            )
        )
        # main re-runs with monkeypatched engine still failing
        assert code == 1


def test_alpha_task_result_empty_status_progress(capsys: pytest.CaptureFixture[str]) -> None:
    from cqros.cli.generate_factor_alpha import _print_progress

    _print_progress(
        AlphaTaskResult(
            year=_YEAR,
            timeframe=_TIMEFRAME,
            symbol=_SYMBOL,
            status="empty",
            rows_generated=0,
        )
    )
    captured = capsys.readouterr()
    assert f"EMPTY {_YEAR}/{_TIMEFRAME}/{_SYMBOL}" in captured.out
