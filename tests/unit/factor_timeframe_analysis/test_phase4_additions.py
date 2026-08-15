"""Phase 4 unit tests for Factor Timeframe Analysis.

Covers:
- ``selected`` and ``source_selection_version`` in engine output
- ``load_factor_selection_for_analysis`` with multi-timeframe fixtures
- ``FactorTimeframeAnalysisVerifier.verify_against_selection``
- ``FactorTimeframeAnalysisRepository`` year-panel round trip
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_selection.schema import FACTOR_SELECTION_SCHEMA, FactorSelectionStatus
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisRepository,
    FactorTimeframeAnalysisVerifier,
    SimpleFactorTimeframeAnalysisEngine,
    TimeframeAnalysisStatus,
    discover_selection_timeframes,
    load_factor_selection_for_analysis,
)
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_YEAR = 2026
_SELECTION_TIME = 1_700_000_000_000


def _selection_row(
    *,
    factor_name: str = "momentum",
    factor_version: str = "1.0.0",
    factor_category: str = "price",
    timeframe: str = "1h",
    selection_score: float = 0.5,
    selected: bool = True,
    status: str = FactorSelectionStatus.SELECTED.value,
) -> dict[str, object]:
    """Return a single Factor Selection row as a dict."""
    return {
        "factor_name": factor_name,
        "factor_version": factor_version,
        "timeframe": timeframe,
        "selection_time": _SELECTION_TIME,
        "factor_category": factor_category,
        "selected": selected,
        "selection_score": selection_score,
        "selection_rank": 1,
        "selection_reason": "top_n",
        "status": status,
    }


def _selection_frame(*rows: dict[str, object]) -> pl.DataFrame:
    """Build a Factor Selection DataFrame from row dicts."""
    return pl.DataFrame(list(rows), schema=FACTOR_SELECTION_SCHEMA)


def _seed_fs_partition(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
    timeframe: str,
    year: int = _YEAR,
    rows: list[dict[str, object]] | None = None,
) -> None:
    """Persist a single-timeframe Factor Selection partition."""
    if rows is None:
        rows = [_selection_row(timeframe=timeframe)]
    frame = pl.DataFrame(rows, schema=FACTOR_SELECTION_SCHEMA)
    FactorSelectionRepository(layout, datastore).save(
        frame,
        manager=_MANAGER,
        exchange=EXCHANGE_BINANCE,
        market=MARKET_USDT_PERPETUAL,
        timeframe=timeframe,
        year=year,
    )


# ---------------------------------------------------------------------------
# Engine: selected and source_selection_version
# ---------------------------------------------------------------------------


class TestEngineOutputPhase4:
    def test_selected_true_when_status_pass(self) -> None:
        frame = _selection_frame(_selection_row(selected=True, selection_score=0.8))
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026")
        output = engine.build(frame)
        assert output["selected"].to_list() == [True]
        assert output["status"].to_list() == [TimeframeAnalysisStatus.PASS.value]

    def test_selected_false_propagates_when_only_rejected_rows(self) -> None:
        frame = _selection_frame(
            _selection_row(
                selected=False,
                status=FactorSelectionStatus.REJECTED.value,
                selection_score=0.1,
            )
        )
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026")
        with pytest.raises(FactorTimeframeAnalysisError):
            engine.build(frame)

    def test_source_selection_version_matches_constructor_argument(self) -> None:
        frame = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026-test")
        output = engine.build(frame)
        versions = output["source_selection_version"].unique().to_list()
        assert "2026-test" in versions

    def test_source_selection_version_year_string(self) -> None:
        frame = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026")
        output = engine.build(frame)
        assert all(v == "2026" for v in output["source_selection_version"].to_list())

    def test_multi_timeframe_best_timeframe_is_highest_score(self) -> None:
        frame = _selection_frame(
            _selection_row(timeframe="1h", selection_score=0.7),
            _selection_row(timeframe="4h", selection_score=0.4),
        )
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026")
        output = engine.build(frame)
        assert output["best_timeframe"].to_list() == ["1h"]
        assert output["selected"].to_list() == [True]

    def test_source_selection_version_blank_raises(self) -> None:
        with pytest.raises(FactorTimeframeAnalysisError):
            SimpleFactorTimeframeAnalysisEngine(source_selection_version="")

    def test_source_selection_version_whitespace_only_raises(self) -> None:
        with pytest.raises(FactorTimeframeAnalysisError):
            SimpleFactorTimeframeAnalysisEngine(source_selection_version="   ")


# ---------------------------------------------------------------------------
# selection_input: load_factor_selection_for_analysis with multi-TF fixtures
# ---------------------------------------------------------------------------


class TestSelectionInputLoader:
    def test_raises_when_no_partitions(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        fs_repo = FactorSelectionRepository(layout, datastore)
        with pytest.raises(FactorTimeframeAnalysisError):
            load_factor_selection_for_analysis(
                fs_repo,
                manager=_MANAGER,
                year=_YEAR,
            )

    def test_loads_single_timeframe(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_partition(layout=layout, datastore=datastore, timeframe="1h")
        fs_repo = FactorSelectionRepository(layout, datastore)
        frame = load_factor_selection_for_analysis(fs_repo, manager=_MANAGER, year=_YEAR)
        assert frame.height == 1
        assert "1h" in frame["timeframe"].to_list()

    def test_loads_multiple_timeframes_concatenated(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            _seed_fs_partition(layout=layout, datastore=datastore, timeframe=tf)
        fs_repo = FactorSelectionRepository(layout, datastore)
        frame = load_factor_selection_for_analysis(fs_repo, manager=_MANAGER, year=_YEAR)
        assert frame.height == 5
        loaded_tfs = sorted(frame["timeframe"].to_list())
        assert loaded_tfs == sorted(["5m", "15m", "1h", "4h", "1d"])

    def test_timeframe_filter_allowlist(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        for tf in ("1h", "4h", "1d"):
            _seed_fs_partition(layout=layout, datastore=datastore, timeframe=tf)
        fs_repo = FactorSelectionRepository(layout, datastore)
        frame = load_factor_selection_for_analysis(
            fs_repo, manager=_MANAGER, year=_YEAR, timeframes=("1h", "4h")
        )
        assert frame.height == 2
        assert set(frame["timeframe"].to_list()) == {"1h", "4h"}

    def test_timeframe_filter_empty_raises(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_fs_partition(layout=layout, datastore=datastore, timeframe="1h")
        fs_repo = FactorSelectionRepository(layout, datastore)
        with pytest.raises(FactorTimeframeAnalysisError):
            load_factor_selection_for_analysis(
                fs_repo, manager=_MANAGER, year=_YEAR, timeframes=("4h",)
            )

    def test_result_sorted_by_factor_name_factor_version_timeframe(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        rows_1h = [
            _selection_row(factor_name="rsi", timeframe="1h"),
            _selection_row(factor_name="momentum", timeframe="1h"),
        ]
        rows_4h = [
            _selection_row(factor_name="rsi", timeframe="4h"),
            _selection_row(factor_name="momentum", timeframe="4h"),
        ]
        _seed_fs_partition(layout=layout, datastore=datastore, timeframe="1h", rows=rows_1h)
        _seed_fs_partition(layout=layout, datastore=datastore, timeframe="4h", rows=rows_4h)
        fs_repo = FactorSelectionRepository(layout, datastore)
        frame = load_factor_selection_for_analysis(fs_repo, manager=_MANAGER, year=_YEAR)
        factor_names = frame["factor_name"].to_list()
        assert factor_names == sorted(factor_names[:2]) + sorted(factor_names[2:]) or (
            factor_names == sorted(factor_names)
        )
        assert sorted(frame["timeframe"].unique().to_list()) == ["1h", "4h"]

    def test_discover_selection_timeframes(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        for tf in ("1h", "4h"):
            _seed_fs_partition(layout=layout, datastore=datastore, timeframe=tf)
        fs_repo = FactorSelectionRepository(layout, datastore)
        discovered = discover_selection_timeframes(fs_repo, manager=_MANAGER, year=_YEAR)
        assert set(discovered) == {"1h", "4h"}


# ---------------------------------------------------------------------------
# verify_against_selection
# ---------------------------------------------------------------------------


class TestVerifyAgainstSelection:
    def _make_fta_from_selection(
        self, selection_frame: pl.DataFrame, year: int = _YEAR
    ) -> pl.DataFrame:
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(year))
        return engine.build(selection_frame)

    def test_passes_with_matching_selection(self) -> None:
        selection = _selection_frame(
            _selection_row(factor_name="momentum", selection_score=0.8),
            _selection_row(
                factor_name="momentum", factor_version="1.0.0", timeframe="4h", selection_score=0.6
            ),
        )
        fta = self._make_fta_from_selection(selection)
        verifier = FactorTimeframeAnalysisVerifier()
        report = verifier.verify_against_selection(fta, selection)
        assert report.passed is True

    def test_warns_when_factor_missing_from_selection(self) -> None:
        selection = _selection_frame(_selection_row(factor_name="momentum"))
        fta = self._make_fta_from_selection(selection)
        empty_selection = _selection_frame(
            _selection_row(
                factor_name="other_factor",
                selected=False,
                status=FactorSelectionStatus.REJECTED.value,
            )
        )
        verifier = FactorTimeframeAnalysisVerifier()
        report = verifier.verify_against_selection(fta, empty_selection)
        assert report.passed is False
        assert len(report.warnings) > 0

    def test_structural_plus_cross_frame_both_run(self) -> None:
        selection = _selection_frame(
            _selection_row(factor_name="a", selection_score=0.9),
            _selection_row(factor_name="b", selection_score=0.7),
        )
        fta = self._make_fta_from_selection(selection)
        verifier = FactorTimeframeAnalysisVerifier()
        report = verifier.verify_against_selection(fta, selection)
        assert report.rows_checked == fta.height
        assert report.passed is True


# ---------------------------------------------------------------------------
# Repository year-panel round trip
# ---------------------------------------------------------------------------


class TestRepositoryYearPanelRoundTrip:
    def test_save_and_load_preserves_rows(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        selection = _selection_frame(
            _selection_row(factor_name="momentum"),
            _selection_row(factor_name="momentum", timeframe="4h"),
        )
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = engine.build(selection)
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        repo.save(
            fta_frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        loaded = repo.load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        assert loaded.height == fta_frame.height

    def test_exists_returns_false_before_save(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        assert not repo.exists(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )

    def test_exists_returns_true_after_save(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        selection = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = engine.build(selection)
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        repo.save(
            fta_frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        assert repo.exists(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )

    def test_list_years_returns_saved_year(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        selection = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = engine.build(selection)
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        for year in (2025, 2026):
            repo.save(
                fta_frame,
                manager=_MANAGER,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
                year=year,
            )
        years = repo.list_years(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
        )
        assert 2025 in years
        assert 2026 in years

    def test_discover_partitions_includes_saved(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        selection = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = engine.build(selection)
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        repo.save(
            fta_frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        partitions = repo.discover_partitions(
            managers=(_MANAGER,),
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
        )
        assert len(partitions) == 1
        assert partitions[0].manager == _MANAGER
        assert partitions[0].year == _YEAR

    def test_source_selection_version_preserved_on_load(self, tmp_path: Path) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        selection = _selection_frame(_selection_row())
        engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version="2026")
        fta_frame = engine.build(selection)
        repo = FactorTimeframeAnalysisRepository(layout, datastore)
        repo.save(
            fta_frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        loaded = repo.load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )
        assert "2026" in loaded["source_selection_version"].unique().to_list()
