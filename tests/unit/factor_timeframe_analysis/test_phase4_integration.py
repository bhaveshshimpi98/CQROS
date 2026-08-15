"""Phase 4 integration test: Factor Selection → FTA → Combination round trip.

Fixture: 5m/15m/1h/4h/1d Factor Selection frames for two factors
→ FTA (via SimpleFactorTimeframeAnalysisEngine)
→ Combination (via SimpleFactorCombinationEngine)

Assertions:
- FTA output has correct structure and lineage
- Combination output traces back to FTA selection scores
- FactorCombinationVerifier passes lineage check
- FactorTimeframeAnalysisVerifier passes cross-frame check
- source_selection_version propagates through FTA column
- Combination timeframe resolves from FTA best_timeframe
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factor_combination import (
    FactorCombinationRepository,
    FactorCombinationVerifier,
    SimpleFactorCombinationEngine,
)
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_selection.schema import FACTOR_SELECTION_SCHEMA, FactorSelectionStatus
from cqros.factor_timeframe_analysis import (
    FactorTimeframeAnalysisRepository,
    FactorTimeframeAnalysisVerifier,
    SimpleFactorTimeframeAnalysisEngine,
    TimeframeAnalysisStatus,
    load_factor_selection_for_analysis,
)
from cqros.storage import ParquetStore, StorageLayout

_MANAGER = "simple"
_YEAR = 2026
_SELECTION_TIME = 1_700_000_000_000

_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
_SCORES_MOMENTUM = {
    "5m": 0.30,
    "15m": 0.45,
    "1h": 0.80,
    "4h": 0.60,
    "1d": 0.40,
}
_SCORES_RSI = {
    "5m": 0.20,
    "15m": 0.35,
    "1h": 0.55,
    "4h": 0.70,
    "1d": 0.50,
}


def _selection_row(
    *,
    factor_name: str,
    timeframe: str,
    selection_score: float,
) -> dict[str, object]:
    """Build a single Factor Selection row dict."""
    return {
        "factor_name": factor_name,
        "factor_version": "1.0.0",
        "timeframe": timeframe,
        "selection_time": _SELECTION_TIME,
        "factor_category": "price",
        "selected": True,
        "selection_score": selection_score,
        "selection_rank": 1,
        "selection_reason": "top_n",
        "status": FactorSelectionStatus.SELECTED.value,
    }


def _build_multi_timeframe_selection() -> pl.DataFrame:
    """Return a Factor Selection frame covering 5m/15m/1h/4h/1d for two factors."""
    rows: list[dict[str, object]] = []
    for timeframe in _TIMEFRAMES:
        rows.append(
            _selection_row(
                factor_name="momentum",
                timeframe=timeframe,
                selection_score=_SCORES_MOMENTUM[timeframe],
            )
        )
        rows.append(
            _selection_row(
                factor_name="rsi",
                timeframe=timeframe,
                selection_score=_SCORES_RSI[timeframe],
            )
        )
    return pl.DataFrame(rows, schema=FACTOR_SELECTION_SCHEMA)


def _seed_selection_partitions(
    *,
    layout: StorageLayout,
    datastore: ParquetStore,
) -> None:
    """Persist one Factor Selection partition per timeframe for two factors."""
    fs_repo = FactorSelectionRepository(layout, datastore)
    for timeframe in _TIMEFRAMES:
        rows = [
            _selection_row(
                factor_name="momentum",
                timeframe=timeframe,
                selection_score=_SCORES_MOMENTUM[timeframe],
            ),
            _selection_row(
                factor_name="rsi",
                timeframe=timeframe,
                selection_score=_SCORES_RSI[timeframe],
            ),
        ]
        frame = pl.DataFrame(rows, schema=FACTOR_SELECTION_SCHEMA)
        fs_repo.save(
            frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            timeframe=timeframe,
            year=_YEAR,
        )


@pytest.fixture()
def fta_frame() -> pl.DataFrame:
    """Return an FTA frame derived from the multi-timeframe selection fixture."""
    selection = _build_multi_timeframe_selection()
    engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
    return engine.build(selection)


@pytest.fixture()
def combination_frame(fta_frame: pl.DataFrame) -> pl.DataFrame:
    """Return a combination frame derived from the FTA fixture."""
    engine = SimpleFactorCombinationEngine()
    return engine.build(fta_frame)


# ---------------------------------------------------------------------------
# FTA correctness
# ---------------------------------------------------------------------------


class TestFTAFromMultiTimeframe:
    def test_one_row_per_factor(self, fta_frame: pl.DataFrame) -> None:
        assert fta_frame.height == 2

    def test_both_factors_present(self, fta_frame: pl.DataFrame) -> None:
        names = set(fta_frame["factor_name"].to_list())
        assert names == {"momentum", "rsi"}

    def test_momentum_best_timeframe_is_1h(self, fta_frame: pl.DataFrame) -> None:
        row = fta_frame.filter(pl.col("factor_name") == "momentum").row(0, named=True)
        assert row["best_timeframe"] == "1h"
        assert row["best_selection_score"] == pytest.approx(_SCORES_MOMENTUM["1h"], rel=1e-6)

    def test_rsi_best_timeframe_is_4h(self, fta_frame: pl.DataFrame) -> None:
        row = fta_frame.filter(pl.col("factor_name") == "rsi").row(0, named=True)
        assert row["best_timeframe"] == "4h"
        assert row["best_selection_score"] == pytest.approx(_SCORES_RSI["4h"], rel=1e-6)

    def test_selected_true_for_both(self, fta_frame: pl.DataFrame) -> None:
        assert all(fta_frame["selected"].to_list())

    def test_status_pass_for_both(self, fta_frame: pl.DataFrame) -> None:
        assert all(s == TimeframeAnalysisStatus.PASS.value for s in fta_frame["status"].to_list())

    def test_source_selection_version_is_year(self, fta_frame: pl.DataFrame) -> None:
        versions = fta_frame["source_selection_version"].unique().to_list()
        assert str(_YEAR) in versions

    def test_fta_verifier_passes_structural_check(self, fta_frame: pl.DataFrame) -> None:
        selection = _build_multi_timeframe_selection()
        verifier = FactorTimeframeAnalysisVerifier()
        report = verifier.verify_against_selection(fta_frame, selection)
        assert report.passed is True, f"Warnings: {report.warnings}"

    def test_fta_repository_round_trip(self, tmp_path: Path, fta_frame: pl.DataFrame) -> None:
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
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
        assert str(_YEAR) in loaded["source_selection_version"].unique().to_list()


# ---------------------------------------------------------------------------
# Combination correctness
# ---------------------------------------------------------------------------


class TestCombinationFromFTA:
    def test_combination_produces_at_least_one_pair(self, combination_frame: pl.DataFrame) -> None:
        assert combination_frame.height >= 1

    def test_combination_contains_momentum_rsi_pair(self, combination_frame: pl.DataFrame) -> None:
        ids = combination_frame["combination_id"].to_list()
        assert any("momentum" in cid and "rsi" in cid for cid in ids)

    def test_combination_timeframe_resolved(self, combination_frame: pl.DataFrame) -> None:
        assert all(len(tf) > 0 for tf in combination_frame["timeframe"].to_list())

    def test_combination_score_is_mean_of_member_scores(
        self,
        fta_frame: pl.DataFrame,
        combination_frame: pl.DataFrame,
    ) -> None:
        """Verify combination_score = mean(best_selection_score_a, best_selection_score_b)."""
        fta_index = {
            row["factor_name"]: row["best_selection_score"]
            for row in fta_frame.iter_rows(named=True)
        }
        for row in combination_frame.iter_rows(named=True):
            names = row["factor_names"]
            expected = (fta_index[names[0]] + fta_index[names[1]]) / 2.0
            assert row["combination_score"] == pytest.approx(expected, rel=1e-6)

    def test_combination_verifier_passes_lineage(
        self,
        fta_frame: pl.DataFrame,
        combination_frame: pl.DataFrame,
    ) -> None:
        verifier = FactorCombinationVerifier()
        verifier.verify(combination_frame)
        verifier.verify_against_fta(combination_frame, fta_frame)

    def test_combination_partitioned_by_timeframe_round_trip(
        self,
        tmp_path: Path,
        fta_frame: pl.DataFrame,
        combination_frame: pl.DataFrame,
    ) -> None:
        """Save each timeframe partition and reload; verify all rows recoverable."""
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        comb_repo = FactorCombinationRepository(layout, datastore)
        timeframes = combination_frame.select("timeframe").unique().to_series().to_list()
        for timeframe in timeframes:
            partition = combination_frame.filter(pl.col("timeframe") == timeframe)
            comb_repo.save(
                partition,
                manager=_MANAGER,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
                timeframe=timeframe,
                year=_YEAR,
            )
        total_rows = 0
        for timeframe in timeframes:
            loaded = comb_repo.load(
                manager=_MANAGER,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
                timeframe=timeframe,
                year=_YEAR,
            )
            total_rows += loaded.height
            verifier = FactorCombinationVerifier()
            verifier.verify(loaded)
            verifier.verify_against_fta(loaded, fta_frame)
        assert total_rows == combination_frame.height


# ---------------------------------------------------------------------------
# Full pipeline from seeded Factor Selection files
# ---------------------------------------------------------------------------


class TestFullPipelineWithRepository:
    def test_selection_to_fta_to_combination(self, tmp_path: Path) -> None:
        """Full round trip: seed FS → load for FTA → build FTA → build combination."""
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_selection_partitions(layout=layout, datastore=datastore)

        fs_repo = FactorSelectionRepository(layout, datastore)
        concatenated_selection = load_factor_selection_for_analysis(
            fs_repo, manager=_MANAGER, year=_YEAR
        )
        assert concatenated_selection.height == 10

        fta_engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = fta_engine.build(concatenated_selection)
        assert fta_frame.height == 2

        fta_repo = FactorTimeframeAnalysisRepository(layout, datastore)
        fta_repo.save(
            fta_frame,
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )

        loaded_fta = fta_repo.load(
            manager=_MANAGER,
            exchange=EXCHANGE_BINANCE,
            market=MARKET_USDT_PERPETUAL,
            year=_YEAR,
        )

        comb_engine = SimpleFactorCombinationEngine()
        combination_output = comb_engine.build(loaded_fta)

        comb_repo = FactorCombinationRepository(layout, datastore)
        timeframes = combination_output.select("timeframe").unique().to_series().to_list()
        for tf in timeframes:
            partition = combination_output.filter(pl.col("timeframe") == tf)
            comb_repo.save(
                partition,
                manager=_MANAGER,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
                timeframe=tf,
                year=_YEAR,
            )

        verifier = FactorCombinationVerifier()
        for tf in timeframes:
            loaded_comb = comb_repo.load(
                manager=_MANAGER,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
                timeframe=tf,
                year=_YEAR,
            )
            verifier.verify(loaded_comb)
            verifier.verify_against_fta(loaded_comb, loaded_fta)

        fta_verifier = FactorTimeframeAnalysisVerifier()
        fta_report = fta_verifier.verify_against_selection(fta_frame, concatenated_selection)
        assert fta_report.passed is True

        ssv = loaded_fta["source_selection_version"].unique().to_list()
        assert str(_YEAR) in ssv

    def test_source_selection_version_lineage_end_to_end(self, tmp_path: Path) -> None:
        """Verify source_selection_version propagates correctly through FTA."""
        layout = StorageLayout(tmp_path)
        datastore = ParquetStore()
        _seed_selection_partitions(layout=layout, datastore=datastore)
        fs_repo = FactorSelectionRepository(layout, datastore)
        selection = load_factor_selection_for_analysis(fs_repo, manager=_MANAGER, year=_YEAR)
        fta_engine = SimpleFactorTimeframeAnalysisEngine(source_selection_version=str(_YEAR))
        fta_frame = fta_engine.build(selection)
        versions = fta_frame["source_selection_version"].unique().to_list()
        assert str(_YEAR) in versions
        assert all(v == str(_YEAR) for v in versions)
