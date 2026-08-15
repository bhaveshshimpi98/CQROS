"""Unit tests for CQROS Factor Combination module contracts and orchestration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_COMBINATION,
)
from cqros.core.types import FilePath
from cqros.factor_combination.detailed_export import (
    COMBINED_DETAILED_CSV_NAME,
    DETAILED_AUDIT_COLUMNS,
    build_detailed_audit_frame,
    combined_detailed_csv_path,
    detailed_csv_path,
    write_combined_detailed_csv,
    write_detailed_csv,
)
from cqros.factor_combination.engine import (
    FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS,
    SimpleFactorCombinationEngine,
    validate_factor_timeframe_analysis_frame,
)
from cqros.factor_combination.exceptions import FactorCombinationError
from cqros.factor_combination.pipeline import FactorCombinationPipeline
from cqros.factor_combination.registry import FactorCombinationRegistry
from cqros.factor_combination.repository import (
    FactorCombinationPartitionRef,
    FactorCombinationRepository,
)
from cqros.factor_combination.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_COMBINATION_COLUMNS,
    FACTOR_COMBINATION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    FactorCombinationStatus,
)
from cqros.factor_combination.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_LINEAGE_DUPLICATE_COMBINATIONS,
    ERROR_LINEAGE_FTA_TYPE,
    ERROR_LINEAGE_MISSING_FACTOR,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorCombinationVerifier,
)
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_TIMEFRAME = "1h"
_YEAR = 2026
_ANALYSIS_TIME = 1_704_067_200_000
_FIXED_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_FIXED_ANALYSIS_TIME_MS = int(_FIXED_NOW.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that stores frames in memory."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}
        self.write_paths: list[Path] = []
        self.read_paths: list[Path] = []
        self.exists_paths: list[Path] = []
        self.delete_paths: list[Path] = []

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        target = Path(path)
        self.write_paths.append(target)
        self.frames[target] = dataframe

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        self.read_paths.append(target)
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
        target = Path(path)
        self.exists_paths.append(target)
        return target in self.frames

    def delete(self, path: FilePath) -> None:
        target = Path(path)
        self.delete_paths.append(target)
        try:
            del self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-002",
                details={"path": str(target)},
            ) from exc

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


class _StubEngine:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[pl.DataFrame] = []

    def build(self, factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(factor_timeframe_analysis)
        return self.output


class _StubRegistry:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[pl.DataFrame] = []

    def build(self, factor_timeframe_analysis: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(factor_timeframe_analysis)
        return self.output


class _StubRepository:
    """Test double that records save calls without persistence."""

    def __init__(self) -> None:
        self.save_calls: list[tuple[pl.DataFrame, dict[str, object]]] = []

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        manager: str,
        exchange: str,
        market: str,
        timeframe: str,
        year: int,
    ) -> None:
        self.save_calls.append(
            (
                dataframe,
                {
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "timeframe": timeframe,
                    "year": year,
                },
            )
        )


def _partition_kwargs(
    *,
    manager: str = _MANAGER,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> dict[str, object]:
    """Build keyword arguments identifying one partition (no symbol)."""
    return {
        "manager": manager,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "timeframe": timeframe,
        "year": year,
    }


def _fta_frame(
    *,
    factor_names: list[str] | None = None,
    factor_versions: list[str] | None = None,
    factor_categories: list[str] | None = None,
    best_timeframes: list[str] | None = None,
    best_selection_scores: list[float] | None = None,
    timeframe_confidences: list[float] | None = None,
    selected: list[bool] | None = None,
) -> pl.DataFrame:
    """Build a Factor Timeframe Analysis-style input for the combination engine."""
    row_count = max(
        len(values)
        for values in (
            factor_names or ["momentum", "rsi"],
            factor_versions or [],
            factor_categories or [],
            best_timeframes or [],
            best_selection_scores or [],
            timeframe_confidences or [],
            selected or [],
        )
    )
    factor_names = (
        factor_names
        if factor_names is not None
        else (
            ["momentum", "rsi"]
            if row_count == 2
            else [f"factor_{index}" for index in range(row_count)]
        )
    )
    factor_versions = factor_versions if factor_versions is not None else ["1.0.0"] * row_count
    factor_categories = (
        factor_categories if factor_categories is not None else ["price"] * row_count
    )
    best_timeframes = best_timeframes if best_timeframes is not None else ["1h"] * row_count
    best_selection_scores = (
        best_selection_scores if best_selection_scores is not None else [0.80] * row_count
    )
    timeframe_confidences = (
        timeframe_confidences if timeframe_confidences is not None else [0.70] * row_count
    )
    selected = selected if selected is not None else [True] * row_count
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": factor_versions,
            "factor_category": factor_categories,
            "best_timeframe": best_timeframes,
            "best_selection_score": best_selection_scores,
            "timeframe_confidence": timeframe_confidences,
            "selected": selected,
        }
    )


def _combination_frame(
    *,
    combination_id: str = "momentum|rsi",
    factor_names: list[str] | None = None,
    combination_score: float = 0.70,
    combination_rank: int = 1,
    status: str = FactorCombinationStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical Factor Combination frame for repository/verifier tests."""
    names = factor_names if factor_names is not None else ["momentum", "rsi"]
    return pl.DataFrame(
        {
            "combination_id": [combination_id],
            "factor_names": [names],
            "factor_versions": [["1.0.0", "1.0.0"]],
            "factor_categories": [["price", "oscillator"]],
            "timeframe": [_TIMEFRAME],
            "combination_size": [2],
            "combination_method": ["equal_weight"],
            "analysis_time": [_ANALYSIS_TIME],
            "information_coefficient": [None],
            "rank_information_coefficient": [None],
            "ic_information_ratio": [None],
            "quantile_spread": [None],
            "hit_rate": [None],
            "turnover": [None],
            "correlation_penalty": [None],
            "diversification_score": [None],
            "stability_score": [0.70],
            "confidence_score": [0.60],
            "combination_score": [combination_score],
            "combination_rank": [combination_rank],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build(frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run ``SimpleFactorCombinationEngine`` with a frozen analysis clock."""
    engine = SimpleFactorCombinationEngine()
    with patch("cqros.factor_combination.engine.datetime") as mock_datetime:
        mock_datetime.now.return_value = _FIXED_NOW
        return engine.build(frame if frame is not None else _fta_frame())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_canonical_columns_match_specification() -> None:
    """Canonical columns follow the declared factor combination contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "combination_id",
        "factor_names",
        "factor_versions",
        "factor_categories",
        "timeframe",
        "combination_size",
        "combination_method",
        "analysis_time",
        "information_coefficient",
        "rank_information_coefficient",
        "ic_information_ratio",
        "quantile_spread",
        "hit_rate",
        "turnover",
        "correlation_penalty",
        "diversification_score",
        "stability_score",
        "confidence_score",
        "combination_score",
        "combination_rank",
        "status",
    )
    assert FACTOR_COMBINATION_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == ("combination_id", "timeframe", "analysis_time")


def test_required_columns_match_canonical_order() -> None:
    """REQUIRED_COLUMNS mirrors CANONICAL_COLUMN_ORDER exactly."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))


def test_schema_dtypes_match_column_dtypes() -> None:
    """FACTOR_COMBINATION_SCHEMA dtypes match COLUMN_DTYPES in canonical order."""
    assert FACTOR_COMBINATION_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert FACTOR_COMBINATION_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["combination_id"] == pl.String
    assert COLUMN_DTYPES["factor_names"] == pl.List(pl.String)
    assert COLUMN_DTYPES["factor_versions"] == pl.List(pl.String)
    assert COLUMN_DTYPES["factor_categories"] == pl.List(pl.String)
    assert COLUMN_DTYPES["combination_size"] == pl.Int32
    assert COLUMN_DTYPES["analysis_time"] == pl.Int64
    assert COLUMN_DTYPES["combination_score"] == pl.Float64
    assert COLUMN_DTYPES["combination_rank"] == pl.Int32
    assert COLUMN_DTYPES["status"] == pl.String


def test_factor_combination_status_enum_members() -> None:
    """FactorCombinationStatus exposes PASS and FAIL members."""
    assert FactorCombinationStatus.PASS.value == "PASS"
    assert FactorCombinationStatus.FAIL.value == "FAIL"


# ---------------------------------------------------------------------------
# Verifier – structural
# ---------------------------------------------------------------------------


def test_verifier_valid_dataframe_passes() -> None:
    """Verifier returns the same frame instance for a valid combination frame."""
    frame = _combination_frame()
    verified = FactorCombinationVerifier().verify(frame)
    assert verified is frame
    assert_frame_equal(verified, frame)


def test_verifier_non_dataframe_fails() -> None:
    """Verifier rejects non-DataFrame inputs."""
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == ERROR_FRAME_TYPE


def test_verifier_empty_dataframe_fails() -> None:
    """Verifier rejects empty combination frames."""
    empty = pl.DataFrame(schema=FACTOR_COMBINATION_SCHEMA)
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify(empty)
    assert exc_info.value.error_code == ERROR_FRAME_EMPTY


def test_verifier_missing_column_fails() -> None:
    """Verifier rejects frames missing required columns."""
    frame = _combination_frame().drop("status")
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_wrong_column_order_fails() -> None:
    """Verifier rejects frames whose column order is non-canonical."""
    frame = _combination_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_COLUMN_ORDER


def test_verifier_wrong_dtype_fails() -> None:
    """Verifier rejects frames whose dtypes differ from the canonical schema."""
    frame = _combination_frame().with_columns(pl.col("combination_rank").cast(pl.Float64))
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Verifier – lineage
# ---------------------------------------------------------------------------


def test_verifier_lineage_valid_passes() -> None:
    """verify_against_fta returns combination_frame unchanged when lineage is clean."""
    fta = _fta_frame(
        factor_names=["momentum", "rsi"],
        factor_versions=["1.0.0", "1.0.0"],
        best_selection_scores=[0.70, 0.70],
        selected=[True, True],
    )
    combination = _combination_frame(
        combination_id="momentum|rsi",
        combination_score=0.70,
    )
    result = FactorCombinationVerifier().verify_against_fta(combination, fta)
    assert result is combination


def test_verifier_lineage_missing_factor_raises() -> None:
    """verify_against_fta raises when a member factor is absent from FTA."""
    fta = _fta_frame(
        factor_names=["momentum"],
        factor_versions=["1.0.0"],
        best_selection_scores=[0.80],
        selected=[True],
    )
    combination = _combination_frame(
        combination_id="momentum|rsi",
        factor_names=["momentum", "rsi"],
    )
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify_against_fta(combination, fta)
    assert exc_info.value.error_code == ERROR_LINEAGE_MISSING_FACTOR


def test_verifier_lineage_unselected_factor_raises() -> None:
    """verify_against_fta raises when a member factor has selected==False."""
    fta = _fta_frame(
        factor_names=["momentum", "rsi"],
        factor_versions=["1.0.0", "1.0.0"],
        best_selection_scores=[0.80, 0.60],
        selected=[True, False],
    )
    combination = _combination_frame(
        combination_id="momentum|rsi",
        factor_names=["momentum", "rsi"],
    )
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify_against_fta(combination, fta)
    assert exc_info.value.error_code == ERROR_LINEAGE_MISSING_FACTOR


def test_verifier_lineage_score_mismatch_raises() -> None:
    """verify_against_fta raises when combination_score deviates from member mean."""
    fta = _fta_frame(
        factor_names=["momentum", "rsi"],
        factor_versions=["1.0.0", "1.0.0"],
        best_selection_scores=[0.80, 0.60],
        selected=[True, True],
    )
    combination = _combination_frame(
        combination_id="momentum|rsi",
        combination_score=0.99,
    )
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify_against_fta(combination, fta)
    assert exc_info.value.error_code == ERROR_LINEAGE_MISSING_FACTOR


def test_verifier_lineage_fta_non_dataframe_raises() -> None:
    """verify_against_fta raises when fta_frame is not a DataFrame."""
    combination = _combination_frame()
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify_against_fta(combination, "not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == ERROR_LINEAGE_FTA_TYPE


def test_verifier_lineage_duplicate_combination_ids_raises() -> None:
    """verify_against_fta raises when combination_id values are duplicated."""
    fta = _fta_frame(
        factor_names=["momentum", "rsi"],
        factor_versions=["1.0.0", "1.0.0"],
        best_selection_scores=[0.70, 0.70],
        selected=[True, True],
    )
    row = _combination_frame(combination_id="momentum|rsi", combination_score=0.70)
    duplicated = pl.concat([row, row])
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationVerifier().verify_against_fta(duplicated, fta)
    assert exc_info.value.error_code == ERROR_LINEAGE_DUPLICATE_COMBINATIONS


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_validates_non_dataframe_input() -> None:
    """Engine rejects non-DataFrame inputs during structural validation."""
    with pytest.raises(FactorCombinationError) as exc_info:
        validate_factor_timeframe_analysis_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FCOMB_FRAME_TYPE"


def test_engine_validates_empty_input() -> None:
    """Engine rejects empty Factor Timeframe Analysis frames."""
    empty = pl.DataFrame(schema={"factor_name": pl.String}).clear()
    with pytest.raises(FactorCombinationError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "FCOMB_FRAME_EMPTY"


def test_engine_validates_missing_columns() -> None:
    """Engine rejects frames missing required input columns."""
    frame = pl.DataFrame({"factor_name": ["momentum"], "selected": [True]})
    with pytest.raises(FactorCombinationError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "FCOMB_MISSING_COLUMNS"


def test_engine_input_columns_contract() -> None:
    """FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS enumerates every consumed column."""
    for column in (
        "factor_name",
        "factor_version",
        "factor_category",
        "best_timeframe",
        "best_selection_score",
        "timeframe_confidence",
        "selected",
    ):
        assert column in FACTOR_TIMEFRAME_ANALYSIS_INPUT_COLUMNS


def test_engine_only_selected_rows_used() -> None:
    """Only rows with selected == True participate in pair generation."""
    result = _build(
        _fta_frame(
            factor_names=["momentum", "rsi", "volume"],
            best_selection_scores=[0.90, 0.50, 0.70],
            timeframe_confidences=[0.60, 0.80, 0.90],
            selected=[True, True, False],
        )
    )
    assert result.height == 1
    assert result["combination_id"].to_list() == ["momentum|rsi"]


def test_engine_deterministic_pair_generation() -> None:
    """Identical inputs produce identical pairwise combination identities."""
    frame = _fta_frame(
        factor_names=["rsi", "momentum", "volume"],
        best_selection_scores=[0.50, 0.90, 0.70],
        timeframe_confidences=[0.80, 0.60, 0.90],
    )
    first = _build(frame)
    second = _build(frame)
    assert first["combination_id"].to_list() == second["combination_id"].to_list()
    assert first.height == 3


def test_engine_equal_weight_combinations() -> None:
    """Every emitted combination uses equal_weight with size 2."""
    result = _build()
    assert result["combination_method"].to_list() == ["equal_weight"]
    assert result["combination_size"].to_list() == [2]


def test_engine_combination_id_deterministic() -> None:
    """combination_id is alphabetical and independent of input row order."""
    forward = _build(
        _fta_frame(
            factor_names=["momentum", "rsi"],
            best_selection_scores=[0.90, 0.50],
        )
    )
    reverse = _build(
        _fta_frame(
            factor_names=["rsi", "momentum"],
            best_selection_scores=[0.50, 0.90],
        )
    )
    assert forward["combination_id"].to_list() == ["momentum|rsi"]
    assert reverse["combination_id"].to_list() == ["momentum|rsi"]


def test_engine_ranking_deterministic() -> None:
    """combination_rank is descending by combination_score with stable ties."""
    result = _build(
        _fta_frame(
            factor_names=["alpha", "bravo", "charlie"],
            best_selection_scores=[0.90, 0.50, 0.70],
            timeframe_confidences=[0.60, 0.80, 0.90],
        )
    )
    assert result["combination_id"].to_list() == [
        "alpha|charlie",
        "alpha|bravo",
        "bravo|charlie",
    ]
    assert result["combination_rank"].to_list() == [1, 2, 3]
    assert result["combination_score"].to_list() == [
        pytest.approx(0.80),
        pytest.approx(0.70),
        pytest.approx(0.60),
    ]


def test_engine_single_factor_raises() -> None:
    """A single selected factor cannot form a pair and raises insufficient factors."""
    with pytest.raises(FactorCombinationError) as exc_info:
        _build(
            _fta_frame(
                factor_names=["momentum"],
                factor_versions=["1.0.0"],
                factor_categories=["price"],
                best_timeframes=["1h"],
                best_selection_scores=[0.80],
                timeframe_confidences=[0.70],
                selected=[True],
            )
        )
    assert exc_info.value.error_code == "FCOMB_INSUFFICIENT_FACTORS"


def test_engine_no_selected_rows_raises() -> None:
    """Frames with no selected rows raise FCOMB_NO_SELECTED."""
    with pytest.raises(FactorCombinationError) as exc_info:
        _build(
            _fta_frame(
                factor_names=["momentum", "rsi"],
                selected=[False, False],
            )
        )
    assert exc_info.value.error_code == "FCOMB_NO_SELECTED"


def test_engine_duplicate_factor_names_handled() -> None:
    """Same factor name with distinct versions forms a deterministic pair."""
    result = _build(
        _fta_frame(
            factor_names=["momentum", "momentum"],
            factor_versions=["1.0.0", "2.0.0"],
            factor_categories=["price", "price"],
            best_timeframes=["1h", "4h"],
            best_selection_scores=[0.40, 0.90],
            timeframe_confidences=[0.50, 0.80],
            selected=[True, True],
        )
    )
    assert result.height == 1
    assert result["combination_id"].to_list() == ["momentum|momentum"]
    assert result["factor_versions"].to_list() == [["1.0.0", "2.0.0"]]
    assert result["timeframe"].to_list() == ["4h"]


def test_engine_duplicate_factor_identity_deduplicated() -> None:
    """Duplicate (factor_name, factor_version) rows keep the highest score."""
    result = _build(
        _fta_frame(
            factor_names=["momentum", "momentum", "rsi"],
            factor_versions=["1.0.0", "1.0.0", "1.0.0"],
            best_selection_scores=[0.40, 0.90, 0.50],
            timeframe_confidences=[0.50, 0.60, 0.80],
            best_timeframes=["1h", "4h", "1h"],
        )
    )
    assert result.height == 1
    assert result["combination_id"].to_list() == ["momentum|rsi"]
    assert result["combination_score"].to_list() == [pytest.approx(0.70)]
    assert result["timeframe"].to_list() == ["4h"]


def test_engine_output_matches_factor_combination_schema() -> None:
    """Engine output conforms to FACTOR_COMBINATION_SCHEMA and canonical order."""
    result = _build()
    assert result.schema == FACTOR_COMBINATION_SCHEMA
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result["status"].to_list() == [FactorCombinationStatus.PASS.value]
    assert result["analysis_time"].to_list() == [_FIXED_ANALYSIS_TIME_MS]
    assert result["information_coefficient"].to_list() == [None]


def test_engine_does_not_mutate_input() -> None:
    """Engine leaves the caller-supplied frame unchanged."""
    frame = _fta_frame()
    before = frame.clone()
    _ = _build(frame)
    assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_default_engine() -> None:
    """Default registry owns a SimpleFactorCombinationEngine instance."""
    registry = FactorCombinationRegistry()
    assert isinstance(registry.engine, SimpleFactorCombinationEngine)


def test_registry_injected_engine_delegation() -> None:
    """Registry.build delegates exclusively to the injected engine."""
    expected = _combination_frame()
    stub = _StubEngine(expected)
    registry = FactorCombinationRegistry(engine=stub)  # type: ignore[arg-type]
    input_frame = _fta_frame()

    result = registry.build(input_frame)

    assert result is expected
    assert stub.calls == [input_frame]
    assert registry.engine is stub


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """FactorCombinationPartitionRef is a frozen immutable dataclass (no symbol)."""
    ref = FactorCombinationPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ref.timeframe = "4h"  # type: ignore[misc]


def test_partition_ref_has_no_symbol_field() -> None:
    """FactorCombinationPartitionRef does not have a symbol field."""
    ref = FactorCombinationPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert not hasattr(ref, "symbol")


def test_repository_save_exists_load_round_trip() -> None:
    """Saved frames can be retrieved by exists() and load()."""
    store = _InMemoryDataStore()
    repository = FactorCombinationRepository(StorageLayout(Path("/data")), store)
    frame = _combination_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == FACTOR_COMBINATION_SCHEMA


def test_repository_empty_load_returns_schema_frame() -> None:
    """load() returns an empty FACTOR_COMBINATION_SCHEMA frame when missing."""
    repository = FactorCombinationRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    loaded = repository.load(**_partition_kwargs())  # type: ignore[arg-type]
    assert loaded.height == 0
    assert loaded.schema == FACTOR_COMBINATION_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


def test_repository_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = FactorCombinationRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_combination_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_repository_delete_absent_succeeds_silently() -> None:
    """Deleting a missing partition succeeds without raising."""
    repository = FactorCombinationRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


def test_repository_save_rejects_missing_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = FactorCombinationRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(FactorCombinationError) as exc_info:
        repository.save(
            pl.DataFrame({"combination_id": ["momentum|rsi"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "FCOMB_REPO_MISSING_COLUMNS"


def test_repository_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs."""
    repository = FactorCombinationRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(FactorCombinationError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "FCOMB_REPO_FRAME_TYPE"


def test_repository_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = FactorCombinationRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    repository.save(_combination_frame(combination_score=0.10), **kwargs)  # type: ignore[arg-type]
    repository.save(_combination_frame(combination_score=0.90), **kwargs)  # type: ignore[arg-type]

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["combination_score"].to_list() == [pytest.approx(0.90)]


def test_repository_parquet_store_path_contains_factor_combination_directory(
    tmp_path: Path,
) -> None:
    """Saved partitions reside under the factor_combination storage directory."""
    layout = StorageLayout(tmp_path)
    repository = FactorCombinationRepository(layout, ParquetStore())
    repository.save(_combination_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_FACTOR_COMBINATION).is_dir()


def test_repository_path_has_no_symbol_segment(tmp_path: Path) -> None:
    """Saved partition path does not contain a symbol segment."""
    layout = StorageLayout(tmp_path)
    repository = FactorCombinationRepository(layout, ParquetStore())
    repository.save(_combination_frame(), **_partition_kwargs())  # type: ignore[arg-type]

    expected_path = (
        tmp_path
        / STORAGE_DIR_FACTOR_COMBINATION
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / _TIMEFRAME
        / f"{_YEAR}.parquet"
    )
    assert expected_path.exists()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_requires_repository() -> None:
    """Pipeline construction without a repository raises FactorCombinationError."""
    with pytest.raises(FactorCombinationError) as exc_info:
        FactorCombinationPipeline(repository=None)
    assert exc_info.value.error_code == "FCOMB_PIPE_REPOSITORY_REQUIRED"


def test_pipeline_registry_build_and_repository_save_called() -> None:
    """Pipeline delegates generation to the registry and persists via repository."""
    expected = _combination_frame()
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = FactorCombinationPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    input_frame = _fta_frame()

    result = pipeline.build(
        input_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert registry.calls == [input_frame]
    assert len(repository.save_calls) == 1
    saved_frame, saved_kwargs = repository.save_calls[0]
    assert saved_frame is expected
    assert saved_kwargs == {
        "manager": _MANAGER,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }
    assert result is expected


def test_pipeline_returned_dataframe_unchanged() -> None:
    """Pipeline returns the registry output frame without transformation."""
    expected = _combination_frame(combination_score=0.42)
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = FactorCombinationPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )

    result = pipeline.build(
        _fta_frame(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert result is expected
    assert_frame_equal(result, expected)


def test_pipeline_end_to_end_with_in_memory_repository() -> None:
    """Pipeline generates pairs and persists them through an in-memory repository."""
    store = _InMemoryDataStore()
    repository = FactorCombinationRepository(StorageLayout(Path("/data")), store)
    pipeline = FactorCombinationPipeline(repository=repository)
    kwargs = _partition_kwargs()

    with patch("cqros.factor_combination.engine.datetime") as mock_datetime:
        mock_datetime.now.return_value = _FIXED_NOW
        result = pipeline.build(
            _fta_frame(
                factor_names=["momentum", "rsi"],
                best_selection_scores=[0.90, 0.50],
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    assert result.schema == FACTOR_COMBINATION_SCHEMA
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, result)


def test_pipeline_build_accepts_no_symbol_kwarg() -> None:
    """Pipeline.build signature accepts no symbol keyword argument."""
    import inspect

    sig = inspect.signature(FactorCombinationPipeline.build)
    assert "symbol" not in sig.parameters


# ---------------------------------------------------------------------------
# Detailed Export
# ---------------------------------------------------------------------------


def test_detailed_audit_columns_includes_lineage_provenance() -> None:
    """DETAILED_AUDIT_COLUMNS includes all combination schema fields plus lineage."""
    for col in CANONICAL_COLUMN_ORDER:
        assert col in DETAILED_AUDIT_COLUMNS, f"missing combination column: {col}"
    assert "source_fta_version" in DETAILED_AUDIT_COLUMNS
    assert "source_selection_version" in DETAILED_AUDIT_COLUMNS
    assert "manager" in DETAILED_AUDIT_COLUMNS
    assert "exchange" in DETAILED_AUDIT_COLUMNS
    assert "market" in DETAILED_AUDIT_COLUMNS
    assert "year" in DETAILED_AUDIT_COLUMNS


def test_build_detailed_audit_frame_attaches_provenance() -> None:
    """build_detailed_audit_frame attaches all provenance columns correctly."""
    combination = _combination_frame()
    audit = build_detailed_audit_frame(
        combination,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        source_fta_version="fta-v1.0.0",
        source_selection_version="sel-v2.0.0",
    )
    assert "source_fta_version" in audit.columns
    assert "source_selection_version" in audit.columns
    assert audit["source_fta_version"].to_list() == ["fta-v1.0.0"]
    assert audit["source_selection_version"].to_list() == ["sel-v2.0.0"]
    assert audit["manager"].to_list() == [_MANAGER]
    assert audit["exchange"].to_list() == [_EXCHANGE]
    assert audit["market"].to_list() == [_MARKET]
    assert audit["year"].to_list() == [_YEAR]
    assert tuple(audit.columns) == DETAILED_AUDIT_COLUMNS


def test_build_detailed_audit_frame_rejects_empty() -> None:
    """build_detailed_audit_frame raises on empty combination frame."""
    empty = pl.DataFrame(schema=FACTOR_COMBINATION_SCHEMA)
    with pytest.raises(FactorCombinationError):
        build_detailed_audit_frame(
            empty,
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            year=_YEAR,
            source_fta_version="v1",
            source_selection_version="v2",
        )


def test_detailed_csv_path_layout(tmp_path: Path) -> None:
    """detailed_csv_path produces the expected path under the combination tier."""
    path = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert path == (
        tmp_path
        / STORAGE_DIR_FACTOR_COMBINATION
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / _TIMEFRAME
        / f"{_YEAR}_detailed.csv"
    )


def test_combined_detailed_csv_path_layout(tmp_path: Path) -> None:
    """combined_detailed_csv_path produces the expected path."""
    path = combined_detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert path == (
        tmp_path
        / STORAGE_DIR_FACTOR_COMBINATION
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / COMBINED_DETAILED_CSV_NAME
    )


def test_write_detailed_csv_creates_file(tmp_path: Path) -> None:
    """write_detailed_csv creates the CSV file with correct content."""
    combination = _combination_frame()
    audit = build_detailed_audit_frame(
        combination,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        source_fta_version="v1",
        source_selection_version="v2",
    )
    path = detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    returned = write_detailed_csv(audit, path)
    assert returned == path
    assert path.exists()
    loaded = pl.read_csv(path)
    assert loaded.height == 1


def test_write_combined_detailed_csv_creates_file(tmp_path: Path) -> None:
    """write_combined_detailed_csv concatenates frames and writes one CSV."""
    combination = _combination_frame()
    audit = build_detailed_audit_frame(
        combination,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
        source_fta_version="v1",
        source_selection_version="v2",
    )
    path = combined_detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    returned = write_combined_detailed_csv([audit, audit], path)
    assert returned == path
    assert path.exists()
    loaded = pl.read_csv(path)
    assert loaded.height == 2


def test_write_combined_detailed_csv_rejects_empty_list(tmp_path: Path) -> None:
    """write_combined_detailed_csv raises when given an empty frame list."""
    path = combined_detailed_csv_path(
        tmp_path,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    with pytest.raises(FactorCombinationError):
        write_combined_detailed_csv([], path)
