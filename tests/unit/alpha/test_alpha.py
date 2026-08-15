"""Unit tests for CQROS Alpha module contracts and orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.alpha.engine import (
    ALPHA_INPUT_COLUMNS,
    SimpleAlphaEngine,
    validate_factor_orthogonalization_frame,
)
from cqros.alpha.exceptions import AlphaError, AlphaException
from cqros.alpha.pipeline import AlphaPipeline
from cqros.alpha.registry import AlphaRegistry
from cqros.alpha.repository import AlphaPartitionRef, AlphaRepository
from cqros.alpha.schema import (
    ALPHA_COLUMNS,
    ALPHA_SCHEMA,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    AlphaStatus,
)
from cqros.alpha.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    AlphaVerifier,
)
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL, STORAGE_DIR_ALPHA
from cqros.core.exceptions import CQROSError, ResearchError
from cqros.core.types import FilePath
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_SYMBOL_ETH = "ETHUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_VALIDATION_START = 1_700_000_000_000
_VALIDATION_END = 1_700_000_003_600_000
_OPEN_TIME_1 = 1_700_000_000_000
_OPEN_TIME_2 = 1_700_000_003_600_000

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


class _RecordingObservationSource:
    """Test double that records load_panel calls and returns a fixed panel."""

    def __init__(self, panel: pl.DataFrame | None = None) -> None:
        self.panel = panel if panel is not None else _empty_observation_panel()
        self.calls: list[dict[str, object]] = []

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        self.calls.append(
            {
                "timeframe": timeframe,
                "factor_names": tuple(factor_names),
                "factor_versions": tuple(factor_versions),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return self.panel


class _StubEngine:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[tuple[pl.DataFrame, str]] = []

    def build(self, factor_orthogonalization: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
        self.calls.append((factor_orthogonalization, symbol))
        return self.output


class _StubRegistry:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[tuple[pl.DataFrame, str]] = []

    def build(self, factor_orthogonalization: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
        self.calls.append((factor_orthogonalization, symbol))
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
        symbol: str,
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
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                },
            )
        )


def _partition_kwargs(
    *,
    manager: str = _MANAGER,
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> dict[str, object]:
    """Build keyword arguments identifying one partition."""
    return {
        "manager": manager,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "symbol": symbol,
        "timeframe": timeframe,
        "year": year,
    }


def _empty_observation_panel() -> pl.DataFrame:
    """Return an empty long-format observation panel."""
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "open_time": pl.Int64,
            "factor_name": pl.String,
            "factor_version": pl.String,
            "factor_value": pl.Float64,
        }
    )


def _observation_panel(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a long-format observation panel from row dictionaries."""
    if len(rows) == 0:
        return _empty_observation_panel()
    return pl.DataFrame(rows).select(
        pl.col("symbol").cast(pl.String),
        pl.col("open_time").cast(pl.Int64),
        pl.col("factor_name").cast(pl.String),
        pl.col("factor_version").cast(pl.String),
        pl.col("factor_value").cast(pl.Float64),
    )


def _default_observation_panel() -> pl.DataFrame:
    """Two symbols × two open_times for ema/sma and rsi/momentum members."""
    return _observation_panel(
        [
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 1.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "sma_distance",
                "factor_version": "1.0.0",
                "factor_value": 3.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 2.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "sma_distance",
                "factor_version": "1.0.0",
                "factor_value": 4.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "rsi",
                "factor_version": "1.0.0",
                "factor_value": 10.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "momentum",
                "factor_version": "2.0.0",
                "factor_value": 30.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "rsi",
                "factor_version": "1.0.0",
                "factor_value": 20.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "momentum",
                "factor_version": "2.0.0",
                "factor_value": 40.0,
            },
            {
                "symbol": _SYMBOL_ETH,
                "open_time": _OPEN_TIME_1,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 5.0,
            },
            {
                "symbol": _SYMBOL_ETH,
                "open_time": _OPEN_TIME_1,
                "factor_name": "sma_distance",
                "factor_version": "1.0.0",
                "factor_value": 7.0,
            },
            {
                "symbol": _SYMBOL_ETH,
                "open_time": _OPEN_TIME_1,
                "factor_name": "rsi",
                "factor_version": "1.0.0",
                "factor_value": 50.0,
            },
            {
                "symbol": _SYMBOL_ETH,
                "open_time": _OPEN_TIME_1,
                "factor_name": "momentum",
                "factor_version": "2.0.0",
                "factor_value": 70.0,
            },
        ]
    )


def _orthogonalization_input_frame(
    *,
    combination_ids: list[str] | None = None,
    factor_names: list[list[str]] | None = None,
    factor_versions: list[list[str]] | None = None,
    factor_categories: list[list[str]] | None = None,
    timeframes: list[str] | None = None,
    combination_methods: list[str] | None = None,
    selected: list[bool] | None = None,
    statuses: list[str] | None = None,
    orthogonalization_ranks: list[int] | None = None,
    validation_start_times: list[int] | None = None,
    validation_end_times: list[int] | None = None,
) -> pl.DataFrame:
    """Build a combination-unit Factor Orthogonalization input frame."""
    combination_ids = (
        combination_ids
        if combination_ids is not None
        else ["ema_distance|sma_distance", "momentum|rsi"]
    )
    row_count = len(combination_ids)
    factor_names = (
        factor_names
        if factor_names is not None
        else [["ema_distance", "sma_distance"], ["momentum", "rsi"]][:row_count]
    )
    if len(factor_names) < row_count:
        factor_names = factor_names + [
            [f"factor_a_{index}", f"factor_b_{index}"]
            for index in range(len(factor_names), row_count)
        ]
    if factor_versions is None:
        factor_versions = []
        for names in factor_names:
            if names == ["momentum", "rsi"]:
                factor_versions.append(["2.0.0", "1.0.0"])
            else:
                factor_versions.append(["1.0.0"] * len(names))
    factor_categories = (
        factor_categories
        if factor_categories is not None
        else [["price", "price"] for _ in range(row_count)]
    )
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    combination_methods = (
        combination_methods if combination_methods is not None else ["equal_weight"] * row_count
    )
    selected = selected if selected is not None else [True] * row_count
    statuses = statuses if statuses is not None else ["PASS"] * row_count
    orthogonalization_ranks = (
        orthogonalization_ranks
        if orthogonalization_ranks is not None
        else list(range(1, row_count + 1))
    )
    validation_start_times = (
        validation_start_times
        if validation_start_times is not None
        else [_VALIDATION_START] * row_count
    )
    validation_end_times = (
        validation_end_times if validation_end_times is not None else [_VALIDATION_END] * row_count
    )
    return pl.DataFrame(
        {
            "combination_id": combination_ids,
            "factor_names": factor_names,
            "factor_versions": factor_versions,
            "factor_categories": factor_categories,
            "timeframe": timeframes,
            "combination_method": combination_methods,
            "selected": selected,
            "status": statuses,
            "orthogonalization_rank": orthogonalization_ranks,
            "validation_start_time": validation_start_times,
            "validation_end_time": validation_end_times,
        }
    )


def _alpha_frame(
    *,
    factor_set_id: str = "ema_distance|sma_distance",
    alpha_model: str = "placeholder",
    alpha_version: str = "1.0",
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    prediction_time: int = _OPEN_TIME_1,
    alpha_score: float | None = 2.0,
    prediction_horizon: int = 1,
    status: str = AlphaStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical Alpha frame for repository/verifier tests."""
    return pl.DataFrame(
        {
            "factor_set_id": [factor_set_id],
            "alpha_model": [alpha_model],
            "alpha_version": [alpha_version],
            "symbol": [symbol],
            "timeframe": [timeframe],
            "prediction_time": [prediction_time],
            "expected_return": [None],
            "alpha_score": [alpha_score],
            "confidence": [None],
            "uncertainty": [None],
            "prediction_horizon": [prediction_horizon],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build(
    frame: pl.DataFrame | None = None,
    *,
    symbol: str = _SYMBOL,
    observation_source: _RecordingObservationSource | None = None,
) -> pl.DataFrame:
    """Run ``SimpleAlphaEngine`` against an input frame."""
    source = (
        observation_source
        if observation_source is not None
        else _RecordingObservationSource(_default_observation_panel())
    )
    engine = SimpleAlphaEngine(observation_source=source)
    return engine.build(
        frame if frame is not None else _orthogonalization_input_frame(),
        symbol=symbol,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_canonical_columns_match_specification() -> None:
    """Canonical columns follow the declared alpha contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "factor_set_id",
        "alpha_model",
        "alpha_version",
        "symbol",
        "timeframe",
        "prediction_time",
        "expected_return",
        "alpha_score",
        "confidence",
        "uncertainty",
        "prediction_horizon",
        "status",
    )
    assert ALPHA_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == (
        "factor_set_id",
        "alpha_model",
        "alpha_version",
        "symbol",
        "timeframe",
        "prediction_time",
    )


def test_required_columns_match_canonical_order() -> None:
    """REQUIRED_COLUMNS mirrors CANONICAL_COLUMN_ORDER exactly."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))


def test_schema_dtypes_match_column_dtypes() -> None:
    """ALPHA_SCHEMA dtypes match COLUMN_DTYPES in order."""
    assert ALPHA_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert ALPHA_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["factor_set_id"] == pl.String
    assert COLUMN_DTYPES["alpha_model"] == pl.String
    assert COLUMN_DTYPES["alpha_version"] == pl.String
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["prediction_time"] == pl.Int64
    assert COLUMN_DTYPES["expected_return"] == pl.Float64
    assert COLUMN_DTYPES["alpha_score"] == pl.Float64
    assert COLUMN_DTYPES["confidence"] == pl.Float64
    assert COLUMN_DTYPES["uncertainty"] == pl.Float64
    assert COLUMN_DTYPES["prediction_horizon"] == pl.Int32
    assert COLUMN_DTYPES["status"] == pl.String


def test_schema_equality_matches_alpha_schema() -> None:
    """A frame built with COLUMN_DTYPES equals ALPHA_SCHEMA."""
    frame = _alpha_frame()
    assert frame.schema == ALPHA_SCHEMA
    assert tuple(frame.columns) == CANONICAL_COLUMN_ORDER


def test_alpha_status_enum_members() -> None:
    """AlphaStatus exposes PASS and FAIL members."""
    assert AlphaStatus.PASS.value == "PASS"
    assert AlphaStatus.FAIL.value == "FAIL"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_alpha_exception_inherits_from_research_error() -> None:
    """AlphaException is a ResearchError specialization."""
    assert issubclass(AlphaException, ResearchError)
    assert issubclass(AlphaException, CQROSError)


def test_alpha_error_inherits_from_alpha_exception() -> None:
    """AlphaError remains under AlphaException."""
    assert issubclass(AlphaError, AlphaException)
    assert issubclass(AlphaError, ResearchError)
    assert issubclass(AlphaError, CQROSError)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verifier_valid_dataframe_passes() -> None:
    """Verifier returns the same frame instance for a valid alpha frame."""
    frame = _alpha_frame()
    verified = AlphaVerifier().verify(frame)
    assert verified is frame
    assert_frame_equal(verified, frame)


def test_verifier_non_dataframe_fails() -> None:
    """Verifier rejects non-DataFrame inputs."""
    with pytest.raises(AlphaError) as exc_info:
        AlphaVerifier().verify("not-a-frame")
    assert exc_info.value.error_code == ERROR_FRAME_TYPE


def test_verifier_empty_dataframe_fails() -> None:
    """Verifier rejects empty alpha frames."""
    empty = pl.DataFrame(schema=ALPHA_SCHEMA)
    with pytest.raises(AlphaError) as exc_info:
        AlphaVerifier().verify(empty)
    assert exc_info.value.error_code == ERROR_FRAME_EMPTY


def test_verifier_missing_column_fails() -> None:
    """Verifier rejects frames missing required columns."""
    frame = _alpha_frame().drop("status")
    with pytest.raises(AlphaError) as exc_info:
        AlphaVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_wrong_column_order_fails() -> None:
    """Verifier rejects frames whose column order is non-canonical."""
    frame = _alpha_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    with pytest.raises(AlphaError) as exc_info:
        AlphaVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_COLUMN_ORDER


def test_verifier_wrong_dtype_fails() -> None:
    """Verifier rejects frames whose dtypes differ from the canonical schema."""
    frame = _alpha_frame().with_columns(pl.col("prediction_horizon").cast(pl.Float64))
    with pytest.raises(AlphaError) as exc_info:
        AlphaVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_requires_observation_source() -> None:
    """SimpleAlphaEngine rejects a missing observation source."""
    with pytest.raises(AlphaError) as exc_info:
        SimpleAlphaEngine(observation_source=None)
    assert exc_info.value.error_code == "ALPHA_OBSERVATION_SOURCE_REQUIRED"


def test_engine_validates_non_dataframe_input() -> None:
    """Engine rejects non-DataFrame inputs during structural validation."""
    with pytest.raises(AlphaError) as exc_info:
        validate_factor_orthogonalization_frame("not-a-frame")
    assert exc_info.value.error_code == "ALPHA_FRAME_TYPE"


def test_engine_validates_empty_input() -> None:
    """Engine rejects empty Factor Orthogonalization frames."""
    empty = pl.DataFrame(schema={"combination_id": pl.String}).clear()
    with pytest.raises(AlphaError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "ALPHA_FRAME_EMPTY"


def test_engine_validates_missing_columns() -> None:
    """Engine rejects frames missing required FO columns."""
    frame = pl.DataFrame({"combination_id": ["ema_distance|sma_distance"], "selected": [True]})
    with pytest.raises(AlphaError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "ALPHA_MISSING_COLUMNS"


def test_engine_input_columns_contract() -> None:
    """ALPHA_INPUT_COLUMNS enumerates every consumed FO column."""
    for column in (
        "combination_id",
        "factor_names",
        "factor_versions",
        "factor_categories",
        "timeframe",
        "combination_method",
        "selected",
        "status",
        "orthogonalization_rank",
        "validation_start_time",
        "validation_end_time",
    ):
        assert column in ALPHA_INPUT_COLUMNS


def test_engine_rejects_empty_symbol() -> None:
    """Engine rejects an empty symbol argument."""
    with pytest.raises(AlphaError) as exc_info:
        _build(symbol="")
    assert exc_info.value.error_code == "ALPHA_SYMBOL_INVALID"


def test_engine_only_selected_and_pass_rows_used() -> None:
    """Only selected PASS combinations participate; rejected rows emit nothing."""
    source = _RecordingObservationSource(_default_observation_panel())
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=[
                "ema_distance|sma_distance",
                "rejected|pair",
                "momentum|rsi",
            ],
            factor_names=[
                ["ema_distance", "sma_distance"],
                ["volume", "vwap"],
                ["momentum", "rsi"],
            ],
            factor_versions=[["1.0.0", "1.0.0"], ["1.0.0", "1.0.0"], ["2.0.0", "1.0.0"]],
            selected=[True, True, True],
            statuses=["PASS", "FAIL", "PASS"],
            orthogonalization_ranks=[1, 2, 3],
        ),
        observation_source=source,
    )
    assert set(result["factor_set_id"].to_list()) == {
        "ema_distance|sma_distance",
        "momentum|rsi",
    }
    assert "rejected|pair" not in result["factor_set_id"].to_list()
    assert "volume" not in result["factor_set_id"].to_list()


def test_engine_rejected_combination_produces_zero_rows() -> None:
    """A fully rejected FO frame raises because no accepted combinations remain."""
    with pytest.raises(AlphaError) as exc_info:
        _build(
            _orthogonalization_input_frame(
                combination_ids=["rejected|pair"],
                factor_names=[["volume", "vwap"]],
                selected=[False],
                statuses=["FAIL"],
            )
        )
    assert exc_info.value.error_code == "ALPHA_NO_COMBINATIONS"


def test_engine_factor_set_id_equals_combination_id() -> None:
    """factor_set_id equals combination_id and never explodes members."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        )
    )
    assert result["factor_set_id"].unique().to_list() == ["ema_distance|sma_distance"]
    assert "ema_distance" not in result["factor_set_id"].to_list()
    assert "sma_distance" not in result["factor_set_id"].to_list()
    assert result.height == 2


def test_engine_timeframe_comes_from_fo() -> None:
    """Alpha timeframe is inherited from FO timeframe, not invented."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            timeframes=["4h"],
            orthogonalization_ranks=[1],
        )
    )
    assert set(result["timeframe"].to_list()) == {"4h"}


def test_engine_symbol_is_populated() -> None:
    """Generated rows carry the requested symbol."""
    result = _build(symbol=_SYMBOL)
    assert set(result["symbol"].to_list()) == {_SYMBOL}


def test_engine_prediction_time_equals_open_time() -> None:
    """prediction_time equals observation open_time."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        )
    )
    assert sorted(result["prediction_time"].to_list()) == [_OPEN_TIME_1, _OPEN_TIME_2]


def test_engine_passes_validation_window_to_observation_source() -> None:
    """Observation loading receives FO validation window bounds."""
    source = _RecordingObservationSource(_default_observation_panel())
    _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        ),
        observation_source=source,
    )
    assert len(source.calls) == 1
    assert source.calls[0]["start_time"] == _VALIDATION_START
    assert source.calls[0]["end_time"] == _VALIDATION_END
    assert source.calls[0]["timeframe"] == _TIMEFRAME


def test_engine_equal_weight_mean_is_correct() -> None:
    """alpha_score is the arithmetic mean of finite member values."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        )
    ).sort("prediction_time")
    assert result["alpha_score"].to_list() == [2.0, 3.0]


def test_engine_paired_name_version_matching() -> None:
    """Member matching uses paired name/version keys, not independent sets."""
    panel = _observation_panel(
        [
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "rsi",
                "factor_version": "1.0.0",
                "factor_value": 10.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "momentum",
                "factor_version": "2.0.0",
                "factor_value": 30.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "momentum",
                "factor_version": "1.0.0",
                "factor_value": 999.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "rsi",
                "factor_version": "2.0.0",
                "factor_value": 999.0,
            },
        ]
    )
    source = _RecordingObservationSource(panel)
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["momentum|rsi"],
            factor_names=[["momentum", "rsi"]],
            factor_versions=[["2.0.0", "1.0.0"]],
            orthogonalization_ranks=[1],
        ),
        observation_source=source,
    )
    assert result.height == 1
    assert result["alpha_score"].to_list() == [20.0]


def test_engine_missing_member_produces_no_signal() -> None:
    """Absent required member identity yields no invented Alpha rows."""
    panel = _observation_panel(
        [
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 1.0,
            }
        ]
    )
    source = _RecordingObservationSource(panel)
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        ),
        observation_source=source,
    )
    assert result.height == 0
    assert result.schema == ALPHA_SCHEMA


def test_engine_non_finite_member_skips_timestamp() -> None:
    """Timestamps with any non-finite member do not emit Alpha rows."""
    panel = _observation_panel(
        [
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 1.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_1,
                "factor_name": "sma_distance",
                "factor_version": "1.0.0",
                "factor_value": float("nan"),
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "ema_distance",
                "factor_version": "1.0.0",
                "factor_value": 2.0,
            },
            {
                "symbol": _SYMBOL,
                "open_time": _OPEN_TIME_2,
                "factor_name": "sma_distance",
                "factor_version": "1.0.0",
                "factor_value": 4.0,
            },
        ]
    )
    source = _RecordingObservationSource(panel)
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        ),
        observation_source=source,
    )
    assert result.height == 1
    assert result["prediction_time"].to_list() == [_OPEN_TIME_2]
    assert result["alpha_score"].to_list() == [3.0]


def test_engine_combination_symbol_identities_are_independent() -> None:
    """Two combinations × two symbols form independent Alpha identities."""
    source = _RecordingObservationSource(_default_observation_panel())
    frame = _orthogonalization_input_frame(
        combination_ids=["ema_distance|sma_distance", "momentum|rsi"],
        factor_names=[["ema_distance", "sma_distance"], ["momentum", "rsi"]],
        factor_versions=[["1.0.0", "1.0.0"], ["2.0.0", "1.0.0"]],
        orthogonalization_ranks=[1, 2],
    )
    btc = _build(frame, symbol=_SYMBOL, observation_source=source)
    eth = _build(frame, symbol=_SYMBOL_ETH, observation_source=source)

    btc_ids = {
        (row["factor_set_id"], row["symbol"], row["prediction_time"])
        for row in btc.select("factor_set_id", "symbol", "prediction_time").to_dicts()
    }
    eth_ids = {
        (row["factor_set_id"], row["symbol"], row["prediction_time"])
        for row in eth.select("factor_set_id", "symbol", "prediction_time").to_dicts()
    }

    assert ("ema_distance|sma_distance", _SYMBOL, _OPEN_TIME_1) in btc_ids
    assert ("ema_distance|sma_distance", _SYMBOL, _OPEN_TIME_2) in btc_ids
    assert ("momentum|rsi", _SYMBOL, _OPEN_TIME_1) in btc_ids
    assert ("momentum|rsi", _SYMBOL, _OPEN_TIME_2) in btc_ids
    assert ("ema_distance|sma_distance", _SYMBOL_ETH, _OPEN_TIME_1) in eth_ids
    assert ("momentum|rsi", _SYMBOL_ETH, _OPEN_TIME_1) in eth_ids
    assert btc_ids.isdisjoint(eth_ids)

    btc_scores = {
        (row["factor_set_id"], row["prediction_time"]): row["alpha_score"]
        for row in btc.select("factor_set_id", "prediction_time", "alpha_score").to_dicts()
    }
    eth_scores = {
        (row["factor_set_id"], row["prediction_time"]): row["alpha_score"]
        for row in eth.select("factor_set_id", "prediction_time", "alpha_score").to_dicts()
    }
    assert btc_scores[("ema_distance|sma_distance", _OPEN_TIME_1)] == 2.0
    assert btc_scores[("momentum|rsi", _OPEN_TIME_1)] == 20.0
    assert eth_scores[("ema_distance|sma_distance", _OPEN_TIME_1)] == 6.0
    assert eth_scores[("momentum|rsi", _OPEN_TIME_1)] == 60.0


def test_engine_placeholder_metrics_remain_null() -> None:
    """expected_return/confidence/uncertainty remain null placeholders."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        )
    )
    assert result["expected_return"].null_count() == result.height
    assert result["confidence"].null_count() == result.height
    assert result["uncertainty"].null_count() == result.height
    assert result["alpha_score"].null_count() == 0


def test_engine_output_matches_alpha_schema() -> None:
    """Engine output conforms to ALPHA_SCHEMA and canonical order."""
    result = _build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance"],
            factor_names=[["ema_distance", "sma_distance"]],
            orthogonalization_ranks=[1],
        )
    )
    assert result.schema == ALPHA_SCHEMA
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert set(result["status"].to_list()) == {AlphaStatus.PASS.value}
    assert set(result["alpha_model"].to_list()) == {"placeholder"}
    assert set(result["alpha_version"].to_list()) == {"1.0"}
    assert set(result["prediction_horizon"].to_list()) == {1}


def test_engine_deterministic_output() -> None:
    """Identical FO + observations + symbol inputs produce identical outputs."""
    frame = _orthogonalization_input_frame()
    source_panel = _default_observation_panel()
    first = _build(frame, observation_source=_RecordingObservationSource(source_panel))
    second = _build(frame, observation_source=_RecordingObservationSource(source_panel))
    assert_frame_equal(first, second)


def test_engine_does_not_mutate_input() -> None:
    """Engine leaves the caller-supplied frame unchanged."""
    frame = _orthogonalization_input_frame()
    before = frame.clone()
    _ = _build(frame)
    assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_requires_engine() -> None:
    """Registry construction without an engine raises AlphaError."""
    with pytest.raises(AlphaError) as exc_info:
        AlphaRegistry(engine=None)
    assert exc_info.value.error_code == "ALPHA_REGISTRY_ENGINE_REQUIRED"


def test_registry_injected_engine_delegation() -> None:
    """Registry.build delegates exclusively to the injected engine with symbol."""
    expected = _alpha_frame()
    stub = _StubEngine(expected)
    registry = AlphaRegistry(engine=stub)
    input_frame = _orthogonalization_input_frame()

    result = registry.build(input_frame, symbol=_SYMBOL)

    assert result is expected
    assert stub.calls == [(input_frame, _SYMBOL)]
    assert registry.engine is stub


def test_registry_engine_property_exposes_owned_engine() -> None:
    """Registry.engine returns the owned engine instance."""
    engine = SimpleAlphaEngine(observation_source=_RecordingObservationSource())
    registry = AlphaRegistry(engine=engine)
    assert registry.engine is engine


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """AlphaPartitionRef is a frozen immutable dataclass."""
    ref = AlphaPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ref.symbol = "ETHUSDT"  # type: ignore[misc]


def test_repository_save_exists_load_round_trip() -> None:
    """Saved frames can be retrieved by exists() and load()."""
    store = _InMemoryDataStore()
    repository = AlphaRepository(StorageLayout(Path("/data")), store)
    frame = _alpha_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == ALPHA_SCHEMA


def test_repository_empty_load_returns_schema_frame() -> None:
    """load() returns an empty ALPHA_SCHEMA frame when missing."""
    repository = AlphaRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    loaded = repository.load(**_partition_kwargs())  # type: ignore[arg-type]
    assert loaded.height == 0
    assert loaded.schema == ALPHA_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


def test_repository_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = AlphaRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_alpha_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_repository_delete_absent_succeeds_silently() -> None:
    """Deleting a missing partition succeeds without raising."""
    repository = AlphaRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


def test_repository_save_rejects_missing_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = AlphaRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(AlphaError) as exc_info:
        repository.save(
            pl.DataFrame({"factor_set_id": ["ema_distance|sma_distance"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "ALPHA_REPO_MISSING_COLUMNS"


def test_repository_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs."""
    repository = AlphaRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(AlphaError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "ALPHA_REPO_FRAME_TYPE"


def test_repository_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = AlphaRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    repository.save(
        _alpha_frame(factor_set_id="ema_distance|sma_distance"),
        **kwargs,  # type: ignore[arg-type]
    )
    repository.save(
        _alpha_frame(factor_set_id="momentum|rsi"),
        **kwargs,  # type: ignore[arg-type]
    )

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["factor_set_id"].to_list() == ["momentum|rsi"]


def test_repository_parquet_store_path_contains_alpha_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the alpha storage directory."""
    layout = StorageLayout(tmp_path)
    repository = AlphaRepository(layout, ParquetStore())
    repository.save(_alpha_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_ALPHA).is_dir()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_requires_registry() -> None:
    """Pipeline construction without a registry raises AlphaError."""
    with pytest.raises(AlphaError) as exc_info:
        AlphaPipeline(registry=None, repository=_StubRepository())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "ALPHA_PIPE_REGISTRY_REQUIRED"


def test_pipeline_requires_repository() -> None:
    """Pipeline construction without a repository raises AlphaError."""
    engine = SimpleAlphaEngine(observation_source=_RecordingObservationSource())
    with pytest.raises(AlphaError) as exc_info:
        AlphaPipeline(registry=AlphaRegistry(engine=engine), repository=None)
    assert exc_info.value.error_code == "ALPHA_PIPE_REPOSITORY_REQUIRED"


def test_pipeline_passes_symbol_into_generation() -> None:
    """Pipeline passes symbol into registry generation, not only persistence."""
    expected = _alpha_frame()
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = AlphaPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    input_frame = _orthogonalization_input_frame()

    result = pipeline.build(
        input_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert registry.calls == [(input_frame, _SYMBOL)]
    assert len(repository.save_calls) == 1
    saved_frame, saved_kwargs = repository.save_calls[0]
    assert saved_frame is expected
    assert saved_kwargs["symbol"] == _SYMBOL
    assert result is expected


def test_pipeline_returned_dataframe_unchanged() -> None:
    """Pipeline returns the registry output frame without transformation."""
    expected = _alpha_frame(factor_set_id="ema_distance|sma_distance")
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = AlphaPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )

    result = pipeline.build(
        _orthogonalization_input_frame(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert result is expected
    assert_frame_equal(result, expected)


def test_pipeline_end_to_end_with_in_memory_repository() -> None:
    """Pipeline generates combination-unit rows and persists them."""
    store = _InMemoryDataStore()
    repository = AlphaRepository(StorageLayout(Path("/data")), store)
    engine = SimpleAlphaEngine(
        observation_source=_RecordingObservationSource(_default_observation_panel())
    )
    pipeline = AlphaPipeline(registry=AlphaRegistry(engine=engine), repository=repository)
    kwargs = _partition_kwargs()

    result = pipeline.build(
        _orthogonalization_input_frame(
            combination_ids=["ema_distance|sma_distance", "momentum|rsi"],
            factor_names=[["ema_distance", "sma_distance"], ["momentum", "rsi"]],
            factor_versions=[["1.0.0", "1.0.0"], ["2.0.0", "1.0.0"]],
            orthogonalization_ranks=[2, 1],
        ),
        **kwargs,  # type: ignore[arg-type]
    )

    assert result.schema == ALPHA_SCHEMA
    assert set(result["symbol"].to_list()) == {_SYMBOL}
    assert set(result["factor_set_id"].to_list()) == {
        "ema_distance|sma_distance",
        "momentum|rsi",
    }
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, result)
