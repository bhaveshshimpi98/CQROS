"""Unit tests for CQROS Regime module contracts and orchestration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL, STORAGE_DIR_REGIME
from cqros.core.exceptions import CQROSError, ResearchError
from cqros.core.types import FilePath
from cqros.regime.engine import (
    REGIME_INPUT_COLUMNS,
    SimpleRegimeEngine,
    validate_alpha_frame,
)
from cqros.regime.exceptions import RegimeError, RegimeException
from cqros.regime.pipeline import RegimePipeline
from cqros.regime.registry import RegimeRegistry
from cqros.regime.repository import RegimePartitionRef, RegimeRepository
from cqros.regime.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REGIME_COLUMNS,
    REGIME_SCHEMA,
    REQUIRED_COLUMNS,
    RegimeStatus,
)
from cqros.regime.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    RegimeVerifier,
)
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026

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

    def build(self, alpha: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(alpha)
        return self.output


class _StubRegistry:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[pl.DataFrame] = []

    def build(self, alpha: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(alpha)
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


def _alpha_input_frame(
    *,
    factor_set_ids: list[str] | None = None,
    alpha_models: list[str] | None = None,
    alpha_versions: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    prediction_times: list[int] | None = None,
    alpha_scores: list[float | None] | None = None,
    statuses: list[str] | None = None,
    include_prediction_score: bool = False,
) -> pl.DataFrame:
    """Build an Alpha-style input for the regime engine."""
    row_count = max(
        len(values)
        for values in (
            factor_set_ids if factor_set_ids is not None else ["momentum|1.0.0", "rsi|1.0.0"],
            alpha_models or [],
            alpha_versions or [],
            symbols or [],
            timeframes or [],
            prediction_times or [],
            alpha_scores or [],
            statuses or [],
        )
    )
    factor_set_ids = (
        factor_set_ids
        if factor_set_ids is not None
        else (
            ["momentum|1.0.0", "rsi|1.0.0"]
            if row_count == 2
            else [f"factor_{index}|1.0.0" for index in range(row_count)]
        )
    )
    alpha_models = alpha_models if alpha_models is not None else ["placeholder"] * row_count
    alpha_versions = alpha_versions if alpha_versions is not None else ["1.0"] * row_count
    symbols = symbols if symbols is not None else [_SYMBOL] * row_count
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    prediction_times = (
        prediction_times
        if prediction_times is not None
        else [1000 * (index + 1) for index in range(row_count)]
    )
    alpha_scores = (
        alpha_scores
        if alpha_scores is not None
        else [0.5 + (0.1 * index) for index in range(row_count)]
    )
    statuses = statuses if statuses is not None else ["PASS"] * row_count
    payload: dict[str, object] = {
        "factor_set_id": factor_set_ids,
        "alpha_model": alpha_models,
        "alpha_version": alpha_versions,
        "symbol": symbols,
        "timeframe": timeframes,
        "prediction_time": prediction_times,
        "alpha_score": alpha_scores,
        "status": statuses,
    }
    if include_prediction_score:
        # Legacy column must be ignored when present; alpha_score is canonical.
        payload["prediction_score"] = [999.0] * row_count
    return pl.DataFrame(payload)


def _regime_frame(
    *,
    regime_id: str = "momentum|1.0.0|placeholder|1.0",
    factor_set_id: str = "momentum|1.0.0",
    alpha_id: str = "momentum|1.0.0|placeholder|1.0",
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    regime_time: datetime | None = None,
    regime_type: str = "UNKNOWN",
    regime_probability: float = 1.0,
    regime_score: float = 0.5,
    regime_version: str = "1.0.0",
    status: str = RegimeStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical Regime frame for repository/verifier tests."""
    if regime_time is None:
        regime_time = datetime(1970, 1, 1, 0, 0, 1)
    empty_metadata: list[str] = []
    return pl.DataFrame(
        {
            "regime_id": [regime_id],
            "factor_set_id": [factor_set_id],
            "alpha_id": [alpha_id],
            "symbol": [symbol],
            "timeframe": [timeframe],
            "regime_time": [regime_time],
            "regime_type": [regime_type],
            "regime_probability": [regime_probability],
            "regime_score": [regime_score],
            "regime_version": [regime_version],
            "status": [status],
            "metadata": [empty_metadata],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build(frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run ``SimpleRegimeEngine`` against an input frame."""
    engine = SimpleRegimeEngine()
    return engine.build(frame if frame is not None else _alpha_input_frame())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_canonical_columns_match_specification() -> None:
    """Canonical columns follow the declared regime contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "regime_id",
        "factor_set_id",
        "alpha_id",
        "symbol",
        "timeframe",
        "regime_time",
        "regime_type",
        "regime_probability",
        "regime_score",
        "regime_version",
        "status",
        "metadata",
    )
    assert REGIME_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == (
        "regime_id",
        "symbol",
        "timeframe",
        "regime_time",
    )


def test_required_columns_match_canonical_order() -> None:
    """REQUIRED_COLUMNS mirrors CANONICAL_COLUMN_ORDER exactly."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))


def test_schema_dtypes_match_column_dtypes() -> None:
    """REGIME_SCHEMA dtypes match COLUMN_DTYPES in order."""
    assert REGIME_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert REGIME_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["regime_id"] == pl.String
    assert COLUMN_DTYPES["factor_set_id"] == pl.String
    assert COLUMN_DTYPES["alpha_id"] == pl.String
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["regime_time"] == pl.Datetime("ms")
    assert COLUMN_DTYPES["regime_type"] == pl.String
    assert COLUMN_DTYPES["regime_probability"] == pl.Float64
    assert COLUMN_DTYPES["regime_score"] == pl.Float64
    assert COLUMN_DTYPES["regime_version"] == pl.String
    assert COLUMN_DTYPES["status"] == pl.String
    assert COLUMN_DTYPES["metadata"] == pl.List(pl.String)


def test_schema_equality_matches_regime_schema() -> None:
    """A frame built with COLUMN_DTYPES equals REGIME_SCHEMA."""
    frame = _regime_frame()
    assert frame.schema == REGIME_SCHEMA
    assert tuple(frame.columns) == CANONICAL_COLUMN_ORDER


def test_regime_status_enum_members() -> None:
    """RegimeStatus exposes PASS and FAIL members."""
    assert RegimeStatus.PASS.value == "PASS"
    assert RegimeStatus.FAIL.value == "FAIL"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_regime_exception_inherits_from_research_error() -> None:
    """RegimeException is a ResearchError specialization."""
    assert issubclass(RegimeException, ResearchError)
    assert issubclass(RegimeException, CQROSError)


def test_regime_error_inherits_from_regime_exception() -> None:
    """RegimeError remains under RegimeException."""
    assert issubclass(RegimeError, RegimeException)
    assert issubclass(RegimeError, ResearchError)
    assert issubclass(RegimeError, CQROSError)


def test_regime_error_supports_structured_fields() -> None:
    """RegimeError accepts message, error_code, and details."""
    error = RegimeError(
        "regime failure",
        error_code="REGIME_TEST",
        details={"dataset": "regime", "rows": 0},
    )
    assert error.message == "regime failure"
    assert error.error_code == "REGIME_TEST"
    assert dict(error.details) == {"dataset": "regime", "rows": 0}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verifier_valid_dataframe_passes() -> None:
    """Verifier returns the same frame instance for a valid regime frame."""
    frame = _regime_frame()
    verified = RegimeVerifier().verify(frame)
    assert verified is frame
    assert_frame_equal(verified, frame)


def test_verifier_non_dataframe_fails() -> None:
    """Verifier rejects non-DataFrame inputs."""
    with pytest.raises(RegimeError) as exc_info:
        RegimeVerifier().verify("not-a-frame")
    assert exc_info.value.error_code == ERROR_FRAME_TYPE


def test_verifier_empty_dataframe_fails() -> None:
    """Verifier rejects empty regime frames."""
    empty = pl.DataFrame(schema=REGIME_SCHEMA)
    with pytest.raises(RegimeError) as exc_info:
        RegimeVerifier().verify(empty)
    assert exc_info.value.error_code == ERROR_FRAME_EMPTY


def test_verifier_missing_column_fails() -> None:
    """Verifier rejects frames missing required columns."""
    frame = _regime_frame().drop("status")
    with pytest.raises(RegimeError) as exc_info:
        RegimeVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_wrong_column_order_fails() -> None:
    """Verifier rejects frames whose column order is non-canonical."""
    frame = _regime_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    with pytest.raises(RegimeError) as exc_info:
        RegimeVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_COLUMN_ORDER


def test_verifier_wrong_dtype_fails() -> None:
    """Verifier rejects frames whose dtypes differ from the canonical schema."""
    frame = _regime_frame().with_columns(pl.col("regime_score").cast(pl.Int64))
    with pytest.raises(RegimeError) as exc_info:
        RegimeVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_validates_non_dataframe_input() -> None:
    """Engine rejects non-DataFrame inputs during structural validation."""
    with pytest.raises(RegimeError) as exc_info:
        validate_alpha_frame("not-a-frame")
    assert exc_info.value.error_code == "REGIME_FRAME_TYPE"


def test_engine_validates_empty_input() -> None:
    """Engine rejects empty Alpha frames."""
    empty = pl.DataFrame(schema={"factor_set_id": pl.String}).clear()
    with pytest.raises(RegimeError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "REGIME_FRAME_EMPTY"


def test_engine_validates_missing_columns() -> None:
    """Engine rejects frames missing required input columns."""
    frame = pl.DataFrame({"factor_set_id": ["momentum|1.0.0"], "status": ["PASS"]})
    with pytest.raises(RegimeError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "REGIME_MISSING_COLUMNS"


def test_engine_input_columns_contract() -> None:
    """REGIME_INPUT_COLUMNS enumerates every consumed Alpha column."""
    assert REGIME_INPUT_COLUMNS == (
        "factor_set_id",
        "alpha_model",
        "alpha_version",
        "symbol",
        "timeframe",
        "prediction_time",
        "alpha_score",
        "status",
    )
    assert "prediction_score" not in REGIME_INPUT_COLUMNS


def test_engine_requires_alpha_score_not_prediction_score() -> None:
    """Engine requires alpha_score and does not require prediction_score."""
    frame = (
        _alpha_input_frame().drop("alpha_score").with_columns(pl.lit(0.5).alias("prediction_score"))
    )
    with pytest.raises(RegimeError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "REGIME_MISSING_COLUMNS"
    assert "alpha_score" in dict(exc_info.value.details)["missing_columns"]


def test_engine_ignores_legacy_prediction_score_when_present() -> None:
    """Legacy prediction_score columns are ignored; regime_score uses alpha_score."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0"],
            alpha_scores=[0.42],
            include_prediction_score=True,
        )
    )
    assert result.height == 1
    assert result["regime_score"].to_list() == [0.42]


def test_engine_only_pass_status_rows_used() -> None:
    """Only rows with status == PASS participate in regime generation."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0", "rsi|1.0.0", "volume|1.0.0"],
            alpha_scores=[0.5, 0.6, 0.7],
            statuses=["PASS", "FAIL", "PASS"],
        )
    )
    assert result.height == 2
    assert result["factor_set_id"].to_list() == ["momentum|1.0.0", "volume|1.0.0"]
    assert result["status"].to_list() == [RegimeStatus.PASS.value, RegimeStatus.PASS.value]


def test_engine_only_finite_non_null_alpha_score_rows_used() -> None:
    """Only rows with finite non-null alpha_score participate in regime generation."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0", "rsi|1.0.0", "volume|1.0.0"],
            alpha_scores=[0.5, None, 0.7],
            statuses=["PASS", "PASS", "PASS"],
        )
    )
    assert result.height == 2
    assert result["factor_set_id"].to_list() == ["momentum|1.0.0", "volume|1.0.0"]


def test_engine_rejects_non_finite_alpha_scores() -> None:
    """NaN and infinite alpha_score values do not become valid regime rows."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0", "rsi|1.0.0", "volume|1.0.0", "macd|1.0.0"],
            alpha_scores=[0.5, float("nan"), float("inf"), float("-inf")],
            statuses=["PASS", "PASS", "PASS", "PASS"],
        )
    )
    assert result.height == 1
    assert result["factor_set_id"].to_list() == ["momentum|1.0.0"]
    assert result["regime_score"].to_list() == [0.5]


def test_engine_no_surviving_alpha_raises() -> None:
    """Frames with no PASS finite non-null alpha_score rows raise REGIME_NO_ALPHA."""
    with pytest.raises(RegimeError) as exc_info:
        _build(
            _alpha_input_frame(
                factor_set_ids=["momentum|1.0.0", "rsi|1.0.0"],
                alpha_scores=[None, 0.5],
                statuses=["PASS", "FAIL"],
            )
        )
    assert exc_info.value.error_code == "REGIME_NO_ALPHA"


def test_engine_only_non_finite_scores_raises() -> None:
    """PASS rows with only non-finite alpha_score values raise REGIME_NO_ALPHA."""
    with pytest.raises(RegimeError) as exc_info:
        _build(
            _alpha_input_frame(
                factor_set_ids=["momentum|1.0.0", "rsi|1.0.0"],
                alpha_scores=[float("nan"), float("inf")],
                statuses=["PASS", "PASS"],
            )
        )
    assert exc_info.value.error_code == "REGIME_NO_ALPHA"


def test_engine_regime_id_and_alpha_id_deterministic() -> None:
    """regime_id and alpha_id are factor_set_id|alpha_model|alpha_version."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0", "rsi|2.0.0"],
            alpha_models=["placeholder", "linear"],
            alpha_versions=["1.0", "2.0"],
            alpha_scores=[0.5, 0.6],
        )
    )
    assert result["regime_id"].to_list() == [
        "momentum|1.0.0|placeholder|1.0",
        "rsi|2.0.0|linear|2.0",
    ]
    assert result["alpha_id"].to_list() == result["regime_id"].to_list()


def test_engine_field_mappings_and_placeholders() -> None:
    """Engine maps alpha fields and applies placeholder regime values."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0"],
            alpha_models=["placeholder"],
            alpha_versions=["1.0"],
            symbols=["ETHUSDT"],
            timeframes=["4h"],
            prediction_times=[1500],
            alpha_scores=[0.42],
            statuses=["PASS"],
        )
    )
    assert result.height == 1
    assert result["factor_set_id"].to_list() == ["momentum|1.0.0"]
    assert result["symbol"].to_list() == ["ETHUSDT"]
    assert result["timeframe"].to_list() == ["4h"]
    assert result["regime_type"].to_list() == ["UNKNOWN"]
    assert result["regime_probability"].to_list() == [1.0]
    assert result["regime_score"].to_list() == [0.42]
    assert result["regime_version"].to_list() == ["1.0.0"]
    assert result["status"].to_list() == [RegimeStatus.PASS.value]
    assert result["metadata"].to_list() == [[]]
    assert result["regime_time"].dtype == pl.Datetime("ms")
    assert result["regime_time"].to_list() == [datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=None)]


def test_engine_output_matches_regime_schema() -> None:
    """Engine output conforms to REGIME_SCHEMA and canonical order."""
    result = _build()
    assert result.schema == REGIME_SCHEMA
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result["status"].to_list() == [RegimeStatus.PASS.value, RegimeStatus.PASS.value]
    assert result["regime_type"].to_list() == ["UNKNOWN", "UNKNOWN"]
    assert result["regime_version"].to_list() == ["1.0.0", "1.0.0"]
    assert result["regime_probability"].to_list() == [1.0, 1.0]


def test_engine_deterministic_output() -> None:
    """Identical inputs produce identical regime outputs."""
    frame = _alpha_input_frame(
        factor_set_ids=["rsi|1.0.0", "momentum|1.0.0", "volume|1.0.0"],
        alpha_scores=[0.3, 0.5, 0.7],
    )
    first = _build(frame)
    second = _build(frame)
    assert_frame_equal(first, second)


def test_engine_preserves_surviving_row_order() -> None:
    """Surviving alpha rows emit regime rows in input order."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["zulu|1.0.0", "alpha|1.0.0", "mike|1.0.0"],
            alpha_scores=[0.1, 0.2, 0.3],
        )
    )
    assert result["factor_set_id"].to_list() == [
        "zulu|1.0.0",
        "alpha|1.0.0",
        "mike|1.0.0",
    ]


def test_engine_does_not_mutate_input() -> None:
    """Engine leaves the caller-supplied frame unchanged."""
    frame = _alpha_input_frame()
    before = frame.clone()
    _ = _build(frame)
    assert_frame_equal(frame, before)


def test_engine_regime_score_equals_alpha_score() -> None:
    """regime_score equals alpha_score for every accepted row."""
    alpha_scores = [0.11, -0.25, 1.5]
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["a|1.0.0", "b|1.0.0", "c|1.0.0"],
            alpha_scores=alpha_scores,
        )
    )
    assert result["regime_score"].to_list() == alpha_scores
    assert result["regime_probability"].to_list() == [1.0, 1.0, 1.0]
    assert result["regime_probability"].to_list() != result["regime_score"].to_list()


def test_engine_preserves_factor_set_id_without_explosion() -> None:
    """factor_set_id is preserved one-to-one; combinations are not exploded."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["combo_a", "combo_b"],
            alpha_scores=[0.2, 0.3],
            prediction_times=[1000, 2000],
        )
    )
    assert result.height == 2
    assert result["factor_set_id"].to_list() == ["combo_a", "combo_b"]


def test_engine_timestamp_maps_prediction_time_to_regime_time() -> None:
    """regime_time represents the same instant as Alpha prediction_time."""
    prediction_times = [1_700_000_000_000, 1_700_000_003_600]
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["combo_a", "combo_b"],
            prediction_times=prediction_times,
            alpha_scores=[0.1, 0.2],
        )
    )
    assert result["regime_time"].cast(pl.Int64).to_list() == prediction_times


def test_engine_symbol_isolation() -> None:
    """BTCUSDT and ETHUSDT rows remain isolated through Regime transformation."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["combo_a", "combo_a"],
            symbols=["BTCUSDT", "ETHUSDT"],
            alpha_scores=[0.1, 0.2],
            prediction_times=[1000, 1000],
        )
    )
    assert result["symbol"].to_list() == ["BTCUSDT", "ETHUSDT"]
    assert set(result["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}
    assert "BTCUSDT" in result["symbol"].to_list()
    btc = result.filter(pl.col("symbol") == "BTCUSDT")
    eth = result.filter(pl.col("symbol") == "ETHUSDT")
    assert btc.height == 1
    assert eth.height == 1
    assert btc["regime_score"].to_list() == [0.1]
    assert eth["regime_score"].to_list() == [0.2]


def test_engine_timeframe_isolation() -> None:
    """1h and 4h rows remain isolated through Regime transformation."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["combo_a", "combo_a"],
            timeframes=["1h", "4h"],
            alpha_scores=[0.1, 0.2],
            prediction_times=[1000, 1000],
        )
    )
    assert result["timeframe"].to_list() == ["1h", "4h"]
    one_h = result.filter(pl.col("timeframe") == "1h")
    four_h = result.filter(pl.col("timeframe") == "4h")
    assert one_h.height == 1
    assert four_h.height == 1
    assert one_h["regime_score"].to_list() == [0.1]
    assert four_h["regime_score"].to_list() == [0.2]


def test_engine_rejected_alpha_rows_do_not_become_pass_regime() -> None:
    """Non-PASS Alpha rows never appear as Regime PASS rows."""
    result = _build(
        _alpha_input_frame(
            factor_set_ids=["pass_combo", "fail_combo"],
            alpha_scores=[0.5, 0.9],
            statuses=["PASS", "FAIL"],
        )
    )
    assert result.height == 1
    assert result["factor_set_id"].to_list() == ["pass_combo"]
    assert result["status"].to_list() == [RegimeStatus.PASS.value]
    assert "fail_combo" not in result["factor_set_id"].to_list()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_default_engine() -> None:
    """Default registry owns a SimpleRegimeEngine instance."""
    registry = RegimeRegistry()
    assert isinstance(registry.engine, SimpleRegimeEngine)


def test_registry_injected_engine_delegation() -> None:
    """Registry.build delegates exclusively to the injected engine."""
    expected = _regime_frame()
    stub = _StubEngine(expected)
    registry = RegimeRegistry(engine=stub)  # type: ignore[arg-type]
    input_frame = _alpha_input_frame()

    result = registry.build(input_frame)

    assert result is expected
    assert stub.calls == [input_frame]
    assert registry.engine is stub


def test_registry_engine_property_exposes_owned_engine() -> None:
    """Registry.engine returns the owned engine instance."""
    engine = SimpleRegimeEngine()
    registry = RegimeRegistry(engine=engine)
    assert registry.engine is engine


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """RegimePartitionRef is a frozen immutable dataclass."""
    ref = RegimePartitionRef(
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
    repository = RegimeRepository(StorageLayout(Path("/data")), store)
    frame = _regime_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == REGIME_SCHEMA


def test_repository_empty_load_returns_schema_frame() -> None:
    """load() returns an empty REGIME_SCHEMA frame when missing."""
    repository = RegimeRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    loaded = repository.load(**_partition_kwargs())  # type: ignore[arg-type]
    assert loaded.height == 0
    assert loaded.schema == REGIME_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


def test_repository_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = RegimeRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_regime_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_repository_delete_absent_succeeds_silently() -> None:
    """Deleting a missing partition succeeds without raising."""
    repository = RegimeRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


def test_repository_save_rejects_missing_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = RegimeRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(RegimeError) as exc_info:
        repository.save(
            pl.DataFrame({"regime_id": ["momentum|1.0.0|placeholder|1.0"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "REGIME_REPO_MISSING_COLUMNS"


def test_repository_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs."""
    repository = RegimeRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(RegimeError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "REGIME_REPO_FRAME_TYPE"


def test_repository_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = RegimeRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    repository.save(
        _regime_frame(regime_id="momentum|1.0.0|placeholder|1.0"),
        **kwargs,  # type: ignore[arg-type]
    )
    repository.save(
        _regime_frame(regime_id="rsi|1.0.0|placeholder|1.0"),
        **kwargs,  # type: ignore[arg-type]
    )

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["regime_id"].to_list() == ["rsi|1.0.0|placeholder|1.0"]


def test_repository_parquet_store_path_contains_regime_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the regime storage directory."""
    layout = StorageLayout(tmp_path)
    repository = RegimeRepository(layout, ParquetStore())
    repository.save(_regime_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_REGIME).is_dir()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_requires_repository() -> None:
    """Pipeline construction without a repository raises RegimeError."""
    with pytest.raises(RegimeError) as exc_info:
        RegimePipeline(repository=None)
    assert exc_info.value.error_code == "REGIME_PIPE_REPOSITORY_REQUIRED"


def test_pipeline_default_registry() -> None:
    """Pipeline without registry uses SimpleRegimeEngine via default RegimeRegistry."""
    repository = _StubRepository()
    pipeline = RegimePipeline(repository=repository)  # type: ignore[arg-type]

    result = pipeline.build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0"],
            alpha_scores=[0.5],
        ),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert result.schema == REGIME_SCHEMA
    assert result["regime_type"].to_list() == ["UNKNOWN"]
    assert len(repository.save_calls) == 1
    assert repository.save_calls[0][0] is result


def test_pipeline_registry_build_and_repository_save_called() -> None:
    """Pipeline delegates generation to the registry and persists via repository."""
    expected = _regime_frame()
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = RegimePipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    input_frame = _alpha_input_frame()

    result = pipeline.build(
        input_frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
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
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }
    assert result is expected


def test_pipeline_returned_dataframe_unchanged() -> None:
    """Pipeline returns the registry output frame without transformation."""
    expected = _regime_frame(regime_id="momentum|1.0.0|placeholder|1.0")
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = RegimePipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )

    result = pipeline.build(
        _alpha_input_frame(),
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
    """Pipeline generates rows and persists them through an in-memory repository."""
    store = _InMemoryDataStore()
    repository = RegimeRepository(StorageLayout(Path("/data")), store)
    pipeline = RegimePipeline(repository=repository)
    kwargs = _partition_kwargs()

    result = pipeline.build(
        _alpha_input_frame(
            factor_set_ids=["momentum|1.0.0", "rsi|1.0.0"],
            alpha_scores=[0.5, 0.6],
        ),
        **kwargs,  # type: ignore[arg-type]
    )

    assert result.schema == REGIME_SCHEMA
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, result)
