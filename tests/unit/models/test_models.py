"""Unit tests for CQROS Models module contracts and orchestration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL, STORAGE_DIR_MODELS
from cqros.core.exceptions import CQROSError, ResearchError
from cqros.core.types import FilePath
from cqros.models.engine import (
    MODEL_INPUT_COLUMNS,
    SimpleModelEngine,
    validate_regime_frame,
)
from cqros.models.exceptions import ModelError, ModelException
from cqros.models.pipeline import ModelPipeline
from cqros.models.registry import ModelRegistry
from cqros.models.repository import ModelPartitionRef, ModelRepository
from cqros.models.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MODELS_COLUMNS,
    MODELS_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ModelStatus,
)
from cqros.models.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    ModelVerifier,
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

    def build(self, regime: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(regime)
        return self.output


class _StubRegistry:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[pl.DataFrame] = []

    def build(self, regime: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(regime)
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


def _regime_input_frame(
    *,
    regime_ids: list[str] | None = None,
    factor_set_ids: list[str] | None = None,
    alpha_ids: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    regime_times: list[int] | None = None,
    regime_types: list[str] | None = None,
    regime_probabilities: list[float | None] | None = None,
    regime_scores: list[float | None] | None = None,
    regime_versions: list[str] | None = None,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    """Build a canonical Regime-style input for the models engine."""
    row_count = max(
        len(values)
        for values in (
            regime_ids if regime_ids is not None else ["regime-a", "regime-b"],
            factor_set_ids or [],
            alpha_ids or [],
            symbols or [],
            timeframes or [],
            regime_times or [],
            regime_types or [],
            regime_probabilities or [],
            regime_scores or [],
            regime_versions or [],
            statuses or [],
        )
    )
    regime_ids = (
        regime_ids
        if regime_ids is not None
        else (
            ["regime-a", "regime-b"]
            if row_count == 2
            else [f"regime-{index}" for index in range(row_count)]
        )
    )
    factor_set_ids = (
        factor_set_ids
        if factor_set_ids is not None
        else (["fs-a", "fs-b"] if row_count == 2 else [f"fs-{index}" for index in range(row_count)])
    )
    alpha_ids = (
        alpha_ids
        if alpha_ids is not None
        else (
            ["alpha-a", "alpha-b"]
            if row_count == 2
            else [f"alpha-{index}" for index in range(row_count)]
        )
    )
    symbols = symbols if symbols is not None else [_SYMBOL] * row_count
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME] * row_count
    regime_times = (
        regime_times
        if regime_times is not None
        else [1000 * (index + 1) for index in range(row_count)]
    )
    regime_types = regime_types if regime_types is not None else ["UNKNOWN"] * row_count
    regime_probabilities = (
        regime_probabilities
        if regime_probabilities is not None
        else [1.0 for _ in range(row_count)]
    )
    regime_scores = (
        regime_scores
        if regime_scores is not None
        else [0.5 + (0.1 * index) for index in range(row_count)]
    )
    regime_versions = regime_versions if regime_versions is not None else ["1.0.0"] * row_count
    statuses = statuses if statuses is not None else ["PASS"] * row_count
    return pl.DataFrame(
        {
            "regime_id": regime_ids,
            "factor_set_id": factor_set_ids,
            "alpha_id": alpha_ids,
            "symbol": symbols,
            "timeframe": timeframes,
            "regime_time": pl.Series("regime_time", regime_times, dtype=pl.Datetime("ms")),
            "regime_type": regime_types,
            "regime_probability": regime_probabilities,
            "regime_score": regime_scores,
            "regime_version": regime_versions,
            "status": statuses,
            "metadata": [[] for _ in range(row_count)],
        }
    )


def _models_frame(
    *,
    model_id: str = "regime-a|baseline",
    regime_id: str = "regime-a",
    symbol: str = _SYMBOL,
    timeframe: str = _TIMEFRAME,
    training_time: int = 1000,
    model_type: str = "baseline",
    model_version: str = "1.0.0",
    prediction_horizon: int = 1,
    validation_score: float = 0.5,
    feature_set_id: str = "alpha-a",
    model_metadata: list[str] | None = None,
    status: str = ModelStatus.PASS.value,
) -> pl.DataFrame:
    """Build a canonical Models frame for repository/verifier tests."""
    metadata = (
        model_metadata
        if model_metadata is not None
        else [
            "factor_set_id=fs-a",
            "alpha_id=alpha-a",
            "regime_type=UNKNOWN",
            "regime_probability=1.0",
            "regime_version=1.0.0",
        ]
    )
    return pl.DataFrame(
        {
            "model_id": [model_id],
            "regime_id": [regime_id],
            "symbol": [symbol],
            "timeframe": [timeframe],
            "training_time": [training_time],
            "model_type": [model_type],
            "model_version": [model_version],
            "prediction_horizon": [prediction_horizon],
            "validation_score": [validation_score],
            "feature_set_id": [feature_set_id],
            "model_metadata": [metadata],
            "status": [status],
        },
        schema=dict(COLUMN_DTYPES),
    ).select(list(CANONICAL_COLUMN_ORDER))


def _build(frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run ``SimpleModelEngine`` against an input frame."""
    engine = SimpleModelEngine()
    return engine.build(frame if frame is not None else _regime_input_frame())


def _metadata_map(entries: list[str]) -> dict[str, str]:
    """Parse ``key=value`` metadata entries into a dictionary."""
    parsed: dict[str, str] = {}
    for entry in entries:
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_canonical_columns_match_specification() -> None:
    """Canonical columns follow the declared models contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "model_id",
        "regime_id",
        "symbol",
        "timeframe",
        "training_time",
        "model_type",
        "model_version",
        "prediction_horizon",
        "validation_score",
        "feature_set_id",
        "model_metadata",
        "status",
    )
    assert MODELS_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == (
        "model_id",
        "regime_id",
        "symbol",
        "timeframe",
        "training_time",
    )


def test_required_columns_match_canonical_order() -> None:
    """REQUIRED_COLUMNS mirrors CANONICAL_COLUMN_ORDER exactly."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))


def test_schema_dtypes_match_column_dtypes() -> None:
    """MODELS_SCHEMA dtypes match COLUMN_DTYPES in order."""
    assert MODELS_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MODELS_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["model_id"] == pl.String
    assert COLUMN_DTYPES["regime_id"] == pl.String
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["training_time"] == pl.Int64
    assert COLUMN_DTYPES["model_type"] == pl.String
    assert COLUMN_DTYPES["model_version"] == pl.String
    assert COLUMN_DTYPES["prediction_horizon"] == pl.Int32
    assert COLUMN_DTYPES["validation_score"] == pl.Float64
    assert COLUMN_DTYPES["feature_set_id"] == pl.String
    assert COLUMN_DTYPES["model_metadata"] == pl.List(pl.String)
    assert COLUMN_DTYPES["status"] == pl.String


def test_schema_equality_matches_models_schema() -> None:
    """A frame built with COLUMN_DTYPES equals MODELS_SCHEMA."""
    frame = _models_frame()
    assert frame.schema == MODELS_SCHEMA
    assert tuple(frame.columns) == CANONICAL_COLUMN_ORDER


def test_model_status_enum_members() -> None:
    """ModelStatus exposes PASS and FAIL members."""
    assert ModelStatus.PASS.value == "PASS"
    assert ModelStatus.FAIL.value == "FAIL"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_model_exception_inherits_from_research_error() -> None:
    """ModelException is a ResearchError specialization."""
    assert issubclass(ModelException, ResearchError)
    assert issubclass(ModelException, CQROSError)


def test_model_error_inherits_from_model_exception() -> None:
    """ModelError remains under ModelException."""
    assert issubclass(ModelError, ModelException)
    assert issubclass(ModelError, ResearchError)
    assert issubclass(ModelError, CQROSError)


def test_model_error_supports_structured_fields() -> None:
    """ModelError accepts message, error_code, and details."""
    error = ModelError(
        "model failure",
        error_code="MODEL_TEST",
        details={"dataset": "models", "rows": 0},
    )
    assert error.message == "model failure"
    assert error.error_code == "MODEL_TEST"
    assert dict(error.details) == {"dataset": "models", "rows": 0}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verifier_valid_dataframe_passes() -> None:
    """Verifier returns the same frame instance for a valid models frame."""
    frame = _models_frame()
    verified = ModelVerifier().verify(frame)
    assert verified is frame
    assert_frame_equal(verified, frame)


def test_verifier_non_dataframe_fails() -> None:
    """Verifier rejects non-DataFrame inputs."""
    with pytest.raises(ModelError) as exc_info:
        ModelVerifier().verify("not-a-frame")
    assert exc_info.value.error_code == ERROR_FRAME_TYPE


def test_verifier_empty_dataframe_fails() -> None:
    """Verifier rejects empty models frames."""
    empty = pl.DataFrame(schema=MODELS_SCHEMA)
    with pytest.raises(ModelError) as exc_info:
        ModelVerifier().verify(empty)
    assert exc_info.value.error_code == ERROR_FRAME_EMPTY


def test_verifier_missing_column_fails() -> None:
    """Verifier rejects frames missing required columns."""
    frame = _models_frame().drop("status")
    with pytest.raises(ModelError) as exc_info:
        ModelVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_wrong_column_order_fails() -> None:
    """Verifier rejects frames whose column order is non-canonical."""
    frame = _models_frame().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    with pytest.raises(ModelError) as exc_info:
        ModelVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_COLUMN_ORDER


def test_verifier_wrong_dtype_fails() -> None:
    """Verifier rejects frames whose dtypes differ from the canonical schema."""
    frame = _models_frame().with_columns(pl.col("validation_score").cast(pl.Int64))
    with pytest.raises(ModelError) as exc_info:
        ModelVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_validates_non_dataframe_input() -> None:
    """Engine rejects non-DataFrame inputs during structural validation."""
    with pytest.raises(ModelError) as exc_info:
        validate_regime_frame("not-a-frame")
    assert exc_info.value.error_code == "MODEL_FRAME_TYPE"


def test_engine_validates_empty_input() -> None:
    """Engine rejects empty Regime frames."""
    empty = pl.DataFrame(schema={"regime_id": pl.String}).clear()
    with pytest.raises(ModelError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "MODEL_FRAME_EMPTY"


def test_engine_validates_missing_columns() -> None:
    """Engine rejects frames missing required input columns."""
    frame = pl.DataFrame({"regime_id": ["regime-a"], "status": ["PASS"]})
    with pytest.raises(ModelError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "MODEL_MISSING_COLUMNS"


def test_engine_input_columns_contract() -> None:
    """MODEL_INPUT_COLUMNS enumerates every consumed column."""
    for column in (
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
    ):
        assert column in MODEL_INPUT_COLUMNS
    assert "regime_confidence" not in MODEL_INPUT_COLUMNS


def test_engine_only_pass_status_rows_used() -> None:
    """Only rows with status == PASS participate in model generation."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b", "regime-c"],
            alpha_ids=["alpha-a", "alpha-b", "alpha-c"],
            regime_scores=[0.5, 0.6, 0.7],
            statuses=["PASS", "FAIL", "PASS"],
        )
    )
    assert result.height == 2
    assert result["regime_id"].to_list() == ["regime-a", "regime-c"]


def test_engine_only_finite_non_null_regime_score_rows_used() -> None:
    """Only rows with finite non-null regime_score participate."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b", "regime-c"],
            alpha_ids=["alpha-a", "alpha-b", "alpha-c"],
            regime_scores=[0.5, None, 0.7],
            statuses=["PASS", "PASS", "PASS"],
        )
    )
    assert result.height == 2
    assert result["regime_id"].to_list() == ["regime-a", "regime-c"]


def test_engine_rejects_non_finite_regime_scores() -> None:
    """NaN and infinite regime_score values do not become ledger rows."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b", "regime-c", "regime-d"],
            alpha_ids=["alpha-a", "alpha-b", "alpha-c", "alpha-d"],
            regime_scores=[0.5, float("nan"), float("inf"), float("-inf")],
            statuses=["PASS", "PASS", "PASS", "PASS"],
        )
    )
    assert result.height == 1
    assert result["validation_score"].to_list() == [0.5]


def test_engine_no_surviving_regimes_raises() -> None:
    """Frames with no PASS finite non-null regime_score rows raise MODEL_NO_REGIMES."""
    with pytest.raises(ModelError) as exc_info:
        _build(
            _regime_input_frame(
                regime_ids=["regime-a", "regime-b"],
                alpha_ids=["alpha-a", "alpha-b"],
                regime_scores=[None, 0.5],
                statuses=["PASS", "FAIL"],
            )
        )
    assert exc_info.value.error_code == "MODEL_NO_REGIMES"


def test_engine_model_id_deterministic() -> None:
    """model_id is regime_id|baseline."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b"],
            alpha_ids=["alpha-a", "alpha-b"],
            regime_scores=[0.5, 0.6],
        )
    )
    assert result["model_id"].to_list() == [
        "regime-a|baseline",
        "regime-b|baseline",
    ]


def test_engine_identity_stable_across_repeated_builds() -> None:
    """Repeated builds preserve model identity fields."""
    frame = _regime_input_frame(
        regime_ids=["regime-a"],
        alpha_ids=["alpha-a"],
        regime_scores=[0.42],
    )
    first = _build(frame)
    second = _build(frame)
    assert first["model_id"].to_list() == second["model_id"].to_list()
    assert first["model_type"].to_list() == second["model_type"].to_list()
    assert first["model_version"].to_list() == second["model_version"].to_list()


def test_engine_field_mappings_and_placeholders() -> None:
    """Engine maps regime fields and applies placeholder model values."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a"],
            factor_set_ids=["fs-xyz"],
            alpha_ids=["alpha-xyz"],
            symbols=["ETHUSDT"],
            timeframes=["4h"],
            regime_times=[1500],
            regime_types=["BULL"],
            regime_probabilities=[0.91],
            regime_scores=[0.42],
            regime_versions=["2.0.0"],
            statuses=["PASS"],
        )
    )
    assert result.height == 1
    assert result["model_id"].to_list() == ["regime-a|baseline"]
    assert result["regime_id"].to_list() == ["regime-a"]
    assert result["symbol"].to_list() == ["ETHUSDT"]
    assert result["timeframe"].to_list() == ["4h"]
    assert result["training_time"].to_list() == [1500]
    assert result["model_type"].to_list() == ["baseline"]
    assert result["model_version"].to_list() == ["1.0.0"]
    assert result["prediction_horizon"].to_list() == [1]
    assert result["validation_score"].to_list() == [0.42]
    assert result["feature_set_id"].to_list() == ["alpha-xyz"]
    assert result["status"].to_list() == [ModelStatus.PASS.value]
    metadata = _metadata_map(result["model_metadata"].to_list()[0])
    assert metadata["factor_set_id"] == "fs-xyz"
    assert metadata["alpha_id"] == "alpha-xyz"
    assert metadata["regime_type"] == "BULL"
    assert metadata["regime_probability"] == "0.91"
    assert metadata["regime_version"] == "2.0.0"


def test_engine_validation_score_equals_regime_score() -> None:
    """validation_score (model score) equals regime_score for every row."""
    regime_scores = [0.11, -0.25, 1.5]
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b", "regime-c"],
            alpha_ids=["alpha-a", "alpha-b", "alpha-c"],
            regime_scores=regime_scores,
            regime_probabilities=[0.9, 0.8, 0.7],
        )
    )
    assert result["validation_score"].to_list() == regime_scores


def test_engine_does_not_use_regime_probability_as_model_score() -> None:
    """regime_probability is contextual and not copied into validation_score."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b"],
            alpha_ids=["alpha-a", "alpha-b"],
            regime_scores=[0.11, 0.99],
            regime_probabilities=[0.42, 0.07],
        )
    )
    assert result["validation_score"].to_list() == [0.11, 0.99]
    assert result["validation_score"].to_list() != [0.42, 0.07]
    for entries, expected_probability in zip(
        result["model_metadata"].to_list(),
        ["0.42", "0.07"],
        strict=True,
    ):
        assert _metadata_map(entries)["regime_probability"] == expected_probability


def test_engine_feature_set_id_equals_alpha_id() -> None:
    """feature_set_id is taken directly from alpha_id."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b"],
            alpha_ids=["alpha-x", "alpha-y"],
            regime_scores=[0.5, 0.6],
        )
    )
    assert result["feature_set_id"].to_list() == ["alpha-x", "alpha-y"]


def test_engine_training_time_maps_from_regime_time() -> None:
    """training_time is the Int64 epoch of regime_time with no shift."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b"],
            alpha_ids=["alpha-a", "alpha-b"],
            regime_times=[1_000, 2_000],
            regime_scores=[0.5, 0.6],
        )
    )
    assert result["training_time"].to_list() == [1000, 2000]


def test_engine_one_pass_regime_row_produces_one_ledger_row() -> None:
    """Grain is one Research Model Ledger row per admitted Regime row."""
    result = _build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b", "regime-c"],
            alpha_ids=["alpha-a", "alpha-b", "alpha-c"],
            regime_scores=[0.1, 0.2, 0.3],
            statuses=["PASS", "FAIL", "PASS"],
        )
    )
    assert result.height == 2
    assert len(result["model_id"].to_list()) == 2


def test_engine_symbol_isolation() -> None:
    """BTC-only and ETH-only builds remain independent."""
    btc = _build(
        _regime_input_frame(
            regime_ids=["regime-btc"],
            alpha_ids=["alpha-btc"],
            symbols=["BTCUSDT"],
            regime_scores=[0.5],
        )
    )
    eth = _build(
        _regime_input_frame(
            regime_ids=["regime-eth"],
            alpha_ids=["alpha-eth"],
            symbols=["ETHUSDT"],
            regime_scores=[0.6],
        )
    )
    assert btc["symbol"].to_list() == ["BTCUSDT"]
    assert eth["symbol"].to_list() == ["ETHUSDT"]
    assert "ETHUSDT" not in btc["symbol"].to_list()
    assert "BTCUSDT" not in eth["symbol"].to_list()


def test_engine_timeframe_isolation() -> None:
    """1h and 4h builds remain independent."""
    one_hour = _build(
        _regime_input_frame(
            regime_ids=["regime-1h"],
            alpha_ids=["alpha-1h"],
            timeframes=["1h"],
            regime_scores=[0.5],
        )
    )
    four_hour = _build(
        _regime_input_frame(
            regime_ids=["regime-4h"],
            alpha_ids=["alpha-4h"],
            timeframes=["4h"],
            regime_scores=[0.6],
        )
    )
    assert one_hour["timeframe"].to_list() == ["1h"]
    assert four_hour["timeframe"].to_list() == ["4h"]


def test_engine_output_matches_models_schema() -> None:
    """Engine output conforms to MODELS_SCHEMA and canonical order."""
    result = _build()
    assert result.schema == MODELS_SCHEMA
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result["status"].to_list() == [ModelStatus.PASS.value, ModelStatus.PASS.value]
    assert result["model_type"].to_list() == ["baseline", "baseline"]
    assert result["model_version"].to_list() == ["1.0.0", "1.0.0"]
    assert result["prediction_horizon"].to_list() == [1, 1]


def test_engine_deterministic_output() -> None:
    """Identical inputs produce identical models outputs."""
    frame = _regime_input_frame(
        regime_ids=["regime-c", "regime-a", "regime-b"],
        alpha_ids=["alpha-c", "alpha-a", "alpha-b"],
        regime_scores=[0.3, 0.5, 0.7],
    )
    first = _build(frame)
    second = _build(frame)
    assert_frame_equal(first, second)


def test_engine_preserves_surviving_row_order() -> None:
    """Surviving regime rows emit model rows in input order."""
    result = _build(
        _regime_input_frame(
            regime_ids=["zulu", "alpha", "mike"],
            alpha_ids=["a-zulu", "a-alpha", "a-mike"],
            regime_scores=[0.1, 0.2, 0.3],
        )
    )
    assert result["regime_id"].to_list() == ["zulu", "alpha", "mike"]


def test_engine_does_not_mutate_input() -> None:
    """Engine leaves the caller-supplied frame unchanged."""
    frame = _regime_input_frame()
    before = frame.clone()
    _ = _build(frame)
    assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_default_engine() -> None:
    """Default registry owns a SimpleModelEngine instance."""
    registry = ModelRegistry()
    assert isinstance(registry.engine, SimpleModelEngine)


def test_registry_injected_engine_delegation() -> None:
    """Registry.build delegates exclusively to the injected engine."""
    expected = _models_frame()
    stub = _StubEngine(expected)
    registry = ModelRegistry(engine=stub)  # type: ignore[arg-type]
    input_frame = _regime_input_frame()

    result = registry.build(input_frame)

    assert result is expected
    assert stub.calls == [input_frame]
    assert registry.engine is stub


def test_registry_engine_property_exposes_owned_engine() -> None:
    """Registry.engine returns the owned engine instance."""
    engine = SimpleModelEngine()
    registry = ModelRegistry(engine=engine)
    assert registry.engine is engine


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_partition_ref_is_frozen_dataclass() -> None:
    """ModelPartitionRef is a frozen immutable dataclass."""
    ref = ModelPartitionRef(
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
    repository = ModelRepository(StorageLayout(Path("/data")), store)
    frame = _models_frame()
    kwargs = _partition_kwargs()

    assert not repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.save(frame, **kwargs)  # type: ignore[arg-type]

    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)
    assert loaded.schema == MODELS_SCHEMA


def test_repository_empty_load_returns_schema_frame() -> None:
    """load() returns an empty MODELS_SCHEMA frame when missing."""
    repository = ModelRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    loaded = repository.load(**_partition_kwargs())  # type: ignore[arg-type]
    assert loaded.height == 0
    assert loaded.schema == MODELS_SCHEMA
    assert tuple(loaded.columns) == CANONICAL_COLUMN_ORDER


def test_repository_delete_removes_partition() -> None:
    """delete() removes the partition so subsequent exists() returns False."""
    store = _InMemoryDataStore()
    repository = ModelRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()
    repository.save(_models_frame(), **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]

    repository.delete(**kwargs)  # type: ignore[arg-type]
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]


def test_repository_delete_absent_succeeds_silently() -> None:
    """Deleting a missing partition succeeds without raising."""
    repository = ModelRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    repository.delete(**_partition_kwargs())  # type: ignore[arg-type]


def test_repository_save_rejects_missing_columns() -> None:
    """save() rejects frames missing required columns."""
    repository = ModelRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(ModelError) as exc_info:
        repository.save(
            pl.DataFrame({"model_id": ["regime-a|baseline"]}),
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "MODEL_REPO_MISSING_COLUMNS"


def test_repository_save_rejects_non_dataframe() -> None:
    """save() rejects non-DataFrame inputs."""
    repository = ModelRepository(
        StorageLayout(Path("/data")),
        _InMemoryDataStore(),
    )
    with pytest.raises(ModelError) as exc_info:
        repository.save(
            "not-a-frame",  # type: ignore[arg-type]
            **_partition_kwargs(),  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "MODEL_REPO_FRAME_TYPE"


def test_repository_save_overwrites_existing_partition() -> None:
    """Saving twice overwrites the existing partition with the new frame."""
    store = _InMemoryDataStore()
    repository = ModelRepository(StorageLayout(Path("/data")), store)
    kwargs = _partition_kwargs()

    repository.save(
        _models_frame(model_id="regime-a|baseline"),
        **kwargs,  # type: ignore[arg-type]
    )
    repository.save(
        _models_frame(model_id="regime-b|baseline"),
        **kwargs,  # type: ignore[arg-type]
    )

    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert loaded["model_id"].to_list() == ["regime-b|baseline"]


def test_repository_parquet_store_path_contains_models_directory(tmp_path: Path) -> None:
    """Saved partitions reside under the models storage directory."""
    layout = StorageLayout(tmp_path)
    repository = ModelRepository(layout, ParquetStore())
    repository.save(_models_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert (tmp_path / STORAGE_DIR_MODELS).is_dir()
    expected = (
        tmp_path
        / STORAGE_DIR_MODELS
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / _SYMBOL
        / _TIMEFRAME
        / f"{_YEAR}.parquet"
    )
    assert expected.is_file()
    assert not (tmp_path / STORAGE_DIR_MODELS / "lightgbm").exists()


def test_repository_discover_returns_same_as_discover_partitions(tmp_path: Path) -> None:
    """discover() delegates to discover_partitions()."""
    layout = StorageLayout(tmp_path)
    repository = ModelRepository(layout, ParquetStore())
    repository.save(_models_frame(), **_partition_kwargs())  # type: ignore[arg-type]
    assert repository.discover() == repository.discover_partitions()


def test_repository_discover_partitions_returns_sorted_refs(tmp_path: Path) -> None:
    """discover_partitions returns deterministically sorted partition references."""
    layout = StorageLayout(tmp_path)
    repository = ModelRepository(layout, ParquetStore())
    for symbol, year in (("ETHUSDT", 2025), ("BTCUSDT", 2026), ("BTCUSDT", 2025)):
        repository.save(
            _models_frame(),
            **_partition_kwargs(symbol=symbol, year=year),  # type: ignore[arg-type]
        )
    partitions = repository.discover_partitions()
    assert len(partitions) == 3
    assert partitions[0].symbol == "BTCUSDT"
    assert partitions[0].year == 2025
    assert partitions[1].symbol == "BTCUSDT"
    assert partitions[1].year == 2026
    assert partitions[2].symbol == "ETHUSDT"
    assert partitions[2].year == 2025


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_requires_repository() -> None:
    """Pipeline construction without a repository raises ModelError."""
    with pytest.raises(ModelError) as exc_info:
        ModelPipeline(repository=None)
    assert exc_info.value.error_code == "MODEL_PIPE_REPOSITORY_REQUIRED"


def test_pipeline_default_registry() -> None:
    """Pipeline without registry uses SimpleModelEngine via default ModelRegistry."""
    repository = _StubRepository()
    pipeline = ModelPipeline(repository=repository)  # type: ignore[arg-type]

    result = pipeline.build(
        _regime_input_frame(
            regime_ids=["regime-a"],
            alpha_ids=["alpha-a"],
            regime_scores=[0.5],
        ),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    assert result.schema == MODELS_SCHEMA
    assert result["model_type"].to_list() == ["baseline"]
    assert len(repository.save_calls) == 1
    assert repository.save_calls[0][0] is result


def test_pipeline_registry_build_and_repository_save_called() -> None:
    """Pipeline delegates generation to the registry and persists via repository."""
    expected = _models_frame()
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = ModelPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    input_frame = _regime_input_frame()

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
    expected = _models_frame(model_id="regime-a|baseline")
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = ModelPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )

    result = pipeline.build(
        _regime_input_frame(),
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
    repository = ModelRepository(StorageLayout(Path("/data")), store)
    pipeline = ModelPipeline(repository=repository)
    kwargs = _partition_kwargs()

    result = pipeline.build(
        _regime_input_frame(
            regime_ids=["regime-a", "regime-b"],
            alpha_ids=["alpha-a", "alpha-b"],
            regime_scores=[0.5, 0.6],
        ),
        **kwargs,  # type: ignore[arg-type]
    )

    assert result.schema == MODELS_SCHEMA
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, result)
