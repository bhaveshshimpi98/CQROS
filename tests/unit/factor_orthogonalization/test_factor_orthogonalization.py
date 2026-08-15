"""Unit tests for CQROS Factor Orthogonalization module contracts and orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_ORTHOGONALIZATION,
)
from cqros.core.types import FilePath
from cqros.factor_orthogonalization.detailed_export import (
    DETAILED_AUDIT_COLUMNS,
    build_detailed_audit_frame,
    write_detailed_csv,
)
from cqros.factor_orthogonalization.engine import (
    FACTOR_COMBINATION_INPUT_COLUMNS,
    LineageContext,
    SimpleFactorOrthogonalizationEngine,
    validate_factor_combination_frame,
)
from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.pipeline import FactorOrthogonalizationPipeline
from cqros.factor_orthogonalization.redundancy import (
    DEFAULT_MAX_COMBINATION_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
    ORTHOGONALIZATION_METHOD,
    REASON_ACCEPTED,
    REASON_REDUNDANT,
    require_max_combination_correlation,
    require_min_correlation_overlap,
)
from cqros.factor_orthogonalization.registry import FactorOrthogonalizationRegistry
from cqros.factor_orthogonalization.repository import (
    FactorOrthogonalizationPartitionRef,
    FactorOrthogonalizationRepository,
)
from cqros.factor_orthogonalization.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_ORTHOGONALIZATION_COLUMNS,
    FACTOR_ORTHOGONALIZATION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    FactorOrthogonalizationStatus,
)
from cqros.factor_orthogonalization.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_LINEAGE_MEMBERSHIP,
    ERROR_LINEAGE_REJECTED_SELECTED,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorOrthogonalizationVerifier,
)
from cqros.storage import DatasetNotFoundError, ParquetStore, StorageLayout

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_TIMEFRAME = "1h"
_YEAR = 2026
_WINDOW_START = 1_700_000_000_000
_WINDOW_END = 1_700_000_086_400_000


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub that stores frames in memory."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        self.frames[Path(path)] = dataframe

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
        target = Path(path)
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


class _StubObservationSource:
    """In-memory FactorObservationSource for correlation tests."""

    def __init__(self, observations: pl.DataFrame | None = None) -> None:
        self.observations = (
            observations
            if observations is not None
            else pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "open_time": pl.Int64,
                    "factor_name": pl.String,
                    "factor_version": pl.String,
                    "factor_value": pl.Float64,
                }
            )
        )
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
        if self.observations.height == 0:
            return self.observations
        name_set = set(factor_names)
        version_set = set(factor_versions)
        return self.observations.filter(
            (pl.col("factor_name").is_in(list(name_set)))
            & (pl.col("factor_version").is_in(list(version_set)))
            & (pl.col("open_time") >= start_time)
            & (pl.col("open_time") <= end_time)
        )


class _StubEngine:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[tuple[pl.DataFrame, LineageContext]] = []

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
    ) -> pl.DataFrame:
        self.calls.append((factor_combination, lineage))
        return self.output


class _StubRegistry:
    """Test double that records build calls and returns a fixed frame."""

    def __init__(self, output: pl.DataFrame) -> None:
        self.output = output
        self.calls: list[tuple[pl.DataFrame, LineageContext]] = []

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
    ) -> pl.DataFrame:
        self.calls.append((factor_combination, lineage))
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


def _lineage(**overrides: object) -> LineageContext:
    """Build a default lineage context."""
    payload: dict[str, object] = {
        "validation_start_time": _WINDOW_START,
        "validation_end_time": _WINDOW_END,
        "source_combination_version": str(_YEAR),
        "source_fta_version": str(_YEAR),
        "source_selection_version": "1.0.0",
        "dataset_version": str(_YEAR),
    }
    payload.update(overrides)
    return LineageContext(**payload)  # type: ignore[arg-type]


def _partition_kwargs(
    *,
    manager: str = _MANAGER,
    timeframe: str = _TIMEFRAME,
    year: int = _YEAR,
) -> dict[str, object]:
    """Build keyword arguments identifying one partition."""
    return {
        "manager": manager,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "timeframe": timeframe,
        "year": year,
    }


def _combination_input_frame(
    *,
    combination_ids: list[str] | None = None,
    factor_names: list[list[str]] | None = None,
    factor_versions: list[list[str]] | None = None,
    factor_categories: list[list[str]] | None = None,
    timeframe: str = _TIMEFRAME,
    combination_ranks: list[int] | None = None,
    combination_scores: list[float] | None = None,
) -> pl.DataFrame:
    """Build a Factor Combination-style input for the orthogonalization engine."""
    if combination_ids is None:
        combination_ids = ["a|b", "a|c", "b|c"]
    row_count = len(combination_ids)
    if factor_names is None:
        factor_names = [["a", "b"], ["a", "c"], ["b", "c"]][:row_count]
        while len(factor_names) < row_count:
            factor_names.append([f"f{len(factor_names)}", f"g{len(factor_names)}"])
    if factor_versions is None:
        factor_versions = [["1.0.0", "1.0.0"] for _ in range(row_count)]
    if factor_categories is None:
        factor_categories = [["Price", "Price"] for _ in range(row_count)]
    if combination_ranks is None:
        combination_ranks = list(range(1, row_count + 1))
    if combination_scores is None:
        combination_scores = [1.0 - (0.1 * index) for index in range(row_count)]
    return pl.DataFrame(
        {
            "combination_id": combination_ids,
            "factor_names": factor_names,
            "factor_versions": factor_versions,
            "factor_categories": factor_categories,
            "timeframe": [timeframe] * row_count,
            "combination_size": [2] * row_count,
            "combination_method": ["equal_weight"] * row_count,
            "combination_rank": combination_ranks,
            "combination_score": combination_scores,
            "stability_score": [0.8] * row_count,
            "confidence_score": [0.7] * row_count,
        }
    )


def _observations_from_series(
    *,
    factors: dict[str, np.ndarray],
    version: str = "1.0.0",
    start_time: int = _WINDOW_START,
    step_ms: int = 60_000,
) -> pl.DataFrame:
    """Build long-format observations from named factor value arrays."""
    rows: list[dict[str, object]] = []
    length = max(len(values) for values in factors.values())
    for index in range(length):
        open_time = start_time + (index * step_ms)
        for name, values in factors.items():
            rows.append(
                {
                    "symbol": "BTCUSDT",
                    "open_time": open_time,
                    "factor_name": name,
                    "factor_version": version,
                    "factor_value": float(values[index]),
                }
            )
    return pl.DataFrame(rows)


def _engine(
    observations: pl.DataFrame | None = None,
    *,
    max_combination_correlation: float = DEFAULT_MAX_COMBINATION_CORRELATION,
    min_overlap: int = 10,
) -> SimpleFactorOrthogonalizationEngine:
    """Build an engine with an in-memory observation source."""
    return SimpleFactorOrthogonalizationEngine(
        observation_source=_StubObservationSource(observations),
        max_combination_correlation=max_combination_correlation,
        min_overlap=min_overlap,
    )


def _build(
    frame: pl.DataFrame | None = None,
    *,
    observations: pl.DataFrame | None = None,
    max_combination_correlation: float = DEFAULT_MAX_COMBINATION_CORRELATION,
    min_overlap: int = 10,
    lineage: LineageContext | None = None,
) -> pl.DataFrame:
    """Run the simple engine against combination input."""
    engine = _engine(
        observations,
        max_combination_correlation=max_combination_correlation,
        min_overlap=min_overlap,
    )
    return engine.build(
        frame if frame is not None else _combination_input_frame(),
        lineage=lineage if lineage is not None else _lineage(),
    )


def _orthogonalization_frame_from_engine() -> pl.DataFrame:
    """Produce one schema-valid orthogonalization frame via the engine."""
    return _build(
        _combination_input_frame(combination_ids=["a|b"], factor_names=[["a", "b"]]),
        observations=pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "factor_value": pl.Float64,
            }
        ),
    )


def test_canonical_columns_match_specification() -> None:
    """Canonical columns follow the declared combination-unit contract."""
    assert "combination_id" in CANONICAL_COLUMN_ORDER
    assert "factor_name" not in CANONICAL_COLUMN_ORDER
    assert FACTOR_ORTHOGONALIZATION_COLUMNS == CANONICAL_COLUMN_ORDER
    assert PRIMARY_KEY_COLUMNS == (
        "combination_id",
        "timeframe",
        "analysis_time",
    )
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER


def test_schema_dtypes_match_column_dtypes() -> None:
    """FACTOR_ORTHOGONALIZATION_SCHEMA dtypes match COLUMN_DTYPES in order."""
    assert FACTOR_ORTHOGONALIZATION_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert FACTOR_ORTHOGONALIZATION_SCHEMA[column] == COLUMN_DTYPES[column]


def test_defaults_match_phase_3b_policy() -> None:
    """Orthogonalization defaults mirror Phase 3B redundancy thresholds."""
    assert DEFAULT_MAX_COMBINATION_CORRELATION == 0.90
    assert DEFAULT_MIN_CORRELATION_OVERLAP == 500
    assert require_max_combination_correlation(0.85) == 0.85
    assert require_min_correlation_overlap(100) == 100


def test_verifier_valid_dataframe_passes() -> None:
    """Verifier returns the same frame instance for a valid frame."""
    frame = _orthogonalization_frame_from_engine()
    verified = FactorOrthogonalizationVerifier().verify(frame)
    assert verified is frame


def test_verifier_non_dataframe_fails() -> None:
    """Verifier rejects non-DataFrame inputs."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == ERROR_FRAME_TYPE


def test_verifier_empty_dataframe_fails() -> None:
    """Verifier rejects empty orthogonalization frames."""
    empty = pl.DataFrame(schema=FACTOR_ORTHOGONALIZATION_SCHEMA)
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify(empty)
    assert exc_info.value.error_code == ERROR_FRAME_EMPTY


def test_verifier_missing_column_fails() -> None:
    """Verifier rejects frames missing required columns."""
    frame = _orthogonalization_frame_from_engine().drop("status")
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS


def test_verifier_wrong_column_order_fails() -> None:
    """Verifier rejects frames whose column order is non-canonical."""
    frame = _orthogonalization_frame_from_engine().select(list(reversed(CANONICAL_COLUMN_ORDER)))
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_COLUMN_ORDER


def test_verifier_wrong_dtype_fails() -> None:
    """Verifier rejects frames whose dtypes differ from the canonical schema."""
    frame = _orthogonalization_frame_from_engine().with_columns(
        pl.col("orthogonalization_rank").cast(pl.Float64)
    )
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH


def test_verifier_lineage_membership() -> None:
    """Lineage verifier checks membership against source Combination."""
    combination = _combination_input_frame(
        combination_ids=["a|b"],
        factor_names=[["a", "b"]],
    )
    result = _build(combination)
    FactorOrthogonalizationVerifier().verify_against_combination(result, combination)
    foreign = result.with_columns(pl.lit("missing|pair").alias("combination_id"))
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify_against_combination(foreign, combination)
    assert exc_info.value.error_code == ERROR_LINEAGE_MEMBERSHIP


def test_verifier_rejects_selected_rejected_row() -> None:
    """Rejected rows marked selected fail substantive verification."""
    frame = _orthogonalization_frame_from_engine().with_columns(
        pl.lit(True).alias("redundancy_rejected"),
        pl.lit(True).alias("selected"),
        pl.lit(REASON_REDUNDANT).alias("orthogonalization_reason"),
        pl.lit(FactorOrthogonalizationStatus.PASS.value).alias("status"),
    )
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationVerifier().verify(frame)
    assert exc_info.value.error_code == ERROR_LINEAGE_REJECTED_SELECTED


def test_engine_validates_non_dataframe_input() -> None:
    """Engine rejects non-DataFrame inputs during structural validation."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        validate_factor_combination_frame("not-a-frame")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FORTH_FRAME_TYPE"


def test_engine_validates_empty_input() -> None:
    """Engine rejects empty Factor Combination frames."""
    empty = pl.DataFrame(schema={"combination_id": pl.String}).clear()
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        _build(empty)
    assert exc_info.value.error_code == "FORTH_FRAME_EMPTY"


def test_engine_validates_missing_columns() -> None:
    """Engine rejects frames missing required input columns."""
    frame = pl.DataFrame({"combination_id": ["a|b"]})
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "FORTH_MISSING_COLUMNS"


def test_engine_input_columns_contract() -> None:
    """FACTOR_COMBINATION_INPUT_COLUMNS enumerates every consumed column."""
    for column in (
        "combination_id",
        "factor_names",
        "factor_versions",
        "timeframe",
        "combination_rank",
        "combination_score",
    ):
        assert column in FACTOR_COMBINATION_INPUT_COLUMNS


def test_engine_duplicate_combination_ids_fail() -> None:
    """Duplicate combination_id values are rejected."""
    frame = _combination_input_frame(
        combination_ids=["a|b", "a|b"],
        factor_names=[["a", "b"], ["a", "b"]],
        combination_ranks=[1, 2],
    )
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        _build(frame)
    assert exc_info.value.error_code == "FORTH_DUPLICATE_COMBINATION_IDS"


def test_engine_invalid_validation_window_fails() -> None:
    """Validation windows with start > end fail explicitly."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        _build(lineage=_lineage(validation_start_time=200, validation_end_time=100))
    assert exc_info.value.error_code == "FORTH_VALIDATION_WINDOW_INVALID"


def test_engine_empty_observations_accept_all() -> None:
    """Empty observation panels do not reject candidates."""
    result = _build(min_overlap=10)
    assert result.height == 3
    assert result["selected"].to_list() == [True, True, True]
    assert result["status"].to_list() == [FactorOrthogonalizationStatus.PASS.value] * 3
    assert result["orthogonalization_method"].to_list() == [ORTHOGONALIZATION_METHOD] * 3


def test_engine_highly_correlated_combinations_reject() -> None:
    """Highly correlated later combinations are rejected against earlier ones."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=100)
    observations = _observations_from_series(
        factors={
            "a": base,
            "b": base + rng.normal(size=100) * 0.001,
            "c": base + rng.normal(size=100) * 0.001,
            "d": rng.normal(size=100),
        }
    )
    combination = _combination_input_frame(
        combination_ids=["a|b", "a|c", "a|d"],
        factor_names=[["a", "b"], ["a", "c"], ["a", "d"]],
        combination_ranks=[1, 2, 3],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.90,
        min_overlap=50,
    )
    by_id = {row["combination_id"]: row for row in result.to_dicts()}
    assert by_id["a|b"]["selected"] is True
    assert by_id["a|c"]["selected"] is False
    assert by_id["a|c"]["orthogonalization_reason"] == REASON_REDUNDANT
    assert by_id["a|c"]["redundancy_reference_combination_id"] == "a|b"
    assert by_id["a|c"]["correlation_score"] >= 0.90
    assert by_id["a|d"]["selected"] is True


def test_engine_perfectly_correlated_copy_factors() -> None:
    """Identical member series produce perfect correlation and rejection."""
    values = np.linspace(-1.0, 1.0, 60)
    observations = _observations_from_series(
        factors={"a": values, "b": values, "c": values, "d": values}
    )
    combination = _combination_input_frame(
        combination_ids=["a|b", "c|d"],
        factor_names=[["a", "b"], ["c", "d"]],
        combination_ranks=[1, 2],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.99,
        min_overlap=20,
    )
    by_id = {row["combination_id"]: row for row in result.to_dicts()}
    assert by_id["a|b"]["selected"] is True
    assert by_id["c|d"]["selected"] is False
    assert by_id["c|d"]["correlation_score"] == pytest.approx(1.0, abs=1e-9)


def test_engine_negatively_correlated_uses_absolute_threshold() -> None:
    """Absolute Pearson correlation rejects strongly negatively correlated pairs."""
    values = np.linspace(-2.0, 2.0, 80)
    observations = _observations_from_series(
        factors={
            "a": values,
            "b": values,
            "c": -values,
            "d": -values,
        }
    )
    combination = _combination_input_frame(
        combination_ids=["a|b", "c|d"],
        factor_names=[["a", "b"], ["c", "d"]],
        combination_ranks=[1, 2],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.90,
        min_overlap=20,
    )
    by_id = {row["combination_id"]: row for row in result.to_dicts()}
    assert by_id["a|b"]["selected"] is True
    assert by_id["c|d"]["selected"] is False
    assert by_id["c|d"]["correlation_score"] == pytest.approx(1.0, abs=1e-9)


def test_engine_independent_combinations_accepted() -> None:
    """Independent combination signals survive the correlation filter."""
    rng = np.random.default_rng(21)
    observations = _observations_from_series(
        factors={
            "a": rng.normal(size=120),
            "b": rng.normal(size=120),
            "c": rng.normal(size=120),
            "d": rng.normal(size=120),
        }
    )
    combination = _combination_input_frame(
        combination_ids=["a|b", "c|d"],
        factor_names=[["a", "b"], ["c", "d"]],
        combination_ranks=[1, 2],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.90,
        min_overlap=50,
    )
    assert result["selected"].to_list() == [True, True]
    assert result["orthogonalization_reason"].to_list() == [REASON_ACCEPTED, REASON_ACCEPTED]


def test_engine_insufficient_overlap_does_not_reject() -> None:
    """Pairs below min_overlap are not rejected even if values would correlate."""
    values = np.linspace(0.0, 1.0, 8)
    observations = _observations_from_series(
        factors={"a": values, "b": values, "c": values, "d": values}
    )
    combination = _combination_input_frame(
        combination_ids=["a|b", "c|d"],
        factor_names=[["a", "b"], ["c", "d"]],
        combination_ranks=[1, 2],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.50,
        min_overlap=50,
    )
    assert result["selected"].to_list() == [True, True]


def test_engine_respects_validation_window_leakage_boundary() -> None:
    """Observations outside the validation window are ignored."""
    inside = np.linspace(0.0, 1.0, 40)
    outside = np.linspace(10.0, 20.0, 40)
    inside_obs = _observations_from_series(
        factors={"a": inside, "b": inside, "c": inside, "d": inside},
        start_time=_WINDOW_START,
    )
    outside_obs = _observations_from_series(
        factors={"a": outside, "b": -outside, "c": outside, "d": -outside},
        start_time=_WINDOW_END + 60_000,
    )
    observations = pl.concat([inside_obs, outside_obs], how="vertical_relaxed")
    combination = _combination_input_frame(
        combination_ids=["a|b", "c|d"],
        factor_names=[["a", "b"], ["c", "d"]],
        combination_ranks=[1, 2],
    )
    source = _StubObservationSource(observations)
    engine = SimpleFactorOrthogonalizationEngine(
        observation_source=source,
        max_combination_correlation=0.90,
        min_overlap=20,
    )
    result = engine.build(combination, lineage=_lineage())
    assert source.calls[0]["start_time"] == _WINDOW_START
    assert source.calls[0]["end_time"] == _WINDOW_END
    by_id = {row["combination_id"]: row for row in result.to_dicts()}
    assert by_id["c|d"]["selected"] is False


def test_engine_timeframe_isolation_in_observation_request() -> None:
    """Engine requests observations only for the combination timeframe."""
    source = _StubObservationSource()
    engine = SimpleFactorOrthogonalizationEngine(
        observation_source=source,
        min_overlap=10,
    )
    engine.build(
        _combination_input_frame(timeframe="4h"),
        lineage=_lineage(),
    )
    assert source.calls[0]["timeframe"] == "4h"


def test_engine_deterministic_output() -> None:
    """Identical inputs produce identical orthogonalization outputs."""
    rng = np.random.default_rng(3)
    observations = _observations_from_series(
        factors={
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
            "c": rng.normal(size=50),
            "d": rng.normal(size=50),
            "e": rng.normal(size=50),
            "f": rng.normal(size=50),
        }
    )
    frame = _combination_input_frame()
    first = _build(frame, observations=observations, min_overlap=10)
    second = _build(frame, observations=observations, min_overlap=10)
    comparable = [column for column in CANONICAL_COLUMN_ORDER if column != "analysis_time"]
    assert_frame_equal(first.select(comparable), second.select(comparable))


def test_engine_does_not_mutate_input() -> None:
    """Engine leaves the caller-supplied frame unchanged."""
    frame = _combination_input_frame()
    before = frame.clone()
    _ = _build(frame)
    assert_frame_equal(frame, before)


def test_engine_requires_observation_source() -> None:
    """Constructing the engine without an observation source fails."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        SimpleFactorOrthogonalizationEngine(observation_source=None)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FORTH_OBSERVATION_SOURCE_REQUIRED"


def test_registry_requires_engine() -> None:
    """Registry construction without an engine raises."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationRegistry(engine=None)  # type: ignore[arg-type]
    assert exc_info.value.error_code == "FORTH_REGISTRY_ENGINE_REQUIRED"


def test_registry_injected_engine_delegation() -> None:
    """Registry.build delegates exclusively to the injected engine."""
    expected = _orthogonalization_frame_from_engine()
    stub = _StubEngine(expected)
    registry = FactorOrthogonalizationRegistry(engine=stub)  # type: ignore[arg-type]
    input_frame = _combination_input_frame()
    lineage = _lineage()
    result = registry.build(input_frame, lineage=lineage)
    assert result is expected
    assert stub.calls == [(input_frame, lineage)]


def test_partition_ref_is_frozen_dataclass() -> None:
    """FactorOrthogonalizationPartitionRef is a frozen immutable dataclass."""
    ref = FactorOrthogonalizationPartitionRef(
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert is_dataclass(ref)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ref.timeframe = "4h"  # type: ignore[misc]


def test_repository_save_exists_load_round_trip() -> None:
    """Saved frames can be retrieved by exists() and load()."""
    store = _InMemoryDataStore()
    repository = FactorOrthogonalizationRepository(StorageLayout(Path("/data")), store)
    frame = _orthogonalization_frame_from_engine()
    kwargs = _partition_kwargs()
    assert not repository.exists(**kwargs)  # type: ignore[arg-type]
    repository.save(frame, **kwargs)  # type: ignore[arg-type]
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, frame)


def test_repository_path_excludes_symbol(tmp_path: Path) -> None:
    """Saved partitions reside under manager/exchange/market/timeframe/year."""
    layout = StorageLayout(tmp_path)
    repository = FactorOrthogonalizationRepository(layout, ParquetStore())
    repository.save(_orthogonalization_frame_from_engine(), **_partition_kwargs())  # type: ignore[arg-type]
    expected = (
        tmp_path
        / STORAGE_DIR_FACTOR_ORTHOGONALIZATION
        / _MANAGER
        / _EXCHANGE
        / _MARKET
        / _TIMEFRAME
        / f"{_YEAR}.parquet"
    )
    assert expected.is_file()
    assert "BTCUSDT" not in expected.as_posix()


def test_pipeline_requires_registry_and_repository() -> None:
    """Pipeline construction without registry/repository raises."""
    with pytest.raises(FactorOrthogonalizationError) as exc_info:
        FactorOrthogonalizationPipeline(repository=None)
    assert exc_info.value.error_code == "FORTH_PIPE_REGISTRY_REQUIRED"


def test_pipeline_registry_build_and_repository_save_called() -> None:
    """Pipeline delegates generation to the registry and persists via repository."""
    expected = _orthogonalization_frame_from_engine()
    registry = _StubRegistry(expected)
    repository = _StubRepository()
    pipeline = FactorOrthogonalizationPipeline(
        registry=registry,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    input_frame = _combination_input_frame()
    lineage = _lineage()
    result = pipeline.build(
        input_frame,
        lineage=lineage,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert registry.calls == [(input_frame, lineage)]
    assert repository.save_calls[0][0] is expected
    assert result is expected


def test_pipeline_end_to_end_with_in_memory_repository() -> None:
    """Pipeline generates rows and persists them through an in-memory repository."""
    store = _InMemoryDataStore()
    repository = FactorOrthogonalizationRepository(StorageLayout(Path("/data")), store)
    engine = _engine()
    registry = FactorOrthogonalizationRegistry(engine=engine)
    pipeline = FactorOrthogonalizationPipeline(registry=registry, repository=repository)
    kwargs = _partition_kwargs()
    result = pipeline.build(
        _combination_input_frame(combination_ids=["a|b"], factor_names=[["a", "b"]]),
        lineage=_lineage(),
        **kwargs,  # type: ignore[arg-type]
    )
    assert result.schema == FACTOR_ORTHOGONALIZATION_SCHEMA
    assert repository.exists(**kwargs)  # type: ignore[arg-type]
    loaded = repository.load(**kwargs)  # type: ignore[arg-type]
    assert_frame_equal(loaded, result)


def test_detailed_export_includes_audit_columns(tmp_path: Path) -> None:
    """Detailed CSV export exposes correlation, overlap, lineage, and decision."""
    frame = _orthogonalization_frame_from_engine()
    audit = build_detailed_audit_frame(
        frame,
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        year=_YEAR,
    )
    assert tuple(audit.columns) == DETAILED_AUDIT_COLUMNS
    path = tmp_path / "audit.csv"
    write_detailed_csv(audit, path)
    loaded = pl.read_csv(path)
    assert "correlation_score" in loaded.columns
    assert "redundancy_reference_combination_id" in loaded.columns
    assert "validation_start_time" in loaded.columns
    assert loaded["manager"].to_list() == [_MANAGER]


def test_combination_to_orthogonalization_integration() -> None:
    """End-to-end Combination → Orthogonalization preserves lineage membership."""
    rng = np.random.default_rng(99)
    observations = _observations_from_series(
        factors={
            "mom": rng.normal(size=90),
            "rsi": rng.normal(size=90),
            "vol": rng.normal(size=90),
            "trend": rng.normal(size=90),
        }
    )
    combination = _combination_input_frame(
        combination_ids=["mom|rsi", "mom|vol", "rsi|trend"],
        factor_names=[["mom", "rsi"], ["mom", "vol"], ["rsi", "trend"]],
        factor_versions=[["1.0.0", "1.0.0"]] * 3,
        combination_ranks=[1, 2, 3],
        combination_scores=[0.9, 0.8, 0.7],
    )
    result = _build(
        combination,
        observations=observations,
        max_combination_correlation=0.95,
        min_overlap=30,
    )
    FactorOrthogonalizationVerifier().verify_against_combination(result, combination)
    assert set(result["combination_id"].to_list()) == set(combination["combination_id"].to_list())
