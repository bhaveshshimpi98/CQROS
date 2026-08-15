"""Unit tests for CQROS ``FactorGenerationPipeline``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import (
    FactorExecutionError,
    FactorRegistrationError,
    FactorValidationError,
)
from cqros.factors.generation_pipeline import (
    FactorGenerationPipeline,
    FactorGenerationStatistics,
)
from cqros.factors.interfaces import Factor
from cqros.factors.pipeline import FactorPipeline
from cqros.factors.registry import FactorRegistry
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_SCHEMA,
)
from cqros.factors.wide_to_long import WideToLongFactorTransformer

_MANAGER = "simple"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2024
_OPEN_TIME = 1_704_067_200_000  # 2024-01-01T00:00:00Z in ms


@dataclass(frozen=True, slots=True)
class _StubFactor(BaseFactor):
    """Minimal deterministic factor for generation-pipeline unit tests."""

    name: str = "stub_alpha"
    version: str = "1.0.0"
    description: str = "Stub factor for generation pipeline tests"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("stub_alpha",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("close").alias(self.produced_columns[0]))


@dataclass(frozen=True, slots=True)
class _SecondStubFactor(BaseFactor):
    """Second stub factor with a distinct produced column."""

    name: str = "stub_beta"
    version: str = "1.0.0"
    description: str = "Second stub factor for generation pipeline tests"
    category: str = "test"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("stub_beta",)
    lookback: int = 0

    def __post_init__(self) -> None:
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns((pl.col("close") * 2.0).alias(self.produced_columns[0]))


class _ListOverrideRegistry(FactorRegistry):
    """FactorRegistry test double that returns a fixed ``list`` result."""

    __slots__ = ("_override",)

    def __init__(self, factors: tuple[Factor, ...]) -> None:
        super().__init__()
        self._override = factors

    def list(self) -> tuple[Factor, ...]:
        return self._override


def _registry_with(*factors: Factor) -> FactorRegistry:
    """Return a registry containing only the supplied factors."""
    registry = FactorRegistry()
    registry.register_many(factors)
    return registry


def _training_frame(*, row_count: int = 3, symbol: str = _SYMBOL) -> pl.DataFrame:
    """Build a small synthetic training frame."""
    return pl.DataFrame(
        {
            "symbol": [symbol] * row_count,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": [_OPEN_TIME + index for index in range(row_count)],
            "close": [float(index + 1) for index in range(row_count)],
        }
    )


def _partition_kwargs() -> dict[str, object]:
    """Build repository partition identity kwargs."""
    return {
        "manager": _MANAGER,
        "exchange": _EXCHANGE,
        "market": _MARKET,
        "symbol": _SYMBOL,
        "timeframe": _TIMEFRAME,
        "year": _YEAR,
    }


def _mock_repository() -> MagicMock:
    """Return a repository mock with a no-op ``save``."""
    repository = MagicMock()
    repository.save = MagicMock(return_value=None)
    return repository


def _build_pipeline(
    *,
    registry: FactorRegistry | None = None,
    factor_pipeline: FactorPipeline | None = None,
    transformer: WideToLongFactorTransformer | None = None,
    repository: MagicMock | None = None,
) -> tuple[FactorGenerationPipeline, FactorRegistry, MagicMock]:
    """Compose a generation pipeline with injectable test doubles."""
    resolved_registry = registry if registry is not None else _registry_with(_StubFactor())
    resolved_transformer = transformer if transformer is not None else WideToLongFactorTransformer()
    resolved_repository = repository if repository is not None else _mock_repository()
    generation = FactorGenerationPipeline(
        resolved_registry,
        factor_pipeline,
        resolved_transformer,
        resolved_repository,
    )
    return generation, resolved_registry, resolved_repository


class TestFactorGenerationPipeline:
    """Unit tests for factor generation orchestration."""

    def test_successful_generation(self) -> None:
        """Training data flows through pipeline → transform → repository save."""
        generation, _registry, repository = _build_pipeline(
            registry=_registry_with(_StubFactor(), _SecondStubFactor())
        )
        frame = _training_frame()

        statistics = generation.run(frame, **_partition_kwargs())  # type: ignore[arg-type]

        assert isinstance(statistics, FactorGenerationStatistics)
        assert statistics.symbols_processed == 1
        assert statistics.rows_generated == frame.height * 2
        assert statistics.factors_generated == 2
        assert statistics.total_registered_factors == 2
        assert statistics.executable_factors == 2
        assert statistics.skipped_factors == 0
        assert statistics.generation_duration >= 0.0
        assert statistics.failed_symbols == ()
        assert statistics.successful_symbols == (_SYMBOL,)

        repository.save.assert_called_once()
        saved_frame = repository.save.call_args.args[0]
        saved_kwargs = repository.save.call_args.kwargs
        assert isinstance(saved_frame, pl.DataFrame)
        assert saved_frame.schema == FACTOR_SCHEMA
        assert saved_frame.columns == list(CANONICAL_COLUMN_ORDER)
        assert saved_frame.height == frame.height * 2
        assert set(saved_frame.get_column("factor_name").to_list()) == {
            "stub_alpha",
            "stub_beta",
        }
        assert saved_frame.get_column("open_time").is_sorted()
        assert saved_kwargs == _partition_kwargs()

    def test_empty_training_dataframe(self) -> None:
        """Empty training frames return zero statistics without persistence."""
        repository = _mock_repository()
        factor_pipeline = MagicMock()
        transformer = MagicMock()
        generation, _, _ = _build_pipeline(
            factor_pipeline=factor_pipeline,
            transformer=transformer,
            repository=repository,
        )
        empty = pl.DataFrame(
            schema={
                "symbol": pl.String,
                "timeframe": pl.String,
                "open_time": pl.Int64,
                "close": pl.Float64,
            }
        )

        statistics = generation.run(empty, **_partition_kwargs())  # type: ignore[arg-type]

        assert statistics.symbols_processed == 0
        assert statistics.rows_generated == 0
        assert statistics.factors_generated == 0
        assert statistics.total_registered_factors == 1
        assert statistics.executable_factors == 0
        assert statistics.skipped_factors == 0
        assert statistics.generation_duration >= 0.0
        assert statistics.failed_symbols == ()
        assert statistics.successful_symbols == ()
        factor_pipeline.run.assert_not_called()
        transformer.transform.assert_not_called()
        repository.save.assert_not_called()

    def test_repository_failure(self) -> None:
        """Repository save failures fail immediately as FactorExecutionError."""
        repository = _mock_repository()
        repository.save.side_effect = RuntimeError("disk full")
        generation, _, _ = _build_pipeline(repository=repository)

        with pytest.raises(
            FactorExecutionError,
            match="factor repository save failed",
        ) as exc_info:
            generation.run(_training_frame(), **_partition_kwargs())  # type: ignore[arg-type]

        assert exc_info.value.error_code == "FACTOR-GEN-006"
        assert exc_info.value.details["symbol"] == _SYMBOL
        assert exc_info.value.details["exception_type"] == "RuntimeError"
        repository.save.assert_called_once()

    def test_metadata_failure(self) -> None:
        """Missing metadata for a produced column fails before persistence."""
        registry = _registry_with(_StubFactor())
        wide = pl.DataFrame(
            {
                "symbol": [_SYMBOL],
                "timeframe": [_TIMEFRAME],
                "open_time": [_OPEN_TIME],
                "stub_alpha": [1.0],
                "unknown_factor": [2.0],
            }
        )
        factor_pipeline = MagicMock()
        factor_pipeline.run.return_value = wide
        repository = _mock_repository()
        generation, _, _ = _build_pipeline(
            registry=registry,
            factor_pipeline=factor_pipeline,
            repository=repository,
        )

        with pytest.raises(
            FactorValidationError,
            match="factor metadata is missing for one or more factor columns",
        ) as exc_info:
            generation.run(_training_frame(), **_partition_kwargs())  # type: ignore[arg-type]

        assert exc_info.value.error_code == "FACTOR-GEN-003"
        assert exc_info.value.details["missing_factor_names"] == ("unknown_factor",)
        repository.save.assert_not_called()

    def test_schema_failure(self) -> None:
        """Invalid long frames fail FACTOR_SCHEMA validation before save."""
        registry = _registry_with(_StubFactor())
        wide = pl.DataFrame(
            {
                "symbol": [_SYMBOL],
                "timeframe": [_TIMEFRAME],
                "open_time": [_OPEN_TIME],
                "stub_alpha": [1.0],
            }
        )
        factor_pipeline = MagicMock()
        factor_pipeline.run.return_value = wide
        transformer = MagicMock()
        transformer.transform.return_value = pl.DataFrame(
            {
                "symbol": [_SYMBOL],
                "timeframe": [_TIMEFRAME],
                "open_time": [_OPEN_TIME],
                "factor_name": ["stub_alpha"],
            }
        )
        repository = _mock_repository()
        generation, _, _ = _build_pipeline(
            registry=registry,
            factor_pipeline=factor_pipeline,
            transformer=transformer,
            repository=repository,
        )

        with pytest.raises(
            FactorValidationError,
            match="generated factors frame is missing required columns",
        ) as exc_info:
            generation.run(_training_frame(), **_partition_kwargs())  # type: ignore[arg-type]

        assert exc_info.value.error_code == "FACTOR-GEN-005"
        repository.save.assert_not_called()

    def test_statistics_correctness(self) -> None:
        """Statistics reflect row counts, factor counts, and symbol outcomes."""
        generation, _, repository = _build_pipeline(
            registry=_registry_with(_StubFactor(), _SecondStubFactor())
        )
        frame = _training_frame(row_count=5)

        statistics = generation.run(frame, **_partition_kwargs())  # type: ignore[arg-type]

        assert statistics.symbols_processed == 1
        assert statistics.rows_generated == 10
        assert statistics.factors_generated == 2
        assert statistics.total_registered_factors == 2
        assert statistics.executable_factors == 2
        assert statistics.skipped_factors == 0
        assert isinstance(statistics.generation_duration, float)
        assert statistics.generation_duration >= 0.0
        assert statistics.failed_symbols == ()
        assert statistics.successful_symbols == (_SYMBOL,)
        assert is_dataclass(statistics)
        with pytest.raises(FrozenInstanceError):
            statistics.rows_generated = 0  # type: ignore[misc]
        repository.save.assert_called_once()

    def test_duplicate_factor_names_fail_immediately(self) -> None:
        """Duplicate catalog names fail before pipeline execution."""
        factor = _StubFactor()
        registry = _ListOverrideRegistry((factor, factor))
        factor_pipeline = MagicMock()
        repository = _mock_repository()
        generation, _, _ = _build_pipeline(
            registry=registry,
            factor_pipeline=factor_pipeline,
            repository=repository,
        )

        with pytest.raises(
            FactorRegistrationError,
            match="duplicate factor name in catalog: stub_alpha",
        ) as exc_info:
            generation.run(_training_frame(), **_partition_kwargs())  # type: ignore[arg-type]

        assert exc_info.value.error_code == "FACTOR-GEN-001"
        factor_pipeline.run.assert_not_called()
        repository.save.assert_not_called()

    def test_repository_factor_errors_propagate(self) -> None:
        """FactorValidationError from repository save is not wrapped."""
        repository = _mock_repository()
        repository.save.side_effect = FactorValidationError(
            "factors schema is missing required columns",
            error_code="FAC_REPO_MISSING_COLUMNS",
        )
        generation, _, _ = _build_pipeline(repository=repository)

        with pytest.raises(
            FactorValidationError,
            match="factors schema is missing required columns",
        ) as exc_info:
            generation.run(_training_frame(), **_partition_kwargs())  # type: ignore[arg-type]

        assert exc_info.value.error_code == "FAC_REPO_MISSING_COLUMNS"

    def test_rejects_non_dataframe_input(self) -> None:
        """Non-DataFrame training inputs fail immediately."""
        generation, _, repository = _build_pipeline()
        with pytest.raises(
            FactorValidationError,
            match="training frame must be a polars DataFrame",
        ) as exc_info:
            generation.run([1, 2, 3], **_partition_kwargs())  # type: ignore[arg-type]
        assert exc_info.value.error_code == "FACTOR-GEN-007"
        repository.save.assert_not_called()


def test_statistics_mapping_contract() -> None:
    """FactorGenerationStatistics exposes the required immutable fields."""
    statistics = FactorGenerationStatistics(
        symbols_processed=1,
        rows_generated=4,
        factors_generated=2,
        total_registered_factors=111,
        executable_factors=73,
        skipped_factors=38,
        generation_duration=0.01,
        failed_symbols=(),
        successful_symbols=(_SYMBOL,),
    )
    as_mapping: Mapping[str, Any] = {
        "symbols_processed": statistics.symbols_processed,
        "rows_generated": statistics.rows_generated,
        "factors_generated": statistics.factors_generated,
        "total_registered_factors": statistics.total_registered_factors,
        "executable_factors": statistics.executable_factors,
        "skipped_factors": statistics.skipped_factors,
        "generation_duration": statistics.generation_duration,
        "failed_symbols": statistics.failed_symbols,
        "successful_symbols": statistics.successful_symbols,
    }
    assert as_mapping["symbols_processed"] == 1
    assert as_mapping["executable_factors"] == 73
    assert as_mapping["skipped_factors"] == 38
    assert as_mapping["successful_symbols"] == (_SYMBOL,)
