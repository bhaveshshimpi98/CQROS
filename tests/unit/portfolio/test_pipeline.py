"""Unit tests for CQROS Portfolio package ``PortfolioPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.portfolio import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_PORTFOLIO_SCHEMA,
    EqualWeightOptimizer,
    PortfolioOptimizer,
    PortfolioOptimizerRegistry,
    PortfolioPipeline,
    PortfolioValidationError,
)
from cqros.portfolio.pipeline import PortfolioPipeline as PortfolioPipelineDirect
from cqros.signals import Signal

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER_NAME = "equal_weight"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _signal_frame(
    *,
    signals: list[str],
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a canonical signal DataFrame for pipeline tests."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                open_times
                if open_times is not None
                else [_open_time(index) for index in range(row_count)]
            ),
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "signal": signals,
        },
        schema={
            "symbol": pl.Utf8,
            "timeframe": pl.Utf8,
            "open_time": pl.Datetime("us", "UTC"),
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "signal": pl.Utf8,
        },
    )


def _portfolio_frame(
    *,
    signals: list[str],
    weights: list[float],
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
    optimizer: str = _OPTIMIZER_NAME,
) -> pl.DataFrame:
    """Build a portfolio-shaped optimizer output frame."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                open_times
                if open_times is not None
                else [_open_time(index) for index in range(row_count)]
            ),
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [optimizer] * row_count,
            "signal": signals,
            "target_weight": weights,
        }
    )


class _RecordingOptimizer:
    """Optimizer stub that records optimize calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[pl.DataFrame] = []

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(signals)
        return self.frame


class _NonDataFrameOptimizer:
    """Optimizer stub that returns a non-DataFrame value."""

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        return {"rows": signals.height}  # type: ignore[return-value]


class _EmptyOutputOptimizer:
    """Optimizer stub that returns an empty portfolio frame."""

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        return _portfolio_frame(signals=[Signal.BUY], weights=[1.0]).clear()


def _make_pipeline(
    *,
    optimizer_name: str = _OPTIMIZER_NAME,
    optimizer: object | None = None,
) -> tuple[PortfolioPipeline, PortfolioOptimizerRegistry, object]:
    """Build a pipeline with a registry containing one optimizer."""
    registry = PortfolioOptimizerRegistry()
    resolved = EqualWeightOptimizer() if optimizer is None else optimizer
    registry.register(optimizer_name, cast(PortfolioOptimizer, resolved))
    return PortfolioPipeline(registry), registry, resolved


def test_portfolio_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert PortfolioPipeline is PortfolioPipelineDirect


def test_successful_equal_weight_execution() -> None:
    """Registered EqualWeightOptimizer produces a finalized portfolio frame."""
    pipeline, _registry, _optimizer = _make_pipeline()
    signals = _signal_frame(
        signals=[Signal.BUY, Signal.BUY, Signal.SELL, Signal.HOLD],
    )

    result = pipeline.run(_OPTIMIZER_NAME, signals)

    assert result.height == 4
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_PORTFOLIO_SCHEMA
    assert result.get_column("target_weight").to_list() == pytest.approx(
        [0.5, 0.5, -1.0, 0.0],
    )
    assert result.get_column("optimizer").to_list() == [
        _OPTIMIZER_NAME,
        _OPTIMIZER_NAME,
        _OPTIMIZER_NAME,
        _OPTIMIZER_NAME,
    ]


def test_unknown_optimizer_raises() -> None:
    """Unknown optimizer names raise PortfolioValidationError."""
    pipeline, _registry, _optimizer = _make_pipeline()
    with pytest.raises(PortfolioValidationError, match="not registered") as exc_info:
        pipeline.run("missing_optimizer", _signal_frame(signals=[Signal.BUY]))
    assert exc_info.value.error_code == "PORTFOLIO_REG_UNKNOWN"


def test_blank_optimizer_name_raises() -> None:
    """Blank optimizer names are rejected before registry lookup."""
    pipeline, _registry, _optimizer = _make_pipeline()
    with pytest.raises(PortfolioValidationError, match="non-blank") as exc_info:
        pipeline.run("   ", _signal_frame(signals=[Signal.BUY]))
    assert exc_info.value.error_code == "PORTFOLIO_PIPE_NAME_BLANK"


def test_empty_signal_dataframe_raises() -> None:
    """Empty signal frames raise PortfolioValidationError."""
    pipeline, _registry, _optimizer = _make_pipeline()
    empty = _signal_frame(signals=[Signal.BUY]).clear()
    with pytest.raises(PortfolioValidationError, match="at least one row") as exc_info:
        pipeline.run(_OPTIMIZER_NAME, empty)
    assert exc_info.value.error_code == "PORTFOLIO_FRAME_EMPTY"


def test_non_dataframe_signals_raise() -> None:
    """Non-DataFrame signal inputs raise PortfolioValidationError."""
    pipeline, _registry, _optimizer = _make_pipeline()
    with pytest.raises(PortfolioValidationError, match="polars DataFrame") as exc_info:
        pipeline.run(_OPTIMIZER_NAME, [{"signal": Signal.BUY}])  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PORTFOLIO_FRAME_TYPE"


def test_invalid_optimizer_output_type_raises() -> None:
    """Non-DataFrame optimizer outputs raise PortfolioValidationError."""
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_NonDataFrameOptimizer(),
    )
    with pytest.raises(PortfolioValidationError, match="optimizer output") as exc_info:
        pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))
    assert exc_info.value.error_code == "PORTFOLIO_PIPE_INVALID_OUTPUT"


def test_empty_optimizer_output_raises() -> None:
    """Empty optimizer outputs raise PortfolioValidationError."""
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_EmptyOutputOptimizer(),
    )
    with pytest.raises(
        PortfolioValidationError,
        match="optimizer output must contain at least one row",
    ) as exc_info:
        pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))
    assert exc_info.value.error_code == "PORTFOLIO_PIPE_OUTPUT_EMPTY"


def test_missing_required_portfolio_columns_raises() -> None:
    """Missing portfolio schema columns on optimizer output are rejected."""
    incomplete = _portfolio_frame(
        signals=[Signal.BUY],
        weights=[1.0],
    ).drop("target_weight")
    pipeline, _registry, optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(incomplete),
    )

    with pytest.raises(
        PortfolioValidationError,
        match="missing required columns",
    ) as exc_info:
        pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))

    assert exc_info.value.error_code == "PORTFOLIO_PIPE_MISSING_COLUMNS"
    assert "target_weight" in cast(
        tuple[str, ...],
        exc_info.value.details["missing_columns"],
    )
    assert isinstance(optimizer, _RecordingOptimizer)
    assert len(optimizer.calls) == 1


def test_missing_optimizer_column_raises() -> None:
    """Missing optimizer lineage on optimizer output is rejected."""
    incomplete = _portfolio_frame(
        signals=[Signal.BUY],
        weights=[1.0],
    ).drop("optimizer")
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(incomplete),
    )

    with pytest.raises(
        PortfolioValidationError,
        match="missing required columns",
    ) as exc_info:
        pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))

    assert exc_info.value.error_code == "PORTFOLIO_PIPE_MISSING_COLUMNS"
    assert "optimizer" in cast(
        tuple[str, ...],
        exc_info.value.details["missing_columns"],
    )


def test_duplicate_primary_keys_on_optimizer_output_raise() -> None:
    """Duplicate primary keys in optimizer output raise PortfolioValidationError."""
    duplicate = _portfolio_frame(
        signals=[Signal.BUY, Signal.SELL],
        weights=[1.0, -1.0],
        symbols=["BTCUSDT", "BTCUSDT"],
        open_times=[_open_time(0), _open_time(0)],
    )
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(duplicate),
    )

    with pytest.raises(
        PortfolioValidationError,
        match="duplicate primary keys",
    ) as exc_info:
        pipeline.run(
            _OPTIMIZER_NAME,
            _signal_frame(
                signals=[Signal.BUY, Signal.SELL],
                symbols=["BTCUSDT", "ETHUSDT"],
            ),
        )

    assert exc_info.value.error_code == "PORTFOLIO_PIPE_DUPLICATE_KEYS"


def test_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    noisy = _portfolio_frame(
        signals=[Signal.BUY, Signal.SELL],
        weights=[1.0, -1.0],
    ).with_columns(pl.lit(1.0).alias("extra_noise"))
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(noisy),
    )

    result = pipeline.run(
        _OPTIMIZER_NAME,
        _signal_frame(signals=[Signal.BUY, Signal.SELL]),
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_noise" not in result.columns


def test_dtype_casting_matches_merged_portfolio_schema() -> None:
    """Finalized columns are cast to MERGED_PORTFOLIO_SCHEMA dtypes."""
    frame = _portfolio_frame(
        signals=[Signal.BUY, Signal.SELL],
        weights=[1.0, -1.0],
    ).with_columns(pl.col("target_weight").cast(pl.Float32))
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(frame),
    )

    result = pipeline.run(
        _OPTIMIZER_NAME,
        _signal_frame(signals=[Signal.BUY, Signal.SELL]),
    )

    assert result.schema == MERGED_PORTFOLIO_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_registry_delegation() -> None:
    """Pipeline resolves and delegates exclusively through the registry."""
    output = _portfolio_frame(
        signals=[Signal.BUY, Signal.SELL],
        weights=[1.0, -1.0],
    )
    optimizer = _RecordingOptimizer(output)
    pipeline, registry, _resolved = _make_pipeline(optimizer=optimizer)
    signals = _signal_frame(signals=[Signal.BUY, Signal.SELL])

    result = pipeline.run(_OPTIMIZER_NAME, signals)

    assert registry.get(_OPTIMIZER_NAME) is optimizer
    assert len(optimizer.calls) == 1
    assert optimizer.calls[0] is signals
    assert_frame_equal(
        result.select(list(CANONICAL_COLUMN_ORDER)),
        output.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_PORTFOLIO_SCHEMA),
    )


def test_input_dataframe_is_not_mutated() -> None:
    """Pipeline never mutates the caller-supplied signal DataFrame."""
    signals = _signal_frame(signals=[Signal.BUY, Signal.HOLD])
    original = signals.clone()
    pipeline, _registry, _optimizer = _make_pipeline()

    pipeline.run(_OPTIMIZER_NAME, signals)

    assert_frame_equal(signals, original)
    assert "target_weight" not in signals.columns
    assert "optimizer" not in signals.columns


def test_returned_frame_is_new() -> None:
    """Pipeline returns a new DataFrame distinct from optimizer output."""
    output = _portfolio_frame(signals=[Signal.BUY], weights=[1.0])
    optimizer = _RecordingOptimizer(output)
    pipeline, _registry, _optimizer = _make_pipeline(optimizer=optimizer)

    result = pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))

    assert result is not output
    assert result.schema == MERGED_PORTFOLIO_SCHEMA


def test_optimizer_failure_propagates() -> None:
    """PortfolioValidationError raised by the optimizer propagates unchanged."""

    class _FailingOptimizer:
        def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
            raise PortfolioValidationError(
                "optimizer refused allocation",
                error_code="PORTFOLIO_OPT_TEST",
                details={"rows": signals.height},
            )

    pipeline, _registry, _optimizer = _make_pipeline(optimizer=_FailingOptimizer())

    with pytest.raises(
        PortfolioValidationError,
        match="optimizer refused allocation",
    ) as exc_info:
        pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))

    assert exc_info.value.error_code == "PORTFOLIO_OPT_TEST"


def test_equal_weight_preserves_signal_metadata() -> None:
    """Equal-weight pipeline output preserves signal and metadata columns."""
    pipeline, _registry, _optimizer = _make_pipeline()
    signals = _signal_frame(
        signals=[Signal.BUY, Signal.SELL],
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    result = pipeline.run(_OPTIMIZER_NAME, signals)

    assert result.get_column("symbol").to_list() == ["BTCUSDT", "ETHUSDT"]
    assert result.get_column("signal").to_list() == [
        Signal.BUY.value,
        Signal.SELL.value,
    ]
    assert result.get_column("model_name").to_list() == [_MODEL_NAME, _MODEL_NAME]
    assert result.get_column("model_version").to_list() == [
        _MODEL_VERSION,
        _MODEL_VERSION,
    ]
    assert result.get_column("optimizer").to_list() == [
        _OPTIMIZER_NAME,
        _OPTIMIZER_NAME,
    ]


def test_extra_optimizer_columns_are_dropped() -> None:
    """Non-canonical optimizer columns are dropped during finalization."""
    frame = _portfolio_frame(
        signals=[Signal.BUY],
        weights=[1.0],
    ).with_columns(
        pl.lit("noise").alias("allocation_note"),
        pl.lit(99).alias("rank"),
    )
    pipeline, _registry, _optimizer = _make_pipeline(
        optimizer=_RecordingOptimizer(frame),
    )

    result = pipeline.run(_OPTIMIZER_NAME, _signal_frame(signals=[Signal.BUY]))

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "allocation_note" not in result.columns
    assert "rank" not in result.columns


def test_multiple_registered_optimizers_resolve_by_name() -> None:
    """Pipeline resolves the requested optimizer among multiple registrations."""
    equal_weight = EqualWeightOptimizer()
    stub_output = _portfolio_frame(signals=[Signal.HOLD], weights=[0.0])
    stub = _RecordingOptimizer(stub_output)
    registry = PortfolioOptimizerRegistry()
    registry.register_many(
        {
            "equal_weight": equal_weight,
            "stub": stub,
        }
    )
    pipeline = PortfolioPipeline(registry)

    result = pipeline.run("stub", _signal_frame(signals=[Signal.HOLD]))

    assert len(stub.calls) == 1
    assert result.get_column("target_weight").to_list() == [0.0]
    assert result.schema == MERGED_PORTFOLIO_SCHEMA
