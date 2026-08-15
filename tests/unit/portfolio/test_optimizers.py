"""Unit tests for the CQROS Equal Weight portfolio optimizer."""

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
    PortfolioValidationError,
)
from cqros.portfolio.optimizers import (
    EqualWeightOptimizer as EqualWeightOptimizerDirect,
)
from cqros.signals import Signal

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER = "equal_weight"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _signal_frame(
    *,
    signals: list[str],
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a canonical signal DataFrame for optimizer tests."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (open_times if open_times is not None else [_open_time(0)] * row_count),
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


def test_equal_weight_optimizer_is_exported_from_package() -> None:
    """Package export matches the optimizers module by identity."""
    assert EqualWeightOptimizer is EqualWeightOptimizerDirect


def test_equal_weight_optimizer_satisfies_protocol() -> None:
    """EqualWeightOptimizer structurally satisfies PortfolioOptimizer."""
    assert isinstance(EqualWeightOptimizer(), PortfolioOptimizer)


def test_all_buy_signals_receive_equal_positive_weights() -> None:
    """Four BUY rows each receive +0.25 and sum to +1.0."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(
        _signal_frame(signals=[Signal.BUY] * 4),
    )
    weights = result.get_column("target_weight").to_list()
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(weights) == pytest.approx(1.0)


def test_all_sell_signals_receive_equal_negative_weights() -> None:
    """Two SELL rows each receive -0.50 and sum to -1.0."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(
        _signal_frame(signals=[Signal.SELL, Signal.SELL]),
    )
    weights = result.get_column("target_weight").to_list()
    assert weights == pytest.approx([-0.5, -0.5])
    assert sum(weights) == pytest.approx(-1.0)


def test_all_hold_signals_receive_zero_weight() -> None:
    """HOLD rows always receive 0.0."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(
        _signal_frame(signals=[Signal.HOLD, Signal.HOLD, Signal.HOLD]),
    )
    assert result.get_column("target_weight").to_list() == [0.0, 0.0, 0.0]


def test_mixed_buy_sell_hold_allocation() -> None:
    """Mixed signals allocate equal longs, equal shorts, and zero holds."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(
        _signal_frame(
            signals=[
                Signal.BUY,
                Signal.BUY,
                Signal.SELL,
                Signal.SELL,
                Signal.HOLD,
            ],
        ),
    )
    assert result.get_column("target_weight").to_list() == pytest.approx(
        [0.5, 0.5, -0.5, -0.5, 0.0],
    )


def test_single_buy_receives_full_positive_weight() -> None:
    """A lone BUY row receives target_weight 1.0."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(_signal_frame(signals=[Signal.BUY]))
    assert result.get_column("target_weight").to_list() == pytest.approx([1.0])


def test_single_sell_receives_full_negative_weight() -> None:
    """A lone SELL row receives target_weight -1.0."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(_signal_frame(signals=[Signal.SELL]))
    assert result.get_column("target_weight").to_list() == pytest.approx([-1.0])


def test_positive_and_negative_weights_normalize_independently() -> None:
    """BUY weights sum to +1.0 and SELL weights sum to -1.0 independently."""
    optimizer = EqualWeightOptimizer()
    result = optimizer.optimize(
        _signal_frame(
            signals=[
                Signal.BUY,
                Signal.BUY,
                Signal.BUY,
                Signal.SELL,
                Signal.SELL,
                Signal.HOLD,
            ],
        ),
    )
    buy_sum = result.filter(pl.col("signal") == Signal.BUY).get_column("target_weight").sum()
    sell_sum = result.filter(pl.col("signal") == Signal.SELL).get_column("target_weight").sum()
    hold_weights = (
        result.filter(pl.col("signal") == Signal.HOLD).get_column("target_weight").to_list()
    )
    assert buy_sum == pytest.approx(1.0)
    assert sell_sum == pytest.approx(-1.0)
    assert hold_weights == [0.0]


def test_duplicate_primary_keys_raise() -> None:
    """Duplicate symbol/timeframe/open_time combinations are rejected."""
    frame = _signal_frame(
        signals=[Signal.BUY, Signal.SELL],
        symbols=["BTCUSDT", "BTCUSDT"],
        open_times=[_open_time(0), _open_time(0)],
    )
    with pytest.raises(PortfolioValidationError) as exc_info:
        EqualWeightOptimizer().optimize(frame)
    assert exc_info.value.error_code == "PORTFOLIO_DUPLICATE_KEYS"


def test_invalid_signal_values_raise() -> None:
    """Signal values outside the Signal enum are rejected."""
    frame = _signal_frame(signals=[Signal.BUY.value, "WAIT"])
    with pytest.raises(PortfolioValidationError) as exc_info:
        EqualWeightOptimizer().optimize(frame)
    assert exc_info.value.error_code == "PORTFOLIO_INVALID_SIGNAL"
    assert "WAIT" in cast(
        tuple[str, ...],
        exc_info.value.details["invalid_values"],
    )


def test_missing_required_columns_raise() -> None:
    """Missing signal-schema columns raise PortfolioValidationError."""
    frame = _signal_frame(signals=[Signal.BUY]).drop("model_version")
    with pytest.raises(PortfolioValidationError) as exc_info:
        EqualWeightOptimizer().optimize(frame)
    assert exc_info.value.error_code == "PORTFOLIO_MISSING_COLUMNS"
    assert "model_version" in cast(
        tuple[str, ...],
        exc_info.value.details["missing_columns"],
    )


def test_empty_dataframe_raise() -> None:
    """Empty signal frames are rejected by shared validation."""
    empty = _signal_frame(signals=[Signal.BUY]).clear()
    with pytest.raises(PortfolioValidationError) as exc_info:
        EqualWeightOptimizer().optimize(empty)
    assert exc_info.value.error_code == "PORTFOLIO_FRAME_EMPTY"


def test_non_dataframe_input_raise() -> None:
    """Non-DataFrame inputs are rejected by shared validation."""
    with pytest.raises(PortfolioValidationError) as exc_info:
        EqualWeightOptimizer().optimize([{"signal": Signal.BUY.value}])  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PORTFOLIO_FRAME_TYPE"


def test_output_uses_canonical_column_order() -> None:
    """Optimizer output columns follow CANONICAL_COLUMN_ORDER."""
    result = EqualWeightOptimizer().optimize(_signal_frame(signals=[Signal.BUY]))
    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_output_matches_merged_portfolio_schema() -> None:
    """Optimizer output schema identity matches MERGED_PORTFOLIO_SCHEMA."""
    result = EqualWeightOptimizer().optimize(
        _signal_frame(signals=[Signal.BUY, Signal.SELL, Signal.HOLD]),
    )
    assert result.schema == MERGED_PORTFOLIO_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_output_preserves_signal_and_metadata_columns() -> None:
    """Signal and metadata columns are preserved alongside target_weight."""
    frame = _signal_frame(
        signals=[Signal.BUY, Signal.SELL],
        symbols=["BTCUSDT", "ETHUSDT"],
    )
    result = EqualWeightOptimizer().optimize(frame)
    expected = (
        frame.with_columns(
            pl.lit(_OPTIMIZER, dtype=pl.Utf8).alias("optimizer"),
            pl.Series("target_weight", [1.0, -1.0], dtype=pl.Float64),
        )
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(MERGED_PORTFOLIO_SCHEMA)
    )
    assert_frame_equal(result, expected)


def test_output_emits_equal_weight_optimizer_lineage() -> None:
    """Every portfolio row carries optimizer provenance equal_weight."""
    result = EqualWeightOptimizer().optimize(
        _signal_frame(signals=[Signal.BUY, Signal.SELL, Signal.HOLD]),
    )
    assert result.get_column("optimizer").to_list() == [
        _OPTIMIZER,
        _OPTIMIZER,
        _OPTIMIZER,
    ]
    assert result.schema["optimizer"] == pl.Utf8


def test_input_dataframe_is_not_mutated() -> None:
    """optimize returns a new frame and leaves the input unchanged."""
    frame = _signal_frame(signals=[Signal.BUY, Signal.HOLD])
    original = frame.clone()
    result = EqualWeightOptimizer().optimize(frame)
    assert_frame_equal(frame, original)
    assert "target_weight" not in frame.columns
    assert "optimizer" not in frame.columns
    assert result is not frame


def test_three_buy_weights_normalize_to_one() -> None:
    """Three BUY weights each equal 1/3 and normalize to +1.0."""
    result = EqualWeightOptimizer().optimize(
        _signal_frame(signals=[Signal.BUY, Signal.BUY, Signal.BUY]),
    )
    weights = result.get_column("target_weight").to_list()
    assert weights == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert sum(weights) == pytest.approx(1.0)


def test_row_order_is_preserved() -> None:
    """Output rows preserve the input row order."""
    frame = _signal_frame(
        signals=[Signal.HOLD, Signal.SELL, Signal.BUY],
        symbols=["AAA", "BBB", "CCC"],
    )
    result = EqualWeightOptimizer().optimize(frame)
    assert result.get_column("symbol").to_list() == ["AAA", "BBB", "CCC"]
    assert result.get_column("signal").to_list() == [
        Signal.HOLD.value,
        Signal.SELL.value,
        Signal.BUY.value,
    ]
    assert result.get_column("target_weight").to_list() == pytest.approx(
        [0.0, -1.0, 1.0],
    )


def test_signal_enum_members_are_accepted() -> None:
    """Signal enum members serialize correctly as optimizer inputs."""
    frame = _signal_frame(
        signals=[Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value],
    )
    result = EqualWeightOptimizer().optimize(frame)
    assert result.height == 3
    assert result.get_column("target_weight").to_list() == pytest.approx(
        [1.0, -1.0, 0.0],
    )
