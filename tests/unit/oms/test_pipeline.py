"""Unit tests for CQROS Order Management System package ``OrderPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import patch
from uuid import UUID

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.oms import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_ORDER_SCHEMA,
    OMSValidationError,
    OrderManager,
    OrderManagerRegistry,
    OrderPipeline,
    OrderSide,
    OrderStatus,
    OrderType,
    SimpleOrderManager,
)
from cqros.oms.pipeline import OrderPipeline as OrderPipelineDirect
from cqros.risk.enums import RiskDecision

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
_MANAGER_NAME = "simple"
_FIXED_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
_UUID_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_UUID_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _risk_frame(
    *,
    decisions: list[str],
    approved_weights: list[float],
    target_weights: list[float] | None = None,
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
    signals: list[str] | None = None,
) -> pl.DataFrame:
    """Build a risk-decision DataFrame enriched with OMS lineage columns."""
    row_count = len(decisions)
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
            "signal": (signals if signals is not None else ["BUY"] * row_count),
            "target_weight": (
                target_weights if target_weights is not None else list(approved_weights)
            ),
            "approved_weight": approved_weights,
            "decision": decisions,
            "reason": ["unit_test"] * row_count,
            "policy": [_POLICY] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
        },
        schema={
            "symbol": pl.Utf8,
            "timeframe": pl.Utf8,
            "open_time": pl.Datetime("us", "UTC"),
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "signal": pl.Utf8,
            "target_weight": pl.Float64,
            "approved_weight": pl.Float64,
            "decision": pl.Utf8,
            "reason": pl.Utf8,
            "policy": pl.Utf8,
            "optimizer": pl.Utf8,
        },
    )


def _order_frame(
    *,
    symbols: list[str],
    sides: list[str],
    quantities: list[float],
    order_ids: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build an order-shaped manager output frame."""
    row_count = len(symbols)
    ids = order_ids if order_ids is not None else [f"order-{index}" for index in range(row_count)]
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                open_times
                if open_times is not None
                else [_open_time(index) for index in range(row_count)]
            ),
            "order_id": ids,
            "parent_order_id": ids,
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "policy": [_POLICY] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "side": sides,
            "order_type": [OrderType.MARKET.value] * row_count,
            "quantity": quantities,
            "limit_price": [None] * row_count,
            "stop_price": [None] * row_count,
            "filled_quantity": [0.0] * row_count,
            "average_fill_price": [None] * row_count,
            "status": [OrderStatus.PENDING.value] * row_count,
            "created_at": [_FIXED_NOW] * row_count,
            "updated_at": [_FIXED_NOW] * row_count,
        }
    )


class _RecordingManager:
    """Order manager stub that records create_orders calls and returns a frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[pl.DataFrame] = []

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(risk_decisions)
        return self.frame


class _NonDataFrameManager:
    """Order manager stub that returns a non-DataFrame value."""

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return {"rows": risk_decisions.height}  # type: ignore[return-value]


class _EmptyOutputManager:
    """Order manager stub that returns an empty order frame."""

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return _order_frame(
            symbols=["BTCUSDT"],
            sides=[OrderSide.BUY.value],
            quantities=[1.0],
        ).clear()


def _make_pipeline(
    *,
    manager_name: str = _MANAGER_NAME,
    manager: object | None = None,
) -> tuple[OrderPipeline, OrderManagerRegistry, object]:
    """Build a pipeline with a registry containing one order manager."""
    registry = OrderManagerRegistry()
    resolved = (
        _RecordingManager(
            _order_frame(
                symbols=["BTCUSDT"],
                sides=[OrderSide.BUY.value],
                quantities=[0.25],
            ),
        )
        if manager is None
        else manager
    )
    registry.register(manager_name, cast(OrderManager, resolved))
    return OrderPipeline(registry), registry, resolved


def test_order_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert OrderPipeline is OrderPipelineDirect


def test_successful_simple_manager_execution() -> None:
    """Registered SimpleOrderManager produces a finalized order frame."""
    pipeline, _registry, _manager = _make_pipeline(manager=SimpleOrderManager())
    risk = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.25, -0.5],
        symbols=["BTCUSDT", "ETHUSDT"],
        signals=["BUY", "SELL"],
    )

    with (
        patch(
            "cqros.oms.managers.uuid4",
            side_effect=[_UUID_A, _UUID_B],
        ),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = pipeline.run(_MANAGER_NAME, risk)

    assert result.height == 2
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_ORDER_SCHEMA
    assert result.get_column("side").to_list() == [
        OrderSide.BUY.value,
        OrderSide.SELL.value,
    ]
    assert result.get_column("quantity").to_list() == pytest.approx([0.25, 0.5])
    assert result.get_column("order_id").to_list() == [_UUID_A.hex, _UUID_B.hex]


def test_unknown_manager_raises() -> None:
    """Unknown manager names raise OMSValidationError."""
    pipeline, _registry, _manager = _make_pipeline()
    with pytest.raises(OMSValidationError, match="not registered") as exc_info:
        pipeline.run(
            "missing_manager",
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.1],
            ),
        )
    assert exc_info.value.error_code == "OMS_REG_UNKNOWN"


def test_blank_manager_name_raises() -> None:
    """Blank manager names are rejected before registry lookup."""
    pipeline, _registry, _manager = _make_pipeline()
    with pytest.raises(OMSValidationError, match="non-blank") as exc_info:
        pipeline.run(
            "   ",
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.1],
            ),
        )
    assert exc_info.value.error_code == "OMS_PIPE_NAME_BLANK"


def test_empty_risk_dataframe_raises() -> None:
    """Empty risk frames raise OMSValidationError."""
    pipeline, _registry, _manager = _make_pipeline()
    empty = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    ).clear()
    with pytest.raises(OMSValidationError, match="at least one row") as exc_info:
        pipeline.run(_MANAGER_NAME, empty)
    assert exc_info.value.error_code == "OMS_FRAME_EMPTY"


def test_non_dataframe_risk_raises() -> None:
    """Non-DataFrame risk inputs raise OMSValidationError."""
    pipeline, _registry, _manager = _make_pipeline()
    with pytest.raises(OMSValidationError, match="polars DataFrame") as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            [{"decision": RiskDecision.APPROVE.value}],  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "OMS_FRAME_TYPE"


def test_invalid_manager_output_type_raises() -> None:
    """Non-DataFrame manager outputs raise OMSValidationError."""
    pipeline, _registry, _manager = _make_pipeline(
        manager=_NonDataFrameManager(),
    )
    with pytest.raises(OMSValidationError, match="manager output") as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.1],
            ),
        )
    assert exc_info.value.error_code == "OMS_PIPE_INVALID_OUTPUT"


def test_empty_manager_output_raises() -> None:
    """Empty manager outputs raise OMSValidationError."""
    pipeline, _registry, _manager = _make_pipeline(
        manager=_EmptyOutputManager(),
    )
    with pytest.raises(
        OMSValidationError,
        match="manager output must contain at least one row",
    ) as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.1],
            ),
        )
    assert exc_info.value.error_code == "OMS_PIPE_OUTPUT_EMPTY"


def test_missing_required_order_columns_raises() -> None:
    """Missing order schema columns on manager output are rejected."""
    incomplete = _order_frame(
        symbols=["BTCUSDT"],
        sides=[OrderSide.BUY.value],
        quantities=[0.25],
    ).drop("quantity")
    pipeline, _registry, manager = _make_pipeline(
        manager=_RecordingManager(incomplete),
    )

    with pytest.raises(
        OMSValidationError,
        match="missing required columns",
    ) as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.25],
            ),
        )

    assert exc_info.value.error_code == "OMS_PIPE_MISSING_COLUMNS"
    assert "quantity" in exc_info.value.details["missing_columns"]
    assert isinstance(manager, _RecordingManager)
    assert len(manager.calls) == 1


def test_duplicate_primary_keys_on_manager_output_raise() -> None:
    """Duplicate primary keys in manager output raise OMSValidationError."""
    duplicate = _order_frame(
        symbols=["BTCUSDT", "BTCUSDT"],
        sides=[OrderSide.BUY.value, OrderSide.SELL.value],
        quantities=[0.25, 0.25],
        order_ids=["same-id", "same-id"],
        open_times=[_open_time(0), _open_time(0)],
    )
    pipeline, _registry, _manager = _make_pipeline(
        manager=_RecordingManager(duplicate),
    )

    with pytest.raises(
        OMSValidationError,
        match="duplicate primary keys",
    ) as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            _risk_frame(
                decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
                approved_weights=[0.25, -0.25],
                symbols=["BTCUSDT", "ETHUSDT"],
            ),
        )

    assert exc_info.value.error_code == "OMS_PIPE_DUPLICATE_KEYS"


def test_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    noisy = _order_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        sides=[OrderSide.BUY.value, OrderSide.SELL.value],
        quantities=[0.25, 0.5],
    ).with_columns(pl.lit(1.0).alias("extra_noise"))
    pipeline, _registry, _manager = _make_pipeline(
        manager=_RecordingManager(noisy),
    )

    result = pipeline.run(
        _MANAGER_NAME,
        _risk_frame(
            decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
            approved_weights=[0.25, -0.5],
        ),
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_noise" not in result.columns


def test_dtype_casting_matches_merged_order_schema() -> None:
    """Finalized columns are cast to MERGED_ORDER_SCHEMA dtypes."""
    frame = _order_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        sides=[OrderSide.BUY.value, OrderSide.SELL.value],
        quantities=[0.25, 0.5],
    ).with_columns(pl.col("quantity").cast(pl.Float32))
    pipeline, _registry, _manager = _make_pipeline(
        manager=_RecordingManager(frame),
    )

    result = pipeline.run(
        _MANAGER_NAME,
        _risk_frame(
            decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
            approved_weights=[0.25, -0.5],
        ),
    )

    assert result.schema == MERGED_ORDER_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_registry_delegation() -> None:
    """Pipeline resolves and delegates exclusively through the registry."""
    output = _order_frame(
        symbols=["BTCUSDT", "ETHUSDT"],
        sides=[OrderSide.BUY.value, OrderSide.SELL.value],
        quantities=[0.25, 0.5],
    )
    manager = _RecordingManager(output)
    pipeline, registry, _resolved = _make_pipeline(manager=manager)
    risk = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.25, -0.5],
    )

    result = pipeline.run(_MANAGER_NAME, risk)

    assert registry.get(_MANAGER_NAME) is manager
    assert len(manager.calls) == 1
    assert manager.calls[0] is risk
    assert_frame_equal(
        result.select(list(CANONICAL_COLUMN_ORDER)),
        output.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_ORDER_SCHEMA),
    )


def test_validate_risk_frame_invoked_before_manager() -> None:
    """Pipeline rejects empty risk frames before manager invocation."""
    manager = _RecordingManager(
        _order_frame(
            symbols=["BTCUSDT"],
            sides=[OrderSide.BUY.value],
            quantities=[0.1],
        ),
    )
    pipeline, _registry, _manager = _make_pipeline(manager=manager)
    empty = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    ).clear()

    with pytest.raises(OMSValidationError) as exc_info:
        pipeline.run(_MANAGER_NAME, empty)

    assert exc_info.value.error_code == "OMS_FRAME_EMPTY"
    assert manager.calls == []


def test_input_dataframe_is_not_mutated() -> None:
    """Pipeline never mutates the caller-supplied risk DataFrame."""
    risk = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    )
    original = risk.clone()
    pipeline, _registry, _manager = _make_pipeline(
        manager=_RecordingManager(
            _order_frame(
                symbols=["SYM0"],
                sides=[OrderSide.BUY.value],
                quantities=[0.1],
            ),
        ),
    )

    pipeline.run(_MANAGER_NAME, risk)

    assert_frame_equal(risk, original)
    assert "order_id" not in risk.columns


def test_returned_frame_is_new() -> None:
    """Pipeline returns a new DataFrame distinct from manager output."""
    output = _order_frame(
        symbols=["BTCUSDT"],
        sides=[OrderSide.BUY.value],
        quantities=[0.1],
    )
    manager = _RecordingManager(output)
    pipeline, _registry, _manager = _make_pipeline(manager=manager)

    result = pipeline.run(
        _MANAGER_NAME,
        _risk_frame(
            decisions=[RiskDecision.APPROVE],
            approved_weights=[0.1],
        ),
    )

    assert result is not output
    assert result.schema == MERGED_ORDER_SCHEMA


def test_manager_failure_propagates() -> None:
    """OMSValidationError raised by the manager propagates unchanged."""

    class _FailingManager:
        def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
            raise OMSValidationError(
                "manager refused order creation",
                error_code="OMS_MGR_TEST",
                details={"rows": risk_decisions.height},
            )

    pipeline, _registry, _manager = _make_pipeline(manager=_FailingManager())

    with pytest.raises(
        OMSValidationError,
        match="manager refused order creation",
    ) as exc_info:
        pipeline.run(
            _MANAGER_NAME,
            _risk_frame(
                decisions=[RiskDecision.APPROVE],
                approved_weights=[0.1],
            ),
        )

    assert exc_info.value.error_code == "OMS_MGR_TEST"


def test_extra_manager_columns_are_dropped() -> None:
    """Non-canonical manager columns are dropped during finalization."""
    frame = _order_frame(
        symbols=["BTCUSDT"],
        sides=[OrderSide.BUY.value],
        quantities=[0.1],
    ).with_columns(
        pl.lit("noise").alias("order_note"),
        pl.lit(99).alias("rank"),
    )
    pipeline, _registry, _manager = _make_pipeline(
        manager=_RecordingManager(frame),
    )

    result = pipeline.run(
        _MANAGER_NAME,
        _risk_frame(
            decisions=[RiskDecision.APPROVE],
            approved_weights=[0.1],
        ),
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "order_note" not in result.columns
    assert "rank" not in result.columns


def test_multiple_registered_managers_resolve_by_name() -> None:
    """Pipeline resolves the requested manager among multiple registrations."""
    buy_output = _order_frame(
        symbols=["BTCUSDT"],
        sides=[OrderSide.BUY.value],
        quantities=[0.1],
        order_ids=["buy-1"],
    )
    sell_output = _order_frame(
        symbols=["ETHUSDT"],
        sides=[OrderSide.SELL.value],
        quantities=[0.2],
        order_ids=["sell-1"],
    )
    buy = _RecordingManager(buy_output)
    sell = _RecordingManager(sell_output)
    registry = OrderManagerRegistry()
    registry.register_many(
        {
            "buy": buy,
            "sell": sell,
        }
    )
    pipeline = OrderPipeline(registry)

    result = pipeline.run(
        "sell",
        _risk_frame(
            decisions=[RiskDecision.APPROVE],
            approved_weights=[-0.2],
            signals=["SELL"],
        ),
    )

    assert len(sell.calls) == 1
    assert len(buy.calls) == 0
    assert result.get_column("side").to_list() == [OrderSide.SELL.value]
    assert result.schema == MERGED_ORDER_SCHEMA
