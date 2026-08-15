"""Unit tests for the CQROS Simple Order Manager."""

from __future__ import annotations

from datetime import UTC, datetime
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
    OrderSide,
    OrderStatus,
    OrderType,
    SimpleOrderManager,
)
from cqros.oms.managers import SimpleOrderManager as SimpleOrderManagerDirect
from cqros.risk.enums import RiskDecision

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "fixed_risk"
_OPTIMIZER = "equal_weight"
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
    if len(approved_weights) != row_count:
        raise ValueError("decisions and approved_weights length mismatch")
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


def test_simple_order_manager_is_exported_from_package() -> None:
    """Package export matches the managers module by identity."""
    assert SimpleOrderManager is SimpleOrderManagerDirect


def test_simple_order_manager_satisfies_protocol() -> None:
    """SimpleOrderManager structurally satisfies OrderManager."""
    assert isinstance(SimpleOrderManager(), OrderManager)


def test_approve_creates_order() -> None:
    """APPROVE rows with non-zero approved_weight produce one order."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.25],
        symbols=["BTCUSDT"],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.height == 1
    assert result.get_column("symbol").to_list() == ["BTCUSDT"]
    assert result.get_column("side").to_list() == [OrderSide.BUY.value]
    assert result.get_column("quantity").to_list() == pytest.approx([0.25])
    assert result.get_column("order_type").to_list() == [OrderType.MARKET.value]
    assert result.get_column("status").to_list() == [OrderStatus.PENDING.value]


def test_reject_is_skipped() -> None:
    """REJECT rows never produce orders."""
    frame = _risk_frame(
        decisions=[RiskDecision.REJECT, RiskDecision.APPROVE],
        approved_weights=[0.5, 0.25],
        symbols=["BTCUSDT", "ETHUSDT"],
        target_weights=[0.5, 0.25],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.height == 1
    assert result.get_column("symbol").to_list() == ["ETHUSDT"]
    assert result.get_column("quantity").to_list() == pytest.approx([0.25])


def test_resize_uses_approved_weight() -> None:
    """RESIZE rows generate orders from approved_weight, not target_weight."""
    frame = _risk_frame(
        decisions=[RiskDecision.RESIZE],
        approved_weights=[0.1],
        target_weights=[0.5],
        symbols=["BTCUSDT"],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.height == 1
    assert result.get_column("quantity").to_list() == pytest.approx([0.1])
    assert result.get_column("side").to_list() == [OrderSide.BUY.value]


def test_positive_weight_maps_to_buy() -> None:
    """Positive approved_weight maps to BUY."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.4],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.get_column("side").to_list() == [OrderSide.BUY.value]


def test_negative_weight_maps_to_sell() -> None:
    """Negative approved_weight maps to SELL with absolute quantity."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[-0.35],
        signals=["SELL"],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.get_column("side").to_list() == [OrderSide.SELL.value]
    assert result.get_column("quantity").to_list() == pytest.approx([0.35])


def test_zero_approved_weight_is_skipped() -> None:
    """Zero approved_weight rows are skipped even when APPROVE."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.0, 0.2],
        symbols=["FLAT", "LONG"],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.height == 1
    assert result.get_column("symbol").to_list() == ["LONG"]


def test_uuid_generation_and_parent_order_id() -> None:
    """order_id uses uuid4().hex and parent_order_id equals order_id."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.2, -0.2],
        symbols=["AAA", "BBB"],
    )
    with (
        patch(
            "cqros.oms.managers.uuid4",
            side_effect=[_UUID_A, _UUID_B],
        ),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    order_ids = result.get_column("order_id").to_list()
    parent_ids = result.get_column("parent_order_id").to_list()
    assert order_ids == [_UUID_A.hex, _UUID_B.hex]
    assert parent_ids == order_ids


def test_timestamps_populated_and_equal() -> None:
    """created_at and updated_at are UTC and equal within a batch."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.1, 0.2],
    )
    with (
        patch(
            "cqros.oms.managers.uuid4",
            side_effect=[_UUID_A, _UUID_B],
        ),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    created = result.get_column("created_at").to_list()
    updated = result.get_column("updated_at").to_list()
    assert created == [_FIXED_NOW, _FIXED_NOW]
    assert updated == created
    assert all(timestamp.tzinfo is not None for timestamp in created)


def test_output_uses_canonical_column_order() -> None:
    """Manager output columns follow CANONICAL_COLUMN_ORDER."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_output_matches_merged_order_schema() -> None:
    """Manager output schema matches MERGED_ORDER_SCHEMA and COLUMN_DTYPES."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.RESIZE],
        approved_weights=[0.1, -0.2],
    )
    with (
        patch(
            "cqros.oms.managers.uuid4",
            side_effect=[_UUID_A, _UUID_B],
        ),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.schema == MERGED_ORDER_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_output_preserves_metadata_and_null_prices() -> None:
    """Preserved metadata and null/zero fill fields match the contract."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.5],
        symbols=["BTCUSDT"],
    )
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    expected = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": [_TIMEFRAME],
            "open_time": [_open_time(0)],
            "order_id": [_UUID_A.hex],
            "parent_order_id": [_UUID_A.hex],
            "model_name": [_MODEL_NAME],
            "model_version": [_MODEL_VERSION],
            "policy": [_POLICY],
            "optimizer": [_OPTIMIZER],
            "side": [OrderSide.BUY.value],
            "order_type": [OrderType.MARKET.value],
            "quantity": [0.5],
            "limit_price": [None],
            "stop_price": [None],
            "filled_quantity": [0.0],
            "average_fill_price": [None],
            "status": [OrderStatus.PENDING.value],
            "created_at": [_FIXED_NOW],
            "updated_at": [_FIXED_NOW],
        },
        schema=MERGED_ORDER_SCHEMA,
    )
    assert_frame_equal(result, expected)


def test_all_rejected_returns_empty_schema_frame() -> None:
    """When every row is ineligible, return an empty MERGED_ORDER_SCHEMA frame."""
    frame = _risk_frame(
        decisions=[RiskDecision.REJECT, RiskDecision.APPROVE],
        approved_weights=[0.5, 0.0],
    )
    result = SimpleOrderManager().create_orders(frame)
    assert result.height == 0
    assert result.schema == MERGED_ORDER_SCHEMA
    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_duplicate_primary_keys_raise() -> None:
    """Duplicate symbol/timeframe/open_time combinations are rejected."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE, RiskDecision.APPROVE],
        approved_weights=[0.1, 0.2],
        symbols=["BTCUSDT", "BTCUSDT"],
        open_times=[_open_time(0), _open_time(0)],
    )
    with pytest.raises(OMSValidationError) as exc_info:
        SimpleOrderManager().create_orders(frame)
    assert exc_info.value.error_code == "OMS_DUPLICATE_KEYS"


def test_missing_required_columns_raise() -> None:
    """Missing risk-schema or lineage columns raise OMSValidationError."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    ).drop("policy")
    with pytest.raises(OMSValidationError) as exc_info:
        SimpleOrderManager().create_orders(frame)
    assert exc_info.value.error_code == "OMS_MISSING_COLUMNS"
    assert "policy" in exc_info.value.details["missing_columns"]


def test_empty_dataframe_raise() -> None:
    """Empty risk frames are rejected by shared validation."""
    empty = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    ).clear()
    with pytest.raises(OMSValidationError) as exc_info:
        SimpleOrderManager().create_orders(empty)
    assert exc_info.value.error_code == "OMS_FRAME_EMPTY"


def test_non_dataframe_input_raise() -> None:
    """Non-DataFrame inputs are rejected by shared validation."""
    with pytest.raises(OMSValidationError) as exc_info:
        SimpleOrderManager().create_orders(
            [{"decision": RiskDecision.APPROVE.value}],  # type: ignore[arg-type]
        )
    assert exc_info.value.error_code == "OMS_FRAME_TYPE"


def test_input_dataframe_is_not_mutated() -> None:
    """create_orders returns a new frame and leaves the input unchanged."""
    frame = _risk_frame(
        decisions=[RiskDecision.APPROVE],
        approved_weights=[0.1],
    )
    original = frame.clone()
    with (
        patch("cqros.oms.managers.uuid4", side_effect=[_UUID_A]),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert_frame_equal(frame, original)
    assert "order_id" not in frame.columns
    assert result is not frame


def test_mixed_decisions_preserve_eligible_row_order() -> None:
    """Eligible rows preserve input order after REJECT and zero skips."""
    frame = _risk_frame(
        decisions=[
            RiskDecision.REJECT,
            RiskDecision.APPROVE,
            RiskDecision.APPROVE,
            RiskDecision.RESIZE,
        ],
        approved_weights=[0.9, 0.0, 0.3, -0.2],
        target_weights=[0.9, 0.0, 0.5, -0.5],
        symbols=["R", "Z", "A", "S"],
    )
    with (
        patch(
            "cqros.oms.managers.uuid4",
            side_effect=[_UUID_A, _UUID_B],
        ),
        patch("cqros.oms.managers.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value = _FIXED_NOW
        result = SimpleOrderManager().create_orders(frame)

    assert result.get_column("symbol").to_list() == ["A", "S"]
    assert result.get_column("side").to_list() == [
        OrderSide.BUY.value,
        OrderSide.SELL.value,
    ]
    assert result.get_column("quantity").to_list() == pytest.approx([0.3, 0.2])
    assert result.get_column("order_id").to_list() == [
        _UUID_A.hex,
        _UUID_B.hex,
    ]
