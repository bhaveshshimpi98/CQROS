"""Unit tests for the CQROS OrderManager interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl
import pytest

from cqros.core.exceptions import CQROSError, ResearchError
from cqros.oms import (
    OMSException,
    OMSValidationError,
    OrderManager,
    validate_risk_frame,
)
from cqros.oms.exceptions import OMSException as OMSExceptionDirect
from cqros.oms.exceptions import OMSValidationError as OMSValidationErrorDirect
from cqros.oms.interfaces import OrderManager as OrderManagerDirect
from cqros.oms.interfaces import validate_risk_frame as validate_risk_frame_direct


class _ConformingManager:
    """Minimal OrderManager-shaped stub for protocol conformance."""

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return risk_decisions


class _IncompleteManager:
    """Stub missing ``create_orders`` so protocol conformance must fail."""

    def evaluate(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return risk_decisions


def test_oms_interface_exports_match_modules() -> None:
    """Package exports match the interface and exception modules by identity."""
    assert OrderManager is OrderManagerDirect
    assert validate_risk_frame is validate_risk_frame_direct
    assert OMSException is OMSExceptionDirect
    assert OMSValidationError is OMSValidationErrorDirect


def test_order_manager_is_runtime_checkable_protocol() -> None:
    """OrderManager is a runtime-checkable Protocol."""
    assert isinstance(OrderManager, type)
    assert issubclass(OrderManager, Protocol)
    assert getattr(OrderManager, "_is_runtime_protocol", False) is True
    assert runtime_checkable(OrderManager) is OrderManager


def test_order_manager_declares_create_orders_method() -> None:
    """OrderManager exposes the create_orders contract method."""
    assert hasattr(OrderManager, "create_orders")
    assert callable(getattr(OrderManager, "create_orders"))


def test_conforming_manager_satisfies_protocol() -> None:
    """A complete OrderManager-shaped object passes isinstance."""
    manager = _ConformingManager()
    assert isinstance(manager, OrderManager)
    frame = pl.DataFrame({"decision": ["APPROVE"]})
    assert manager.create_orders(frame).equals(frame)


def test_incomplete_manager_fails_protocol() -> None:
    """Objects without ``create_orders`` are not OrderManagers."""
    assert not isinstance(_IncompleteManager(), OrderManager)


def test_non_manager_object_fails_protocol() -> None:
    """Unrelated objects are not OrderManagers."""
    assert not isinstance(object(), OrderManager)
    assert not isinstance(
        {"create_orders": lambda frame: frame},
        OrderManager,
    )


def test_oms_exception_hierarchy() -> None:
    """OMS exceptions follow the CQROS research hierarchy."""
    assert issubclass(OMSException, ResearchError)
    assert issubclass(OMSException, CQROSError)
    assert issubclass(OMSValidationError, OMSException)
    assert issubclass(OMSValidationError, ResearchError)
    assert issubclass(OMSValidationError, CQROSError)


def test_oms_validation_error_construction() -> None:
    """OMSValidationError stores structured diagnostic fields."""
    error = OMSValidationError(
        "invalid oms input",
        error_code="OMS_FRAME_TYPE",
        details={"actual_type": "list"},
        recovery_suggestion="Pass a polars DataFrame.",
    )
    assert error.message == "invalid oms input"
    assert error.error_code == "OMS_FRAME_TYPE"
    assert dict(error.details) == {"actual_type": "list"}
    assert error.recovery_suggestion == "Pass a polars DataFrame."
    assert isinstance(error, OMSException)


def test_validate_risk_frame_accepts_non_empty_dataframe() -> None:
    """validate_risk_frame returns the input DataFrame unchanged."""
    frame = pl.DataFrame({"decision": ["APPROVE", "REJECT"]})
    assert validate_risk_frame(frame) is frame


def test_validate_risk_frame_rejects_non_dataframe() -> None:
    """Non-DataFrame inputs raise OMSValidationError."""
    with pytest.raises(OMSValidationError) as exc_info:
        validate_risk_frame([{"decision": "APPROVE"}])

    error = exc_info.value
    assert error.error_code == "OMS_FRAME_TYPE"
    assert error.details["actual_type"] == "list"
    assert "polars DataFrame" in error.message


def test_validate_risk_frame_rejects_empty_dataframe() -> None:
    """Empty DataFrames raise OMSValidationError."""
    empty = pl.DataFrame({"decision": []})
    with pytest.raises(OMSValidationError) as exc_info:
        validate_risk_frame(empty)

    error = exc_info.value
    assert error.error_code == "OMS_FRAME_EMPTY"
    assert error.details["rows"] == 0
    assert "at least one row" in error.message
