"""Unit tests for the CQROS RiskManager interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl
import pytest

from cqros.core.exceptions import CQROSError, ResearchError
from cqros.risk import (
    RiskError,
    RiskManager,
    RiskValidationError,
    validate_portfolio_frame,
)
from cqros.risk.exceptions import RiskError as RiskErrorDirect
from cqros.risk.exceptions import (
    RiskValidationError as RiskValidationErrorDirect,
)
from cqros.risk.interfaces import RiskManager as RiskManagerDirect
from cqros.risk.interfaces import (
    validate_portfolio_frame as validate_portfolio_frame_direct,
)


class _ConformingManager:
    """Minimal RiskManager-shaped stub for protocol conformance."""

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return portfolios


class _IncompleteManager:
    """Stub missing ``evaluate`` so protocol conformance must fail."""

    def optimize(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return portfolios


def test_risk_interface_exports_match_modules() -> None:
    """Package exports match the interface and exception modules by identity."""
    assert RiskManager is RiskManagerDirect
    assert validate_portfolio_frame is validate_portfolio_frame_direct
    assert RiskError is RiskErrorDirect
    assert RiskValidationError is RiskValidationErrorDirect


def test_risk_manager_is_runtime_checkable_protocol() -> None:
    """RiskManager is a runtime-checkable Protocol."""
    assert isinstance(RiskManager, type)
    assert issubclass(RiskManager, Protocol)
    assert getattr(RiskManager, "_is_runtime_protocol", False) is True
    assert runtime_checkable(RiskManager) is RiskManager


def test_risk_manager_declares_evaluate_method() -> None:
    """RiskManager exposes the evaluate contract method."""
    assert hasattr(RiskManager, "evaluate")
    assert callable(getattr(RiskManager, "evaluate"))


def test_conforming_manager_satisfies_protocol() -> None:
    """A complete RiskManager-shaped object passes isinstance."""
    manager = _ConformingManager()
    assert isinstance(manager, RiskManager)
    frame = pl.DataFrame({"signal": ["BUY"]})
    assert manager.evaluate(frame).equals(frame)


def test_incomplete_manager_fails_protocol() -> None:
    """Objects without ``evaluate`` are not RiskManagers."""
    assert not isinstance(_IncompleteManager(), RiskManager)


def test_non_manager_object_fails_protocol() -> None:
    """Unrelated objects are not RiskManagers."""
    assert not isinstance(object(), RiskManager)
    assert not isinstance({"evaluate": lambda frame: frame}, RiskManager)


def test_risk_error_hierarchy() -> None:
    """Risk exceptions follow the CQROS research hierarchy."""
    assert issubclass(RiskError, ResearchError)
    assert issubclass(RiskError, CQROSError)
    assert issubclass(RiskValidationError, RiskError)
    assert issubclass(RiskValidationError, ResearchError)
    assert issubclass(RiskValidationError, CQROSError)


def test_risk_validation_error_construction() -> None:
    """RiskValidationError stores structured diagnostic fields."""
    error = RiskValidationError(
        "invalid risk input",
        error_code="RISK_FRAME_TYPE",
        details={"actual_type": "list"},
        recovery_suggestion="Pass a polars DataFrame.",
    )
    assert error.message == "invalid risk input"
    assert error.error_code == "RISK_FRAME_TYPE"
    assert dict(error.details) == {"actual_type": "list"}
    assert error.recovery_suggestion == "Pass a polars DataFrame."
    assert isinstance(error, RiskError)


def test_validate_portfolio_frame_accepts_non_empty_dataframe() -> None:
    """validate_portfolio_frame returns the input DataFrame unchanged."""
    frame = pl.DataFrame({"signal": ["BUY", "SELL"]})
    assert validate_portfolio_frame(frame) is frame


def test_validate_portfolio_frame_rejects_non_dataframe() -> None:
    """Non-DataFrame inputs raise RiskValidationError."""
    with pytest.raises(RiskValidationError) as exc_info:
        validate_portfolio_frame([{"signal": "BUY"}])

    error = exc_info.value
    assert error.error_code == "RISK_FRAME_TYPE"
    assert error.details["actual_type"] == "list"
    assert "polars DataFrame" in error.message


def test_validate_portfolio_frame_rejects_empty_dataframe() -> None:
    """Empty DataFrames raise RiskValidationError."""
    empty = pl.DataFrame({"signal": []})
    with pytest.raises(RiskValidationError) as exc_info:
        validate_portfolio_frame(empty)

    error = exc_info.value
    assert error.error_code == "RISK_FRAME_EMPTY"
    assert error.details["rows"] == 0
    assert "at least one row" in error.message
