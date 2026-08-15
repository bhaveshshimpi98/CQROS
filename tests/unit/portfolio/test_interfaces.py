"""Unit tests for the CQROS PortfolioOptimizer interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl
import pytest

from cqros.core.exceptions import CQROSError, ResearchError
from cqros.portfolio import (
    PortfolioError,
    PortfolioOptimizer,
    PortfolioValidationError,
    validate_signals_frame,
)
from cqros.portfolio.exceptions import PortfolioError as PortfolioErrorDirect
from cqros.portfolio.exceptions import (
    PortfolioValidationError as PortfolioValidationErrorDirect,
)
from cqros.portfolio.interfaces import (
    PortfolioOptimizer as PortfolioOptimizerDirect,
)
from cqros.portfolio.interfaces import (
    validate_signals_frame as validate_signals_frame_direct,
)


class _ConformingOptimizer:
    """Minimal PortfolioOptimizer-shaped stub for protocol conformance."""

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        return signals


class _IncompleteOptimizer:
    """Stub missing ``optimize`` so protocol conformance must fail."""

    def generate(self, signals: pl.DataFrame) -> pl.DataFrame:
        return signals


def test_portfolio_interface_exports_match_modules() -> None:
    """Package exports match the interface and exception modules by identity."""
    assert PortfolioOptimizer is PortfolioOptimizerDirect
    assert validate_signals_frame is validate_signals_frame_direct
    assert PortfolioError is PortfolioErrorDirect
    assert PortfolioValidationError is PortfolioValidationErrorDirect


def test_portfolio_optimizer_is_runtime_checkable_protocol() -> None:
    """PortfolioOptimizer is a runtime-checkable Protocol."""
    assert isinstance(PortfolioOptimizer, type)
    assert issubclass(PortfolioOptimizer, Protocol)
    assert getattr(PortfolioOptimizer, "_is_runtime_protocol", False) is True
    assert runtime_checkable(PortfolioOptimizer) is PortfolioOptimizer


def test_portfolio_optimizer_declares_optimize_method() -> None:
    """PortfolioOptimizer exposes the optimize contract method."""
    assert hasattr(PortfolioOptimizer, "optimize")
    assert callable(PortfolioOptimizer.optimize)


def test_conforming_optimizer_satisfies_protocol() -> None:
    """A complete PortfolioOptimizer-shaped object passes isinstance."""
    optimizer = _ConformingOptimizer()
    assert isinstance(optimizer, PortfolioOptimizer)
    frame = pl.DataFrame({"signal": ["BUY"]})
    assert optimizer.optimize(frame).equals(frame)


def test_incomplete_optimizer_fails_protocol() -> None:
    """Objects without ``optimize`` are not PortfolioOptimizers."""
    assert not isinstance(_IncompleteOptimizer(), PortfolioOptimizer)


def test_non_optimizer_object_fails_protocol() -> None:
    """Unrelated objects are not PortfolioOptimizers."""
    assert not isinstance(object(), PortfolioOptimizer)

    def _optimize(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    assert not isinstance({"optimize": _optimize}, PortfolioOptimizer)


def test_portfolio_error_hierarchy() -> None:
    """Portfolio exceptions follow the CQROS research hierarchy."""
    assert issubclass(PortfolioError, ResearchError)
    assert issubclass(PortfolioError, CQROSError)
    assert issubclass(PortfolioValidationError, PortfolioError)
    assert issubclass(PortfolioValidationError, ResearchError)
    assert issubclass(PortfolioValidationError, CQROSError)


def test_portfolio_validation_error_construction() -> None:
    """PortfolioValidationError stores structured diagnostic fields."""
    error = PortfolioValidationError(
        "invalid portfolio input",
        error_code="PORTFOLIO_FRAME_TYPE",
        details={"actual_type": "list"},
        recovery_suggestion="Pass a polars DataFrame.",
    )
    assert error.message == "invalid portfolio input"
    assert error.error_code == "PORTFOLIO_FRAME_TYPE"
    assert dict(error.details) == {"actual_type": "list"}
    assert error.recovery_suggestion == "Pass a polars DataFrame."
    assert isinstance(error, PortfolioError)


def test_validate_signals_frame_accepts_non_empty_dataframe() -> None:
    """validate_signals_frame returns the input DataFrame unchanged."""
    frame = pl.DataFrame({"signal": ["BUY", "SELL"]})
    assert validate_signals_frame(frame) is frame


def test_validate_signals_frame_rejects_non_dataframe() -> None:
    """Non-DataFrame inputs raise PortfolioValidationError."""
    with pytest.raises(PortfolioValidationError) as exc_info:
        validate_signals_frame([{"signal": "BUY"}])

    error = exc_info.value
    assert error.error_code == "PORTFOLIO_FRAME_TYPE"
    assert error.details["actual_type"] == "list"
    assert "polars DataFrame" in error.message


def test_validate_signals_frame_rejects_empty_dataframe() -> None:
    """Empty DataFrames raise PortfolioValidationError."""
    empty = pl.DataFrame({"signal": []})
    with pytest.raises(PortfolioValidationError) as exc_info:
        validate_signals_frame(empty)

    error = exc_info.value
    assert error.error_code == "PORTFOLIO_FRAME_EMPTY"
    assert error.details["rows"] == 0
    assert "at least one row" in error.message
