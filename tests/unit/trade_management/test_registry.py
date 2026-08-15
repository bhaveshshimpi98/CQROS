"""Unit tests for CQROS ``TradeManagementManagerRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.trade_management import (
    SimpleTradeManagementManager,
    TradeManagementManagerRegistry,
    TradeManagementValidationError,
)
from cqros.trade_management.registry import (
    TradeManagementManagerRegistry as TradeManagementManagerRegistryDirect,
)


class _StubManager:
    """Minimal TradeManagementManager-shaped stub for registry unit tests."""

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        return accounting


class _IncompleteManager:
    """Stub missing ``evaluate`` so protocol checks must fail."""

    def execute(self, accounting: pl.DataFrame) -> pl.DataFrame:
        return accounting


def test_trade_management_manager_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert TradeManagementManagerRegistry is TradeManagementManagerRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no managers."""
    registry = TradeManagementManagerRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores a manager that get can retrieve by name."""
    registry = TradeManagementManagerRegistry()
    manager = SimpleTradeManagementManager()
    registry.register("simple", manager)
    assert registry.get("simple") is manager
    assert registry.exists("simple") is True
    assert registry.list() == ("simple",)


def test_register_rejects_duplicates() -> None:
    """Duplicate manager names raise TradeManagementValidationError."""
    registry = TradeManagementManagerRegistry()
    first = SimpleTradeManagementManager()
    registry.register("simple", first)
    with pytest.raises(TradeManagementValidationError, match="already registered") as exc_info:
        registry.register("simple", SimpleTradeManagementManager())
    assert exc_info.value.error_code == "TME_REG_DUPLICATE"
    assert registry.get("simple") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise TradeManagementValidationError."""
    registry = TradeManagementManagerRegistry()
    manager = SimpleTradeManagementManager()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(TradeManagementValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, manager)
        assert exc_info.value.error_code == "TME_REG_NAME_BLANK"


def test_register_rejects_invalid_managers() -> None:
    """Objects that do not implement TradeManagementManager are rejected."""
    registry = TradeManagementManagerRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteManager()):
        with pytest.raises(
            TradeManagementValidationError,
            match="TradeManagementManager protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "TME_REG_NOT_MANAGER"


def test_register_many_and_clear() -> None:
    """register_many stores every provided manager; clear empties the registry."""
    registry = TradeManagementManagerRegistry()
    simple = SimpleTradeManagementManager()
    stub = _StubManager()
    registry.register_many({"simple": simple, "stub": stub})
    assert registry.list() == ("simple", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_many_is_atomic() -> None:
    """A failing register_many leaves the registry unchanged."""
    registry = TradeManagementManagerRegistry()
    registry.register("simple", SimpleTradeManagementManager())
    with pytest.raises(TradeManagementValidationError) as exc_info:
        registry.register_many({"other": SimpleTradeManagementManager(), "simple": _StubManager()})
    assert exc_info.value.error_code == "TME_REG_DUPLICATE"
    assert registry.list() == ("simple",)


def test_get_unknown_raises() -> None:
    """get raises when the manager name is unknown."""
    registry = TradeManagementManagerRegistry()
    with pytest.raises(TradeManagementValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "TME_REG_UNKNOWN"
