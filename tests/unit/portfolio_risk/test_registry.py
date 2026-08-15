"""Unit tests for CQROS ``PortfolioRiskManagerRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.portfolio_risk import (
    PortfolioRiskManagerRegistry,
    PortfolioRiskValidationError,
    SimplePortfolioRiskManager,
)
from cqros.portfolio_risk.registry import (
    PortfolioRiskManagerRegistry as PortfolioRiskManagerRegistryDirect,
)


class _StubManager:
    """Minimal PortfolioRiskManager-shaped stub for registry unit tests."""

    def evaluate(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        return accounting


class _IncompleteManager:
    """Stub missing ``evaluate`` so protocol checks must fail."""

    def execute(self, accounting: pl.DataFrame) -> pl.DataFrame:
        return accounting


def test_portfolio_risk_manager_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert PortfolioRiskManagerRegistry is PortfolioRiskManagerRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no managers."""
    registry = PortfolioRiskManagerRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores a manager that get can retrieve by name."""
    registry = PortfolioRiskManagerRegistry()
    manager = SimplePortfolioRiskManager()
    registry.register("simple", manager)
    assert registry.get("simple") is manager
    assert registry.exists("simple") is True
    assert registry.list() == ("simple",)


def test_register_rejects_duplicates() -> None:
    """Duplicate manager names raise PortfolioRiskValidationError."""
    registry = PortfolioRiskManagerRegistry()
    first = SimplePortfolioRiskManager()
    registry.register("simple", first)
    with pytest.raises(PortfolioRiskValidationError, match="already registered") as exc_info:
        registry.register("simple", SimplePortfolioRiskManager())
    assert exc_info.value.error_code == "PRISK_REG_DUPLICATE"
    assert registry.get("simple") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise PortfolioRiskValidationError."""
    registry = PortfolioRiskManagerRegistry()
    manager = SimplePortfolioRiskManager()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(PortfolioRiskValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, manager)
        assert exc_info.value.error_code == "PRISK_REG_NAME_BLANK"


def test_register_rejects_invalid_managers() -> None:
    """Objects that do not implement PortfolioRiskManager are rejected."""
    registry = PortfolioRiskManagerRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteManager()):
        with pytest.raises(
            PortfolioRiskValidationError,
            match="PortfolioRiskManager protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "PRISK_REG_NOT_MANAGER"


def test_register_many_and_clear() -> None:
    """register_many stores every provided manager; clear empties the registry."""
    registry = PortfolioRiskManagerRegistry()
    simple = SimplePortfolioRiskManager()
    stub = _StubManager()
    registry.register_many({"simple": simple, "stub": stub})
    assert registry.list() == ("simple", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_many_is_atomic() -> None:
    """A failing register_many leaves the registry unchanged."""
    registry = PortfolioRiskManagerRegistry()
    registry.register("simple", SimplePortfolioRiskManager())
    with pytest.raises(PortfolioRiskValidationError) as exc_info:
        registry.register_many({"other": SimplePortfolioRiskManager(), "simple": _StubManager()})
    assert exc_info.value.error_code == "PRISK_REG_DUPLICATE"
    assert registry.list() == ("simple",)


def test_get_unknown_raises() -> None:
    """get raises when the manager name is unknown."""
    registry = PortfolioRiskManagerRegistry()
    with pytest.raises(PortfolioRiskValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "PRISK_REG_UNKNOWN"
