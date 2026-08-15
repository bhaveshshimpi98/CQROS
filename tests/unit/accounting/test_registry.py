"""Unit tests for CQROS ``AccountingEngineRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.accounting import (
    AccountingEngineRegistry,
    AccountingValidationError,
    SimplePortfolioAccountingEngine,
)
from cqros.accounting.registry import AccountingEngineRegistry as AccountingEngineRegistryDirect


class _StubEngine:
    """Minimal AccountingEngine-shaped stub for registry unit tests."""

    def build(self, positions: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        return positions


class _IncompleteEngine:
    """Stub missing ``build`` so protocol checks must fail."""

    def execute(self, positions: pl.DataFrame) -> pl.DataFrame:
        return positions


def test_accounting_engine_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert AccountingEngineRegistry is AccountingEngineRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no engines."""
    registry = AccountingEngineRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores an engine that get can retrieve by name."""
    registry = AccountingEngineRegistry()
    engine = SimplePortfolioAccountingEngine()
    registry.register("simple", engine)
    assert registry.get("simple") is engine


def test_register_rejects_duplicates() -> None:
    """Duplicate engine names raise AccountingValidationError."""
    registry = AccountingEngineRegistry()
    first = SimplePortfolioAccountingEngine()
    registry.register("simple", first)
    with pytest.raises(AccountingValidationError, match="already registered") as exc_info:
        registry.register("simple", SimplePortfolioAccountingEngine())
    assert exc_info.value.error_code == "ACC_REG_DUPLICATE"
    assert registry.get("simple") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise AccountingValidationError."""
    registry = AccountingEngineRegistry()
    engine = SimplePortfolioAccountingEngine()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(AccountingValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, engine)
        assert exc_info.value.error_code == "ACC_REG_NAME_BLANK"


def test_register_rejects_invalid_engine() -> None:
    """Objects that do not implement AccountingEngine are rejected."""
    registry = AccountingEngineRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteEngine()):
        with pytest.raises(
            AccountingValidationError,
            match="AccountingEngine protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "ACC_REG_NOT_ENGINE"


def test_register_many_and_clear() -> None:
    """register_many stores every provided engine; clear empties the registry."""
    registry = AccountingEngineRegistry()
    simple = SimplePortfolioAccountingEngine()
    stub = _StubEngine()
    registry.register_many({"simple": simple, "stub": stub})
    assert registry.list() == ("simple", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()


def test_register_many_is_atomic() -> None:
    """A failing register_many leaves the registry unchanged."""
    registry = AccountingEngineRegistry()
    registry.register("simple", SimplePortfolioAccountingEngine())
    with pytest.raises(AccountingValidationError) as exc_info:
        registry.register_many(
            {"other": SimplePortfolioAccountingEngine(), "simple": _StubEngine()}
        )
    assert exc_info.value.error_code == "ACC_REG_DUPLICATE"
    assert registry.list() == ("simple",)


def test_get_unknown_raises() -> None:
    """get raises when the engine name is unknown."""
    registry = AccountingEngineRegistry()
    with pytest.raises(AccountingValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "ACC_REG_UNKNOWN"
