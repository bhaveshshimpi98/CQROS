"""Unit tests for CQROS ``PyramidingRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.pyramiding import (
    PyramidingRegistry,
    PyramidingValidationError,
    SimplePyramidingEngine,
)
from cqros.pyramiding.registry import PyramidingRegistry as PyramidingRegistryDirect


class _StubEngine:
    """Minimal PyramidingEngine-shaped stub for registry unit tests."""

    def evaluate(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
    ) -> pl.DataFrame:
        return accounting


class _IncompleteEngine:
    """Stub missing ``evaluate`` so protocol checks must fail."""

    def execute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


def test_pyramiding_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert PyramidingRegistry is PyramidingRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no engines."""
    registry = PyramidingRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores an engine that get can retrieve by name."""
    registry = PyramidingRegistry()
    engine = SimplePyramidingEngine()
    registry.register("simple", engine)
    assert registry.get("simple") is engine
    assert registry.exists("simple") is True
    assert registry.list() == ("simple",)


def test_register_preserves_insertion_order() -> None:
    """Engine names are returned in insertion order."""
    registry = PyramidingRegistry()
    registry.register("alpha", SimplePyramidingEngine())
    registry.register("beta", _StubEngine())
    assert registry.list() == ("alpha", "beta")


def test_register_rejects_duplicates() -> None:
    """Duplicate engine names raise PyramidingValidationError."""
    registry = PyramidingRegistry()
    first = SimplePyramidingEngine()
    registry.register("simple", first)
    with pytest.raises(PyramidingValidationError, match="already registered") as exc_info:
        registry.register("simple", SimplePyramidingEngine())
    assert exc_info.value.error_code == "PYR_REG_DUPLICATE"
    assert registry.get("simple") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise PyramidingValidationError."""
    registry = PyramidingRegistry()
    engine = SimplePyramidingEngine()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(PyramidingValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, engine)
        assert exc_info.value.error_code == "PYR_REG_NAME_BLANK"


def test_register_rejects_invalid_engines() -> None:
    """Objects that do not implement PyramidingEngine are rejected."""
    registry = PyramidingRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteEngine()):
        with pytest.raises(
            PyramidingValidationError,
            match="PyramidingEngine protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "PYR_REG_NOT_ENGINE"


def test_get_unknown_raises() -> None:
    """get raises when the engine name is unknown."""
    registry = PyramidingRegistry()
    with pytest.raises(PyramidingValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "PYR_REG_UNKNOWN"


def test_exists_returns_false_for_unknown() -> None:
    """exists returns False for unregistered names without raising."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    assert registry.exists("other") is False
    assert registry.exists("simple") is True


def test_clear_removes_all_engines() -> None:
    """clear empties the registry."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    registry.register("stub", _StubEngine())
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("simple") is False
    assert registry.exists("stub") is False


def test_register_many_and_clear() -> None:
    """register_many stores every provided engine; clear empties the registry."""
    registry = PyramidingRegistry()
    simple = SimplePyramidingEngine()
    stub = _StubEngine()
    registry.register_many({"simple": simple, "stub": stub})
    assert registry.list() == ("simple", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_many_is_atomic() -> None:
    """A failing register_many leaves the registry unchanged."""
    registry = PyramidingRegistry()
    registry.register("simple", SimplePyramidingEngine())
    with pytest.raises(PyramidingValidationError) as exc_info:
        registry.register_many({"other": SimplePyramidingEngine(), "simple": _StubEngine()})
    assert exc_info.value.error_code == "PYR_REG_DUPLICATE"
    assert registry.list() == ("simple",)


def test_register_many_rejects_blank_name_in_mapping() -> None:
    """register_many rejects blank names inside the mapping."""
    registry = PyramidingRegistry()
    with pytest.raises(PyramidingValidationError) as exc_info:
        registry.register_many({"": SimplePyramidingEngine()})
    assert exc_info.value.error_code == "PYR_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_many_rejects_non_engine_in_mapping() -> None:
    """register_many rejects objects that do not implement PyramidingEngine."""
    registry = PyramidingRegistry()
    with pytest.raises(PyramidingValidationError) as exc_info:
        registry.register_many({"stub": object()})  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PYR_REG_NOT_ENGINE"
    assert registry.list() == ()
