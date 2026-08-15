"""Unit tests for CQROS ``PositionEngineRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.positions import (
    AverageCostPositionEngine,
    PositionEngineRegistry,
    PositionValidationError,
)
from cqros.positions.registry import PositionEngineRegistry as PositionEngineRegistryDirect


class _StubEngine:
    """Minimal PositionEngine-shaped stub for registry unit tests."""

    def build(self, trades: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        return trades


class _IncompleteEngine:
    """Stub missing ``build`` so protocol checks must fail."""

    def execute(self, trades: pl.DataFrame) -> pl.DataFrame:
        return trades


def test_position_engine_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert PositionEngineRegistry is PositionEngineRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no engines."""
    registry = PositionEngineRegistry()
    assert registry.list() == ()
    assert registry.exists("average_cost") is False


def test_register_and_get() -> None:
    """register stores an engine that get can retrieve by name."""
    registry = PositionEngineRegistry()
    engine = AverageCostPositionEngine()
    registry.register("average_cost", engine)
    assert registry.get("average_cost") is engine


def test_register_rejects_duplicates() -> None:
    """Duplicate engine names raise PositionValidationError."""
    registry = PositionEngineRegistry()
    first = AverageCostPositionEngine()
    registry.register("average_cost", first)
    with pytest.raises(PositionValidationError, match="already registered") as exc_info:
        registry.register("average_cost", AverageCostPositionEngine())
    assert exc_info.value.error_code == "POS_REG_DUPLICATE"
    assert registry.get("average_cost") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise PositionValidationError."""
    registry = PositionEngineRegistry()
    engine = AverageCostPositionEngine()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(PositionValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, engine)
        assert exc_info.value.error_code == "POS_REG_NAME_BLANK"


def test_register_rejects_invalid_engine() -> None:
    """Objects that do not implement PositionEngine are rejected."""
    registry = PositionEngineRegistry()
    for invalid in (None, "average_cost", 123, object(), _IncompleteEngine()):
        with pytest.raises(
            PositionValidationError,
            match="PositionEngine protocol",
        ) as exc_info:
            registry.register("average_cost", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "POS_REG_NOT_ENGINE"


def test_register_many_and_clear() -> None:
    """register_many stores every provided engine; clear empties the registry."""
    registry = PositionEngineRegistry()
    average_cost = AverageCostPositionEngine()
    stub = _StubEngine()
    registry.register_many({"average_cost": average_cost, "stub": stub})
    assert registry.list() == ("average_cost", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises when the engine name is unknown."""
    registry = PositionEngineRegistry()
    with pytest.raises(PositionValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "POS_REG_UNKNOWN"
