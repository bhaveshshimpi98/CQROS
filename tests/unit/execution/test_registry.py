"""Unit tests for CQROS ``ExecutionSimulatorRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.execution import (
    ExecutionSimulatorRegistry,
    ExecutionValidationError,
    SimpleExecutionSimulator,
)
from cqros.execution.registry import (
    ExecutionSimulatorRegistry as ExecutionSimulatorRegistryDirect,
)


class _StubSimulator:
    """Minimal ExecutionSimulator-shaped stub for registry unit tests."""

    def execute(self, orders: pl.DataFrame, *, manager: str) -> pl.DataFrame:
        return orders


class _IncompleteSimulator:
    """Stub missing ``execute`` so protocol checks must fail."""

    def simulate(self, orders: pl.DataFrame) -> pl.DataFrame:
        return orders


def test_execution_simulator_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert ExecutionSimulatorRegistry is ExecutionSimulatorRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no simulators."""
    registry = ExecutionSimulatorRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores a simulator that get can retrieve by name."""
    registry = ExecutionSimulatorRegistry()
    simulator = SimpleExecutionSimulator()
    registry.register("simple", simulator)
    assert registry.get("simple") is simulator


def test_register_rejects_duplicates() -> None:
    """Duplicate simulator names raise ExecutionValidationError."""
    registry = ExecutionSimulatorRegistry()
    first = SimpleExecutionSimulator()
    registry.register("simple", first)
    with pytest.raises(ExecutionValidationError, match="already registered") as exc_info:
        registry.register("simple", SimpleExecutionSimulator())
    assert exc_info.value.error_code == "EXEC_REG_DUPLICATE"
    assert registry.get("simple") is first


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise ExecutionValidationError."""
    registry = ExecutionSimulatorRegistry()
    simulator = SimpleExecutionSimulator()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(ExecutionValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, simulator)
        assert exc_info.value.error_code == "EXEC_REG_NAME_BLANK"


def test_register_rejects_invalid_simulator() -> None:
    """Objects that do not implement ExecutionSimulator are rejected."""
    registry = ExecutionSimulatorRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteSimulator()):
        with pytest.raises(
            ExecutionValidationError,
            match="ExecutionSimulator protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "EXEC_REG_NOT_SIMULATOR"


def test_register_many_and_clear() -> None:
    """register_many stores every provided simulator; clear empties the registry."""
    registry = ExecutionSimulatorRegistry()
    simple = SimpleExecutionSimulator()
    stub = _StubSimulator()
    registry.register_many({"simple": simple, "stub": stub})
    assert registry.list() == ("simple", "stub")
    assert registry.get("stub") is stub
    registry.clear()
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises when the simulator name is unknown."""
    registry = ExecutionSimulatorRegistry()
    with pytest.raises(ExecutionValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "EXEC_REG_UNKNOWN"
