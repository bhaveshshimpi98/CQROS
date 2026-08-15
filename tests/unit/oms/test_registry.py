"""Unit tests for CQROS ``OrderManagerRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.oms import (
    OMSValidationError,
    OrderManager,
    OrderManagerRegistry,
    SimpleOrderManager,
)
from cqros.oms.registry import OrderManagerRegistry as OrderManagerRegistryDirect


class _StubManager:
    """Minimal OrderManager-shaped stub for registry unit tests."""

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return risk_decisions


class _IncompleteManager:
    """Stub missing ``create_orders`` so protocol checks must fail."""

    def evaluate(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        return risk_decisions


def test_order_manager_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert OrderManagerRegistry is OrderManagerRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no managers."""
    registry = OrderManagerRegistry()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_and_get() -> None:
    """register stores a manager that get can retrieve by name."""
    registry = OrderManagerRegistry()
    manager = SimpleOrderManager()
    registry.register("simple", manager)
    assert registry.get("simple") is manager


def test_register_rejects_duplicates() -> None:
    """Duplicate manager names raise OMSValidationError."""
    registry = OrderManagerRegistry()
    first = SimpleOrderManager()
    registry.register("simple", first)
    with pytest.raises(OMSValidationError, match="already registered") as exc_info:
        registry.register("simple", SimpleOrderManager())
    assert exc_info.value.error_code == "OMS_REG_DUPLICATE"
    assert registry.get("simple") is first
    assert registry.list() == ("simple",)


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise OMSValidationError."""
    registry = OrderManagerRegistry()
    manager = SimpleOrderManager()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(OMSValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, manager)
        assert exc_info.value.error_code == "OMS_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_non_string_names() -> None:
    """Non-string names raise OMSValidationError."""
    registry = OrderManagerRegistry()
    with pytest.raises(OMSValidationError, match="non-blank") as exc_info:
        registry.register(123, SimpleOrderManager())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "OMS_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_invalid_manager() -> None:
    """Objects that do not implement OrderManager are rejected."""
    registry = OrderManagerRegistry()
    for invalid in (None, "simple", 123, object(), _IncompleteManager()):
        with pytest.raises(
            OMSValidationError,
            match="OrderManager protocol",
        ) as exc_info:
            registry.register("simple", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "OMS_REG_NOT_MANAGER"
    assert registry.list() == ()


def test_register_many_registers_all() -> None:
    """register_many stores every provided manager."""
    registry = OrderManagerRegistry()
    simple = SimpleOrderManager()
    stub = _StubManager()
    registry.register_many(
        {
            "simple": simple,
            "stub": stub,
        }
    )
    assert registry.get("simple") is simple
    assert registry.get("stub") is stub
    assert isinstance(registry.get("stub"), OrderManager)
    assert registry.list() == ("simple", "stub")


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = OrderManagerRegistry()
    registry.register("simple", SimpleOrderManager())
    with pytest.raises(OMSValidationError, match="already registered"):
        registry.register_many(
            {
                "twap": _StubManager(),
                "simple": SimpleOrderManager(),
            }
        )
    assert registry.list() == ("simple",)
    assert registry.exists("twap") is False


def test_register_many_is_atomic_on_invalid_name() -> None:
    """register_many leaves the registry unchanged when a name is invalid."""
    registry = OrderManagerRegistry()
    with pytest.raises(OMSValidationError, match="non-blank"):
        registry.register_many(
            {
                "simple": SimpleOrderManager(),
                "": SimpleOrderManager(),
            }
        )
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_register_many_is_atomic_on_invalid_manager() -> None:
    """register_many leaves the registry unchanged when an invalid object appears."""
    registry = OrderManagerRegistry()
    with pytest.raises(
        OMSValidationError,
        match="OrderManager protocol",
    ):
        registry.register_many(
            {
                "simple": SimpleOrderManager(),
                "bad": object(),  # type: ignore[dict-item]
            }
        )
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises OMSValidationError for missing names."""
    registry = OrderManagerRegistry()
    with pytest.raises(OMSValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "OMS_REG_UNKNOWN"


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = OrderManagerRegistry()
    assert registry.exists("simple") is False
    registry.register("simple", SimpleOrderManager())
    assert registry.exists("simple") is True
    assert registry.exists("twap") is False


def test_clear() -> None:
    """clear removes all registered managers."""
    registry = OrderManagerRegistry()
    registry.register_many(
        {
            "simple": SimpleOrderManager(),
            "stub": _StubManager(),
        }
    )
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("simple") is False


def test_list_preserves_insertion_order() -> None:
    """list returns names in registration insertion order."""
    registry = OrderManagerRegistry()
    registry.register("zeta", _StubManager())
    registry.register("alpha", SimpleOrderManager())
    registry.register("mu", _StubManager())
    assert registry.list() == ("zeta", "alpha", "mu")


def test_list_returns_immutable_snapshot() -> None:
    """list returns a new tuple unaffected by later registry mutation."""
    registry = OrderManagerRegistry()
    registry.register_many(
        {
            "simple": SimpleOrderManager(),
            "stub": _StubManager(),
        }
    )
    names = registry.list()
    assert isinstance(names, tuple)
    with pytest.raises(TypeError):
        names[0] = "changed"  # type: ignore[index]
    registry.clear()
    assert names == ("simple", "stub")
    assert registry.list() == ()


def test_list_returns_independent_snapshots() -> None:
    """Repeated list calls return independent tuple objects."""
    registry = OrderManagerRegistry()
    registry.register("simple", SimpleOrderManager())
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_register_does_not_mutate_manager() -> None:
    """Registry stores the manager reference without altering it."""
    registry = OrderManagerRegistry()
    manager = SimpleOrderManager()
    registry.register("simple", manager)
    assert registry.get("simple") is manager
    assert isinstance(manager, OrderManager)


def test_register_many_preserves_mapping_insertion_order() -> None:
    """register_many preserves mapping insertion order in list()."""
    registry = OrderManagerRegistry()
    registry.register_many(
        {
            "twap": _StubManager(),
            "simple": SimpleOrderManager(),
            "vwap": _StubManager(),
        }
    )
    assert registry.list() == ("twap", "simple", "vwap")
