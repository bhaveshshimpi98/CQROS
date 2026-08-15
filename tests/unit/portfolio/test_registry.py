"""Unit tests for CQROS ``PortfolioOptimizerRegistry``."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.portfolio import (
    EqualWeightOptimizer,
    PortfolioOptimizer,
    PortfolioOptimizerRegistry,
    PortfolioValidationError,
)
from cqros.portfolio.registry import (
    PortfolioOptimizerRegistry as PortfolioOptimizerRegistryDirect,
)


class _StubOptimizer:
    """Minimal PortfolioOptimizer-shaped stub for registry unit tests."""

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        return signals


class _IncompleteOptimizer:
    """Stub missing ``optimize`` so protocol checks must fail."""

    def generate(self, signals: pl.DataFrame) -> pl.DataFrame:
        return signals


def test_portfolio_optimizer_registry_is_exported_from_package() -> None:
    """Package export matches the registry module by identity."""
    assert PortfolioOptimizerRegistry is PortfolioOptimizerRegistryDirect


def test_empty_registry() -> None:
    """A new registry contains no optimizers."""
    registry = PortfolioOptimizerRegistry()
    assert registry.list() == ()
    assert registry.exists("equal_weight") is False


def test_register_and_get() -> None:
    """register stores an optimizer that get can retrieve by name."""
    registry = PortfolioOptimizerRegistry()
    optimizer = EqualWeightOptimizer()
    registry.register("equal_weight", optimizer)
    assert registry.get("equal_weight") is optimizer


def test_register_rejects_duplicates() -> None:
    """Duplicate optimizer names raise PortfolioValidationError."""
    registry = PortfolioOptimizerRegistry()
    first = EqualWeightOptimizer()
    registry.register("equal_weight", first)
    with pytest.raises(PortfolioValidationError, match="already registered") as exc_info:
        registry.register("equal_weight", EqualWeightOptimizer())
    assert exc_info.value.error_code == "PORTFOLIO_REG_DUPLICATE"
    assert registry.get("equal_weight") is first
    assert registry.list() == ("equal_weight",)


def test_register_rejects_blank_names() -> None:
    """Empty or whitespace-only names raise PortfolioValidationError."""
    registry = PortfolioOptimizerRegistry()
    optimizer = EqualWeightOptimizer()
    for invalid_name in ("", "   ", "\t"):
        with pytest.raises(PortfolioValidationError, match="non-blank") as exc_info:
            registry.register(invalid_name, optimizer)
        assert exc_info.value.error_code == "PORTFOLIO_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_non_string_names() -> None:
    """Non-string names raise PortfolioValidationError."""
    registry = PortfolioOptimizerRegistry()
    with pytest.raises(PortfolioValidationError, match="non-blank") as exc_info:
        registry.register(123, EqualWeightOptimizer())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "PORTFOLIO_REG_NAME_BLANK"
    assert registry.list() == ()


def test_register_rejects_invalid_optimizer() -> None:
    """Objects that do not implement PortfolioOptimizer are rejected."""
    registry = PortfolioOptimizerRegistry()
    for invalid in (None, "equal_weight", 123, object(), _IncompleteOptimizer()):
        with pytest.raises(
            PortfolioValidationError,
            match="PortfolioOptimizer protocol",
        ) as exc_info:
            registry.register("equal_weight", invalid)  # type: ignore[arg-type]
        assert exc_info.value.error_code == "PORTFOLIO_REG_NOT_OPTIMIZER"
    assert registry.list() == ()


def test_register_many_registers_all() -> None:
    """register_many stores every provided optimizer."""
    registry = PortfolioOptimizerRegistry()
    equal_weight = EqualWeightOptimizer()
    stub = _StubOptimizer()
    registry.register_many(
        {
            "equal_weight": equal_weight,
            "stub": stub,
        }
    )
    assert registry.get("equal_weight") is equal_weight
    assert registry.get("stub") is stub
    assert isinstance(registry.get("stub"), PortfolioOptimizer)
    assert registry.list() == ("equal_weight", "stub")


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = PortfolioOptimizerRegistry()
    registry.register("equal_weight", EqualWeightOptimizer())
    with pytest.raises(PortfolioValidationError, match="already registered"):
        registry.register_many(
            {
                "fixed_weight": _StubOptimizer(),
                "equal_weight": EqualWeightOptimizer(),
            }
        )
    assert registry.list() == ("equal_weight",)
    assert registry.exists("fixed_weight") is False


def test_register_many_is_atomic_on_invalid_name() -> None:
    """register_many leaves the registry unchanged when a name is invalid."""
    registry = PortfolioOptimizerRegistry()
    with pytest.raises(PortfolioValidationError, match="non-blank"):
        registry.register_many(
            {
                "equal_weight": EqualWeightOptimizer(),
                "": EqualWeightOptimizer(),
            }
        )
    assert registry.list() == ()
    assert registry.exists("equal_weight") is False


def test_register_many_is_atomic_on_invalid_optimizer() -> None:
    """register_many leaves the registry unchanged when an invalid object appears."""
    registry = PortfolioOptimizerRegistry()
    with pytest.raises(
        PortfolioValidationError,
        match="PortfolioOptimizer protocol",
    ):
        registry.register_many(
            {
                "equal_weight": EqualWeightOptimizer(),
                "bad": object(),  # type: ignore[dict-item]
            }
        )
    assert registry.list() == ()


def test_get_unknown_raises() -> None:
    """get raises PortfolioValidationError for missing names."""
    registry = PortfolioOptimizerRegistry()
    with pytest.raises(PortfolioValidationError, match="not registered") as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "PORTFOLIO_REG_UNKNOWN"


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = PortfolioOptimizerRegistry()
    assert registry.exists("equal_weight") is False
    registry.register("equal_weight", EqualWeightOptimizer())
    assert registry.exists("equal_weight") is True
    assert registry.exists("fixed_weight") is False


def test_clear() -> None:
    """clear removes all registered optimizers."""
    registry = PortfolioOptimizerRegistry()
    registry.register_many(
        {
            "equal_weight": EqualWeightOptimizer(),
            "stub": _StubOptimizer(),
        }
    )
    registry.clear()
    assert registry.list() == ()
    assert registry.exists("equal_weight") is False


def test_list_preserves_insertion_order() -> None:
    """list returns names in registration insertion order."""
    registry = PortfolioOptimizerRegistry()
    registry.register("zeta", _StubOptimizer())
    registry.register("alpha", EqualWeightOptimizer())
    registry.register("mu", _StubOptimizer())
    assert registry.list() == ("zeta", "alpha", "mu")


def test_list_returns_immutable_snapshot() -> None:
    """list returns a new tuple unaffected by later registry mutation."""
    registry = PortfolioOptimizerRegistry()
    registry.register_many(
        {
            "equal_weight": EqualWeightOptimizer(),
            "stub": _StubOptimizer(),
        }
    )
    names = registry.list()
    assert isinstance(names, tuple)
    with pytest.raises(TypeError):
        names[0] = "changed"  # type: ignore[index]
    registry.clear()
    assert names == ("equal_weight", "stub")
    assert registry.list() == ()


def test_list_returns_independent_snapshots() -> None:
    """Repeated list calls return independent tuple objects."""
    registry = PortfolioOptimizerRegistry()
    registry.register("equal_weight", EqualWeightOptimizer())
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_register_does_not_mutate_optimizer() -> None:
    """Registry stores the optimizer reference without altering it."""
    registry = PortfolioOptimizerRegistry()
    optimizer = EqualWeightOptimizer()
    registry.register("equal_weight", optimizer)
    assert registry.get("equal_weight") is optimizer
    assert isinstance(optimizer, PortfolioOptimizer)


def test_register_many_preserves_mapping_insertion_order() -> None:
    """register_many preserves mapping insertion order in list()."""
    registry = PortfolioOptimizerRegistry()
    registry.register_many(
        {
            "risk_parity": _StubOptimizer(),
            "equal_weight": EqualWeightOptimizer(),
            "kelly": _StubOptimizer(),
        }
    )
    assert registry.list() == ("risk_parity", "equal_weight", "kelly")
