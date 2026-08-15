"""Unit tests for CQROS production Factor Registry bootstrap."""

from __future__ import annotations

from types import ModuleType

import pytest

from cqros.factors import (
    composite,
    funding,
    liquidation,
    microstructure,
    open_interest,
    price,
    relative,
    volume,
)
from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import FactorRegistrationError
from cqros.factors.metadata import FactorMetadata
from cqros.factors.registry import FactorRegistry

_PRODUCTION_FACTOR_COUNT = 111
_PRODUCTION_CATEGORIES = (
    "composite",
    "funding",
    "liquidation",
    "microstructure",
    "open_interest",
    "price",
    "relative",
    "volume",
)
_CATEGORY_PACKAGES: tuple[ModuleType, ...] = (
    price,
    volume,
    microstructure,
    funding,
    open_interest,
    liquidation,
    relative,
    composite,
)


def _expected_production_factor_names() -> set[str]:
    """Derive expected factor names from every category public API."""
    names: set[str] = set()
    for package in _CATEGORY_PACKAGES:
        for class_name in package.__all__:
            factor_cls = getattr(package, class_name)
            names.add(factor_cls().name)
    return names


def test_build_default_registry_returns_factor_registry() -> None:
    """build_default_registry returns a FactorRegistry instance."""
    registry = build_default_registry()
    assert isinstance(registry, FactorRegistry)


def test_build_default_registry_contains_every_production_factor() -> None:
    """The production registry contains every implemented factor exactly once."""
    registry = build_default_registry()
    expected = _expected_production_factor_names()
    assert len(expected) == _PRODUCTION_FACTOR_COUNT
    assert set(registry.names()) == expected
    assert len(registry.list()) == _PRODUCTION_FACTOR_COUNT


def test_build_default_registry_factor_names_are_unique() -> None:
    """Every production factor name is unique."""
    registry = build_default_registry()
    names = registry.names()
    assert len(names) == len(set(names))
    assert len(names) == _PRODUCTION_FACTOR_COUNT


def test_build_default_registry_produced_columns_are_unique() -> None:
    """Every produced column across the production catalog is unique."""
    registry = build_default_registry()
    columns: list[str] = []
    for factor in registry.list():
        columns.extend(factor.produced_columns)
    assert len(columns) == len(set(columns))
    assert len(columns) >= _PRODUCTION_FACTOR_COUNT


def test_build_default_registry_category_lookup() -> None:
    """by_category returns production factors for every registered category."""
    registry = build_default_registry()
    assert registry.categories() == _PRODUCTION_CATEGORIES
    total = 0
    for category in registry.categories():
        factors = registry.by_category(category)
        assert factors
        assert all(factor.category == category for factor in factors)
        assert tuple(factor.name for factor in factors) == tuple(
            sorted(factor.name for factor in factors)
        )
        total += len(factors)
    assert total == _PRODUCTION_FACTOR_COUNT


def test_build_default_registry_metadata_lookup() -> None:
    """metadata_for returns FactorMetadata for every production factor."""
    registry = build_default_registry()
    for name in registry.names():
        metadata = registry.metadata_for(name)
        factor = registry.get(name)
        assert isinstance(metadata, FactorMetadata)
        assert metadata.name == factor.name
        assert metadata.version == factor.version
        assert metadata.category == factor.category
        assert metadata.produced_columns == tuple(factor.produced_columns)
        assert metadata.lookback == factor.lookback


def test_build_default_registry_duplicate_name_protection() -> None:
    """Re-registering a production factor name raises FactorRegistrationError."""
    registry = build_default_registry()
    existing = registry.get(registry.names()[0])
    with pytest.raises(FactorRegistrationError, match="already registered"):
        registry.register(existing)


def test_build_default_registry_duplicate_column_protection() -> None:
    """Re-registering a claimed produced column raises FactorRegistrationError."""
    registry = build_default_registry()
    existing = registry.get(registry.names()[0])
    column = existing.produced_columns[0]

    from dataclasses import dataclass

    import polars as pl

    from cqros.factors.base import BaseFactor

    @dataclass(frozen=True, slots=True)
    class _DuplicateColumnFactor(BaseFactor):
        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame

    duplicate = _DuplicateColumnFactor(
        name="duplicate_column_probe",
        version="1.0.0",
        description="probe duplicate produced columns",
        category="price",
        required_features=(),
        produced_columns=(column,),
        lookback=0,
    )
    with pytest.raises(FactorRegistrationError, match="produced column already registered"):
        registry.register(duplicate)


def test_build_default_registry_alphabetical_iteration_is_deterministic() -> None:
    """names, list, and metadata remain alphabetically ordered and stable."""
    first = build_default_registry()
    second = build_default_registry()
    assert first.names() == second.names()
    assert first.names() == tuple(sorted(first.names()))
    assert tuple(factor.name for factor in first.list()) == first.names()
    assert tuple(item.name for item in first.metadata()) == first.names()
    assert first.categories() == tuple(sorted(first.categories()))


def test_package_exports_build_default_registry() -> None:
    """build_default_registry is exported from the factors package."""
    import cqros.factors as factors_package

    assert "build_default_registry" in factors_package.__all__
    assert factors_package.build_default_registry is build_default_registry
