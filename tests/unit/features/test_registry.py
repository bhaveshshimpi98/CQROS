"""Unit tests for CQROS Feature Engine ``FeatureRegistry``."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from cqros.features.base import BaseFeature
from cqros.features.exceptions import (
    DuplicateFeatureError,
    FeatureRegistrationError,
    UnknownFeatureError,
)
from cqros.features.metadata import FeatureMetadata
from cqros.features.registry import FeatureRegistry


@dataclass(frozen=True, slots=True)
class _StubFeature(BaseFeature):
    """Minimal concrete feature used only for registry unit tests."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


@dataclass(frozen=True, slots=True)
class _UncheckedFeature:
    """Feature-shaped stub that allows blank names for registration tests."""

    name: str
    version: str = "1.0.0"
    category: str = "price"
    description: str = "stub"
    required_columns: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("returns",)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


def _feature(
    name: str,
    *,
    version: str = "1.0.0",
    category: str = "price",
    description: str = "stub",
    required_columns: tuple[str, ...] = ("close",),
    produced_columns: tuple[str, ...] | None = None,
    lookback: int = 1,
    dependencies: tuple[str, ...] = (),
) -> _StubFeature:
    """Build a stub feature with the given name and optional overrides."""
    return _StubFeature(
        name=name,
        version=version,
        category=category,
        description=description,
        required_columns=required_columns,
        produced_columns=produced_columns if produced_columns is not None else (name,),
        lookback=lookback,
        dependencies=dependencies,
    )


def test_register_and_get() -> None:
    """register stores a feature that get can retrieve by name."""
    registry = FeatureRegistry()
    feature = _feature("returns")
    registry.register(feature)
    assert registry.get("returns") is feature


def test_register_rejects_blank_names() -> None:
    """Blank feature names are rejected at registration."""
    registry = FeatureRegistry()
    for name in ("", "   "):
        with pytest.raises(FeatureRegistrationError, match="non-blank"):
            registry.register(_UncheckedFeature(name=name))
    assert registry.names() == ()


def test_register_rejects_duplicates() -> None:
    """Duplicate feature names raise DuplicateFeatureError."""
    registry = FeatureRegistry()
    registry.register(_feature("returns"))
    with pytest.raises(DuplicateFeatureError, match="already registered"):
        registry.register(_feature("returns", version="2.0.0"))
    assert registry.get("returns").version == "1.0.0"


def test_register_many_registers_all() -> None:
    """register_many stores every provided feature."""
    registry = FeatureRegistry()
    returns = _feature("returns")
    ema = _feature("ema")
    registry.register_many((returns, ema))
    assert registry.get("returns") is returns
    assert registry.get("ema") is ema


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = FeatureRegistry()
    registry.register(_feature("returns"))
    with pytest.raises(DuplicateFeatureError):
        registry.register_many((_feature("ema"), _feature("returns")))
    assert registry.names() == ("returns",)
    assert not registry.exists("ema")


def test_register_many_is_atomic_on_duplicate_within_batch() -> None:
    """register_many rejects duplicate names within the same batch."""
    registry = FeatureRegistry()
    with pytest.raises(DuplicateFeatureError):
        registry.register_many((_feature("returns"), _feature("returns", version="2.0.0")))
    assert registry.names() == ()


def test_register_many_is_atomic_on_blank_name() -> None:
    """register_many leaves the registry unchanged when a blank name appears."""
    registry = FeatureRegistry()
    with pytest.raises(FeatureRegistrationError):
        registry.register_many((_feature("ema"), _UncheckedFeature(name="")))
    assert registry.names() == ()


def test_get_unknown_raises() -> None:
    """get raises UnknownFeatureError for missing names."""
    registry = FeatureRegistry()
    with pytest.raises(UnknownFeatureError, match="not registered"):
        registry.get("missing")


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = FeatureRegistry()
    assert registry.exists("returns") is False
    registry.register(_feature("returns"))
    assert registry.exists("returns") is True
    assert registry.exists("ema") is False


def test_remove() -> None:
    """remove deletes a registered feature and rejects missing names."""
    registry = FeatureRegistry()
    registry.register(_feature("returns"))
    registry.remove("returns")
    assert registry.exists("returns") is False
    with pytest.raises(UnknownFeatureError):
        registry.remove("returns")


def test_clear() -> None:
    """clear removes all registered features."""
    registry = FeatureRegistry()
    registry.register_many((_feature("returns"), _feature("ema")))
    registry.clear()
    assert registry.names() == ()
    assert registry.list() == ()


def test_list_and_names_are_alphabetical() -> None:
    """list and names return features sorted alphabetically by name."""
    registry = FeatureRegistry()
    registry.register_many((_feature("zeta"), _feature("alpha"), _feature("mu")))
    assert registry.names() == ("alpha", "mu", "zeta")
    assert tuple(feature.name for feature in registry.list()) == ("alpha", "mu", "zeta")


def test_metadata_generation() -> None:
    """metadata projects registered features into FeatureMetadata tuples."""
    registry = FeatureRegistry()
    registry.register_many(
        (
            _feature(
                "returns",
                version="1.2.0",
                category="price",
                description="Close returns",
                required_columns=("close",),
                produced_columns=("returns",),
                lookback=1,
                dependencies=(),
            ),
            _feature(
                "ema",
                version="2.0.0",
                category="trend",
                description="EMA",
                required_columns=("close",),
                produced_columns=("ema",),
                lookback=20,
                dependencies=("returns",),
            ),
        )
    )
    metadata = registry.metadata()
    assert isinstance(metadata, tuple)
    assert len(metadata) == 2
    assert all(isinstance(item, FeatureMetadata) for item in metadata)
    assert metadata[0].name == "ema"
    assert metadata[0].version == "2.0.0"
    assert metadata[0].category == "trend"
    assert metadata[0].description == "EMA"
    assert metadata[0].author == ""
    assert metadata[0].lookback == 20
    assert metadata[0].warmup_rows == 19
    assert metadata[0].required_columns == ("close",)
    assert metadata[0].produced_columns == ("ema",)
    assert metadata[0].dependencies == ("returns",)
    assert metadata[1].name == "returns"
    assert metadata[1].version == "1.2.0"


def test_returned_collections_are_immutable_snapshots() -> None:
    """Returned tuples are snapshots unaffected by later registry mutation."""
    registry = FeatureRegistry()
    registry.register_many((_feature("returns"), _feature("ema")))
    names = registry.names()
    features = registry.list()
    metadata = registry.metadata()
    assert isinstance(names, tuple)
    assert isinstance(features, tuple)
    assert isinstance(metadata, tuple)
    registry.clear()
    assert names == ("ema", "returns")
    assert tuple(feature.name for feature in features) == ("ema", "returns")
    assert tuple(item.name for item in metadata) == ("ema", "returns")
    assert registry.names() == ()


def test_register_does_not_mutate_feature() -> None:
    """Registry stores the feature reference without altering its metadata."""
    registry = FeatureRegistry()
    feature = _feature("returns", version="1.0.0")
    registry.register(feature)
    assert feature.name == "returns"
    assert feature.version == "1.0.0"
    assert registry.get("returns") is feature


def test_package_exports_feature_registry() -> None:
    """FeatureRegistry is exported from the features package."""
    import cqros.features as features_package

    assert "FeatureRegistry" in features_package.__all__
    assert features_package.FeatureRegistry is FeatureRegistry
