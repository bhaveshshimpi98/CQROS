"""Unit tests for CQROS Feature Engine metadata models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cqros.features import metadata as metadata_module
from cqros.features.metadata import (
    FeatureCategory,
    FeatureGroup,
    FeatureManifest,
    FeatureMetadata,
)

_TS = datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC)

_METADATA_TYPES: tuple[type[object], ...] = (
    FeatureMetadata,
    FeatureCategory,
    FeatureGroup,
    FeatureManifest,
)


def _feature_metadata(**overrides: object) -> FeatureMetadata:
    """Build a FeatureMetadata fixture with optional overrides."""
    values: dict[str, object] = {
        "name": "returns",
        "version": "1.0.0",
        "category": "price",
        "description": "Close-to-close returns",
        "author": "research",
        "lookback": 1,
        "tags": ("price", "return"),
        "required_columns": ("close",),
        "produced_columns": ("returns",),
        "dependencies": (),
        "created_at": _TS,
        "deprecated": False,
        "experimental": False,
        "extension_data": {"family": "price"},
    }
    values.update(overrides)
    return FeatureMetadata(**values)  # type: ignore[arg-type]


def _feature_category(**overrides: object) -> FeatureCategory:
    """Build a FeatureCategory fixture with optional overrides."""
    values: dict[str, object] = {
        "name": "momentum",
        "description": "Momentum indicators",
        "display_name": "Momentum",
    }
    values.update(overrides)
    return FeatureCategory(**values)  # type: ignore[arg-type]


def _feature_group(**overrides: object) -> FeatureGroup:
    """Build a FeatureGroup fixture with optional overrides."""
    values: dict[str, object] = {
        "name": "core-price",
        "description": "Core price features",
        "features": ("returns", "log_returns"),
    }
    values.update(overrides)
    return FeatureGroup(**values)  # type: ignore[arg-type]


def _feature_manifest(**overrides: object) -> FeatureManifest:
    """Build a FeatureManifest fixture with optional overrides."""
    values: dict[str, object] = {
        "version": "1.0.0",
        "created_at": _TS,
        "features": (_feature_metadata(),),
    }
    values.update(overrides)
    return FeatureManifest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("metadata_type", _METADATA_TYPES)
def test_metadata_types_are_frozen_slotted_dataclasses(
    metadata_type: type[object],
) -> None:
    """Metadata models are immutable slotted dataclasses."""
    assert is_dataclass(metadata_type)
    assert hasattr(metadata_type, "__slots__")


@pytest.mark.parametrize("metadata_type", _METADATA_TYPES)
def test_metadata_types_are_exported(metadata_type: type[object]) -> None:
    """Each metadata model is listed in the module public API."""
    assert metadata_type.__name__ in metadata_module.__all__
    assert getattr(metadata_module, metadata_type.__name__) is metadata_type


def test_feature_metadata_construction() -> None:
    """FeatureMetadata stores identity, contracts, and lifecycle fields."""
    meta = _feature_metadata(
        name="ema",
        version="2.0.0",
        category="trend",
        description="Exponential moving average",
        author="alice",
        lookback=20,
        tags=("trend",),
        required_columns=("close",),
        produced_columns=("ema_20",),
        dependencies=("returns",),
        deprecated=True,
        experimental=True,
    )
    assert meta.name == "ema"
    assert meta.version == "2.0.0"
    assert meta.category == "trend"
    assert meta.description == "Exponential moving average"
    assert meta.author == "alice"
    assert meta.lookback == 20
    assert meta.tags == ("trend",)
    assert meta.required_columns == ("close",)
    assert meta.produced_columns == ("ema_20",)
    assert meta.dependencies == ("returns",)
    assert meta.created_at == _TS
    assert meta.deprecated is True
    assert meta.experimental is True
    assert isinstance(meta.extension_data, Mapping)
    assert meta.extension_data["family"] == "price"


def test_feature_metadata_defaults() -> None:
    """Optional FeatureMetadata collections and flags use safe defaults."""
    meta = FeatureMetadata(
        name="returns",
        version="1.0.0",
        category="price",
        description="returns",
        author="research",
        lookback=1,
    )
    assert meta.tags == ()
    assert meta.required_columns == ()
    assert meta.produced_columns == ()
    assert meta.dependencies == ()
    assert meta.created_at is None
    assert meta.deprecated is False
    assert meta.experimental is False
    assert meta.extension_data == MappingProxyType({})


def test_feature_metadata_tuple_normalization() -> None:
    """Sequence inputs are stored as independent immutable tuples."""
    tags: list[str] = ["price"]
    required: list[str] = ["close"]
    produced: list[str] = ["returns"]
    dependencies: list[str] = ["base"]
    meta = _feature_metadata(
        tags=tags,
        required_columns=required,
        produced_columns=produced,
        dependencies=dependencies,
    )
    assert isinstance(meta.tags, tuple)
    assert isinstance(meta.required_columns, tuple)
    assert isinstance(meta.produced_columns, tuple)
    assert isinstance(meta.dependencies, tuple)
    assert meta.tags == ("price",)
    assert meta.required_columns == ("close",)
    assert meta.produced_columns == ("returns",)
    assert meta.dependencies == ("base",)
    tags.append("extra")
    required.append("volume")
    produced.append("other")
    dependencies.append("rsi")
    assert meta.tags == ("price",)
    assert meta.required_columns == ("close",)
    assert meta.produced_columns == ("returns",)
    assert meta.dependencies == ("base",)


def test_feature_metadata_extension_data_is_immutable_mapping() -> None:
    """extension_data is exposed as an immutable mapping copy."""
    payload: dict[str, object] = {"family": "price"}
    meta = _feature_metadata(extension_data=payload)
    assert isinstance(meta.extension_data, Mapping)
    assert meta.extension_data["family"] == "price"
    payload["family"] = "mutated"
    assert meta.extension_data["family"] == "price"
    with pytest.raises(TypeError):
        meta.extension_data["family"] = "blocked"  # type: ignore[index]


def test_feature_category_construction() -> None:
    """FeatureCategory stores name, description, and display name."""
    category = _feature_category()
    assert category.name == "momentum"
    assert category.description == "Momentum indicators"
    assert category.display_name == "Momentum"


def test_feature_group_construction_and_tuple_normalization() -> None:
    """FeatureGroup freezes feature names into an immutable tuple."""
    names: list[str] = ["returns", "log_returns"]
    group = _feature_group(features=names)
    assert group.name == "core-price"
    assert group.description == "Core price features"
    assert isinstance(group.features, tuple)
    assert group.features == ("returns", "log_returns")
    names.append("rsi")
    assert group.features == ("returns", "log_returns")


def test_feature_manifest_construction_and_tuple_normalization() -> None:
    """FeatureManifest freezes feature metadata into an immutable tuple."""
    features: list[FeatureMetadata] = [_feature_metadata(), _feature_metadata(name="ema")]
    manifest = _feature_manifest(features=features)
    assert manifest.version == "1.0.0"
    assert manifest.created_at == _TS
    assert isinstance(manifest.features, tuple)
    assert len(manifest.features) == 2
    assert manifest.features[0].name == "returns"
    assert manifest.features[1].name == "ema"
    features.append(_feature_metadata(name="rsi"))
    assert len(manifest.features) == 2


@pytest.mark.parametrize(
    ("factory", "attr", "value"),
    [
        (_feature_metadata, "name", "other"),
        (_feature_metadata, "tags", ("x",)),
        (_feature_metadata, "lookback", 99),
        (_feature_category, "display_name", "Other"),
        (_feature_group, "features", ("x",)),
        (_feature_manifest, "version", "2.0.0"),
        (_feature_manifest, "features", ()),
    ],
)
def test_metadata_instances_are_frozen(
    factory: Callable[..., object],
    attr: str,
    value: object,
) -> None:
    """Metadata instances reject attribute mutation."""
    instance = factory()
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attr, value)


def test_feature_metadata_equality() -> None:
    """Equal FeatureMetadata values compare equal; differences do not."""
    left = _feature_metadata()
    right = _feature_metadata()
    assert left == right
    assert left != _feature_metadata(name="ema")
    assert left != _feature_metadata(version="2.0.0")
    assert left != _feature_metadata(lookback=2)
    assert left != _feature_metadata(tags=("other",))
    assert left != _feature_metadata(extension_data={"family": "other"})


def test_feature_category_equality() -> None:
    """FeatureCategory equality is value-based."""
    assert _feature_category() == _feature_category()
    assert _feature_category() != _feature_category(name="trend")


def test_feature_group_equality() -> None:
    """FeatureGroup equality is value-based."""
    assert _feature_group() == _feature_group()
    assert _feature_group() != _feature_group(features=("returns",))


def test_feature_manifest_equality() -> None:
    """FeatureManifest equality is value-based."""
    assert _feature_manifest() == _feature_manifest()
    assert _feature_manifest() != _feature_manifest(version="2.0.0")
    assert _feature_manifest() != _feature_manifest(features=())


def test_feature_metadata_hashability() -> None:
    """FeatureMetadata instances are hashable for set and mapping use."""
    left = _feature_metadata(extension_data={})
    right = _feature_metadata(extension_data={})
    different = _feature_metadata(name="ema", extension_data={})
    assert hash(left) == hash(right)
    assert len({left, right, different}) == 2
    mapping = {left: "a", different: "b"}
    assert mapping[right] == "a"


def test_feature_category_group_manifest_hashability() -> None:
    """Category, group, and manifest values are hashable."""
    category = _feature_category()
    group = _feature_group()
    manifest = _feature_manifest(features=(_feature_metadata(extension_data={}),))
    assert hash(category) == hash(_feature_category())
    assert hash(group) == hash(_feature_group())
    assert hash(manifest) == hash(
        _feature_manifest(features=(_feature_metadata(extension_data={}),))
    )
    assert len({category, _feature_category(), _feature_category(name="trend")}) == 2


def test_feature_metadata_repr() -> None:
    """FeatureMetadata repr includes key identifying fields."""
    text = repr(_feature_metadata())
    assert text.startswith("FeatureMetadata(")
    assert "name='returns'" in text
    assert "version='1.0.0'" in text
    assert "category='price'" in text
    assert "author='research'" in text
    assert "lookback=1" in text


def test_feature_category_repr() -> None:
    """FeatureCategory repr includes name and display name."""
    text = repr(_feature_category())
    assert text.startswith("FeatureCategory(")
    assert "name='momentum'" in text
    assert "display_name='Momentum'" in text


def test_feature_group_repr() -> None:
    """FeatureGroup repr includes name and features."""
    text = repr(_feature_group())
    assert text.startswith("FeatureGroup(")
    assert "name='core-price'" in text
    assert "features=('returns', 'log_returns')" in text


def test_feature_manifest_repr() -> None:
    """FeatureManifest repr includes version and features collection."""
    text = repr(_feature_manifest())
    assert text.startswith("FeatureManifest(")
    assert "version='1.0.0'" in text
    assert "features=" in text


def test_package_exports_metadata_models() -> None:
    """The features package re-exports metadata models."""
    import cqros.features as features_package

    for name in (
        "FeatureMetadata",
        "FeatureCategory",
        "FeatureGroup",
        "FeatureManifest",
    ):
        assert name in features_package.__all__
        assert getattr(features_package, name).__name__ == name
