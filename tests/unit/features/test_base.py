"""Unit tests for CQROS Feature Engine ``BaseFeature``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.interfaces import Feature


@dataclass(frozen=True, slots=True)
class _ConcreteFeature(BaseFeature):
    """Minimal concrete feature used only for unit tests."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


@dataclass(frozen=True, slots=True)
class _OtherConcreteFeature(BaseFeature):
    """Second concrete type used to verify type-aware equality."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


def _feature(**overrides: object) -> _ConcreteFeature:
    """Build a concrete feature with optional field overrides."""
    values: dict[str, object] = {
        "name": "returns",
        "version": "1.0.0",
        "category": "price",
        "description": "Simple close-to-close returns",
        "required_columns": ("close",),
        "produced_columns": ("returns",),
        "lookback": 1,
        "dependencies": (),
    }
    values.update(overrides)
    return _ConcreteFeature(**values)  # type: ignore[arg-type]


def test_base_feature_is_frozen_slotted_dataclass() -> None:
    """BaseFeature is an immutable slotted dataclass."""
    feature = _feature()
    assert is_dataclass(feature)
    assert is_dataclass(BaseFeature)
    with pytest.raises(FrozenInstanceError):
        feature.name = "other"  # type: ignore[misc]


def test_metadata_properties_are_exposed() -> None:
    """Constructor arguments are exposed as immutable metadata attributes."""
    feature = _feature(
        name="ema",
        version="2.1.0",
        category="trend",
        description="Exponential moving average",
        required_columns=("close",),
        produced_columns=("ema_20",),
        lookback=20,
        dependencies=("returns",),
    )
    assert feature.name == "ema"
    assert feature.version == "2.1.0"
    assert feature.category == "trend"
    assert feature.description == "Exponential moving average"
    assert feature.required_columns == ("close",)
    assert feature.produced_columns == ("ema_20",)
    assert feature.lookback == 20
    assert feature.warmup_rows == 19
    assert feature.dependencies == ("returns",)


def test_default_warmup_rows_follows_lookback_minus_one() -> None:
    """BaseFeature warmup_rows defaults to max(0, lookback - 1)."""
    assert _feature(lookback=0).warmup_rows == 0
    assert _feature(lookback=1).warmup_rows == 0
    assert _feature(lookback=20).warmup_rows == 19


def test_overridden_warmup_rows_on_shift_features() -> None:
    """Momentum and change features report shift warm-up equal to lookback."""
    from cqros.features.funding.funding_momentum import FundingMomentumFeature
    from cqros.features.long_short.ratio_momentum import RatioMomentumFeature
    from cqros.features.open_interest.oi_momentum import OIMomentumFeature
    from cqros.features.price.returns import ReturnsFeature

    assert FundingMomentumFeature().warmup_rows == 20
    assert OIMomentumFeature().warmup_rows == 20
    assert RatioMomentumFeature().warmup_rows == 20
    assert ReturnsFeature().warmup_rows == 1
    assert FundingMomentumFeature(lookback=5).warmup_rows == 5


def test_sequence_fields_are_frozen_as_tuples() -> None:
    """Sequence inputs are stored as independent immutable tuples."""
    required: list[str] = ["close", "volume"]
    produced: list[str] = ["feature_a"]
    dependencies: list[str] = ["returns"]
    feature = _feature(
        required_columns=required,
        produced_columns=produced,
        dependencies=dependencies,
    )
    assert isinstance(feature.required_columns, tuple)
    assert isinstance(feature.produced_columns, tuple)
    assert isinstance(feature.dependencies, tuple)
    assert feature.required_columns == ("close", "volume")
    assert feature.produced_columns == ("feature_a",)
    assert feature.dependencies == ("returns",)
    required.append("open")
    produced.append("feature_b")
    dependencies.append("rsi")
    assert feature.required_columns == ("close", "volume")
    assert feature.produced_columns == ("feature_a",)
    assert feature.dependencies == ("returns",)


def test_dependencies_default_to_empty_tuple() -> None:
    """Dependencies default to an empty immutable tuple."""
    feature = _ConcreteFeature(
        name="returns",
        version="1.0.0",
        category="price",
        description="returns",
        required_columns=("close",),
        produced_columns=("returns",),
        lookback=1,
    )
    assert feature.dependencies == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("version", ""),
        ("version", "   "),
        ("category", ""),
        ("category", "   "),
    ],
)
def test_constructor_rejects_empty_identity_fields(field_name: str, value: str) -> None:
    """Name, version, and category must be non-empty strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a non-empty string"):
        _feature(**{field_name: value})


def test_constructor_rejects_non_string_description() -> None:
    """Description must be a string."""
    with pytest.raises(ValidationError, match="description must be a string"):
        _feature(description=123)  # type: ignore[arg-type]


def test_constructor_allows_empty_description() -> None:
    """Empty description is permitted."""
    feature = _feature(description="")
    assert feature.description == ""


def test_constructor_rejects_empty_produced_columns() -> None:
    """Produced columns must contain at least one entry."""
    with pytest.raises(ValidationError, match="produced_columns must contain at least one entry"):
        _feature(produced_columns=())


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("required_columns", "close"),
        ("produced_columns", "returns"),
        ("dependencies", "returns"),
        ("required_columns", 1),
        ("produced_columns", None),
        ("dependencies", {"a"}),
    ],
)
def test_constructor_rejects_non_sequence_column_fields(
    field_name: str,
    value: object,
) -> None:
    """Column and dependency fields must be sequences of strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a sequence of strings"):
        _feature(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("required_columns", ("",)),
        ("required_columns", ("  ",)),
        ("produced_columns", ("ok", "")),
        ("dependencies", ("returns", " ")),
        ("produced_columns", (123,)),
    ],
)
def test_constructor_rejects_invalid_sequence_entries(
    field_name: str,
    value: object,
) -> None:
    """Sequence entries must be non-empty strings."""
    with pytest.raises(ValidationError, match=f"{field_name} entries must be non-empty strings"):
        _feature(**{field_name: value})


@pytest.mark.parametrize("lookback", [-1, True, False, 1.5, "1"])
def test_constructor_rejects_invalid_lookback(lookback: object) -> None:
    """Lookback must be a non-negative integer (bool excluded)."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 0",
    ):
        _feature(lookback=lookback)


@pytest.mark.parametrize("lookback", [0, 1, 20])
def test_constructor_accepts_valid_lookback(lookback: int) -> None:
    """Non-negative integer lookback values are accepted."""
    feature = _feature(lookback=lookback)
    assert feature.lookback == lookback


def test_equality_compares_metadata_for_same_concrete_type() -> None:
    """Equal metadata on the same concrete type yields equality."""
    left = _feature()
    right = _feature()
    assert left == right
    assert left != _feature(name="ema")
    assert left != _feature(version="2.0.0")
    assert left != _feature(category="momentum")
    assert left != _feature(description="other")
    assert left != _feature(required_columns=("open",))
    assert left != _feature(produced_columns=("other",))
    assert left != _feature(lookback=2)
    assert left != _feature(dependencies=("rsi",))


def test_equality_is_type_aware() -> None:
    """Different concrete feature types are unequal even with identical metadata."""
    left = _feature()
    right = _OtherConcreteFeature(
        name=left.name,
        version=left.version,
        category=left.category,
        description=left.description,
        required_columns=left.required_columns,
        produced_columns=left.produced_columns,
        lookback=left.lookback,
        dependencies=left.dependencies,
    )
    assert left != right


def test_hashability_and_set_membership() -> None:
    """Equal features hash identically and can live in sets and mappings."""
    left = _feature()
    right = _feature()
    different = _feature(name="ema")
    assert hash(left) == hash(right)
    assert len({left, right, different}) == 2
    mapping = {left: "a", different: "b"}
    assert mapping[right] == "a"


def test_repr_includes_all_metadata() -> None:
    """Repr is unambiguous and includes every metadata field."""
    feature = _feature()
    text = repr(feature)
    assert text.startswith("_ConcreteFeature(")
    assert "name='returns'" in text
    assert "version='1.0.0'" in text
    assert "category='price'" in text
    assert "description='Simple close-to-close returns'" in text
    assert "required_columns=('close',)" in text
    assert "produced_columns=('returns',)" in text
    assert "lookback=1" in text
    assert "dependencies=()" in text


def test_str_is_compact_identity() -> None:
    """Str returns name@version."""
    feature = _feature(name="rsi", version="3.2.1")
    assert str(feature) == "rsi@3.2.1"


def test_cannot_instantiate_abstract_base_feature() -> None:
    """BaseFeature cannot be instantiated without transform."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseFeature"):
        BaseFeature(  # type: ignore[abstract]
            name="returns",
            version="1.0.0",
            category="price",
            description="returns",
            required_columns=("close",),
            produced_columns=("returns",),
            lookback=1,
        )


def test_concrete_feature_satisfies_feature_protocol() -> None:
    """Concrete BaseFeature subclasses structurally satisfy Feature."""
    feature = _feature()
    assert isinstance(feature, Feature)
    assert isinstance(feature.required_columns, Sequence)
    assert isinstance(feature.produced_columns, Sequence)
    assert isinstance(feature.dependencies, Sequence)


def test_transform_is_invoked_on_concrete_feature() -> None:
    """Concrete transform receives the frame and returns a DataFrame."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    result = _feature().transform(frame)
    assert result.equals(frame)


def test_sequence_fields_are_immutable_at_attribute_level() -> None:
    """Metadata attributes cannot be reassigned after construction."""
    feature = _feature()
    with pytest.raises(FrozenInstanceError):
        feature.required_columns = ("open",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        feature.produced_columns = ("x",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        feature.dependencies = ("y",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        feature.lookback = 99  # type: ignore[misc]
