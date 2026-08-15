"""Unit tests for CQROS Factor Research Engine ``BaseFactor``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.interfaces import Factor
from cqros.factors.metadata import FactorMetadata


@dataclass(frozen=True, slots=True)
class _ConcreteFactor(BaseFactor):
    """Minimal concrete factor used only for unit tests."""

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


@dataclass(frozen=True, slots=True)
class _OtherConcreteFactor(BaseFactor):
    """Second concrete type used to verify type-aware equality."""

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


def _factor(**overrides: object) -> _ConcreteFactor:
    """Build a concrete factor with optional field overrides."""
    values: dict[str, object] = {
        "name": "momentum",
        "version": "1.0.0",
        "description": "Cross-sectional momentum",
        "category": "momentum",
        "required_features": ("returns",),
        "produced_columns": ("momentum_score",),
        "lookback": 20,
    }
    values.update(overrides)
    return _ConcreteFactor(**values)  # type: ignore[arg-type]


# --- abstract class ---


def test_cannot_instantiate_abstract_base_factor() -> None:
    """BaseFactor cannot be instantiated without compute."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseFactor"):
        BaseFactor(  # type: ignore[abstract]
            name="momentum",
            version="1.0.0",
            description="momentum",
            category="momentum",
            required_features=("returns",),
            produced_columns=("momentum_score",),
            lookback=20,
        )


def test_base_factor_is_abc() -> None:
    """BaseFactor exposes an abstract compute method."""
    assert getattr(BaseFactor.compute, "__isabstractmethod__", False) is True


def test_concrete_factor_can_be_instantiated() -> None:
    """Concrete subclasses that implement compute can be constructed."""
    factor = _factor()
    assert isinstance(factor, BaseFactor)


# --- immutability ---


def test_base_factor_is_frozen_slotted_dataclass() -> None:
    """BaseFactor is an immutable slotted dataclass."""
    factor = _factor()
    assert is_dataclass(factor)
    assert is_dataclass(BaseFactor)
    with pytest.raises(FrozenInstanceError):
        factor.name = "other"  # type: ignore[misc]


def test_metadata_fields_cannot_be_reassigned() -> None:
    """Metadata attributes cannot be reassigned after construction."""
    factor = _factor()
    with pytest.raises(FrozenInstanceError):
        factor.version = "9.9.9"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        factor.description = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        factor.category = "value"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        factor.required_features = ("rsi",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        factor.produced_columns = ("x",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        factor.lookback = 99  # type: ignore[misc]


def test_sequence_fields_are_frozen_as_tuples() -> None:
    """Sequence inputs are stored as independent immutable tuples."""
    required: list[str] = ["returns", "volume"]
    produced: list[str] = ["momentum_score"]
    factor = _factor(required_features=required, produced_columns=produced)
    assert isinstance(factor.required_features, tuple)
    assert isinstance(factor.produced_columns, tuple)
    assert factor.required_features == ("returns", "volume")
    assert factor.produced_columns == ("momentum_score",)
    required.append("open")
    produced.append("other")
    assert factor.required_features == ("returns", "volume")
    assert factor.produced_columns == ("momentum_score",)


def test_required_features_may_be_empty() -> None:
    """An empty required_features tuple is permitted."""
    factor = _factor(required_features=())
    assert factor.required_features == ()


# --- metadata property ---


def test_metadata_properties_are_exposed() -> None:
    """Constructor arguments are exposed as immutable metadata attributes."""
    factor = _factor(
        name="value",
        version="2.1.0",
        description="Book-to-market proxy",
        category="value",
        required_features=("book_value", "market_cap"),
        produced_columns=("value_score",),
        lookback=5,
    )
    assert factor.name == "value"
    assert factor.version == "2.1.0"
    assert factor.description == "Book-to-market proxy"
    assert factor.category == "value"
    assert factor.required_features == ("book_value", "market_cap")
    assert factor.produced_columns == ("value_score",)
    assert factor.lookback == 5


def test_metadata_property_returns_factor_metadata() -> None:
    """The metadata property returns an immutable FactorMetadata snapshot."""
    factor = _factor()
    meta = factor.metadata
    assert isinstance(meta, FactorMetadata)
    assert meta.name == factor.name
    assert meta.version == factor.version
    assert meta.description == factor.description
    assert meta.category == factor.category
    assert meta.required_features == factor.required_features
    assert meta.produced_columns == factor.produced_columns
    assert meta.lookback == factor.lookback


def test_metadata_property_is_frozen() -> None:
    """FactorMetadata snapshots cannot be mutated."""
    meta = _factor().metadata
    with pytest.raises(FrozenInstanceError):
        meta.name = "other"  # type: ignore[misc]


def test_metadata_property_is_stable_across_calls() -> None:
    """Repeated metadata access yields equal immutable snapshots."""
    factor = _factor()
    left = factor.metadata
    right = factor.metadata
    assert left == right
    assert hash(left) == hash(right)


def test_metadata_reflects_constructor_overrides() -> None:
    """Metadata mirrors non-default constructor values."""
    factor = _factor(
        name="quality",
        version="3.0.0",
        description="Quality composite",
        category="quality",
        required_features=("roe", "leverage"),
        produced_columns=("quality_score", "quality_rank"),
        lookback=60,
    )
    meta = factor.metadata
    assert meta.name == "quality"
    assert meta.version == "3.0.0"
    assert meta.description == "Quality composite"
    assert meta.category == "quality"
    assert meta.required_features == ("roe", "leverage")
    assert meta.produced_columns == ("quality_score", "quality_rank")
    assert meta.lookback == 60


def test_metadata_collections_are_tuples() -> None:
    """FactorMetadata collection fields are immutable tuples."""
    meta = _factor(required_features=["returns"], produced_columns=["score"]).metadata
    assert isinstance(meta.required_features, tuple)
    assert isinstance(meta.produced_columns, tuple)


# --- constructor validation ---


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("version", ""),
        ("version", "   "),
        ("description", ""),
        ("description", "   "),
        ("category", ""),
        ("category", "   "),
    ],
)
def test_constructor_rejects_blank_identity_fields(field_name: str, value: str) -> None:
    """Name, version, description, and category must be non-empty strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a non-empty string"):
        _factor(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", None),
        ("version", 1),
        ("description", 123),
        ("category", ["momentum"]),
    ],
)
def test_constructor_rejects_non_string_identity_fields(
    field_name: str,
    value: object,
) -> None:
    """Identity fields must be strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a non-empty string"):
        _factor(**{field_name: value})


def test_constructor_rejects_empty_produced_columns() -> None:
    """Produced columns must contain at least one entry."""
    with pytest.raises(ValidationError, match="produced_columns must contain at least one entry"):
        _factor(produced_columns=())


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("required_features", "returns"),
        ("produced_columns", "momentum_score"),
        ("required_features", 1),
        ("produced_columns", None),
        ("required_features", {"a"}),
        ("produced_columns", b"score"),
    ],
)
def test_constructor_rejects_non_sequence_collection_fields(
    field_name: str,
    value: object,
) -> None:
    """Feature and column fields must be sequences of strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a sequence of strings"):
        _factor(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("required_features", ("",)),
        ("required_features", ("  ",)),
        ("produced_columns", ("ok", "")),
        ("produced_columns", (" ",)),
        ("required_features", (123,)),
        ("produced_columns", (None,)),
    ],
)
def test_constructor_rejects_invalid_sequence_entries(
    field_name: str,
    value: object,
) -> None:
    """Sequence entries must be non-empty strings."""
    with pytest.raises(ValidationError, match=f"{field_name} entries must be non-empty strings"):
        _factor(**{field_name: value})


@pytest.mark.parametrize("lookback", [-1, -100, True, False, 1.5, "1", None])
def test_constructor_rejects_invalid_lookback(lookback: object) -> None:
    """Lookback must be a non-negative integer (bool excluded)."""
    with pytest.raises(
        ValidationError,
        match="lookback must be an integer greater than or equal to 0",
    ):
        _factor(lookback=lookback)


@pytest.mark.parametrize("lookback", [0, 1, 20, 252])
def test_constructor_accepts_valid_lookback(lookback: int) -> None:
    """Non-negative integer lookback values are accepted."""
    factor = _factor(lookback=lookback)
    assert factor.lookback == lookback


def test_constructor_accepts_list_inputs_for_collections() -> None:
    """List inputs are accepted and frozen into tuples."""
    factor = _factor(
        required_features=["returns", "volatility"],
        produced_columns=["alpha"],
    )
    assert factor.required_features == ("returns", "volatility")
    assert factor.produced_columns == ("alpha",)


def test_validation_error_includes_error_code_for_blank_name() -> None:
    """Blank name raises ValidationError with a stable error code."""
    with pytest.raises(ValidationError) as exc_info:
        _factor(name="")
    assert exc_info.value.error_code == "FACTOR-BASE-001"


def test_validation_error_includes_error_code_for_empty_produced() -> None:
    """Empty produced_columns raises ValidationError with a stable error code."""
    with pytest.raises(ValidationError) as exc_info:
        _factor(produced_columns=())
    assert exc_info.value.error_code == "FACTOR-BASE-006"


# --- equality ---


def test_equality_compares_metadata_for_same_concrete_type() -> None:
    """Equal metadata on the same concrete type yields equality."""
    left = _factor()
    right = _factor()
    assert left == right
    assert left != _factor(name="value")
    assert left != _factor(version="2.0.0")
    assert left != _factor(description="other")
    assert left != _factor(category="value")
    assert left != _factor(required_features=("rsi",))
    assert left != _factor(produced_columns=("other",))
    assert left != _factor(lookback=2)


def test_equality_is_type_aware() -> None:
    """Different concrete factor types are unequal even with identical metadata."""
    left = _factor()
    right = _OtherConcreteFactor(
        name=left.name,
        version=left.version,
        description=left.description,
        category=left.category,
        required_features=left.required_features,
        produced_columns=left.produced_columns,
        lookback=left.lookback,
    )
    assert left != right


def test_equality_with_unrelated_object() -> None:
    """Factors are unequal to unrelated objects."""
    assert _factor() != object()
    assert _factor() != "momentum@1.0.0"
    assert _factor() != {"name": "momentum"}


def test_metadata_equality_matches_factor_fields() -> None:
    """Equal factors produce equal FactorMetadata snapshots."""
    assert _factor().metadata == _factor().metadata
    assert _factor().metadata != _factor(name="value").metadata


# --- hashing ---


def test_hashability_and_set_membership() -> None:
    """Equal factors hash identically and can live in sets and mappings."""
    left = _factor()
    right = _factor()
    different = _factor(name="value")
    assert hash(left) == hash(right)
    assert len({left, right, different}) == 2
    mapping = {left: "a", different: "b"}
    assert mapping[right] == "a"


def test_metadata_is_hashable() -> None:
    """FactorMetadata snapshots are hashable value objects."""
    left = _factor().metadata
    right = _factor().metadata
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_hash_differs_when_metadata_differs() -> None:
    """Factors with different metadata do not collide as equal hashes in sets."""
    factors = {
        _factor(),
        _factor(name="value"),
        _factor(version="2.0.0"),
        _factor(lookback=1),
    }
    assert len(factors) == 4


# --- repr / str ---


def test_repr_includes_all_metadata() -> None:
    """Repr is unambiguous and includes every metadata field."""
    factor = _factor()
    text = repr(factor)
    assert text.startswith("_ConcreteFactor(")
    assert "name='momentum'" in text
    assert "version='1.0.0'" in text
    assert "description='Cross-sectional momentum'" in text
    assert "category='momentum'" in text
    assert "required_features=('returns',)" in text
    assert "produced_columns=('momentum_score',)" in text
    assert "lookback=20" in text
    assert "factor_group='alpha'" in text
    assert "prediction_horizon=1" in text
    assert "enabled=True" in text
    assert "status=" in text


def test_repr_uses_concrete_class_name() -> None:
    """Repr uses the concrete subclass name, not BaseFactor."""
    assert repr(
        _OtherConcreteFactor(
            name="x",
            version="1.0.0",
            description="d",
            category="c",
            required_features=(),
            produced_columns=("y",),
            lookback=0,
        )
    ).startswith("_OtherConcreteFactor(")


def test_str_is_compact_identity() -> None:
    """Str returns name@version."""
    factor = _factor(name="quality", version="3.2.1")
    assert str(factor) == "quality@3.2.1"


def test_str_differs_from_repr() -> None:
    """Str is compact while repr includes full metadata."""
    factor = _factor()
    assert str(factor) == "momentum@1.0.0"
    assert str(factor) != repr(factor)
    assert "required_features" not in str(factor)
    assert "required_features" in repr(factor)


# --- protocol / compute ---


def test_concrete_factor_satisfies_factor_protocol() -> None:
    """Concrete BaseFactor subclasses structurally satisfy Factor."""
    factor = _factor()
    assert isinstance(factor, Factor)
    assert isinstance(factor.required_features, Sequence)
    assert isinstance(factor.produced_columns, Sequence)


def test_compute_is_invoked_on_concrete_factor() -> None:
    """Concrete compute receives the frame and returns a DataFrame."""
    frame = pl.DataFrame({"returns": [0.1, -0.2, 0.05]})
    result = _factor().compute(frame)
    assert result.equals(frame)


def test_base_factor_does_not_validate_dataframe() -> None:
    """Constructor and compute do not inspect or validate dataframe content."""
    frame = pl.DataFrame({"unrelated": [1, 2, 3]})
    factor = _factor()
    assert factor.compute(frame).equals(frame)


def test_slots_prevent_dynamic_attributes() -> None:
    """Slotted factors reject unexpected attribute assignment."""
    factor = _factor()
    # Frozen dataclass + ABC may raise AttributeError or TypeError here.
    with pytest.raises((AttributeError, TypeError)):
        factor.unexpected = "value"  # type: ignore[attr-defined]
