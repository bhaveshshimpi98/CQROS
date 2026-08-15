"""Unit tests for CQROS Data Processing Framework ``BaseProcessingStep``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import ValidationError
from cqros.processing.base import BaseProcessingStep
from cqros.processing.interfaces import ProcessingStep
from cqros.processing.metadata import ProcessingMetadata


@dataclass(frozen=True, slots=True)
class _ConcreteStep(BaseProcessingStep):
    """Minimal concrete processing step used only for unit tests."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


@dataclass(frozen=True, slots=True)
class _OtherConcreteStep(BaseProcessingStep):
    """Second concrete type used to verify type-aware equality."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged for abstract-base coverage."""
        return frame


def _step(**overrides: object) -> _ConcreteStep:
    """Build a concrete processing step with optional field overrides."""
    values: dict[str, object] = {
        "name": "dedupe",
        "version": "1.0.0",
        "description": "Drop duplicate rows",
    }
    values.update(overrides)
    return _ConcreteStep(**values)  # type: ignore[arg-type]


def test_cannot_instantiate_abstract_base_processing_step() -> None:
    """BaseProcessingStep cannot be instantiated without process."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseProcessingStep"):
        BaseProcessingStep(  # type: ignore[abstract]
            name="dedupe",
            version="1.0.0",
            description="dedupe",
        )


def test_base_processing_step_is_abc() -> None:
    """BaseProcessingStep exposes an abstract process method."""
    assert getattr(BaseProcessingStep.process, "__isabstractmethod__", False) is True


def test_concrete_step_can_be_instantiated() -> None:
    """Concrete subclasses that implement process can be constructed."""
    step = _step()
    assert isinstance(step, BaseProcessingStep)


def test_base_processing_step_is_frozen_slotted_dataclass() -> None:
    """BaseProcessingStep is an immutable slotted dataclass."""
    step = _step()
    assert is_dataclass(step)
    assert is_dataclass(BaseProcessingStep)
    with pytest.raises(FrozenInstanceError):
        step.name = "other"  # type: ignore[misc]


def test_metadata_properties_are_exposed() -> None:
    """Constructor arguments are exposed as immutable metadata attributes."""
    step = _step(
        name="sort",
        version="2.1.0",
        description="Sort by timestamp",
    )
    assert step.name == "sort"
    assert step.version == "2.1.0"
    assert step.description == "Sort by timestamp"


def test_metadata_property_returns_processing_metadata() -> None:
    """metadata returns an immutable ProcessingMetadata snapshot."""
    step = _step(name="align", version="1.2.0", description="Align timestamps")
    meta = step.metadata
    assert isinstance(meta, ProcessingMetadata)
    assert meta == ProcessingMetadata(
        name="align",
        version="1.2.0",
        description="Align timestamps",
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("version", ""),
        ("version", "   "),
        ("description", ""),
        ("description", "   "),
    ],
)
def test_constructor_rejects_empty_identity_fields(field_name: str, value: str) -> None:
    """Name, version, and description must be non-empty strings."""
    with pytest.raises(ValidationError, match=f"{field_name} must be a non-empty string"):
        _step(**{field_name: value})


def test_equality_compares_metadata_for_same_concrete_type() -> None:
    """Equal metadata on the same concrete type yields equality."""
    left = _step()
    right = _step()
    assert left == right
    assert left != _step(name="sort")
    assert left != _step(version="2.0.0")
    assert left != _step(description="other")


def test_equality_is_type_aware() -> None:
    """Different concrete step types are unequal even with identical metadata."""
    left = _step()
    right = _OtherConcreteStep(
        name=left.name,
        version=left.version,
        description=left.description,
    )
    assert left != right


def test_hashability_and_set_membership() -> None:
    """Equal steps hash identically and can live in sets and mappings."""
    left = _step()
    right = _step()
    different = _step(name="sort")
    assert hash(left) == hash(right)
    assert len({left, right, different}) == 2
    mapping = {left: "a", different: "b"}
    assert mapping[right] == "a"


def test_repr_includes_all_metadata() -> None:
    """Repr is unambiguous and includes every metadata field."""
    step = _step()
    text = repr(step)
    assert text.startswith("_ConcreteStep(")
    assert "name='dedupe'" in text
    assert "version='1.0.0'" in text
    assert "description='Drop duplicate rows'" in text


def test_str_is_compact_identity() -> None:
    """Str returns name@version."""
    step = _step(name="sort", version="3.2.1")
    assert str(step) == "sort@3.2.1"


def test_concrete_step_satisfies_processing_step_protocol() -> None:
    """Concrete BaseProcessingStep subclasses structurally satisfy ProcessingStep."""
    step = _step()
    assert isinstance(step, ProcessingStep)


def test_process_is_invoked_on_concrete_step() -> None:
    """Concrete process receives the frame and returns a DataFrame."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    result = _step().process(frame)
    assert result.equals(frame)


def test_metadata_fields_cannot_be_reassigned() -> None:
    """Metadata attributes cannot be reassigned after construction."""
    step = _step()
    with pytest.raises(FrozenInstanceError):
        step.version = "9.9.9"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        step.description = "other"  # type: ignore[misc]
