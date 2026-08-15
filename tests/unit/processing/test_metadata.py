"""Unit tests for CQROS Data Processing Framework metadata models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from cqros.processing import metadata as metadata_module
from cqros.processing.metadata import ProcessingMetadata


def _processing_metadata(**overrides: object) -> ProcessingMetadata:
    """Build a ProcessingMetadata fixture with optional overrides."""
    values: dict[str, object] = {
        "name": "dedupe",
        "version": "1.0.0",
        "description": "Drop duplicate rows",
    }
    values.update(overrides)
    return ProcessingMetadata(**values)  # type: ignore[arg-type]


def test_processing_metadata_is_frozen_slotted_dataclass() -> None:
    """ProcessingMetadata is an immutable slotted dataclass."""
    assert is_dataclass(ProcessingMetadata)
    assert hasattr(ProcessingMetadata, "__slots__")
    meta = _processing_metadata()
    with pytest.raises(FrozenInstanceError):
        meta.name = "other"  # type: ignore[misc]


def test_processing_metadata_is_exported() -> None:
    """ProcessingMetadata is listed in the module public API."""
    assert ProcessingMetadata.__name__ in metadata_module.__all__
    assert getattr(metadata_module, ProcessingMetadata.__name__) is ProcessingMetadata


def test_processing_metadata_construction() -> None:
    """ProcessingMetadata stores identity and description fields."""
    meta = _processing_metadata(
        name="sort",
        version="2.0.0",
        description="Sort by timestamp",
    )
    assert meta.name == "sort"
    assert meta.version == "2.0.0"
    assert meta.description == "Sort by timestamp"


def test_processing_metadata_equality() -> None:
    """Equal ProcessingMetadata values compare equal; differences do not."""
    left = _processing_metadata()
    right = _processing_metadata()
    assert left == right
    assert left != _processing_metadata(name="sort")
    assert left != _processing_metadata(version="2.0.0")
    assert left != _processing_metadata(description="other")


def test_processing_metadata_hashability() -> None:
    """ProcessingMetadata instances are hashable for set and mapping use."""
    left = _processing_metadata()
    right = _processing_metadata()
    different = _processing_metadata(name="sort")
    assert hash(left) == hash(right)
    assert len({left, right, different}) == 2
    mapping = {left: "a", different: "b"}
    assert mapping[right] == "a"


def test_processing_metadata_repr() -> None:
    """ProcessingMetadata repr includes identifying fields."""
    text = repr(_processing_metadata())
    assert text.startswith("ProcessingMetadata(")
    assert "name='dedupe'" in text
    assert "version='1.0.0'" in text
    assert "description='Drop duplicate rows'" in text


def test_package_exports_metadata_model() -> None:
    """The processing package re-exports ProcessingMetadata."""
    import cqros.processing as processing_package

    assert "ProcessingMetadata" in processing_package.__all__
    assert processing_package.ProcessingMetadata is ProcessingMetadata
