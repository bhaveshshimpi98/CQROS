"""Unit tests for CQROS Data Processing Framework ``ProcessingRegistry``."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from cqros.processing.base import BaseProcessingStep
from cqros.processing.exceptions import (
    DuplicateProcessingStepError,
    ProcessingRegistrationError,
    UnknownProcessingStepError,
)
from cqros.processing.metadata import ProcessingMetadata
from cqros.processing.registry import ProcessingRegistry


@dataclass(frozen=True, slots=True)
class _StubStep(BaseProcessingStep):
    """Minimal concrete processing step used only for registry unit tests."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


@dataclass(frozen=True, slots=True)
class _UncheckedStep:
    """Step-shaped stub that allows blank names for registration tests."""

    name: str
    version: str = "1.0.0"
    description: str = "stub"

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


def _step(
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "stub",
) -> _StubStep:
    """Build a stub processing step with the given name and optional overrides."""
    return _StubStep(
        name=name,
        version=version,
        description=description,
    )


def test_register_and_get() -> None:
    """register stores a step that get can retrieve by name."""
    registry = ProcessingRegistry()
    step = _step("dedupe")
    registry.register(step)
    assert registry.get("dedupe") is step


def test_register_rejects_blank_names() -> None:
    """Blank step names are rejected at registration."""
    registry = ProcessingRegistry()
    for name in ("", "   "):
        with pytest.raises(ProcessingRegistrationError, match="non-blank"):
            registry.register(_UncheckedStep(name=name))
    assert registry.names() == ()


def test_register_rejects_duplicates() -> None:
    """Duplicate step names raise DuplicateProcessingStepError."""
    registry = ProcessingRegistry()
    registry.register(_step("dedupe"))
    with pytest.raises(DuplicateProcessingStepError, match="already registered"):
        registry.register(_step("dedupe", version="2.0.0"))
    assert registry.get("dedupe").version == "1.0.0"


def test_register_many_registers_all() -> None:
    """register_many stores every provided step."""
    registry = ProcessingRegistry()
    dedupe = _step("dedupe")
    sort = _step("sort")
    registry.register_many((dedupe, sort))
    assert registry.get("dedupe") is dedupe
    assert registry.get("sort") is sort


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = ProcessingRegistry()
    registry.register(_step("dedupe"))
    with pytest.raises(DuplicateProcessingStepError):
        registry.register_many((_step("sort"), _step("dedupe")))
    assert registry.names() == ("dedupe",)
    assert not registry.exists("sort")


def test_register_many_is_atomic_on_duplicate_within_batch() -> None:
    """register_many rejects duplicate names within the same batch."""
    registry = ProcessingRegistry()
    with pytest.raises(DuplicateProcessingStepError):
        registry.register_many((_step("dedupe"), _step("dedupe", version="2.0.0")))
    assert registry.names() == ()


def test_register_many_is_atomic_on_blank_name() -> None:
    """register_many leaves the registry unchanged when a blank name appears."""
    registry = ProcessingRegistry()
    with pytest.raises(ProcessingRegistrationError):
        registry.register_many((_step("sort"), _UncheckedStep(name="")))
    assert registry.names() == ()


def test_get_unknown_raises() -> None:
    """get raises UnknownProcessingStepError for missing names."""
    registry = ProcessingRegistry()
    with pytest.raises(UnknownProcessingStepError, match="not registered"):
        registry.get("missing")


def test_exists() -> None:
    """exists reports registration presence without raising."""
    registry = ProcessingRegistry()
    assert registry.exists("dedupe") is False
    registry.register(_step("dedupe"))
    assert registry.exists("dedupe") is True
    assert registry.exists("sort") is False


def test_remove() -> None:
    """remove deletes a registered step and rejects missing names."""
    registry = ProcessingRegistry()
    registry.register(_step("dedupe"))
    registry.remove("dedupe")
    assert registry.exists("dedupe") is False
    with pytest.raises(UnknownProcessingStepError):
        registry.remove("dedupe")


def test_clear() -> None:
    """clear removes all registered steps."""
    registry = ProcessingRegistry()
    registry.register_many((_step("dedupe"), _step("sort")))
    registry.clear()
    assert registry.names() == ()
    assert registry.list() == ()


def test_list_and_names_are_alphabetical() -> None:
    """list and names return steps sorted alphabetically by name."""
    registry = ProcessingRegistry()
    registry.register_many((_step("zeta"), _step("alpha"), _step("mu")))
    assert registry.names() == ("alpha", "mu", "zeta")
    assert tuple(step.name for step in registry.list()) == ("alpha", "mu", "zeta")


def test_metadata_generation() -> None:
    """metadata projects registered steps into ProcessingMetadata tuples."""
    registry = ProcessingRegistry()
    registry.register_many(
        (
            _step(
                "dedupe",
                version="1.2.0",
                description="Drop duplicates",
            ),
            _step(
                "sort",
                version="2.0.0",
                description="Sort rows",
            ),
        )
    )
    metadata = registry.metadata()
    assert isinstance(metadata, tuple)
    assert len(metadata) == 2
    assert all(isinstance(item, ProcessingMetadata) for item in metadata)
    assert metadata[0].name == "dedupe"
    assert metadata[0].version == "1.2.0"
    assert metadata[0].description == "Drop duplicates"
    assert metadata[1].name == "sort"
    assert metadata[1].version == "2.0.0"
    assert metadata[1].description == "Sort rows"


def test_returned_collections_are_immutable_snapshots() -> None:
    """Returned tuples are snapshots unaffected by later registry mutation."""
    registry = ProcessingRegistry()
    registry.register_many((_step("dedupe"), _step("sort")))
    names = registry.names()
    steps = registry.list()
    metadata = registry.metadata()
    assert isinstance(names, tuple)
    assert isinstance(steps, tuple)
    assert isinstance(metadata, tuple)
    registry.clear()
    assert names == ("dedupe", "sort")
    assert tuple(step.name for step in steps) == ("dedupe", "sort")
    assert tuple(item.name for item in metadata) == ("dedupe", "sort")
    assert registry.names() == ()


def test_register_does_not_mutate_step() -> None:
    """Registry stores the step reference without altering its metadata."""
    registry = ProcessingRegistry()
    step = _step("dedupe", version="1.0.0")
    registry.register(step)
    assert step.name == "dedupe"
    assert step.version == "1.0.0"
    assert registry.get("dedupe") is step


def test_package_exports_processing_registry() -> None:
    """ProcessingRegistry is exported from the processing package."""
    import cqros.processing as processing_package

    assert "ProcessingRegistry" in processing_package.__all__
    assert processing_package.ProcessingRegistry is ProcessingRegistry
