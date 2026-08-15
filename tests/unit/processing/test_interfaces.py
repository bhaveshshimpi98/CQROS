"""Unit tests for CQROS Data Processing Framework protocol conformance."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from cqros.processing.interfaces import ProcessingStep


class _ConformingStep:
    """Minimal ProcessingStep-shaped stub for protocol conformance tests."""

    @property
    def name(self) -> str:
        return "dedupe"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Drop duplicate rows"

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


class _IncompleteStep:
    """Stub missing ``process`` so ProcessingStep conformance must fail."""

    @property
    def name(self) -> str:
        return "incomplete"

    @property
    def version(self) -> str:
        return "0.0.0"

    @property
    def description(self) -> str:
        return "Incomplete"


def test_processing_step_is_runtime_checkable_protocol() -> None:
    """ProcessingStep is a runtime-checkable Protocol."""
    assert isinstance(ProcessingStep, type)
    assert issubclass(ProcessingStep, Protocol)
    assert getattr(ProcessingStep, "_is_runtime_protocol", False) is True


def test_conforming_step_satisfies_processing_step_protocol() -> None:
    """A structurally complete stub satisfies ProcessingStep."""
    step = _ConformingStep()
    assert isinstance(step, ProcessingStep)
    assert step.name == "dedupe"
    assert step.version == "1.0.0"
    assert step.description == "Drop duplicate rows"
    frame = pl.DataFrame({"a": [1, 2]})
    assert step.process(frame).equals(frame)


def test_incomplete_step_does_not_satisfy_protocol() -> None:
    """Missing process prevents ProcessingStep conformance."""
    assert not isinstance(_IncompleteStep(), ProcessingStep)


def test_processing_step_is_runtime_checkable_decorator() -> None:
    """ProcessingStep carries the runtime_checkable marker."""
    assert runtime_checkable is not None
    assert isinstance(_ConformingStep(), ProcessingStep)
