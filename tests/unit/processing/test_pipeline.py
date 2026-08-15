"""Unit tests for CQROS Data Processing Framework ``ProcessingPipeline``."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from cqros.processing.base import BaseProcessingStep
from cqros.processing.exceptions import ProcessingExecutionError, ProcessingValidationError
from cqros.processing.pipeline import ProcessingPipeline

_execution_log: list[str] = []


@dataclass(frozen=True, slots=True)
class _AddColumnStep(BaseProcessingStep):
    """Appends a constant column named after the step."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` with ``name`` column added."""
        series = pl.Series(self.name, [1.0] * frame.height)
        return frame.hstack([series])


@dataclass(frozen=True, slots=True)
class _RecordingStep(BaseProcessingStep):
    """Appends a column and records execution into ``_execution_log``."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Record execution order and append the produced column."""
        _execution_log.append(self.name)
        series = pl.Series(self.name, [float(len(_execution_log))] * frame.height)
        return frame.hstack([series])


@dataclass(frozen=True, slots=True)
class _FailingStep(BaseProcessingStep):
    """Raises a RuntimeError from process."""

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Always fail."""
        raise RuntimeError("boom")


def _step(
    name: str,
    *,
    description: str | None = None,
    cls: type[_AddColumnStep] | type[_RecordingStep] | type[_FailingStep] = _AddColumnStep,
) -> BaseProcessingStep:
    """Build a concrete processing step for pipeline tests."""
    return cls(
        name=name,
        version="1.0.0",
        description=description if description is not None else f"{name} stub",
    )


def _frame() -> pl.DataFrame:
    """Build a minimal input frame."""
    return pl.DataFrame({"close": [1.0, 2.0, 3.0]})


def _reset_execution_log() -> None:
    """Clear the shared execution-order log."""
    _execution_log.clear()


def test_run_single_step() -> None:
    """A single step adds its produced column."""
    pipeline = ProcessingPipeline((_step("dedupe"),))
    result = pipeline.run(_frame())
    assert "dedupe" in result.columns
    assert "close" in result.columns


def test_run_multiple_steps() -> None:
    """Multiple steps are all applied in order."""
    pipeline = ProcessingPipeline((_step("dedupe"), _step("sort")))
    result = pipeline.run(_frame())
    assert "dedupe" in result.columns
    assert "sort" in result.columns


def test_execution_order_is_preserved() -> None:
    """Steps execute in the declared pipeline order."""
    _reset_execution_log()
    pipeline = ProcessingPipeline(
        (
            _step("first", cls=_RecordingStep),
            _step("second", cls=_RecordingStep),
            _step("third", cls=_RecordingStep),
        )
    )
    result = pipeline.run(_frame())
    assert _execution_log == ["first", "second", "third"]
    assert set(result.columns) >= {"close", "first", "second", "third"}


def test_empty_pipeline_returns_clone() -> None:
    """An empty pipeline returns a clone of the input frame."""
    original = _frame()
    pipeline = ProcessingPipeline(())
    result = pipeline.run(original)
    assert result.equals(original)
    assert result is not original


def test_steps_property_is_immutable_tuple() -> None:
    """Pipeline exposes steps as an immutable tuple snapshot."""
    steps = [_step("dedupe"), _step("sort")]
    pipeline = ProcessingPipeline(steps)
    assert isinstance(pipeline.steps, tuple)
    assert tuple(step.name for step in pipeline.steps) == ("dedupe", "sort")
    steps.append(_step("filter"))
    assert tuple(step.name for step in pipeline.steps) == ("dedupe", "sort")


def test_process_exception_wrapping() -> None:
    """Unexpected process errors are wrapped in ProcessingExecutionError."""
    pipeline = ProcessingPipeline((_step("broken", cls=_FailingStep),))
    with pytest.raises(ProcessingExecutionError, match="processing step failed") as exc_info:
        pipeline.run(_frame())
    error = exc_info.value
    assert error.error_code == "PROCESSING-PIPE-001"
    assert error.details["step"] == "broken"
    assert error.details["version"] == "1.0.0"
    assert error.details["exception_type"] == "RuntimeError"
    assert error.details["exception_message"] == "boom"
    assert isinstance(error.__cause__, RuntimeError)


def test_processing_error_is_propagated_unchanged() -> None:
    """ProcessingError subclasses raised by steps are not re-wrapped."""

    @dataclass(frozen=True, slots=True)
    class _ValidationFailingStep(BaseProcessingStep):
        def process(self, frame: pl.DataFrame) -> pl.DataFrame:
            raise ProcessingValidationError(
                "invalid input",
                error_code="PROCESSING-TEST-001",
                details={"step": self.name},
            )

    pipeline = ProcessingPipeline(
        (
            _ValidationFailingStep(
                name="invalid",
                version="1.0.0",
                description="fails validation",
            ),
        )
    )
    with pytest.raises(ProcessingValidationError, match="invalid input") as exc_info:
        pipeline.run(_frame())
    assert exc_info.value.error_code == "PROCESSING-TEST-001"
    assert not isinstance(exc_info.value, ProcessingExecutionError)


def test_pipeline_stops_on_first_failure() -> None:
    """Execution stops at the first failing step."""
    _reset_execution_log()
    pipeline = ProcessingPipeline(
        (
            _step("ok", cls=_RecordingStep),
            _step("broken", cls=_FailingStep),
            _step("never", cls=_RecordingStep),
        )
    )
    with pytest.raises(ProcessingExecutionError):
        pipeline.run(_frame())
    assert _execution_log == ["ok"]


def test_input_dataframe_immutability() -> None:
    """Pipeline execution does not mutate the caller-supplied DataFrame."""
    pipeline = ProcessingPipeline((_step("dedupe"),))
    original = _frame()
    original_columns = list(original.columns)
    original_values = original.get_column("close").to_list()
    result = pipeline.run(original)
    assert list(original.columns) == original_columns
    assert original.get_column("close").to_list() == original_values
    assert "dedupe" not in original.columns
    assert "dedupe" in result.columns
    assert result is not original


def test_package_exports_processing_pipeline() -> None:
    """ProcessingPipeline is exported from the processing package."""
    import cqros.processing as processing_package

    assert processing_package.ProcessingPipeline is ProcessingPipeline
    assert "ProcessingPipeline" in processing_package.__all__
