"""CQROS Data Processing Framework pipeline.

Purpose:
    Orchestrate deterministic sequential execution of processing steps over
    a Polars DataFrame.

Responsibilities:
    - Execute ``ProcessingStep`` instances in declared order
    - Clone the input DataFrame before execution
    - Stop on the first failure
    - Wrap unexpected step failures in ``ProcessingExecutionError``
    - Remain free of registration, storage, persistence, caching, and
      dataset-specific logic

Dependencies:
    ``polars``, ``cqros.processing.exceptions``, and
    ``cqros.processing.interfaces.ProcessingStep``.

Public API:
    ``ProcessingPipeline``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.exceptions import ProcessingError, ProcessingExecutionError
from cqros.processing.interfaces import ProcessingStep

__all__ = ["ProcessingPipeline"]

_ERROR_EXECUTION: Final[str] = "PROCESSING-PIPE-001"


class ProcessingPipeline:
    """Deterministic orchestrator for ordered processing-step execution.

    The pipeline executes each step sequentially, returns a new DataFrame,
    and never mutates the caller-supplied input frame. It does not register
    steps, persist results, or apply dataset-specific cleaning rules.

    Args:
        steps: Ordered processing steps to execute. Stored as an immutable
            tuple snapshot.
    """

    __slots__ = ("_steps",)

    def __init__(self, steps: Sequence[ProcessingStep]) -> None:
        """Initialize the pipeline with an ordered step sequence.

        Args:
            steps: Processing steps to execute in the given order.
        """
        self._steps: tuple[ProcessingStep, ...] = tuple(steps)

    @property
    def steps(self) -> tuple[ProcessingStep, ...]:
        """Return the immutable ordered step sequence."""
        return self._steps

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute all steps against ``frame`` and return the result.

        The original ``frame`` is cloned before any step runs and is never
        mutated. Execution stops at the first failure.

        Args:
            frame: Input market DataFrame.

        Returns:
            A new DataFrame produced by sequential step execution. When no
            steps are configured, returns a clone of ``frame``.

        Raises:
            ProcessingExecutionError: If a step ``process`` raises an unexpected
                non-``ProcessingError`` exception.
            ProcessingError: Propagated unchanged when raised by a step.
        """
        current = frame.clone()
        for step in self._steps:
            current = self._process_step(step, current)
        return current

    def _process_step(self, step: ProcessingStep, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute one processing step and wrap unexpected failures.

        Args:
            step: Processing step to execute.
            frame: Current pipeline DataFrame.

        Returns:
            DataFrame returned by ``step.process``.

        Raises:
            ProcessingError: Propagated unchanged from ``process``.
            ProcessingExecutionError: If ``process`` raises any non-processing
                exception.
        """
        try:
            return step.process(frame)
        except ProcessingError:
            raise
        except Exception as exc:
            raise ProcessingExecutionError(
                f"processing step failed: {step.name}",
                error_code=_ERROR_EXECUTION,
                details={
                    "step": step.name,
                    "version": step.version,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc
