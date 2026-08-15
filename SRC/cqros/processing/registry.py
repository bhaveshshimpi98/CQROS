"""CQROS Data Processing Framework registry.

Purpose:
    Provide the authoritative in-memory catalog of available processing
    steps for registration and lookup.

Responsibilities:
    - Register immutable ``ProcessingStep`` instances by unique name
    - Provide deterministic lookup, listing, and metadata projection
    - Reject duplicate and blank step names
    - Remain free of execution, storage, pipeline, and dataframe logic

Dependencies:
    ``cqros.processing.exceptions``,
    ``cqros.processing.interfaces.ProcessingStep``, and
    ``cqros.processing.metadata.ProcessingMetadata``.

Public API:
    ``ProcessingRegistry``

Notes:
    This registry is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from cqros.processing.exceptions import (
    DuplicateProcessingStepError,
    ProcessingRegistrationError,
    UnknownProcessingStepError,
)
from cqros.processing.interfaces import ProcessingStep
from cqros.processing.metadata import ProcessingMetadata

__all__ = ["ProcessingRegistry"]

_ERROR_NAME_BLANK: Final[str] = "PROCESSING-REG-001"
_ERROR_DUPLICATE: Final[str] = "PROCESSING-REG-002"
_ERROR_UNKNOWN: Final[str] = "PROCESSING-REG-003"


class ProcessingRegistry:
    """Authoritative catalog of registered CQROS processing steps.

    Steps are indexed by name. The registry stores references to the
    supplied ``ProcessingStep`` instances and never mutates them. Returned
    collections are new tuples and do not expose the internal mapping.

    Notes:
        This registry is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_steps",)

    def __init__(self) -> None:
        """Initialize an empty processing-step registry."""
        self._steps: dict[str, ProcessingStep] = {}

    def register(self, step: ProcessingStep) -> None:
        """Register one processing step by name.

        Args:
            step: Processing step to register. Must not be mutated by the
                registry after registration.

        Raises:
            ProcessingRegistrationError: If ``step.name`` is blank.
            DuplicateProcessingStepError: If a step with the same name exists.
        """
        name = _require_step_name(step.name)
        if name in self._steps:
            raise DuplicateProcessingStepError(
                f"processing step already registered: {name}",
                error_code=_ERROR_DUPLICATE,
                details={"name": name},
            )
        self._steps[name] = step

    def register_many(self, steps: Iterable[ProcessingStep]) -> None:
        """Register multiple processing steps atomically.

        Either every step in ``steps`` is registered, or the registry
        remains unchanged when any registration would fail.

        Args:
            steps: Processing steps to register.

        Raises:
            ProcessingRegistrationError: If any step name is blank.
            DuplicateProcessingStepError: If any name is already registered or
                duplicated within ``steps``.
        """
        pending: dict[str, ProcessingStep] = {}
        for step in steps:
            name = _require_step_name(step.name)
            if name in self._steps or name in pending:
                raise DuplicateProcessingStepError(
                    f"processing step already registered: {name}",
                    error_code=_ERROR_DUPLICATE,
                    details={"name": name},
                )
            pending[name] = step
        self._steps.update(pending)

    def get(self, name: str) -> ProcessingStep:
        """Return the registered processing step for ``name``.

        Args:
            name: Processing-step name to look up.

        Returns:
            The registered processing-step instance.

        Raises:
            UnknownProcessingStepError: If no step is registered under ``name``.
        """
        step = self._steps.get(name)
        if step is None:
            raise UnknownProcessingStepError(
                f"processing step not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        return step

    def exists(self, name: str) -> bool:
        """Return whether a processing step is registered under ``name``.

        Args:
            name: Processing-step name to check.

        Returns:
            ``True`` when the name is registered, otherwise ``False``.
        """
        return name in self._steps

    def remove(self, name: str) -> None:
        """Remove a registered processing step by name.

        Args:
            name: Processing-step name to remove.

        Raises:
            UnknownProcessingStepError: If no step is registered under ``name``.
        """
        if name not in self._steps:
            raise UnknownProcessingStepError(
                f"processing step not registered: {name}",
                error_code=_ERROR_UNKNOWN,
                details={"name": name},
            )
        del self._steps[name]

    def clear(self) -> None:
        """Remove all registered processing steps."""
        self._steps.clear()

    def names(self) -> tuple[str, ...]:
        """Return registered processing-step names in alphabetical order.

        Returns:
            A new tuple of processing-step names.
        """
        return tuple(sorted(self._steps))

    def list(self) -> tuple[ProcessingStep, ...]:
        """Return registered processing steps sorted alphabetically by name.

        Returns:
            A new tuple of registered processing-step instances.
        """
        return tuple(self._steps[name] for name in sorted(self._steps))

    def metadata(self) -> tuple[ProcessingMetadata, ...]:
        """Return metadata snapshots for all registered processing steps.

        Metadata is projected from each step's public attributes.

        Returns:
            A new tuple of ``ProcessingMetadata`` objects sorted alphabetically
            by step name.
        """
        return tuple(_to_processing_metadata(step) for step in self.list())


def _require_step_name(name: object) -> str:
    """Validate and return a non-blank processing-step name.

    Args:
        name: Candidate processing-step name.

    Returns:
        The validated processing-step name.

    Raises:
        ProcessingRegistrationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ProcessingRegistrationError(
            "processing step name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"name": name},
        )
    return name


def _to_processing_metadata(step: ProcessingStep) -> ProcessingMetadata:
    """Project a registered processing step into immutable metadata."""
    return ProcessingMetadata(
        name=step.name,
        version=step.version,
        description=step.description,
    )
