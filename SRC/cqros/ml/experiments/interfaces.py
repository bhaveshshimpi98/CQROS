"""CQROS ML Experiment public interfaces.

Purpose:
    Define structural contracts for experiment-tracking orchestration so every
    tracker implementation shares one public surface.

Responsibilities:
    - Expose ``ExperimentTracker`` as the shared experiment-catalog contract
    - Remain free of validation details and concrete storage logic

Dependencies:
    ``cqros.ml.experiments.schema.ExperimentRecord``.

Public API:
    ``ExperimentTracker``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cqros.ml.experiments.schema import ExperimentRecord

__all__ = [
    "ExperimentTracker",
]


@runtime_checkable
class ExperimentTracker(Protocol):
    """Structural contract for framework-independent experiment tracking.

    Implementations store immutable ``ExperimentRecord`` metadata, preserve
    insertion order, and return immutable list snapshots. They must not train,
    evaluate, or persist model binaries.
    """

    def record(self, record: ExperimentRecord) -> None:
        """Store one experiment record by ``experiment_id``.

        Args:
            record: Immutable experiment metadata to store.
        """
        ...

    def get(self, experiment_id: str) -> ExperimentRecord:
        """Return the stored record for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to look up.

        Returns:
            The stored ``ExperimentRecord``.
        """
        ...

    def list(self) -> tuple[ExperimentRecord, ...]:
        """Return stored records in insertion order.

        Returns:
            A new tuple of ``ExperimentRecord`` instances.
        """
        ...

    def delete(self, experiment_id: str) -> None:
        """Remove the stored record for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to remove.
        """
        ...

    def exists(self, experiment_id: str) -> bool:
        """Return whether an experiment is stored under ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to check.

        Returns:
            ``True`` when the identifier is stored, otherwise ``False``.
        """
        ...
