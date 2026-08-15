"""CQROS ML ExperimentTracker orchestration.

Purpose:
    Provide an in-memory catalog of immutable experiment metadata for CQROS
    ML workflows without coupling to training, evaluation, or persistence
    backends.

Responsibilities:
    - Record ``ExperimentRecord`` instances by unique ``experiment_id``
    - Provide deterministic lookup, listing, existence checks, and deletion
    - Reject duplicates and invalid records
    - Preserve insertion order and return immutable list snapshots
    - Remain free of training, evaluation, model I/O, and external trackers

Dependencies:
    ``cqros.ml.experiments.exceptions`` and ``cqros.ml.experiments.schema``.

Public API:
    ``ExperimentTracker``

Notes:
    This tracker is not thread-safe. Callers that share one instance across
    threads must provide their own synchronization.
"""

from __future__ import annotations

from typing import Final

from cqros.ml.experiments.exceptions import ModelValidationError
from cqros.ml.experiments.schema import ExperimentRecord

__all__ = ["ExperimentTracker"]

_ERROR_RECORD_TYPE: Final[str] = "ML-EXP-TRACK-001"
_ERROR_ID_EMPTY: Final[str] = "ML-EXP-TRACK-002"
_ERROR_DUPLICATE: Final[str] = "ML-EXP-TRACK-003"
_ERROR_UNKNOWN: Final[str] = "ML-EXP-TRACK-004"


class ExperimentTracker:
    """In-memory catalog of immutable CQROS ML experiment records.

    Experiments are indexed by ``experiment_id``. The tracker stores references
    to immutable ``ExperimentRecord`` instances and never mutates them.
    Returned collections are new tuples and do not expose the internal
    mapping. Insertion order is preserved.

    Notes:
        This tracker is not thread-safe. Concurrent mutation from multiple
        threads requires external synchronization.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        """Initialize an empty experiment tracker."""
        self._records: dict[str, ExperimentRecord] = {}

    def record(self, record: ExperimentRecord) -> None:
        """Store one experiment record by ``experiment_id``.

        Args:
            record: Immutable experiment metadata to store.

        Raises:
            ModelValidationError: If ``record`` is invalid or an experiment
                with the same ID already exists.
        """
        validated = _require_record(record)
        experiment_id = validated.experiment_id
        if experiment_id in self._records:
            raise ModelValidationError(
                f"experiment already recorded: {experiment_id}",
                error_code=_ERROR_DUPLICATE,
                details={"experiment_id": experiment_id},
            )
        self._records[experiment_id] = validated

    def get(self, experiment_id: str) -> ExperimentRecord:
        """Return the stored record for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to look up.

        Returns:
            The stored ``ExperimentRecord``.

        Raises:
            ModelValidationError: If ``experiment_id`` is empty or unknown.
        """
        key = _require_experiment_id(experiment_id)
        record = self._records.get(key)
        if record is None:
            raise ModelValidationError(
                f"experiment not recorded: {key}",
                error_code=_ERROR_UNKNOWN,
                details={"experiment_id": key},
            )
        return record

    def list(self) -> tuple[ExperimentRecord, ...]:
        """Return stored records in insertion order.

        Returns:
            A new tuple of ``ExperimentRecord`` instances.
        """
        return tuple(self._records.values())

    def delete(self, experiment_id: str) -> None:
        """Remove the stored record for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to remove.

        Raises:
            ModelValidationError: If ``experiment_id`` is empty or unknown.
        """
        key = _require_experiment_id(experiment_id)
        if key not in self._records:
            raise ModelValidationError(
                f"experiment not recorded: {key}",
                error_code=_ERROR_UNKNOWN,
                details={"experiment_id": key},
            )
        del self._records[key]

    def exists(self, experiment_id: str) -> bool:
        """Return whether an experiment is stored under ``experiment_id``.

        Args:
            experiment_id: Experiment identifier to check.

        Returns:
            ``True`` when the identifier is stored, otherwise ``False``.

        Raises:
            ModelValidationError: If ``experiment_id`` is empty.
        """
        key = _require_experiment_id(experiment_id)
        return key in self._records


def _require_record(record: object) -> ExperimentRecord:
    """Validate that ``record`` is an ``ExperimentRecord`` instance."""
    if not isinstance(record, ExperimentRecord):
        raise ModelValidationError(
            "record must be an ExperimentRecord instance",
            error_code=_ERROR_RECORD_TYPE,
            details={
                "parameter": "record",
                "value_type": type(record).__name__,
            },
        )
    return record


def _require_experiment_id(experiment_id: object) -> str:
    """Validate that ``experiment_id`` is a non-empty string."""
    if not isinstance(experiment_id, str) or experiment_id.strip() == "":
        raise ModelValidationError(
            "experiment_id must be a non-empty string",
            error_code=_ERROR_ID_EMPTY,
            details={"parameter": "experiment_id", "value": experiment_id},
        )
    return experiment_id
