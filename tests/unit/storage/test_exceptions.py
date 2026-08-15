"""Unit tests for CQROS storage exception hierarchy."""

from __future__ import annotations

from cqros.core.exceptions import CQROSError
from cqros.storage.exceptions import (
    ArtifactExistsError,
    BackupFailureError,
    CompressionError,
    CorruptedDatasetError,
    DatasetNotFoundError,
    StorageError,
    StorageSerializationError,
    VersionConflictError,
)


def test_storage_error_inherits_from_cqros_error() -> None:
    """StorageError is part of the CQROS exception taxonomy."""
    error = StorageError("storage failed", error_code="STORAGE-001")
    assert isinstance(error, CQROSError)
    assert error.error_code == "STORAGE-001"


def test_specialized_storage_exceptions_inherit_from_storage_error() -> None:
    """Specialized storage exceptions remain under StorageError."""
    specialized = (
        DatasetNotFoundError,
        ArtifactExistsError,
        VersionConflictError,
        StorageSerializationError,
        CompressionError,
        BackupFailureError,
        CorruptedDatasetError,
    )
    for error_type in specialized:
        error = error_type("failure", error_code="STORAGE-X")
        assert isinstance(error, StorageError)
        assert isinstance(error, CQROSError)
