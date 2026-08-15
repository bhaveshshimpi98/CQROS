"""CQROS storage-layer exception hierarchy.

Purpose:
    Provide storage-specific exception types used by parquet and future
    storage backends when persistence, retrieval, or serialization fails.

Responsibilities:
    - Define the ``StorageError`` root for all storage failures
    - Expose specialized errors for missing datasets, conflicts, corruption,
      serialization, compression, and backup failures
    - Remain free of I/O and business logic

Dependencies:
    ``cqros.core.exceptions.CQROSError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import CQROSError

__all__ = [
    "StorageError",
    "DatasetNotFoundError",
    "ArtifactExistsError",
    "VersionConflictError",
    "StorageSerializationError",
    "CompressionError",
    "BackupFailureError",
    "CorruptedDatasetError",
]


class StorageError(CQROSError):
    """Raised when a storage-layer operation fails."""

    __slots__ = ()


class DatasetNotFoundError(StorageError):
    """Raised when a required dataset or storage artifact does not exist."""

    __slots__ = ()


class ArtifactExistsError(StorageError):
    """Raised when creating an artifact that already exists is forbidden."""

    __slots__ = ()


class VersionConflictError(StorageError):
    """Raised when a storage version conflict prevents a safe write."""

    __slots__ = ()


class StorageSerializationError(StorageError):
    """Raised when encoding or decoding a storage artifact fails."""

    __slots__ = ()


class CompressionError(StorageError):
    """Raised when compression or decompression of an artifact fails."""

    __slots__ = ()


class BackupFailureError(StorageError):
    """Raised when a storage backup or restore operation fails."""

    __slots__ = ()


class CorruptedDatasetError(StorageError):
    """Raised when a stored dataset is unreadable or fails integrity checks."""

    __slots__ = ()
