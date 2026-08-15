"""CQROS Parquet dataset store backed by Polars.

Purpose:
    Persist and retrieve tabular datasets as Parquet files using Polars I/O
    with deterministic defaults suitable for institutional research storage.

Responsibilities:
    - Write DataFrames to Parquet with configurable compression (ZSTD default)
    - Perform atomic writes via temporary files and ``os.replace``
    - Read, scan, inspect schema, and count rows without domain transforms
    - Translate Polars and filesystem I/O failures into storage exceptions

Dependencies:
    ``polars``, ``cqros.core`` constants/types, and ``cqros.storage``
    exceptions/interfaces.

Public API:
    ``ParquetStore`` and ``DEFAULT_PARQUET_COMPRESSION``.

Notes:
    This module contains no validation, feature engineering, or downloader
    logic. ``ParquetStore`` satisfies ``IDataStore`` so alternate formats can
    share the same caller-facing contract.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final, Literal, cast
from uuid import uuid4

import polars as pl
from polars.exceptions import ComputeError, PolarsError

from cqros.core.constants import (
    DEFAULT_DATASET_COMPRESSION,
    SUPPORTED_COMPRESSION_CODECS,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import CompressionCodec, FilePath
from cqros.storage.exceptions import (
    CorruptedDatasetError,
    DatasetNotFoundError,
    StorageError,
    StorageSerializationError,
)

__all__ = [
    "DEFAULT_PARQUET_COMPRESSION",
    "ParquetStore",
]

DEFAULT_PARQUET_COMPRESSION: Final[CompressionCodec] = cast(
    CompressionCodec,
    DEFAULT_DATASET_COMPRESSION,
)

type _ParquetCompression = Literal[
    "lz4",
    "uncompressed",
    "snappy",
    "gzip",
    "brotli",
    "zstd",
]

_POLARS_COMPRESSION_ALIASES: Final[dict[str, _ParquetCompression]] = {
    "none": "uncompressed",
}

_logger = logging.getLogger(__name__)


class ParquetStore:
    """Parquet storage backend implementing the shared ``IDataStore`` contract.

    Writes use ZSTD compression by default and are performed atomically by
    writing to a temporary sibling file and renaming into place. Schema is
    preserved by Polars Parquet serialization without alteration.

    Args:
        compression: Parquet compression codec. Defaults to ZSTD.
        logger: Optional logger instance. Defaults to the module logger.

    Raises:
        ValidationError: If ``compression`` is not a supported codec.
    """

    __slots__ = ("_compression", "_logger")

    _compression: CompressionCodec
    _logger: logging.Logger

    def __init__(
        self,
        *,
        compression: CompressionCodec = DEFAULT_PARQUET_COMPRESSION,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize store configuration.

        Args:
            compression: Parquet compression codec. Defaults to ZSTD.
            logger: Optional logger instance.

        Raises:
            ValidationError: If ``compression`` is not a supported codec.
        """
        if compression not in SUPPORTED_COMPRESSION_CODECS:
            raise ValidationError(
                "compression must be a supported codec",
                error_code="STORAGE-PARQUET-001",
                details={
                    "parameter": "compression",
                    "value": compression,
                    "supported": sorted(SUPPORTED_COMPRESSION_CODECS),
                },
            )

        self._compression = compression
        self._logger = logger if logger is not None else _logger

    @property
    def compression(self) -> CompressionCodec:
        """Return the configured Parquet compression codec."""
        return self._compression

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        """Atomically write a DataFrame to a Parquet file.

        Parent directories are created when missing. The frame is first written
        to a temporary sibling path, then renamed onto ``path`` so readers
        never observe a partial file.

        Args:
            path: Destination Parquet path.
            dataframe: Frame to persist. Schema is preserved as-is.

        Raises:
            StorageSerializationError: If Polars fails to serialize the frame.
            StorageError: If the filesystem write or rename fails.
        """
        target = _as_path(path)
        temporary = _temporary_path(target)
        polars_compression = _to_polars_compression(self._compression)

        self._logger.debug(
            "Writing Parquet dataset",
            extra={
                "path": str(target),
                "rows": dataframe.height,
                "columns": dataframe.width,
                "compression": self._compression,
            },
        )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            dataframe.write_parquet(
                temporary,
                compression=polars_compression,
            )
            os.replace(temporary, target)
        except Exception as exc:
            _cleanup_temporary(temporary)
            raise _translate_error(exc, path=target, operation="write") from exc

        self._logger.info(
            "Wrote Parquet dataset",
            extra={
                "path": str(target),
                "rows": dataframe.height,
                "columns": dataframe.width,
                "compression": self._compression,
            },
        )

    def read(self, path: FilePath) -> pl.DataFrame:
        """Read a Parquet file into an eager DataFrame.

        Args:
            path: Source Parquet path.

        Returns:
            Loaded Polars DataFrame with the stored schema.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            CorruptedDatasetError: If the file cannot be decoded.
            StorageSerializationError: If Polars fails to read the file.
            StorageError: If a filesystem error occurs.
        """
        target = _require_file(path, operation="read")
        self._logger.debug("Reading Parquet dataset", extra={"path": str(target)})

        try:
            frame = pl.read_parquet(target)
        except Exception as exc:
            raise _translate_error(exc, path=target, operation="read") from exc

        self._logger.info(
            "Read Parquet dataset",
            extra={"path": str(target), "rows": frame.height, "columns": frame.width},
        )
        return frame

    def scan(self, path: FilePath) -> pl.LazyFrame:
        """Open a lazy scan over a Parquet file.

        Args:
            path: Source Parquet path.

        Returns:
            Polars LazyFrame bound to the dataset at ``path``.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            CorruptedDatasetError: If the file header cannot be decoded.
            StorageSerializationError: If Polars fails to open the scan.
            StorageError: If a filesystem error occurs.
        """
        target = _require_file(path, operation="scan")
        self._logger.debug("Scanning Parquet dataset", extra={"path": str(target)})

        try:
            return pl.scan_parquet(target)
        except Exception as exc:
            raise _translate_error(exc, path=target, operation="scan") from exc

    def exists(self, path: FilePath) -> bool:
        """Return whether a regular Parquet file exists at ``path``.

        Args:
            path: Filesystem path to check.

        Returns:
            ``True`` when ``path`` refers to an existing regular file.
        """
        return _as_path(path).is_file()

    def delete(self, path: FilePath) -> None:
        """Delete the Parquet file at ``path``.

        Args:
            path: Filesystem path to delete.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            StorageError: If the filesystem delete fails.
        """
        target = _require_file(path, operation="delete")
        self._logger.debug("Deleting Parquet dataset", extra={"path": str(target)})

        try:
            target.unlink()
        except Exception as exc:
            raise _translate_error(exc, path=target, operation="delete") from exc

        self._logger.info("Deleted Parquet dataset", extra={"path": str(target)})

    def schema(self, path: FilePath) -> pl.Schema:
        """Return the Parquet schema without loading row data.

        Args:
            path: Source Parquet path.

        Returns:
            Polars schema preserved from the stored file.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            CorruptedDatasetError: If the schema cannot be decoded.
            StorageSerializationError: If Polars fails to read the schema.
            StorageError: If a filesystem error occurs.
        """
        target = _require_file(path, operation="schema")
        self._logger.debug("Reading Parquet schema", extra={"path": str(target)})

        try:
            return pl.Schema(pl.read_parquet_schema(target))
        except Exception as exc:
            raise _translate_error(exc, path=target, operation="schema") from exc

    def row_count(self, path: FilePath) -> int:
        """Return the number of rows in the Parquet file at ``path``.

        Args:
            path: Source Parquet path.

        Returns:
            Non-negative row count.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            CorruptedDatasetError: If the file cannot be decoded.
            StorageSerializationError: If Polars fails to count rows.
            StorageError: If a filesystem error occurs.
        """
        target = _require_file(path, operation="row_count")
        self._logger.debug("Counting Parquet rows", extra={"path": str(target)})

        try:
            # LazyFrame.select stubs include Unknown parameter aliases under
            # pyright strict; the runtime expression is a single ``pl.len()``.
            count = (
                pl.scan_parquet(target)
                .select(pl.len())  # pyright: ignore[reportUnknownMemberType]
                .collect()
                .item()
            )
        except Exception as exc:
            raise _translate_error(exc, path=target, operation="row_count") from exc

        return int(count)


def _as_path(path: FilePath) -> Path:
    """Normalize a path-like value to ``pathlib.Path``."""
    return Path(path)


def _temporary_path(target: Path) -> Path:
    """Return a unique temporary sibling path for an atomic write."""
    return target.with_name(f".{target.name}.{uuid4().hex}.tmp")


def _cleanup_temporary(temporary: Path) -> None:
    """Best-effort removal of a temporary write artifact."""
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        _logger.warning(
            "Failed to remove temporary Parquet write artifact",
            extra={"path": str(temporary)},
            exc_info=True,
        )


def _require_file(path: FilePath, *, operation: str) -> Path:
    """Return ``path`` when it refers to an existing regular file.

    Args:
        path: Candidate filesystem path.
        operation: Name of the calling operation for error context.

    Returns:
        Normalized path to an existing regular file.

    Raises:
        DatasetNotFoundError: If the path is missing or is not a regular file.
    """
    target = _as_path(path)
    if not target.is_file():
        raise DatasetNotFoundError(
            f"Dataset not found for {operation}",
            error_code="STORAGE-PARQUET-002",
            details={"path": str(target), "operation": operation},
            recovery_suggestion="Verify the storage path and dataset version.",
        )
    return target


def _to_polars_compression(compression: CompressionCodec) -> _ParquetCompression:
    """Map a CQROS compression codec name to a Polars Parquet codec string."""
    aliased = _POLARS_COMPRESSION_ALIASES.get(compression)
    if aliased is not None:
        return aliased
    return cast(_ParquetCompression, compression)


def _translate_error(
    exc: BaseException,
    *,
    path: Path,
    operation: str,
) -> StorageError:
    """Translate filesystem and Polars failures into storage exceptions.

    Args:
        exc: Original exception raised by I/O or Polars.
        path: Path involved in the failed operation.
        operation: Name of the failed store operation.

    Returns:
        A ``StorageError`` (or subclass) suitable for callers.
    """
    if isinstance(exc, StorageError):
        return exc

    details: dict[str, object] = {
        "path": str(path),
        "operation": operation,
        "error_type": type(exc).__name__,
    }

    if isinstance(exc, FileNotFoundError):
        return DatasetNotFoundError(
            f"Dataset not found for {operation}",
            error_code="STORAGE-PARQUET-002",
            details=details,
            recovery_suggestion="Verify the storage path and dataset version.",
        )

    if isinstance(exc, ComputeError):
        return CorruptedDatasetError(
            f"Parquet dataset appears corrupted during {operation}",
            error_code="STORAGE-PARQUET-003",
            details=details,
            recovery_suggestion="Restore the dataset from a verified backup.",
        )

    if isinstance(exc, PolarsError):
        return StorageSerializationError(
            f"Parquet {operation} failed",
            error_code="STORAGE-PARQUET-004",
            details=details,
            recovery_suggestion="Inspect the dataset schema and file integrity.",
        )

    if isinstance(exc, OSError):
        return StorageError(
            f"Filesystem error during Parquet {operation}",
            error_code="STORAGE-PARQUET-005",
            details=details,
            recovery_suggestion="Check filesystem permissions and free space.",
        )

    return StorageSerializationError(
        f"Unexpected failure during Parquet {operation}",
        error_code="STORAGE-PARQUET-006",
        details=details,
        recovery_suggestion="Retry the operation and inspect logs for context.",
    )
