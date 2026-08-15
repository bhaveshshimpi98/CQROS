"""CQROS storage backend interfaces.

Purpose:
    Define format-agnostic structural contracts for dataset persistence so
    higher layers depend on interfaces rather than concrete file formats.

Responsibilities:
    - Expose ``IDataStore`` as the shared read/write contract for tabular
      datasets
    - Remain free of I/O, compression, and business logic

Dependencies:
    ``polars`` and ``cqros.core.types``.

Public API:
    ``IDataStore``
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from cqros.core.types import FilePath

__all__ = [
    "IDataStore",
]


class IDataStore(Protocol):
    """Structural contract for tabular dataset storage backends.

    Implementations may persist data as Parquet or other approved formats
    while preserving the same path-oriented public API. Callers should depend
    on this protocol rather than concrete store classes.
    """

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        """Persist a DataFrame to ``path``.

        Args:
            path: Destination filesystem path.
            dataframe: In-memory frame to persist.
        """
        ...

    def read(self, path: FilePath) -> pl.DataFrame:
        """Load a DataFrame from ``path``.

        Args:
            path: Source filesystem path.

        Returns:
            Eagerly loaded Polars DataFrame.
        """
        ...

    def scan(self, path: FilePath) -> pl.LazyFrame:
        """Open a lazy scan over the dataset at ``path``.

        Args:
            path: Source filesystem path.

        Returns:
            Polars LazyFrame bound to the stored dataset.
        """
        ...

    def exists(self, path: FilePath) -> bool:
        """Return whether a dataset file exists at ``path``.

        Args:
            path: Filesystem path to check.

        Returns:
            ``True`` when a regular file exists at ``path``.
        """
        ...

    def delete(self, path: FilePath) -> None:
        """Delete the dataset file at ``path``.

        Args:
            path: Filesystem path to delete.
        """
        ...

    def schema(self, path: FilePath) -> pl.Schema:
        """Return the stored schema for the dataset at ``path``.

        Args:
            path: Source filesystem path.

        Returns:
            Polars schema describing column names and dtypes.
        """
        ...

    def row_count(self, path: FilePath) -> int:
        """Return the number of rows stored at ``path``.

        Args:
            path: Source filesystem path.

        Returns:
            Non-negative row count.
        """
        ...
