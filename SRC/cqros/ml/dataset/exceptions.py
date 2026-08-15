"""CQROS ML Dataset exception hierarchy.

Purpose:
    Provide dataset exception types used by the ML Dataset package.

Responsibilities:
    - Expose ``DatasetSchemaError`` for ML dataset schema contract failures
    - Expose ``DatasetLoaderError`` for ML dataset loading failures
    - Expose ``DatasetSplitterError`` for ML dataset splitting failures
    - Expose ``DatasetStatisticsError`` for ML dataset analysis failures
    - Expose ``DatasetScalerError`` for ML dataset scaling failures
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.DatasetError``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import DatasetError

__all__ = [
    "DatasetError",
    "DatasetLoaderError",
    "DatasetScalerError",
    "DatasetSchemaError",
    "DatasetSplitterError",
    "DatasetStatisticsError",
]


class DatasetSchemaError(DatasetError):
    """Raised when an ML dataset schema contract is missing or invalid."""

    __slots__ = ()


class DatasetLoaderError(DatasetError):
    """Raised when ML dataset loading inputs or assembly fail."""

    __slots__ = ()


class DatasetSplitterError(DatasetError):
    """Raised when ML dataset splitting inputs or ratios are invalid."""

    __slots__ = ()


class DatasetStatisticsError(DatasetError):
    """Raised when ML dataset statistics analysis inputs are invalid."""

    __slots__ = ()


class DatasetScalerError(DatasetError):
    """Raised when ML dataset feature scaling inputs or state are invalid."""

    __slots__ = ()
