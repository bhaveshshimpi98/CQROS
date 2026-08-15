"""Shared required-feature validation for composite factors.

Purpose:
    Provide a single fail-fast helper that verifies feature columns produced
    by the Feature Engine are present before a composite factor computes.

Responsibilities:
    - Validate that every required feature column exists on a DataFrame
    - Raise ``FactorError`` with stable error codes and context
    - Remain free of formula logic

Dependencies:
    ``polars`` and ``cqros.factors.exceptions.FactorError``.

Public API:
    ``require_feature_columns``
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from cqros.factors.exceptions import FactorError

__all__ = ["require_feature_columns"]


def require_feature_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    factor: str,
    error_code: str,
) -> None:
    """Raise ``FactorError`` when any required feature column is missing.

    Args:
        frame: Input research DataFrame expected to contain feature outputs.
        columns: Feature column names that must be present.
        factor: Factor name recorded in error details.
        error_code: Stable CQROS error code for the missing-column failure.

    Raises:
        FactorError: If any entry in ``columns`` is absent from ``frame``.
    """
    for column in columns:
        if column not in frame.columns:
            raise FactorError(
                f"required feature missing: {column}",
                error_code=error_code,
                details={
                    "factor": factor,
                    "required_feature": column,
                    "available_columns": tuple(frame.columns),
                },
            )
