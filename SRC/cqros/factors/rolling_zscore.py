"""Shared rolling z-score expression for CQROS research factors.

Purpose:
    Provide one reusable Polars expression for rolling population z-scores so
    every z-score factor handles warmup, positive variance, and zero-variance
    windows identically.

Responsibilities:
    - Compute ``(value - rolling_mean) / rolling_std`` with ``ddof=0``
    - Return null for incomplete warmup windows
    - Return ``0.0`` when rolling standard deviation is exactly zero
    - Remain free of factor metadata, registry, and pipeline concerns

Dependencies:
    ``polars``

Public API:
    ``rolling_zscore_expr``
"""

from __future__ import annotations

import polars as pl

__all__ = ["rolling_zscore_expr"]


def rolling_zscore_expr(
    value: pl.Expr,
    *,
    window_size: int,
    ddof: int = 0,
) -> pl.Expr:
    """Build a rolling z-score Polars expression.

    Args:
        value: Input series expression to standardize.
        window_size: Rolling window length (must be >= 2 at the call site).
        ddof: Delta degrees of freedom for ``rolling_std`` (default ``0``).

    Returns:
        Expression that yields:

        - ``null`` while the rolling window is incomplete (warmup)
        - ``(value - rolling_mean) / rolling_std`` when ``rolling_std > 0``
        - ``0.0`` when the window is complete and ``rolling_std == 0``

    Notes:
        Zero-variance windows are mathematically valid: every observation equals
        the rolling mean, so the standardized deviation is ``0``. Returning
        null incorrectly marks those observations as missing.
    """
    mean = value.rolling_mean(window_size=window_size)  # pyright: ignore[reportUnknownMemberType]
    std = value.rolling_std(  # pyright: ignore[reportUnknownMemberType]
        window_size=window_size,
        ddof=ddof,
    )
    # Chained when: null std (warmup) falls through to otherwise(None).
    # False for std == 0 takes the second branch. True takes the normal ratio.
    return (
        pl.when(std > 0).then((value - mean) / std).when(std == 0).then(0.0).otherwise(None)
    )  # pyright: ignore[reportUnknownMemberType]
