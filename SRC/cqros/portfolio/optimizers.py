"""CQROS Equal Weight portfolio optimizer.

Purpose:
    Provide the baseline ``PortfolioOptimizer`` implementation that converts
    canonical signal datasets into canonical portfolio datasets using
    equal-weight long and short allocation.

Responsibilities:
    - Validate canonical signal DataFrame structure and signal vocabulary
    - Reject duplicate primary keys
    - Allocate equal positive weights across ``BUY`` rows summing to ``+1.0``
    - Allocate equal negative weights across ``SELL`` rows summing to ``-1.0``
    - Assign ``0.0`` weight to every ``HOLD`` row
    - Stamp every output row with ``optimizer = "equal_weight"`` lineage
    - Return newly constructed portfolio DataFrames
    - Remain free of persistence, repositories, CLI, and trading execution

Dependencies:
    ``polars``, ``cqros.portfolio.enums``, ``cqros.portfolio.exceptions``,
    ``cqros.portfolio.interfaces``, ``cqros.portfolio.schema``,
    ``cqros.signals.enums``, and ``cqros.signals.schema``.

Public API:
    ``EqualWeightOptimizer``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.portfolio.enums import OptimizerStrategy
from cqros.portfolio.exceptions import PortfolioValidationError
from cqros.portfolio.interfaces import validate_signals_frame
from cqros.portfolio.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PORTFOLIO_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)
from cqros.signals.enums import Signal
from cqros.signals.enums import values as signal_values
from cqros.signals.schema import REQUIRED_COLUMNS as SIGNAL_REQUIRED_COLUMNS

__all__ = [
    "EqualWeightOptimizer",
]

_ERROR_MISSING_COLUMNS: Final[str] = "PORTFOLIO_MISSING_COLUMNS"
_ERROR_INVALID_SIGNAL: Final[str] = "PORTFOLIO_INVALID_SIGNAL"
_ERROR_DUPLICATE_KEYS: Final[str] = "PORTFOLIO_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)
_SIGNAL_VALUES: Final[tuple[str, ...]] = signal_values()
_OUTPUT_BASE_COLUMNS: Final[tuple[str, ...]] = SIGNAL_REQUIRED_COLUMNS
_EQUAL_WEIGHT_OPTIMIZER: Final[str] = OptimizerStrategy.EQUAL_WEIGHT.value


class EqualWeightOptimizer:
    """Allocate equal long and short weights from discrete trading signals.

    Rules:
        - Each ``BUY`` row receives ``+1.0 / n_buy`` when ``n_buy > 0``
        - Each ``SELL`` row receives ``-1.0 / n_sell`` when ``n_sell > 0``
        - Each ``HOLD`` row receives ``0.0``
        - Positive weights sum to ``+1.0`` when any ``BUY`` exists
        - Negative weights sum to ``-1.0`` when any ``SELL`` exists

    Notes:
        Allocation is computed over the full input frame. Every output row
        carries ``optimizer = "equal_weight"`` as portfolio provenance.
        Implementations must not mutate the caller-supplied DataFrame.
    """

    __slots__ = ()

    def optimize(self, signals: pl.DataFrame) -> pl.DataFrame:
        """Convert a signal frame into a canonical equal-weight portfolio frame.

        Args:
            signals: Canonical signal dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``MERGED_PORTFOLIO_SCHEMA``.

        Raises:
            PortfolioValidationError: If ``signals`` fails structural
                validation, contains invalid signal values, or has duplicate
                primary keys.
        """
        frame = validate_signals_frame(signals)
        _require_signal_columns(frame)
        _require_valid_signal_values(frame)
        _require_unique_primary_keys(frame)
        return _build_portfolio_frame(frame)


def _build_portfolio_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Assemble a canonical portfolio DataFrame with equal-weight targets.

    Args:
        frame: Validated signal DataFrame.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_PORTFOLIO_SCHEMA``.
    """
    buy_count = frame.filter(pl.col("signal") == Signal.BUY).height
    sell_count = frame.filter(pl.col("signal") == Signal.SELL).height
    buy_weight = 1.0 / buy_count if buy_count > 0 else 0.0
    sell_weight = -1.0 / sell_count if sell_count > 0 else 0.0

    weight_expr = (
        pl.when(pl.col("signal") == Signal.BUY)
        .then(pl.lit(buy_weight, dtype=pl.Float64))
        .when(pl.col("signal") == Signal.SELL)
        .then(pl.lit(sell_weight, dtype=pl.Float64))
        .when(pl.col("signal") == Signal.HOLD)
        .then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("target_weight")
    )
    optimizer_expr = pl.lit(_EQUAL_WEIGHT_OPTIMIZER, dtype=pl.Utf8).alias("optimizer")
    assembled = frame.select(
        *[pl.col(column) for column in _OUTPUT_BASE_COLUMNS],
        optimizer_expr,
        weight_expr,
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_PORTFOLIO_SCHEMA)


def _require_signal_columns(frame: pl.DataFrame) -> None:
    """Raise when any required signal-schema column is missing.

    Args:
        frame: Candidate signal DataFrame.

    Raises:
        PortfolioValidationError: If one or more signal columns are absent.
    """
    missing = [column for column in SIGNAL_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PortfolioValidationError(
            "signal frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": SIGNAL_REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_valid_signal_values(frame: pl.DataFrame) -> None:
    """Raise when any signal value is outside the canonical ``Signal`` enum.

    Args:
        frame: Candidate signal DataFrame.

    Raises:
        PortfolioValidationError: If one or more signal values are invalid.
    """
    invalid = (
        frame.filter(~pl.col("signal").is_in(list(_SIGNAL_VALUES)))
        .get_column("signal")
        .unique()
        .to_list()
    )
    if invalid:
        raise PortfolioValidationError(
            "signal frame contains invalid signal values",
            error_code=_ERROR_INVALID_SIGNAL,
            details={
                "invalid_values": tuple(invalid),
                "allowed_values": _SIGNAL_VALUES,
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing primary-key columns.

    Raises:
        PortfolioValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise PortfolioValidationError(
            "signal frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
