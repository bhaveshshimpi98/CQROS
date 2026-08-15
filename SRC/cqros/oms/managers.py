"""CQROS Simple Order Manager.

Purpose:
    Provide the baseline ``OrderManager`` implementation that converts
    canonical risk-decision datasets into canonical OMS order datasets using
    one market order per eligible risk decision.

Responsibilities:
    - Validate canonical risk DataFrame structure
    - Reject duplicate risk primary keys
    - Create one ``PENDING`` market order for every non-``REJECT`` decision
      with a non-zero ``approved_weight``
    - Map positive weights to ``BUY`` and negative weights to ``SELL``
    - Return newly constructed order DataFrames
    - Remain free of persistence, repositories, CLI, exchange APIs, and
      execution

Dependencies:
    ``polars``, ``cqros.oms.enums``, ``cqros.oms.exceptions``,
    ``cqros.oms.interfaces``, ``cqros.oms.schema``, ``cqros.risk.enums``, and
    ``cqros.risk.schema``.

Public API:
    ``SimpleOrderManager``
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final
from uuid import uuid4

import polars as pl

from cqros.oms.enums import OrderSide, OrderStatus, OrderType
from cqros.oms.exceptions import OMSValidationError
from cqros.oms.interfaces import validate_risk_frame
from cqros.oms.schema import CANONICAL_COLUMN_ORDER, MERGED_ORDER_SCHEMA
from cqros.risk.enums import RiskDecision
from cqros.risk.schema import (
    PRIMARY_KEY_COLUMNS as RISK_PRIMARY_KEY_COLUMNS,
)
from cqros.risk.schema import (
    REQUIRED_COLUMNS as RISK_REQUIRED_COLUMNS,
)

__all__ = [
    "SimpleOrderManager",
]

_ERROR_MISSING_COLUMNS: Final[str] = "OMS_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "OMS_DUPLICATE_KEYS"

_RISK_PRIMARY_KEY_LIST: Final[list[str]] = list(RISK_PRIMARY_KEY_COLUMNS)

# Risk Decision columns plus OMS lineage metadata preserved onto each order.
_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    *RISK_REQUIRED_COLUMNS,
    "policy",
    "optimizer",
)


class SimpleOrderManager:
    """Create one market order per eligible risk-decision row.

    Rules:
        - ``REJECT`` rows never produce orders
        - ``APPROVE`` and ``RESIZE`` rows use ``approved_weight``
        - ``approved_weight == 0`` rows are skipped
        - Positive ``approved_weight`` maps to ``BUY``
        - Negative ``approved_weight`` maps to ``SELL``
        - Quantity is the absolute value of ``approved_weight``
        - Orders are ``MARKET`` / ``PENDING`` with null prices and zero fills

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        ``policy`` and ``optimizer`` must already be present on the risk frame
        so OMS lineage metadata can be preserved onto each order row.
    """

    __slots__ = ()

    def create_orders(self, risk_decisions: pl.DataFrame) -> pl.DataFrame:
        """Convert a risk-decision frame into a canonical OMS order frame.

        Args:
            risk_decisions: Canonical risk-decision dataset enriched with
                ``policy`` and ``optimizer`` lineage columns. Must not be
                mutated.

        Returns:
            A new DataFrame matching ``MERGED_ORDER_SCHEMA``. May be empty when
            every input row is ineligible for order creation.

        Raises:
            OMSValidationError: If ``risk_decisions`` fails structural
                validation, is missing required columns, or has duplicate
                primary keys.
        """
        frame = validate_risk_frame(risk_decisions)
        _require_risk_columns(frame)
        _require_unique_primary_keys(frame)
        return _build_order_frame(frame)


def _build_order_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Assemble a canonical OMS order DataFrame from eligible risk rows.

    Args:
        frame: Validated risk-decision DataFrame.

    Returns:
        A new DataFrame ordered and cast to ``MERGED_ORDER_SCHEMA``.
    """
    eligible = frame.filter(
        (pl.col("decision") != RiskDecision.REJECT.value) & (pl.col("approved_weight") != 0.0)
    )
    if eligible.height == 0:
        return pl.DataFrame(schema=MERGED_ORDER_SCHEMA)

    order_ids = [uuid4().hex for _ in range(eligible.height)]
    created_at = datetime.now(UTC)
    assembled = eligible.with_columns(
        pl.Series("order_id", order_ids, dtype=pl.Utf8),
    ).select(
        pl.col("symbol"),
        pl.col("timeframe"),
        pl.col("open_time"),
        pl.col("order_id"),
        pl.col("order_id").alias("parent_order_id"),
        pl.col("model_name"),
        pl.col("model_version"),
        pl.col("policy"),
        pl.col("optimizer"),
        pl.when(pl.col("approved_weight") > 0.0)
        .then(pl.lit(OrderSide.BUY.value))
        .otherwise(pl.lit(OrderSide.SELL.value))
        .alias("side"),
        pl.lit(OrderType.MARKET.value).alias("order_type"),
        pl.col("approved_weight").abs().alias("quantity"),
        pl.lit(None, dtype=pl.Float64).alias("limit_price"),
        pl.lit(None, dtype=pl.Float64).alias("stop_price"),
        pl.lit(0.0, dtype=pl.Float64).alias("filled_quantity"),
        pl.lit(None, dtype=pl.Float64).alias("average_fill_price"),
        pl.lit(OrderStatus.PENDING.value).alias("status"),
        pl.lit(created_at).alias("created_at"),
        pl.lit(created_at).alias("updated_at"),
    )
    return assembled.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_ORDER_SCHEMA)


def _require_risk_columns(frame: pl.DataFrame) -> None:
    """Raise when any required risk or OMS lineage column is missing.

    Args:
        frame: Candidate risk-decision DataFrame.

    Raises:
        OMSValidationError: If one or more required columns are absent.
    """
    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise OMSValidationError(
            "risk frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _REQUIRED_INPUT_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when risk primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing risk primary-key columns.

    Raises:
        OMSValidationError: If any primary-key combination appears more than
            once.
    """
    unique_keys = frame.select(_RISK_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise OMSValidationError(
            "risk frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": RISK_PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
