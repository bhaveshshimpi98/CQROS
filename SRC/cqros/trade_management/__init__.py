"""CQROS Trade Management Engine package public API."""

from cqros.processing.verification.report import VerificationReport
from cqros.trade_management.exceptions import (
    TradeManagementException,
    TradeManagementValidationError,
)
from cqros.trade_management.manager import (
    ACCOUNTING_INPUT_COLUMNS,
    MARKET_PRICE_INPUT_COLUMNS,
    PORTFOLIO_RISK_INPUT_COLUMNS,
    POSITION_INPUT_COLUMNS,
    SimpleTradeManagementManager,
    TradeManagementManager,
    validate_accounting_frame,
    validate_market_price_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
)
from cqros.trade_management.pipeline import TradeManagementPipeline
from cqros.trade_management.registry import TradeManagementManagerRegistry
from cqros.trade_management.repository import (
    TradeManagementPartitionRef,
    TradeManagementRepository,
)
from cqros.trade_management.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    DEFAULT_INITIAL_RISK_PERCENT,
    DEFAULT_TRAIL_PERCENT,
    MERGED_TRADE_MANAGEMENT_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TRADE_MANAGEMENT_COLUMNS,
    ManagementAction,
    ShutdownReason,
    management_actions,
    shutdown_reasons,
    values,
)
from cqros.trade_management.verifier import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    TradeManagementVerifier,
)

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "DEFAULT_INITIAL_RISK_PERCENT",
    "DEFAULT_TRAIL_PERCENT",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "MARKET_PRICE_INPUT_COLUMNS",
    "MERGED_TRADE_MANAGEMENT_SCHEMA",
    "METADATA_COLUMNS",
    "PORTFOLIO_RISK_INPUT_COLUMNS",
    "POSITION_INPUT_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "TRADE_MANAGEMENT_COLUMNS",
    "ManagementAction",
    "ShutdownReason",
    "SimpleTradeManagementManager",
    "TradeManagementException",
    "TradeManagementManager",
    "TradeManagementManagerRegistry",
    "TradeManagementPartitionRef",
    "TradeManagementPipeline",
    "TradeManagementRepository",
    "TradeManagementValidationError",
    "TradeManagementVerifier",
    "VerificationReport",
    "management_actions",
    "shutdown_reasons",
    "validate_accounting_frame",
    "validate_market_price_frame",
    "validate_portfolio_risk_frame",
    "validate_position_frame",
    "values",
]
