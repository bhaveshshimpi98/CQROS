"""CQROS Execution Engine package public API."""

from cqros.execution.exceptions import ExecutionException, ExecutionValidationError
from cqros.execution.pipeline import ExecutionPipeline
from cqros.execution.registry import ExecutionSimulatorRegistry
from cqros.execution.repository import ExecutionPartitionRef, TradeRepository
from cqros.execution.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_TRADE_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TRADE_COLUMNS,
    ExecutionStatus,
    execution_statuses,
    values,
)
from cqros.execution.simulator import (
    ORDER_INPUT_COLUMNS,
    ExecutionSimulator,
    SimpleExecutionSimulator,
    validate_order_frame,
)
from cqros.execution.verifier import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    ExecutionVerifier,
)
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "ExecutionException",
    "ExecutionPartitionRef",
    "ExecutionPipeline",
    "ExecutionSimulator",
    "ExecutionSimulatorRegistry",
    "ExecutionStatus",
    "ExecutionValidationError",
    "ExecutionVerifier",
    "MERGED_TRADE_SCHEMA",
    "METADATA_COLUMNS",
    "ORDER_INPUT_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "SimpleExecutionSimulator",
    "TRADE_COLUMNS",
    "TradeRepository",
    "VerificationReport",
    "execution_statuses",
    "validate_order_frame",
    "values",
]
