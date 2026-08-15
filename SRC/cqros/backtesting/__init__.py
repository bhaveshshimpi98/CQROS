"""CQROS Backtesting Engine package public API."""

from cqros.backtesting.engine import (
    ACCOUNTING_INPUT_COLUMNS,
    EXIT_ENGINE_INPUT_COLUMNS,
    POSITION_INPUT_COLUMNS,
    BacktestingEngine,
    SimpleBacktestingEngine,
    validate_accounting_frame,
    validate_exit_engine_frame,
    validate_position_frame,
)
from cqros.backtesting.exceptions import (
    BacktestingException,
    BacktestingValidationError,
)
from cqros.backtesting.pipeline import BacktestingPipeline
from cqros.backtesting.registry import BacktestingRegistry
from cqros.backtesting.repository import (
    BacktestingPartitionRef,
    BacktestingRepository,
)
from cqros.backtesting.schema import (
    BACKTESTING_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_BACKTESTING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    BacktestingStatus,
    backtesting_statuses,
    values,
)
from cqros.backtesting.verifier import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    BacktestingVerifier,
)
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ACCOUNTING_INPUT_COLUMNS",
    "BACKTESTING_COLUMNS",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "EXIT_ENGINE_INPUT_COLUMNS",
    "MERGED_BACKTESTING_SCHEMA",
    "POSITION_INPUT_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "BacktestingEngine",
    "BacktestingException",
    "BacktestingPartitionRef",
    "BacktestingPipeline",
    "BacktestingRegistry",
    "BacktestingRepository",
    "BacktestingStatus",
    "BacktestingValidationError",
    "BacktestingVerifier",
    "SimpleBacktestingEngine",
    "VerificationReport",
    "backtesting_statuses",
    "validate_accounting_frame",
    "validate_exit_engine_frame",
    "validate_position_frame",
    "values",
]
