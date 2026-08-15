"""CQROS Reporting package public API."""

from cqros.reporting.engine import (
    ANALYTICS_INPUT_COLUMNS,
    ReportingEngine,
    SimpleReportingEngine,
    validate_analytics_frame,
)
from cqros.reporting.exceptions import ReportingException, ReportingValidationError
from cqros.reporting.factor_orientation_diagnostic import (
    FACTOR_ORIENTATION_SUMMARY_COLUMNS,
    FACTOR_ORIENTATION_SUMMARY_CSV_NAME,
    ORIENTATION_FACTOR_DETAIL_COLUMNS,
    FactorOrientationReporter,
    build_factor_orientation_details,
    build_orientation_summary,
)
from cqros.reporting.factor_validation_report import (
    DEFAULT_OUTPUT_ROOT,
    REPORT_FORMATS,
    SUMMARY_COLUMNS,
    TOP_FACTOR_COUNT,
    FactorValidationReporter,
)
from cqros.reporting.pipeline import ReportingPipeline
from cqros.reporting.registry import ReportingEngineRegistry
from cqros.reporting.repository import ReportingPartitionRef, ReportingRepository
from cqros.reporting.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REPORTING_COLUMNS,
    REPORTING_SCHEMA,
    REQUIRED_COLUMNS,
    ReportingStatus,
    reporting_status_values,
    reporting_statuses,
)
from cqros.reporting.verifier import ReportingVerifier
from cqros.reporting.walk_forward_audit_report import (
    ALL_TIMEFRAMES_CSV_NAME,
    DETAIL_COLUMNS,
    GLOBAL_SUMMARY_CSV_NAME,
    TIMEFRAME_SUMMARY_COLUMNS,
    TIMEFRAME_SUMMARY_CSV_NAME,
    WalkForwardAuditReporter,
    aggregate_partition_frame,
    build_global_summary,
    build_timeframe_summary,
    forbidden_import_violations,
    format_discovery_table,
)

__all__ = [
    "ALL_TIMEFRAMES_CSV_NAME",
    "ANALYTICS_INPUT_COLUMNS",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "DEFAULT_OUTPUT_ROOT",
    "DETAIL_COLUMNS",
    "FACTOR_ORIENTATION_SUMMARY_COLUMNS",
    "FACTOR_ORIENTATION_SUMMARY_CSV_NAME",
    "FactorOrientationReporter",
    "FactorValidationReporter",
    "GLOBAL_SUMMARY_CSV_NAME",
    "ORIENTATION_FACTOR_DETAIL_COLUMNS",
    "PRIMARY_KEY_COLUMNS",
    "REPORTING_COLUMNS",
    "REPORTING_SCHEMA",
    "REPORT_FORMATS",
    "REQUIRED_COLUMNS",
    "ReportingEngine",
    "ReportingEngineRegistry",
    "ReportingException",
    "ReportingPartitionRef",
    "ReportingPipeline",
    "ReportingRepository",
    "ReportingStatus",
    "ReportingValidationError",
    "ReportingVerifier",
    "SUMMARY_COLUMNS",
    "SimpleReportingEngine",
    "TIMEFRAME_SUMMARY_COLUMNS",
    "TIMEFRAME_SUMMARY_CSV_NAME",
    "TOP_FACTOR_COUNT",
    "WalkForwardAuditReporter",
    "aggregate_partition_frame",
    "build_factor_orientation_details",
    "build_global_summary",
    "build_orientation_summary",
    "build_timeframe_summary",
    "forbidden_import_violations",
    "format_discovery_table",
    "reporting_status_values",
    "reporting_statuses",
    "validate_analytics_frame",
]
