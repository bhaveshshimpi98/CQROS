"""CQROS Factor Research Engine package public API."""

from cqros.factors.base import BaseFactor
from cqros.factors.composite import (
    BreakoutConfirmationFactor,
    CrowdingFactor,
    FlowConfirmationFactor,
    FundingDivergenceFactor,
    LeveragedLongBuildUpFactor,
    LeveragedShortBuildUpFactor,
    LongSqueezeFactor,
    PositionBuildUpFactor,
    ShortSqueezeFactor,
    TrendConfirmationFactor,
)
from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import (
    FactorError,
    FactorExecutionError,
    FactorRegistrationError,
    FactorValidationError,
    UnknownFactorError,
)
from cqros.factors.executable_catalog import ExecutableFactorCatalog
from cqros.factors.generation_pipeline import (
    FactorGenerationPipeline,
    FactorGenerationStatistics,
)
from cqros.factors.input_partition import (
    FactorInputPartition,
    classify_dependency_class,
    required_companion_columns,
    required_datasets,
)
from cqros.factors.interfaces import Factor, FactorValidator
from cqros.factors.metadata import FactorMetadata
from cqros.factors.pipeline import FactorPipeline
from cqros.factors.price import (
    BreakoutStrengthFactor,
    DistanceFromHighFactor,
    DistanceFromLowFactor,
    MaximumDrawdownFactor,
    MomentumFactor,
    MultiHorizonMomentumFactor,
    PriceAccelerationFactor,
    RecoveryStrengthFactor,
    RollingReturnMeanFactor,
    RollingReturnMedianFactor,
    TrendPersistenceFactor,
)
from cqros.factors.registry import FactorRegistry
from cqros.factors.repository import FactorPartitionRef, FactorsRepository
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_COLUMNS,
    FACTOR_SCHEMA,
    METADATA_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    FactorStatus,
    factor_status_values,
    factor_statuses,
)
from cqros.factors.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    FactorVerifier,
    VerificationReport,
)
from cqros.factors.wide_to_long import WideToLongFactorTransformer

__all__ = [
    "BaseFactor",
    "BreakoutConfirmationFactor",
    "BreakoutStrengthFactor",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "CrowdingFactor",
    "DistanceFromHighFactor",
    "DistanceFromLowFactor",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "ExecutableFactorCatalog",
    "FACTOR_COLUMNS",
    "FACTOR_SCHEMA",
    "Factor",
    "FactorError",
    "FactorExecutionError",
    "FactorGenerationPipeline",
    "FactorGenerationStatistics",
    "FactorInputPartition",
    "FactorMetadata",
    "FactorPartitionRef",
    "FactorPipeline",
    "FactorRegistrationError",
    "FactorRegistry",
    "FactorStatus",
    "FactorValidationError",
    "FactorValidator",
    "FactorVerifier",
    "FactorsRepository",
    "FlowConfirmationFactor",
    "FundingDivergenceFactor",
    "LeveragedLongBuildUpFactor",
    "LeveragedShortBuildUpFactor",
    "LongSqueezeFactor",
    "METADATA_COLUMNS",
    "MaximumDrawdownFactor",
    "MomentumFactor",
    "MultiHorizonMomentumFactor",
    "PRIMARY_KEY_COLUMNS",
    "PositionBuildUpFactor",
    "PriceAccelerationFactor",
    "REQUIRED_COLUMNS",
    "RecoveryStrengthFactor",
    "RollingReturnMeanFactor",
    "RollingReturnMedianFactor",
    "ShortSqueezeFactor",
    "TrendConfirmationFactor",
    "TrendPersistenceFactor",
    "UnknownFactorError",
    "VerificationReport",
    "WideToLongFactorTransformer",
    "build_default_registry",
    "classify_dependency_class",
    "factor_status_values",
    "factor_statuses",
    "required_companion_columns",
    "required_datasets",
]
