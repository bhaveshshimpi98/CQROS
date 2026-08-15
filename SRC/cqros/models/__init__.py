"""CQROS Models package public API.

The Research Model Ledger records regime-conditioned model identity rows.
It is not the supervised ``cqros.ml`` training / prediction path.
"""

from cqros.models.engine import (
    MODEL_INPUT_COLUMNS,
    ModelEngine,
    SimpleModelEngine,
    validate_regime_frame,
)
from cqros.models.exceptions import ModelError, ModelException
from cqros.models.pipeline import ModelPipeline
from cqros.models.registry import ModelRegistry
from cqros.models.repository import ModelPartitionRef, ModelRepository
from cqros.models.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MODELS_COLUMNS,
    MODELS_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ModelStatus,
)
from cqros.models.verifier import (
    ERROR_COLUMN_ORDER,
    ERROR_FRAME_EMPTY,
    ERROR_FRAME_TYPE,
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    ModelVerifier,
)

__all__ = [
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "ERROR_COLUMN_ORDER",
    "ERROR_FRAME_EMPTY",
    "ERROR_FRAME_TYPE",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "MODEL_INPUT_COLUMNS",
    "MODELS_COLUMNS",
    "MODELS_SCHEMA",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "ModelEngine",
    "ModelError",
    "ModelException",
    "ModelPartitionRef",
    "ModelPipeline",
    "ModelRegistry",
    "ModelRepository",
    "ModelStatus",
    "ModelVerifier",
    "SimpleModelEngine",
    "validate_regime_frame",
]
