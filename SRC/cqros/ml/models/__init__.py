"""CQROS ML Models package public API."""

from cqros.ml.models.base import BaseModel
from cqros.ml.models.catboost import CatBoostModel
from cqros.ml.models.exceptions import ModelError, ModelValidationError
from cqros.ml.models.interfaces import Model
from cqros.ml.models.lightgbm import LightGBMModel
from cqros.ml.models.metadata import ModelFramework, ModelMetadata, ModelTaskType
from cqros.ml.models.persistence import ModelPersistence
from cqros.ml.models.registry import ModelRegistry
from cqros.ml.models.repository import ModelArtifactRef, ModelArtifactRepository
from cqros.ml.models.xgboost import XGBoostModel

__all__ = [
    "BaseModel",
    "CatBoostModel",
    "LightGBMModel",
    "Model",
    "ModelArtifactRef",
    "ModelArtifactRepository",
    "ModelError",
    "ModelFramework",
    "ModelMetadata",
    "ModelPersistence",
    "ModelRegistry",
    "ModelTaskType",
    "ModelValidationError",
    "XGBoostModel",
]
