"""CQROS storage package public API."""

from cqros.storage.exceptions import (
    ArtifactExistsError,
    BackupFailureError,
    CompressionError,
    CorruptedDatasetError,
    DatasetNotFoundError,
    StorageError,
    StorageSerializationError,
    VersionConflictError,
)
from cqros.storage.feature_repository import FeaturePartitionRef, FeatureRepository
from cqros.storage.interfaces import IDataStore
from cqros.storage.label_repository import LabelPartitionRef, LabelRepository
from cqros.storage.layout import StorageLayout
from cqros.storage.parquet import DEFAULT_PARQUET_COMPRESSION, ParquetStore
from cqros.storage.order_repository import OrderPartitionRef, OrderRepository
from cqros.storage.portfolio_repository import (
    PortfolioPartitionRef,
    PortfolioRepository,
)
from cqros.storage.prediction_repository import (
    PredictionPartitionRef,
    PredictionRepository,
)
from cqros.storage.processed_repository import (
    PROCESSED_DATASETS,
    ProcessedMarketDataRepository,
    ProcessedPartitionRef,
)
from cqros.storage.repository import MarketDataRepository
from cqros.storage.risk_repository import RiskPartitionRef, RiskRepository
from cqros.storage.signal_repository import SignalPartitionRef, SignalRepository
from cqros.storage.threshold_repository import (
    ThresholdPartitionRef,
    ThresholdRepository,
)
from cqros.storage.training_repository import TrainingPartitionRef, TrainingRepository

__all__ = [
    "ArtifactExistsError",
    "BackupFailureError",
    "CompressionError",
    "CorruptedDatasetError",
    "DEFAULT_PARQUET_COMPRESSION",
    "DatasetNotFoundError",
    "FeaturePartitionRef",
    "FeatureRepository",
    "IDataStore",
    "LabelPartitionRef",
    "LabelRepository",
    "MarketDataRepository",
    "OrderPartitionRef",
    "OrderRepository",
    "PROCESSED_DATASETS",
    "ParquetStore",
    "PortfolioPartitionRef",
    "PortfolioRepository",
    "PredictionPartitionRef",
    "PredictionRepository",
    "ProcessedMarketDataRepository",
    "ProcessedPartitionRef",
    "RiskPartitionRef",
    "RiskRepository",
    "SignalPartitionRef",
    "SignalRepository",
    "StorageError",
    "StorageLayout",
    "StorageSerializationError",
    "ThresholdPartitionRef",
    "ThresholdRepository",
    "TrainingPartitionRef",
    "TrainingRepository",
    "VersionConflictError",
]
