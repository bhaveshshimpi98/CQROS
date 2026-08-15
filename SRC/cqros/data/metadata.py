"""CQROS research and artifact metadata models.

Purpose:
    Provide immutable, exchange-agnostic value objects that record identity,
    provenance, and reproducibility metadata for research artifacts used
    throughout CQROS.

Responsibilities:
    - Define metadata structures for datasets, feature sets, models,
      experiments, and backtests
    - Define lineage and generic artifact metadata value objects
    - Remain free of business logic, validation, and I/O

Dependencies:
    Python standard library and ``cqros.core.types``.

Public API:
    The dataclasses listed in ``__all__``.

Notes:
    Collections that form part of an immutable value object use ``tuple``
    rather than ``list``. Arbitrary extension fields use
    ``Mapping[str, object]`` so future exchanges and research workflows can
    attach structured context without changing the core models.
    ``cqros.data.schemas.DatasetDescriptor`` is the market-storage
    dataset descriptor; ``DatasetMetadata`` in this module is the research
    reproducibility and lineage-oriented record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cqros.core.types import (
    CompressionCodec,
    Exchange,
    FilePath,
    Id,
    Symbol,
    Timeframe,
    Timestamp,
)

__all__ = [
    "LineageMetadata",
    "ArtifactMetadata",
    "DatasetMetadata",
    "FeatureSetMetadata",
    "ModelMetadata",
    "ExperimentMetadata",
    "BacktestMetadata",
]


@dataclass(frozen=True, slots=True)
class LineageMetadata:
    """Directed provenance links for a CQROS research artifact.

    Lineage forms a directed acyclic graph of parents, children, inputs,
    outputs, and other dependency relationships. Relationship identifiers
    refer to other artifact IDs; this model does not resolve or validate
    those references.

    Attributes:
        artifact_id: Identifier of the artifact this lineage describes.
        parents: Immutable sequence of direct parent artifact IDs.
        children: Immutable sequence of direct child artifact IDs.
        inputs: Immutable sequence of input artifact IDs consumed.
        outputs: Immutable sequence of output artifact IDs produced.
        dependencies: Immutable sequence of dependency artifact IDs.
        producers: Immutable sequence of producer artifact or process IDs.
        consumers: Immutable sequence of consumer artifact or process IDs.
        created_at: Lineage record creation timestamp (UTC), if recorded.
        metadata: Additional structured lineage context, if recorded.
    """

    artifact_id: Id
    parents: tuple[Id, ...] = ()
    children: tuple[Id, ...] = ()
    inputs: tuple[Id, ...] = ()
    outputs: tuple[Id, ...] = ()
    dependencies: tuple[Id, ...] = ()
    producers: tuple[Id, ...] = ()
    consumers: tuple[Id, ...] = ()
    created_at: Timestamp | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Generic identity and audit metadata for a CQROS artifact.

    Captures the fields shared across datasets, features, models, reports,
    experiments, and related research products so nothing is anonymous and
    every artifact remains auditable.

    Attributes:
        artifact_id: Stable globally unique artifact identifier.
        version: Artifact version string.
        name: Human-readable artifact name.
        artifact_type: Artifact category identifier (for example
            ``dataset``, ``model``, or ``experiment``).
        owner: Researcher, system, or service that owns the artifact.
        created_at: Creation timestamp (UTC).
        status: Lifecycle status identifier (for example ``draft`` or
            ``approved``).
        description: Free-text description of the artifact, if assigned.
        tags: Immutable sequence of classification tags.
        source: Origin system or process identifier, if recorded.
        checksum: Content checksum of the artifact payload, if recorded.
        parent_ids: Immutable sequence of parent artifact IDs, if recorded
            outside a full lineage object.
        child_ids: Immutable sequence of child artifact IDs, if recorded
            outside a full lineage object.
        lineage: Structured lineage relationships, if recorded.
        git_commit: Source-control commit associated with creation, if
            recorded.
        metadata: Additional structured artifact context, if recorded.
    """

    artifact_id: Id
    version: str
    name: str
    artifact_type: str
    owner: str
    created_at: Timestamp
    status: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    source: str | None = None
    checksum: str | None = None
    parent_ids: tuple[Id, ...] = ()
    child_ids: tuple[Id, ...] = ()
    lineage: LineageMetadata | None = None
    git_commit: str | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Research metadata describing a versioned CQROS dataset.

    Records identity, coverage, integrity, and provenance required for
    reproducible research. Optional fields support multi-exchange and
    multi-horizon workflows without encoding venue-specific semantics.

    Attributes:
        dataset_id: Stable dataset identifier.
        version: Dataset version string.
        created_at: Creation timestamp (UTC).
        symbols: Immutable sequence of instrument symbols covered.
        intervals: Immutable sequence of timeframes / intervals covered.
        rows: Number of rows in the dataset.
        columns: Immutable sequence of column names.
        checksum: Content checksum of the dataset artifact.
        name: Human-readable dataset name, if assigned.
        description: Free-text description, if assigned.
        created_by: Researcher or system that created the dataset, if
            recorded.
        exchange: Primary exchange identifier, if the dataset is
            single-venue.
        exchanges: Immutable sequence of exchange identifiers covered,
            if multi-venue.
        schema_version: Schema version string, if recorded.
        storage_location: Storage path or URI, if recorded.
        compression: Compression codec used for the artifact, if recorded.
        hash_algorithm: Algorithm used to compute ``checksum``, if
            recorded.
        quality_score: Dataset quality score, if computed.
        validation_report_id: Identifier of a validation report, if
            recorded.
        start_time: Inclusive dataset time range start (UTC), if known.
        end_time: Exclusive or inclusive dataset time range end (UTC), if
            known.
        lineage: Structured lineage relationships, if recorded.
        git_commit: Source-control commit associated with creation, if
            recorded.
        tags: Immutable sequence of classification tags.
        metadata: Additional structured dataset context, if recorded.
    """

    dataset_id: Id
    version: str
    created_at: Timestamp
    symbols: tuple[Symbol, ...]
    intervals: tuple[Timeframe, ...]
    rows: int
    columns: tuple[str, ...]
    checksum: str
    name: str | None = None
    description: str | None = None
    created_by: str | None = None
    exchange: Exchange | None = None
    exchanges: tuple[Exchange, ...] = ()
    schema_version: str | None = None
    storage_location: FilePath | None = None
    compression: CompressionCodec | None = None
    hash_algorithm: str | None = None
    quality_score: float | None = None
    validation_report_id: Id | None = None
    start_time: Timestamp | None = None
    end_time: Timestamp | None = None
    lineage: LineageMetadata | None = None
    git_commit: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class FeatureSetMetadata:
    """Research metadata describing a versioned feature set.

    Attributes:
        feature_set_id: Stable feature-set identifier.
        version: Feature-set version string.
        name: Human-readable feature-set name.
        created_at: Creation timestamp (UTC).
        feature_ids: Immutable sequence of feature identifiers included.
        feature_names: Immutable sequence of feature names included.
        created_by: Researcher or system that created the feature set, if
            recorded.
        description: Free-text description, if assigned.
        formulas: Mapping of feature name to formula or definition payload,
            if recorded.
        parameters: Mapping of configuration parameters used to build the
            feature set, if recorded.
        dependencies: Immutable sequence of dependency artifact IDs.
        input_dataset_ids: Immutable sequence of input dataset IDs.
        output_schema: Immutable sequence of output column or field names,
            if recorded.
        schema_version: Output schema version string, if recorded.
        lineage: Structured lineage relationships, if recorded.
        git_commit: Source-control commit associated with creation, if
            recorded.
        tags: Immutable sequence of classification tags.
        metadata: Additional structured feature-set context, if recorded.
    """

    feature_set_id: Id
    version: str
    name: str
    created_at: Timestamp
    feature_ids: tuple[Id, ...]
    feature_names: tuple[str, ...]
    created_by: str | None = None
    description: str | None = None
    formulas: Mapping[str, object] | None = None
    parameters: Mapping[str, object] | None = None
    dependencies: tuple[Id, ...] = ()
    input_dataset_ids: tuple[Id, ...] = ()
    output_schema: tuple[str, ...] = ()
    schema_version: str | None = None
    lineage: LineageMetadata | None = None
    git_commit: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Research metadata describing a trained model artifact.

    Attributes:
        model_id: Stable model identifier.
        version: Model version string.
        name: Human-readable model name.
        created_at: Creation timestamp (UTC).
        algorithm: Algorithm or model-family identifier.
        hyperparameters: Mapping of hyperparameter names to values.
        training_dataset_id: Identifier of the training dataset.
        feature_set_version: Version of the feature set used for training.
        target_version: Version of the target definition used for training.
        metrics: Mapping of evaluation metric names to values.
        random_seed: Random seed used during training.
        created_by: Researcher or system that trained the model, if
            recorded.
        description: Free-text description, if assigned.
        framework: ML framework identifier, if recorded.
        framework_version: ML framework version string, if recorded.
        training_duration_seconds: Wall-clock training duration in seconds,
            if recorded.
        lineage: Structured lineage relationships, if recorded.
        git_commit: Source-control commit associated with training, if
            recorded.
        python_version: Python interpreter version used, if recorded.
        library_versions: Mapping of library names to versions, if
            recorded.
        container_version: Execution container or image version, if
            recorded.
        tags: Immutable sequence of classification tags.
        metadata: Additional structured model context, if recorded.
    """

    model_id: Id
    version: str
    name: str
    created_at: Timestamp
    algorithm: str
    hyperparameters: Mapping[str, object]
    training_dataset_id: Id
    feature_set_version: str
    target_version: str
    metrics: Mapping[str, object]
    random_seed: int
    created_by: str | None = None
    description: str | None = None
    framework: str | None = None
    framework_version: str | None = None
    training_duration_seconds: float | None = None
    lineage: LineageMetadata | None = None
    git_commit: str | None = None
    python_version: str | None = None
    library_versions: Mapping[str, object] | None = None
    container_version: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Research metadata describing a tracked experiment run.

    Attributes:
        experiment_id: Stable experiment identifier.
        version: Experiment version string.
        name: Human-readable experiment name.
        created_at: Creation timestamp (UTC).
        objective: Research objective or question under test.
        configuration: Mapping of experiment configuration values.
        git_commit: Source-control commit associated with the run.
        created_by: Researcher or system that ran the experiment, if
            recorded.
        hypothesis: Explicit hypothesis statement, if recorded.
        dataset_versions: Immutable sequence of dataset version strings.
        feature_versions: Immutable sequence of feature-set version
            strings.
        target_versions: Immutable sequence of target version strings.
        model_versions: Immutable sequence of model version strings.
        results: Mapping of result metric or artifact references, if
            recorded.
        environment: Runtime environment identifier, if recorded.
        python_version: Python interpreter version used, if recorded.
        library_versions: Mapping of library names to versions, if
            recorded.
        container_version: Execution container or image version, if
            recorded.
        random_seed: Random seed used for the experiment, if recorded.
        executed_at: Execution timestamp (UTC), if distinct from
            ``created_at``.
        status: Experiment lifecycle status identifier, if recorded.
        lineage: Structured lineage relationships, if recorded.
        tags: Immutable sequence of classification tags.
        metadata: Additional structured experiment context, if recorded.
    """

    experiment_id: Id
    version: str
    name: str
    created_at: Timestamp
    objective: str
    configuration: Mapping[str, object]
    git_commit: str
    created_by: str | None = None
    hypothesis: str | None = None
    dataset_versions: tuple[str, ...] = ()
    feature_versions: tuple[str, ...] = ()
    target_versions: tuple[str, ...] = ()
    model_versions: tuple[str, ...] = ()
    results: Mapping[str, object] | None = None
    environment: str | None = None
    python_version: str | None = None
    library_versions: Mapping[str, object] | None = None
    container_version: str | None = None
    random_seed: int | None = None
    executed_at: Timestamp | None = None
    status: str | None = None
    lineage: LineageMetadata | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BacktestMetadata:
    """Research metadata describing a backtest run.

    Attributes:
        backtest_id: Stable backtest identifier.
        version: Backtest version string.
        name: Human-readable backtest name.
        created_at: Creation timestamp (UTC).
        start_time: Inclusive backtest window start (UTC).
        end_time: Exclusive or inclusive backtest window end (UTC).
        configuration: Mapping of backtest configuration values.
        created_by: Researcher or system that ran the backtest, if
            recorded.
        description: Free-text description, if assigned.
        strategy_id: Strategy identifier under test, if recorded.
        model_ids: Immutable sequence of model IDs used in the backtest.
        dataset_ids: Immutable sequence of dataset IDs used in the
            backtest.
        initial_capital: Starting capital in quote-asset units, if
            recorded.
        metrics: Mapping of performance metric names to values, if
            recorded.
        random_seed: Random seed used for the backtest, if recorded.
        git_commit: Source-control commit associated with the run, if
            recorded.
        python_version: Python interpreter version used, if recorded.
        library_versions: Mapping of library names to versions, if
            recorded.
        container_version: Execution container or image version, if
            recorded.
        lineage: Structured lineage relationships, if recorded.
        tags: Immutable sequence of classification tags.
        metadata: Additional structured backtest context, if recorded.
    """

    backtest_id: Id
    version: str
    name: str
    created_at: Timestamp
    start_time: Timestamp
    end_time: Timestamp
    configuration: Mapping[str, object]
    created_by: str | None = None
    description: str | None = None
    strategy_id: Id | None = None
    model_ids: tuple[Id, ...] = ()
    dataset_ids: tuple[Id, ...] = ()
    initial_capital: float | None = None
    metrics: Mapping[str, object] | None = None
    random_seed: int | None = None
    git_commit: str | None = None
    python_version: str | None = None
    library_versions: Mapping[str, object] | None = None
    container_version: str | None = None
    lineage: LineageMetadata | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None
