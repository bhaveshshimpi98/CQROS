"""Unit tests for CQROS research and artifact metadata models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from cqros.data import metadata as metadata_module
from cqros.data.metadata import (
    ArtifactMetadata,
    BacktestMetadata,
    DatasetMetadata,
    ExperimentMetadata,
    FeatureSetMetadata,
    LineageMetadata,
    ModelMetadata,
)

_TS = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
_TS_END = datetime(2026, 7, 25, 18, 0, 0, tzinfo=UTC)

_METADATA_TYPES: tuple[type[object], ...] = (
    LineageMetadata,
    ArtifactMetadata,
    DatasetMetadata,
    FeatureSetMetadata,
    ModelMetadata,
    ExperimentMetadata,
    BacktestMetadata,
)


def _sample_lineage(**overrides: object) -> LineageMetadata:
    values: dict[str, object] = {
        "artifact_id": "art-001",
        "parents": ("ds-raw-v1",),
        "children": ("feat-v1",),
        "inputs": ("ds-raw-v1",),
        "outputs": ("ds-research-v1",),
        "dependencies": ("cfg-v1",),
        "producers": ("dataset-builder",),
        "consumers": ("feature-engine",),
        "created_at": _TS,
        "metadata": {"edge_count": 3},
    }
    values.update(overrides)
    return LineageMetadata(**values)  # type: ignore[arg-type]


def _sample_artifact(**overrides: object) -> ArtifactMetadata:
    values: dict[str, object] = {
        "artifact_id": "art-001",
        "version": "1.0.0",
        "name": "btc-research-dataset",
        "artifact_type": "dataset",
        "owner": "research",
        "created_at": _TS,
        "status": "validated",
        "description": "BTC research dataset",
        "tags": ("crypto", "ohlcv"),
        "source": "dataset-builder",
        "checksum": "sha256:abc",
        "parent_ids": ("ds-raw-v1",),
        "child_ids": ("feat-v1",),
        "lineage": _sample_lineage(),
        "git_commit": "deadbeef",
        "metadata": {"priority": "high"},
    }
    values.update(overrides)
    return ArtifactMetadata(**values)  # type: ignore[arg-type]


def _sample_dataset(**overrides: object) -> DatasetMetadata:
    values: dict[str, object] = {
        "dataset_id": "ds-btc-1m",
        "version": "1.0.0",
        "created_at": _TS,
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "intervals": ("1m", "5m"),
        "rows": 1_000,
        "columns": ("open_time", "open", "high", "low", "close", "volume"),
        "checksum": "sha256:deadbeef",
        "name": "btc-eth-ohlcv",
        "description": "Multi-symbol OHLCV",
        "created_by": "alice",
        "exchange": "binance",
        "exchanges": ("binance",),
        "schema_version": "1.0.0",
        "storage_location": Path("data/processed/ds-btc-1m"),
        "compression": "zstd",
        "hash_algorithm": "sha256",
        "quality_score": 0.99,
        "validation_report_id": "vr-001",
        "start_time": _TS,
        "end_time": _TS_END,
        "lineage": _sample_lineage(artifact_id="ds-btc-1m"),
        "git_commit": "deadbeef",
        "tags": ("research",),
        "metadata": {"venue": "binance"},
    }
    values.update(overrides)
    return DatasetMetadata(**values)  # type: ignore[arg-type]


def _sample_feature_set(**overrides: object) -> FeatureSetMetadata:
    values: dict[str, object] = {
        "feature_set_id": "fs-001",
        "version": "1.0.0",
        "name": "momentum-features",
        "created_at": _TS,
        "feature_ids": ("f-ret-1", "f-vol-1"),
        "feature_names": ("return_1m", "realized_vol_1h"),
        "created_by": "alice",
        "description": "Short-horizon momentum",
        "formulas": {"return_1m": "close.pct_change(1)"},
        "parameters": {"vol_window": 60},
        "dependencies": ("ds-btc-1m",),
        "input_dataset_ids": ("ds-btc-1m",),
        "output_schema": ("return_1m", "realized_vol_1h"),
        "schema_version": "1.0.0",
        "lineage": _sample_lineage(artifact_id="fs-001"),
        "git_commit": "deadbeef",
        "tags": ("momentum",),
        "metadata": {"family": "price"},
    }
    values.update(overrides)
    return FeatureSetMetadata(**values)  # type: ignore[arg-type]


def _sample_model(**overrides: object) -> ModelMetadata:
    values: dict[str, object] = {
        "model_id": "mdl-001",
        "version": "1.0.0",
        "name": "lgbm-momentum",
        "created_at": _TS,
        "algorithm": "lightgbm",
        "hyperparameters": {"num_leaves": 31, "learning_rate": 0.05},
        "training_dataset_id": "ds-btc-1m",
        "feature_set_version": "1.0.0",
        "target_version": "1.0.0",
        "metrics": {"ic": 0.04, "sharpe": 1.2},
        "random_seed": 42,
        "created_by": "alice",
        "description": "Baseline LGBM",
        "framework": "lightgbm",
        "framework_version": "4.5.0",
        "training_duration_seconds": 12.5,
        "lineage": _sample_lineage(artifact_id="mdl-001"),
        "git_commit": "deadbeef",
        "python_version": "3.13.0",
        "library_versions": {"lightgbm": "4.5.0", "numpy": "2.1.0"},
        "container_version": "cqros-research:1.0.0",
        "tags": ("baseline",),
        "metadata": {"objective": "regression"},
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


def _sample_experiment(**overrides: object) -> ExperimentMetadata:
    values: dict[str, object] = {
        "experiment_id": "exp-001",
        "version": "1.0.0",
        "name": "momentum-ic-study",
        "created_at": _TS,
        "objective": "Measure IC of short-horizon momentum",
        "configuration": {"horizon": "1h", "universe": "top10"},
        "git_commit": "deadbeef",
        "created_by": "alice",
        "hypothesis": "Momentum IC is positive OOS",
        "dataset_versions": ("1.0.0",),
        "feature_versions": ("1.0.0",),
        "target_versions": ("1.0.0",),
        "model_versions": ("1.0.0",),
        "results": {"rank_ic": 0.03},
        "environment": "research",
        "python_version": "3.13.0",
        "library_versions": {"polars": "1.9.0"},
        "container_version": "cqros-research:1.0.0",
        "random_seed": 42,
        "executed_at": _TS,
        "status": "validated",
        "lineage": _sample_lineage(artifact_id="exp-001"),
        "tags": ("ic", "momentum"),
        "metadata": {"reviewer": "bob"},
    }
    values.update(overrides)
    return ExperimentMetadata(**values)  # type: ignore[arg-type]


def _sample_backtest(**overrides: object) -> BacktestMetadata:
    values: dict[str, object] = {
        "backtest_id": "bt-001",
        "version": "1.0.0",
        "name": "momentum-walkforward",
        "created_at": _TS,
        "start_time": _TS,
        "end_time": _TS_END,
        "configuration": {"cost_bps": 2.0, "rebalance": "1h"},
        "created_by": "alice",
        "description": "Walk-forward momentum backtest",
        "strategy_id": "strat-momentum",
        "model_ids": ("mdl-001",),
        "dataset_ids": ("ds-btc-1m",),
        "initial_capital": 100_000.0,
        "metrics": {"sharpe": 1.1, "max_drawdown": 0.12},
        "random_seed": 42,
        "git_commit": "deadbeef",
        "python_version": "3.13.0",
        "library_versions": {"numpy": "2.1.0"},
        "container_version": "cqros-research:1.0.0",
        "lineage": _sample_lineage(artifact_id="bt-001"),
        "tags": ("walkforward",),
        "metadata": {"benchmark": "btc"},
    }
    values.update(overrides)
    return BacktestMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("metadata_type", _METADATA_TYPES)
def test_metadata_types_are_frozen_slotted_dataclasses(
    metadata_type: type[object],
) -> None:
    """Metadata models are immutable slotted dataclasses."""
    assert is_dataclass(metadata_type)
    assert hasattr(metadata_type, "__slots__")


@pytest.mark.parametrize("metadata_type", _METADATA_TYPES)
def test_metadata_types_are_exported(metadata_type: type[object]) -> None:
    """Each metadata model is listed in the module public API."""
    assert metadata_type.__name__ in metadata_module.__all__
    assert getattr(metadata_module, metadata_type.__name__) is metadata_type


def test_lineage_metadata_defaults_and_tuples() -> None:
    """LineageMetadata defaults relationship collections to empty tuples."""
    lineage = LineageMetadata(artifact_id="art-001")
    assert lineage.parents == ()
    assert lineage.children == ()
    assert lineage.inputs == ()
    assert lineage.outputs == ()
    assert lineage.dependencies == ()
    assert lineage.producers == ()
    assert lineage.consumers == ()
    assert lineage.created_at is None
    assert lineage.metadata is None


def test_lineage_metadata_stores_relationships() -> None:
    """LineageMetadata preserves immutable relationship identifiers."""
    lineage = _sample_lineage()
    assert lineage.parents == ("ds-raw-v1",)
    assert lineage.consumers == ("feature-engine",)
    assert isinstance(lineage.parents, tuple)
    assert isinstance(lineage.metadata, Mapping)


def test_artifact_metadata_required_and_optional_defaults() -> None:
    """ArtifactMetadata requires identity fields; optional fields default safely."""
    artifact = ArtifactMetadata(
        artifact_id="art-001",
        version="0.1.0",
        name="raw-ticks",
        artifact_type="dataset",
        owner="ingestion",
        created_at=_TS,
        status="draft",
    )
    assert artifact.description is None
    assert artifact.tags == ()
    assert artifact.parent_ids == ()
    assert artifact.child_ids == ()
    assert artifact.lineage is None
    assert artifact.metadata is None


def test_dataset_metadata_uses_tuples_for_collections() -> None:
    """DatasetMetadata stores symbols, intervals, columns, and tags as tuples."""
    dataset = _sample_dataset()
    assert isinstance(dataset.symbols, tuple)
    assert isinstance(dataset.intervals, tuple)
    assert isinstance(dataset.columns, tuple)
    assert isinstance(dataset.exchanges, tuple)
    assert isinstance(dataset.tags, tuple)
    assert dataset.symbols == ("BTCUSDT", "ETHUSDT")
    assert dataset.compression == "zstd"


def test_dataset_metadata_supports_multi_exchange_coverage() -> None:
    """DatasetMetadata can omit a primary exchange and list multiple venues."""
    dataset = DatasetMetadata(
        dataset_id="ds-multi",
        version="1.0.0",
        created_at=_TS,
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1h",),
        rows=100,
        columns=("close",),
        checksum="sha256:abc",
        exchanges=("binance", "bybit"),
    )
    assert dataset.exchange is None
    assert dataset.exchanges == ("binance", "bybit")


def test_feature_set_metadata_mapping_fields() -> None:
    """FeatureSetMetadata accepts Mapping values for formulas and parameters."""
    feature_set = _sample_feature_set(
        formulas=MappingProxyType({"return_1m": "close.pct_change(1)"}),
        parameters=MappingProxyType({"vol_window": 60}),
    )
    assert isinstance(feature_set.formulas, Mapping)
    assert feature_set.parameters is not None
    assert feature_set.parameters["vol_window"] == 60
    assert feature_set.feature_names == ("return_1m", "realized_vol_1h")


def test_model_metadata_records_reproducibility_fields() -> None:
    """ModelMetadata records seed, versions, and environment details."""
    model = _sample_model()
    assert model.random_seed == 42
    assert model.feature_set_version == "1.0.0"
    assert model.target_version == "1.0.0"
    assert model.python_version == "3.13.0"
    assert model.library_versions is not None
    assert model.library_versions["lightgbm"] == "4.5.0"


def test_experiment_metadata_records_version_lineage() -> None:
    """ExperimentMetadata stores version tuples for related artifacts."""
    experiment = _sample_experiment()
    assert experiment.dataset_versions == ("1.0.0",)
    assert experiment.feature_versions == ("1.0.0",)
    assert experiment.target_versions == ("1.0.0",)
    assert experiment.model_versions == ("1.0.0",)
    assert experiment.git_commit == "deadbeef"
    assert experiment.objective.startswith("Measure IC")


def test_backtest_metadata_window_and_metrics() -> None:
    """BacktestMetadata records the evaluation window and optional metrics."""
    backtest = _sample_backtest()
    assert backtest.start_time == _TS
    assert backtest.end_time == _TS_END
    assert backtest.model_ids == ("mdl-001",)
    assert backtest.dataset_ids == ("ds-btc-1m",)
    assert backtest.metrics is not None
    assert backtest.metrics["sharpe"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    ("factory", "attr", "value"),
    [
        (_sample_lineage, "parents", ("x",)),
        (_sample_artifact, "tags", ("a",)),
        (_sample_dataset, "rows", 0),
        (_sample_feature_set, "name", "other"),
        (_sample_model, "random_seed", 0),
        (_sample_experiment, "status", "rejected"),
        (_sample_backtest, "initial_capital", 0.0),
    ],
)
def test_metadata_instances_are_frozen(
    factory: object,
    attr: str,
    value: object,
) -> None:
    """Metadata instances reject attribute mutation."""
    instance = factory()  # type: ignore[operator]
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attr, value)


def test_lineage_metadata_field_names_are_stable() -> None:
    """LineageMetadata public field names remain stable for consumers."""
    names = tuple(field.name for field in fields(LineageMetadata))
    assert names == (
        "artifact_id",
        "parents",
        "children",
        "inputs",
        "outputs",
        "dependencies",
        "producers",
        "consumers",
        "created_at",
        "metadata",
    )


def test_dataset_metadata_field_names_are_stable() -> None:
    """DatasetMetadata public field names remain stable for consumers."""
    names = tuple(field.name for field in fields(DatasetMetadata))
    assert names == (
        "dataset_id",
        "version",
        "created_at",
        "symbols",
        "intervals",
        "rows",
        "columns",
        "checksum",
        "name",
        "description",
        "created_by",
        "exchange",
        "exchanges",
        "schema_version",
        "storage_location",
        "compression",
        "hash_algorithm",
        "quality_score",
        "validation_report_id",
        "start_time",
        "end_time",
        "lineage",
        "git_commit",
        "tags",
        "metadata",
    )


def test_package_exports_metadata_models() -> None:
    """The data package re-exports research metadata models."""
    import cqros.data as data_package

    for name in (
        "LineageMetadata",
        "ArtifactMetadata",
        "DatasetMetadata",
        "FeatureSetMetadata",
        "ModelMetadata",
        "ExperimentMetadata",
        "BacktestMetadata",
    ):
        assert name in data_package.__all__
        assert getattr(data_package, name).__name__ == name

    # Research DatasetMetadata is the package-level DatasetMetadata export.
    assert data_package.DatasetMetadata is DatasetMetadata
    assert data_package.FeatureSetMetadata is FeatureSetMetadata
    assert data_package.ModelMetadata is ModelMetadata
    assert data_package.ExperimentMetadata is ExperimentMetadata
    assert data_package.BacktestMetadata is BacktestMetadata
    assert data_package.LineageMetadata is LineageMetadata
    assert data_package.ArtifactMetadata is ArtifactMetadata
