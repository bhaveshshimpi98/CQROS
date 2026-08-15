"""Unit tests for CQROS Factor Research Engine protocol conformance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import polars as pl

from cqros.factors.interfaces import Factor, FactorPipeline, FactorValidator


class _ConformingFactor:
    """Minimal Factor-shaped stub used only for protocol conformance tests."""

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Cross-sectional momentum"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def required_features(self) -> Sequence[str]:
        return ("returns",)

    @property
    def produced_columns(self) -> Sequence[str]:
        return ("momentum_score",)

    @property
    def lookback(self) -> int:
        return 20

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


class _ConformingFactorValidator:
    """Minimal FactorValidator-shaped stub for protocol conformance tests."""

    def validate(self, frame: pl.DataFrame, factor: Factor) -> None:
        return None


class _ConformingFactorPipeline:
    """Minimal FactorPipeline-shaped stub for protocol conformance tests."""

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


class _IncompleteFactor:
    """Stub missing ``compute`` so Factor conformance must fail."""

    @property
    def name(self) -> str:
        return "incomplete"

    @property
    def version(self) -> str:
        return "0.0.0"

    @property
    def description(self) -> str:
        return "Incomplete"

    @property
    def category(self) -> str:
        return "test"

    @property
    def required_features(self) -> Sequence[str]:
        return ()

    @property
    def produced_columns(self) -> Sequence[str]:
        return ()

    @property
    def lookback(self) -> int:
        return 0


def test_factor_protocols_are_runtime_checkable() -> None:
    """Factor contracts are Protocols marked runtime-checkable."""
    assert isinstance(Factor, type)
    assert isinstance(FactorValidator, type)
    assert isinstance(FactorPipeline, type)
    assert issubclass(Factor, Protocol)
    assert issubclass(FactorValidator, Protocol)
    assert issubclass(FactorPipeline, Protocol)
    assert getattr(Factor, "_is_runtime_protocol", False) is True
    assert getattr(FactorValidator, "_is_runtime_protocol", False) is True
    assert getattr(FactorPipeline, "_is_runtime_protocol", False) is True


def test_factor_protocols_use_runtime_checkable_decorator() -> None:
    """Each Factor contract is decorated with ``runtime_checkable``."""
    assert runtime_checkable(Factor) is Factor
    assert runtime_checkable(FactorValidator) is FactorValidator
    assert runtime_checkable(FactorPipeline) is FactorPipeline


def test_conforming_factor_satisfies_protocol() -> None:
    """A complete Factor-shaped object passes runtime isinstance checks."""
    assert isinstance(_ConformingFactor(), Factor)


def test_incomplete_factor_fails_protocol() -> None:
    """A Factor-shaped object missing ``compute`` fails isinstance."""
    assert not isinstance(_IncompleteFactor(), Factor)


def test_non_factor_object_fails_protocol() -> None:
    """Unrelated objects are not Factors."""
    assert not isinstance(object(), Factor)
    assert not isinstance({"name": "momentum"}, Factor)


def test_conforming_factor_validator_satisfies_protocol() -> None:
    """A complete FactorValidator-shaped object passes isinstance."""
    assert isinstance(_ConformingFactorValidator(), FactorValidator)


def test_incomplete_factor_validator_fails_protocol() -> None:
    """Objects without ``validate`` are not FactorValidators."""
    assert not isinstance(object(), FactorValidator)
    assert not isinstance(_ConformingFactor(), FactorValidator)


def test_conforming_factor_pipeline_satisfies_protocol() -> None:
    """A complete FactorPipeline-shaped object passes isinstance."""
    assert isinstance(_ConformingFactorPipeline(), FactorPipeline)


def test_incomplete_factor_pipeline_fails_protocol() -> None:
    """Objects without ``run`` are not FactorPipelines."""
    assert not isinstance(object(), FactorPipeline)
    assert not isinstance(_ConformingFactor(), FactorPipeline)


def test_factor_compute_signature_is_callable() -> None:
    """Conforming Factor ``compute`` accepts a DataFrame and returns one."""
    factor = _ConformingFactor()
    frame = pl.DataFrame({"returns": [0.1, -0.2]})
    result = factor.compute(frame)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(frame)


def test_factor_validator_validate_accepts_factor() -> None:
    """Conforming FactorValidator ``validate`` accepts frame and Factor."""
    validator = _ConformingFactorValidator()
    factor: Factor = _ConformingFactor()
    frame = pl.DataFrame({"momentum_score": [1.0]})
    assert validator.validate(frame, factor) is None


def test_factor_pipeline_run_accepts_frame() -> None:
    """Conforming FactorPipeline ``run`` accepts a training DataFrame."""
    pipeline = _ConformingFactorPipeline()
    frame = pl.DataFrame({"returns": [0.1]})
    result = pipeline.run(frame)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(frame)
