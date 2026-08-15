"""Unit tests for CQROS Risk Management package ``RiskPipeline``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.risk import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_RISK_SCHEMA,
    RiskDecision,
    RiskManager,
    RiskPipeline,
    RiskPolicyRegistry,
    RiskValidationError,
)
from cqros.risk.pipeline import RiskPipeline as RiskPipelineDirect

_TIMEFRAME = "1h"
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_OPTIMIZER = "equal_weight"
_POLICY_NAME = "fixed_risk"


def _open_time(index: int) -> datetime:
    """Build a deterministic UTC open_time for row ``index``."""
    return datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=index)


def _portfolio_frame(
    *,
    signals: list[str],
    weights: list[float],
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a canonical portfolio DataFrame for pipeline tests."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                open_times
                if open_times is not None
                else [_open_time(index) for index in range(row_count)]
            ),
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [_OPTIMIZER] * row_count,
            "signal": signals,
            "target_weight": weights,
        },
        schema={
            "symbol": pl.Utf8,
            "timeframe": pl.Utf8,
            "open_time": pl.Datetime("us", "UTC"),
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "optimizer": pl.Utf8,
            "signal": pl.Utf8,
            "target_weight": pl.Float64,
        },
    )


def _risk_frame(
    *,
    signals: list[str],
    target_weights: list[float],
    approved_weights: list[float] | None = None,
    decisions: list[str] | None = None,
    reasons: list[str] | None = None,
    symbols: list[str] | None = None,
    open_times: list[datetime] | None = None,
    optimizer: str = _OPTIMIZER,
    policy: str = _POLICY_NAME,
) -> pl.DataFrame:
    """Build a risk-decision-shaped manager output frame."""
    row_count = len(signals)
    return pl.DataFrame(
        {
            "symbol": (
                symbols if symbols is not None else [f"SYM{index}" for index in range(row_count)]
            ),
            "timeframe": [_TIMEFRAME] * row_count,
            "open_time": (
                open_times
                if open_times is not None
                else [_open_time(index) for index in range(row_count)]
            ),
            "model_name": [_MODEL_NAME] * row_count,
            "model_version": [_MODEL_VERSION] * row_count,
            "optimizer": [optimizer] * row_count,
            "policy": [policy] * row_count,
            "signal": signals,
            "target_weight": target_weights,
            "approved_weight": (
                approved_weights if approved_weights is not None else list(target_weights)
            ),
            "decision": (
                decisions if decisions is not None else [RiskDecision.APPROVE.value] * row_count
            ),
            "reason": (reasons if reasons is not None else ["ok"] * row_count),
        }
    )


class _RecordingManager:
    """Risk manager stub that records evaluate calls and returns a fixed frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls: list[pl.DataFrame] = []

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        self.calls.append(portfolios)
        return self.frame


class _NonDataFrameManager:
    """Risk manager stub that returns a non-DataFrame value."""

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return {"rows": portfolios.height}  # type: ignore[return-value]


class _EmptyOutputManager:
    """Risk manager stub that returns an empty risk frame."""

    def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
        return _risk_frame(signals=["BUY"], target_weights=[1.0]).clear()


def _make_pipeline(
    *,
    policy_name: str = _POLICY_NAME,
    policy: object | None = None,
) -> tuple[RiskPipeline, RiskPolicyRegistry, object]:
    """Build a pipeline with a registry containing one risk manager."""
    registry = RiskPolicyRegistry()
    resolved = (
        _RecordingManager(
            _risk_frame(signals=["BUY"], target_weights=[1.0]),
        )
        if policy is None
        else policy
    )
    registry.register(policy_name, cast(RiskManager, resolved))
    return RiskPipeline(registry), registry, resolved


def test_risk_pipeline_is_exported_from_package() -> None:
    """Package export matches the pipeline module class."""
    assert RiskPipeline is RiskPipelineDirect


def test_successful_execution() -> None:
    """Registered manager produces a finalized risk-decision frame."""
    output = _risk_frame(
        signals=["BUY", "SELL", "HOLD"],
        target_weights=[0.5, -0.5, 0.0],
        approved_weights=[0.4, -0.4, 0.0],
        decisions=[
            RiskDecision.RESIZE.value,
            RiskDecision.RESIZE.value,
            RiskDecision.APPROVE.value,
        ],
        reasons=["resized", "resized", "ok"],
    )
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(output),
    )
    portfolios = _portfolio_frame(
        signals=["BUY", "SELL", "HOLD"],
        weights=[0.5, -0.5, 0.0],
    )

    result = pipeline.run(_POLICY_NAME, portfolios)

    assert result.height == 3
    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert result.schema == MERGED_RISK_SCHEMA
    assert result.get_column("approved_weight").to_list() == pytest.approx(
        [0.4, -0.4, 0.0],
    )
    assert result.get_column("decision").to_list() == [
        RiskDecision.RESIZE.value,
        RiskDecision.RESIZE.value,
        RiskDecision.APPROVE.value,
    ]


def test_unknown_policy_raises() -> None:
    """Unknown policy names raise RiskValidationError."""
    pipeline, _registry, _policy = _make_pipeline()
    with pytest.raises(RiskValidationError, match="not registered") as exc_info:
        pipeline.run(
            "missing_policy",
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )
    assert exc_info.value.error_code == "RISK_REG_UNKNOWN"


def test_blank_policy_name_raises() -> None:
    """Blank policy names are rejected before registry lookup."""
    pipeline, _registry, _policy = _make_pipeline()
    with pytest.raises(RiskValidationError, match="non-blank") as exc_info:
        pipeline.run(
            "   ",
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )
    assert exc_info.value.error_code == "RISK_PIPE_NAME_BLANK"


def test_empty_portfolio_dataframe_raises() -> None:
    """Empty portfolio frames raise RiskValidationError."""
    pipeline, _registry, _policy = _make_pipeline()
    empty = _portfolio_frame(signals=["BUY"], weights=[1.0]).clear()
    with pytest.raises(RiskValidationError, match="at least one row") as exc_info:
        pipeline.run(_POLICY_NAME, empty)
    assert exc_info.value.error_code == "RISK_FRAME_EMPTY"


def test_non_dataframe_portfolios_raise() -> None:
    """Non-DataFrame portfolio inputs raise RiskValidationError."""
    pipeline, _registry, _policy = _make_pipeline()
    with pytest.raises(RiskValidationError, match="polars DataFrame") as exc_info:
        pipeline.run(_POLICY_NAME, [{"signal": "BUY"}])  # type: ignore[arg-type]
    assert exc_info.value.error_code == "RISK_FRAME_TYPE"


def test_invalid_policy_output_type_raises() -> None:
    """Non-DataFrame policy outputs raise RiskValidationError."""
    pipeline, _registry, _policy = _make_pipeline(
        policy=_NonDataFrameManager(),
    )
    with pytest.raises(RiskValidationError, match="policy output") as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )
    assert exc_info.value.error_code == "RISK_PIPE_INVALID_OUTPUT"


def test_empty_policy_output_raises() -> None:
    """Empty policy outputs raise RiskValidationError."""
    pipeline, _registry, _policy = _make_pipeline(
        policy=_EmptyOutputManager(),
    )
    with pytest.raises(
        RiskValidationError,
        match="policy output must contain at least one row",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )
    assert exc_info.value.error_code == "RISK_PIPE_OUTPUT_EMPTY"


def test_missing_required_risk_columns_raises() -> None:
    """Missing risk schema columns on policy output are rejected."""
    incomplete = _risk_frame(
        signals=["BUY"],
        target_weights=[1.0],
    ).drop("approved_weight")
    pipeline, _registry, policy = _make_pipeline(
        policy=_RecordingManager(incomplete),
    )

    with pytest.raises(
        RiskValidationError,
        match="missing required columns",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )

    assert exc_info.value.error_code == "RISK_PIPE_MISSING_COLUMNS"
    assert "approved_weight" in exc_info.value.details["missing_columns"]
    assert isinstance(policy, _RecordingManager)
    assert len(policy.calls) == 1


def test_missing_optimizer_and_policy_columns_raise() -> None:
    """Missing optimizer or policy lineage columns are rejected."""
    incomplete = _risk_frame(
        signals=["BUY"],
        target_weights=[1.0],
    ).drop("optimizer", "policy")
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(incomplete),
    )

    with pytest.raises(
        RiskValidationError,
        match="missing required columns",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )

    assert exc_info.value.error_code == "RISK_PIPE_MISSING_COLUMNS"
    missing = exc_info.value.details["missing_columns"]
    assert "optimizer" in missing
    assert "policy" in missing


def test_successful_execution_preserves_lineage() -> None:
    """Finalized risk frames retain optimizer and policy lineage columns."""
    output = _risk_frame(
        signals=["BUY"],
        target_weights=[1.0],
        optimizer="equal_weight",
        policy="fixed_risk",
    )
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(output),
    )
    result = pipeline.run(
        _POLICY_NAME,
        _portfolio_frame(signals=["BUY"], weights=[1.0]),
    )
    assert result.get_column("optimizer").to_list() == ["equal_weight"]
    assert result.get_column("policy").to_list() == ["fixed_risk"]
    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_duplicate_primary_keys_on_policy_output_raise() -> None:
    """Duplicate primary keys in policy output raise RiskValidationError."""
    duplicate = _risk_frame(
        signals=["BUY", "SELL"],
        target_weights=[1.0, -1.0],
        symbols=["BTCUSDT", "BTCUSDT"],
        open_times=[_open_time(0), _open_time(0)],
    )
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(duplicate),
    )

    with pytest.raises(
        RiskValidationError,
        match="duplicate primary keys",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(
                signals=["BUY", "SELL"],
                weights=[1.0, -1.0],
                symbols=["BTCUSDT", "ETHUSDT"],
            ),
        )

    assert exc_info.value.error_code == "RISK_PIPE_DUPLICATE_KEYS"


def test_canonical_column_order() -> None:
    """Finalized output columns follow CANONICAL_COLUMN_ORDER exactly."""
    noisy = _risk_frame(
        signals=["BUY", "SELL"],
        target_weights=[1.0, -1.0],
    ).with_columns(pl.lit(1.0).alias("extra_noise"))
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(noisy),
    )

    result = pipeline.run(
        _POLICY_NAME,
        _portfolio_frame(signals=["BUY", "SELL"], weights=[1.0, -1.0]),
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "extra_noise" not in result.columns


def test_dtype_casting_matches_merged_risk_schema() -> None:
    """Finalized columns are cast to MERGED_RISK_SCHEMA dtypes."""
    frame = _risk_frame(
        signals=["BUY", "SELL"],
        target_weights=[1.0, -1.0],
    ).with_columns(
        pl.col("target_weight").cast(pl.Float32),
        pl.col("approved_weight").cast(pl.Float32),
    )
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(frame),
    )

    result = pipeline.run(
        _POLICY_NAME,
        _portfolio_frame(signals=["BUY", "SELL"], weights=[1.0, -1.0]),
    )

    assert result.schema == MERGED_RISK_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_registry_delegation() -> None:
    """Pipeline resolves and delegates exclusively through the registry."""
    output = _risk_frame(
        signals=["BUY", "SELL"],
        target_weights=[1.0, -1.0],
    )
    policy = _RecordingManager(output)
    pipeline, registry, _resolved = _make_pipeline(policy=policy)
    portfolios = _portfolio_frame(signals=["BUY", "SELL"], weights=[1.0, -1.0])

    result = pipeline.run(_POLICY_NAME, portfolios)

    assert registry.get(_POLICY_NAME) is policy
    assert len(policy.calls) == 1
    assert policy.calls[0] is portfolios
    assert_frame_equal(
        result.select(list(CANONICAL_COLUMN_ORDER)),
        output.select(list(CANONICAL_COLUMN_ORDER)).cast(MERGED_RISK_SCHEMA),
    )


def test_input_dataframe_is_not_mutated() -> None:
    """Pipeline never mutates the caller-supplied portfolio DataFrame."""
    portfolios = _portfolio_frame(signals=["BUY", "HOLD"], weights=[1.0, 0.0])
    original = portfolios.clone()
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(
            _risk_frame(
                signals=["BUY", "HOLD"],
                target_weights=[1.0, 0.0],
            ),
        ),
    )

    pipeline.run(_POLICY_NAME, portfolios)

    assert_frame_equal(portfolios, original)
    assert "approved_weight" not in portfolios.columns


def test_returned_frame_is_new() -> None:
    """Pipeline returns a new DataFrame distinct from policy output."""
    output = _risk_frame(signals=["BUY"], target_weights=[1.0])
    policy = _RecordingManager(output)
    pipeline, _registry, _policy = _make_pipeline(policy=policy)

    result = pipeline.run(
        _POLICY_NAME,
        _portfolio_frame(signals=["BUY"], weights=[1.0]),
    )

    assert result is not output
    assert result.schema == MERGED_RISK_SCHEMA


def test_policy_failure_propagates() -> None:
    """RiskValidationError raised by the policy propagates unchanged."""

    class _FailingManager:
        def evaluate(self, portfolios: pl.DataFrame) -> pl.DataFrame:
            raise RiskValidationError(
                "policy refused evaluation",
                error_code="RISK_POL_TEST",
                details={"rows": portfolios.height},
            )

    pipeline, _registry, _policy = _make_pipeline(policy=_FailingManager())

    with pytest.raises(
        RiskValidationError,
        match="policy refused evaluation",
    ) as exc_info:
        pipeline.run(
            _POLICY_NAME,
            _portfolio_frame(signals=["BUY"], weights=[1.0]),
        )

    assert exc_info.value.error_code == "RISK_POL_TEST"


def test_extra_policy_columns_are_dropped() -> None:
    """Non-canonical policy columns are dropped during finalization."""
    frame = _risk_frame(
        signals=["BUY"],
        target_weights=[1.0],
    ).with_columns(
        pl.lit("noise").alias("risk_note"),
        pl.lit(99).alias("rank"),
    )
    pipeline, _registry, _policy = _make_pipeline(
        policy=_RecordingManager(frame),
    )

    result = pipeline.run(
        _POLICY_NAME,
        _portfolio_frame(signals=["BUY"], weights=[1.0]),
    )

    assert result.columns == list(CANONICAL_COLUMN_ORDER)
    assert "risk_note" not in result.columns
    assert "rank" not in result.columns


def test_multiple_registered_policies_resolve_by_name() -> None:
    """Pipeline resolves the requested policy among multiple registrations."""
    approve_output = _risk_frame(
        signals=["HOLD"],
        target_weights=[0.0],
        decisions=[RiskDecision.APPROVE.value],
    )
    reject_output = _risk_frame(
        signals=["BUY"],
        target_weights=[1.0],
        approved_weights=[0.0],
        decisions=[RiskDecision.REJECT.value],
        reasons=["limit"],
    )
    approve = _RecordingManager(approve_output)
    reject = _RecordingManager(reject_output)
    registry = RiskPolicyRegistry()
    registry.register_many(
        {
            "approve": approve,
            "reject": reject,
        }
    )
    pipeline = RiskPipeline(registry)

    result = pipeline.run(
        "reject",
        _portfolio_frame(signals=["BUY"], weights=[1.0]),
    )

    assert len(reject.calls) == 1
    assert len(approve.calls) == 0
    assert result.get_column("decision").to_list() == [RiskDecision.REJECT.value]
    assert result.schema == MERGED_RISK_SCHEMA
