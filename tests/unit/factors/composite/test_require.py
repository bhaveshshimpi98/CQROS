"""Unit tests for composite factor required-feature validation helper."""

from __future__ import annotations

import polars as pl
import pytest

from cqros.factors.composite._require import require_feature_columns
from cqros.factors.exceptions import FactorError


def test_require_feature_columns_passes_when_all_present() -> None:
    """Helper accepts frames that contain every required feature."""
    frame = pl.DataFrame({"returns": [0.1], "flow_imbalance": [0.2]})
    require_feature_columns(
        frame,
        ("returns", "flow_imbalance"),
        factor="flow_confirmation",
        error_code="FACTOR-FLOW-CONFIRMATION-001",
    )


def test_require_feature_columns_raises_on_first_missing() -> None:
    """Helper fails on the first missing feature with structured details."""
    frame = pl.DataFrame({"returns": [0.1]})
    with pytest.raises(FactorError, match="required feature missing: flow_imbalance") as (exc_info):
        require_feature_columns(
            frame,
            ("returns", "flow_imbalance"),
            factor="flow_confirmation",
            error_code="FACTOR-FLOW-CONFIRMATION-001",
        )
    error = exc_info.value
    assert error.error_code == "FACTOR-FLOW-CONFIRMATION-001"
    assert error.details["factor"] == "flow_confirmation"
    assert error.details["required_feature"] == "flow_imbalance"
    assert error.details["available_columns"] == ("returns",)
