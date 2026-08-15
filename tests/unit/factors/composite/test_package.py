"""Unit tests for CQROS composite factor package exports."""

from __future__ import annotations

import cqros.factors as factors_package
import cqros.factors.composite as composite_package
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


def test_composite_package_exports_all_factors() -> None:
    """Composite package exports every initial composite factor family."""
    expected = {
        "BreakoutConfirmationFactor",
        "CrowdingFactor",
        "FlowConfirmationFactor",
        "FundingDivergenceFactor",
        "LeveragedLongBuildUpFactor",
        "LeveragedShortBuildUpFactor",
        "LongSqueezeFactor",
        "PositionBuildUpFactor",
        "ShortSqueezeFactor",
        "TrendConfirmationFactor",
    }
    assert set(composite_package.__all__) == expected


def test_factors_package_reexports_composite_factors() -> None:
    """Top-level factors package re-exports composite factor classes."""
    assert factors_package.TrendConfirmationFactor is TrendConfirmationFactor
    assert factors_package.BreakoutConfirmationFactor is BreakoutConfirmationFactor
    assert factors_package.PositionBuildUpFactor is PositionBuildUpFactor
    assert factors_package.LeveragedLongBuildUpFactor is LeveragedLongBuildUpFactor
    assert factors_package.LeveragedShortBuildUpFactor is LeveragedShortBuildUpFactor
    assert factors_package.CrowdingFactor is CrowdingFactor
    assert factors_package.ShortSqueezeFactor is ShortSqueezeFactor
    assert factors_package.LongSqueezeFactor is LongSqueezeFactor
    assert factors_package.FlowConfirmationFactor is FlowConfirmationFactor
    assert factors_package.FundingDivergenceFactor is FundingDivergenceFactor
