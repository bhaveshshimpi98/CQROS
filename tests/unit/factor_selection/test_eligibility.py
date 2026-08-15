"""Unit, regression, and integration tests for FactorEligibilityPolicy.

Tests cover:
    - Zero observations → hard ineligible
    - 100% null / FAIL status → hard ineligible (via zero observations)
    - Sufficient coverage → eligible
    - Insufficient warmup → ineligible
    - Sufficient warmup → eligible
    - Timeframe-aware evaluation (bar duration derivation)
    - Selection-window boundary enforcement (OOS data cannot affect eligibility)
    - Orientation metadata unchanged by eligibility gate
    - abs(IC) ranking unchanged for eligible factors
    - Eligible factors ranked normally
    - Ineligible factors cannot be selected
    - Legacy metadata fails closed
    - Deterministic eligibility decisions
    - Regression fixture: real 1d situation (37 bars, late companion, lookback 20/39)
    - Integration: Factor Validation → Eligibility → Factor Selection → orientation metadata
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factor_selection.eligibility import (
    ELIGIBILITY_METADATA_COLUMNS,
    ELIGIBILITY_POLICY_VERSION,
    FACTOR_ELIGIBILITY_POLICY,
    LEGACY_ELIGIBILITY_ERROR_CODE,
    TIMEFRAME_BAR_MILLISECONDS,
    EligibilityDecision,
    EligibilityStatus,
    FactorEligibilityPolicy,
    _EFFECTIVE_WARMUP_OVERRIDES,
    evaluate_eligibility,
    require_eligibility_metadata,
)
from cqros.factor_selection.engine import SimpleFactorSelectionEngine
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.schema import ELIGIBILITY_COLUMNS

_TIMEFRAME = "1d"
_FACTOR_V = "1.0.0"
_POLICY = FactorEligibilityPolicy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val_frame(
    *,
    names: list[str],
    observations: list[int],
    ics: list[float | None] | None = None,
    timeframe: str = _TIMEFRAME,
    validation_start_ms: int | None = None,
    validation_end_ms: int | None = None,
) -> pl.DataFrame:
    """Build a minimal Factor Validation frame for selection engine tests."""
    count = len(names)
    ics_resolved = ics if ics is not None else [0.05 if o > 0 else None for o in observations]
    abs_ics = [abs(v) if v is not None else None for v in ics_resolved]
    statuses = ["PASS" if o > 0 else "FAIL" for o in observations]
    data: dict[str, list[object]] = {
        "factor_name": names,
        "factor_version": [_FACTOR_V] * count,
        "factor_category": ["price"] * count,
        "timeframe": [timeframe] * count,
        "validation_time": [1_750_000_000_000] * count,
        "information_coefficient": ics_resolved,
        "rank_information_coefficient": abs_ics,
        "ic_information_ratio": [0.8 if o > 0 else None for o in observations],
        "ic_p_value": [0.01 if o > 0 else None for o in observations],
        "ic_t_stat": [3.0 if o > 0 else None for o in observations],
        "ic_decay": [0.6 if o > 0 else None for o in observations],
        "turnover": [0.2 if o > 0 else None for o in observations],
        "monotonicity_score": [0.7 if o > 0 else None for o in observations],
        "quantile_spread": [0.03 if o > 0 else None for o in observations],
        "observations": observations,
        "status": statuses,
    }
    if validation_start_ms is not None and validation_end_ms is not None:
        data["validation_start_time"] = [validation_start_ms] * count
        data["validation_end_time"] = [validation_end_ms] * count
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Unit: EligibilityStatus values
# ---------------------------------------------------------------------------


def test_eligibility_status_values_are_unique() -> None:
    """All EligibilityStatus values are unique strings."""
    values = [s.value for s in EligibilityStatus]
    assert len(values) == len(set(values))


def test_factor_eligibility_policy_version_constant() -> None:
    """FACTOR_ELIGIBILITY_POLICY matches default policy_version."""
    assert FactorEligibilityPolicy().policy_version == FACTOR_ELIGIBILITY_POLICY
    assert FACTOR_ELIGIBILITY_POLICY == ELIGIBILITY_POLICY_VERSION


# ---------------------------------------------------------------------------
# Unit: zero observations → hard ineligible
# ---------------------------------------------------------------------------


def test_zero_observations_is_hard_ineligible() -> None:
    """usable_observations=0 must produce INELIGIBLE_ZERO_OBSERVATIONS."""
    decision = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="1d",
        usable_observations=0,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS
    assert not decision.is_eligible
    assert "usable_observations=0" in decision.reason


def test_zero_observations_overrides_warmup_check() -> None:
    """Zero observations is caught before warmup check."""
    decision = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="1d",
        usable_observations=0,
        available_history=100,
        declared_lookback=20,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS


def test_100_pct_null_factor_has_zero_usable_obs() -> None:
    """A factor whose entire selection-window is null has 0 usable observations."""
    decision = evaluate_eligibility(
        factor_name="bollinger_bandwidth",
        timeframe="1d",
        usable_observations=0,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS


# ---------------------------------------------------------------------------
# Unit: coverage → eligible
# ---------------------------------------------------------------------------


def test_nonzero_observations_without_warmup_info_is_eligible() -> None:
    """Factor with non-zero obs and no warmup info is ELIGIBLE."""
    decision = _POLICY.evaluate(
        factor_name="accumulation_distribution",
        timeframe="1d",
        usable_observations=3691,
    )
    assert decision.status == EligibilityStatus.ELIGIBLE
    assert decision.is_eligible


def test_coverage_ratio_computed_when_total_available() -> None:
    """coverage_ratio and null_rate are correct when total_observations provided."""
    decision = _POLICY.evaluate(
        factor_name="rsi",
        timeframe="1d",
        usable_observations=50,
        total_observations=100,
    )
    assert decision.coverage_ratio == pytest.approx(0.5)
    assert decision.null_rate == pytest.approx(0.5)


def test_coverage_ratio_none_when_total_unavailable() -> None:
    """coverage_ratio is None when total_observations not provided."""
    decision = _POLICY.evaluate(
        factor_name="rsi",
        timeframe="1d",
        usable_observations=50,
    )
    assert decision.coverage_ratio is None
    assert decision.null_rate is None


# ---------------------------------------------------------------------------
# Unit: warmup check
# ---------------------------------------------------------------------------


def test_insufficient_warmup_blocks_factor() -> None:
    """effective_warmup > available_history → INELIGIBLE_INSUFFICIENT_WARMUP."""
    decision = _POLICY.evaluate(
        factor_name="some_factor",
        timeframe="1d",
        usable_observations=10,
        declared_lookback=20,
        available_history=15,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_INSUFFICIENT_WARMUP
    assert not decision.is_eligible
    assert decision.warmup_sufficient is False


def test_exact_warmup_equals_history_is_eligible() -> None:
    """effective_warmup == available_history → ELIGIBLE (boundary inclusive)."""
    decision = _POLICY.evaluate(
        factor_name="some_factor",
        timeframe="1d",
        usable_observations=10,
        declared_lookback=20,
        available_history=20,
    )
    assert decision.status == EligibilityStatus.ELIGIBLE
    assert decision.warmup_sufficient is True


def test_sufficient_warmup_is_eligible() -> None:
    """effective_warmup < available_history → ELIGIBLE."""
    decision = _POLICY.evaluate(
        factor_name="some_factor",
        timeframe="1d",
        usable_observations=200,
        declared_lookback=14,
        available_history=500,
    )
    assert decision.status == EligibilityStatus.ELIGIBLE
    assert decision.warmup_sufficient is True


def test_missing_available_history_skips_warmup_check() -> None:
    """When available_history is None, warmup check is skipped → eligible."""
    decision = _POLICY.evaluate(
        factor_name="some_factor",
        timeframe="1d",
        usable_observations=10,
        declared_lookback=100,
        available_history=None,
    )
    assert decision.status == EligibilityStatus.ELIGIBLE
    assert decision.warmup_sufficient is None


# ---------------------------------------------------------------------------
# Unit: atr_slope effective warmup override
# ---------------------------------------------------------------------------


def test_atr_slope_effective_warmup_is_39() -> None:
    """atr_slope has effective warmup 39, not declared lookback 20."""
    policy = FactorEligibilityPolicy()
    assert policy.effective_warmup_bars("atr_slope", 20) == 39


def test_atr_slope_39_bars_available_history_37_is_ineligible() -> None:
    """Regression: atr_slope with 39 warmup and 37 bars → insufficient warmup."""
    decision = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="1d",
        usable_observations=10,
        declared_lookback=20,
        available_history=37,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_INSUFFICIENT_WARMUP
    assert decision.effective_warmup == 39
    assert decision.available_history == 37


def test_atr_slope_effective_warmup_overridden_by_caller() -> None:
    """Caller-supplied effective_warmup_overrides take precedence."""
    policy = FactorEligibilityPolicy(
        effective_warmup_overrides=MappingProxyType({"atr_slope": 10})
    )
    assert policy.effective_warmup_bars("atr_slope", 20) == 10


def test_breakout_strength_effective_warmup_is_21() -> None:
    """breakout_strength uses lookback+1 warmup."""
    policy = FactorEligibilityPolicy()
    assert policy.effective_warmup_bars("breakout_strength", 20) == 21


# ---------------------------------------------------------------------------
# Unit: timeframe-aware evaluation
# ---------------------------------------------------------------------------


def test_timeframe_bar_milliseconds_covers_research_timeframes() -> None:
    """TIMEFRAME_BAR_MILLISECONDS has entries for all research timeframes."""
    for tf in ("5m", "15m", "1h", "4h", "1d"):
        assert tf in TIMEFRAME_BAR_MILLISECONDS
        assert TIMEFRAME_BAR_MILLISECONDS[tf] > 0


def test_eligibility_independent_per_timeframe() -> None:
    """Same factor can be eligible in one timeframe, ineligible in another."""
    decision_5m = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="5m",
        usable_observations=8000,
        declared_lookback=20,
        available_history=8000,
    )
    decision_1d = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="1d",
        usable_observations=0,
        declared_lookback=20,
        available_history=37,
    )
    assert decision_5m.status == EligibilityStatus.ELIGIBLE
    assert decision_1d.status == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS


# ---------------------------------------------------------------------------
# Unit: companion dependencies
# ---------------------------------------------------------------------------


def test_companion_dependencies_extracted_correctly() -> None:
    """Required companion columns are reported separately from OHLCV."""
    decision = _POLICY.evaluate(
        factor_name="aggressive_buy_ratio",
        timeframe="1d",
        usable_observations=0,
        required_features=("taker_buy_volume", "volume"),
    )
    assert "taker_buy_volume" in decision.companion_dependencies
    assert "volume" not in decision.companion_dependencies


def test_ohlcv_only_factor_has_no_companion_deps() -> None:
    """OHLCV-only factors report empty companion dependencies."""
    decision = _POLICY.evaluate(
        factor_name="rsi",
        timeframe="1d",
        usable_observations=200,
        required_features=("close",),
    )
    assert decision.companion_dependencies == ()
    assert decision.companion_coverage_status is None


# ---------------------------------------------------------------------------
# Unit: fail-closed legacy metadata
# ---------------------------------------------------------------------------


def test_require_eligibility_metadata_raises_when_missing() -> None:
    """Legacy Factor Selection artifacts without eligibility columns raise."""
    with pytest.raises(FactorSelectionError) as exc_info:
        require_eligibility_metadata(["factor_name", "selected", "selection_score"])
    assert exc_info.value.error_code == LEGACY_ELIGIBILITY_ERROR_CODE


def test_require_eligibility_metadata_passes_when_present() -> None:
    """Frame with all eligibility metadata columns passes without raising."""
    cols = list(ELIGIBILITY_METADATA_COLUMNS) + ["factor_name", "selected"]
    require_eligibility_metadata(cols)  # must not raise


# ---------------------------------------------------------------------------
# Unit: determinism
# ---------------------------------------------------------------------------


def test_eligibility_decisions_are_deterministic() -> None:
    """Repeated evaluation with identical inputs produces identical results."""
    kwargs = dict(
        factor_name="rsi",
        timeframe="1d",
        usable_observations=369,
        declared_lookback=14,
        available_history=17,
    )
    d1 = _POLICY.evaluate(**kwargs)  # type: ignore[arg-type]
    d2 = _POLICY.evaluate(**kwargs)  # type: ignore[arg-type]
    assert d1.status == d2.status
    assert d1.reason == d2.reason
    assert d1.effective_warmup == d2.effective_warmup


# ---------------------------------------------------------------------------
# Unit: OOS data cannot influence eligibility
# ---------------------------------------------------------------------------


def test_eligibility_uses_only_selection_window_observations() -> None:
    """Eligibility does not consume observations beyond the selection window.

    This is structural: evaluate() takes only usable_observations (a scalar
    derived from the selection-window validation metrics). No OOS frame is
    ever accepted.
    """
    import inspect

    sig = inspect.signature(_POLICY.evaluate)
    param_names = list(sig.parameters.keys())
    # Confirm no OOS parameter exists
    for banned in ("oos_observations", "oos_ic", "oos_frame", "test_frame"):
        assert banned not in param_names, f"OOS parameter '{banned}' found in evaluate()"


def test_eligibility_module_does_not_import_walk_forward() -> None:
    """Eligibility module must not import Walk-Forward or Purged-CV."""
    source = Path("src/cqros/factor_selection/eligibility.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            else:
                module = ""
            for alias in getattr(node, "names", []):
                full = f"{module}.{alias.name}" if module else alias.name
                for banned in ("walk_forward", "purged_cv", "alpha", "regime", "ml"):
                    assert banned not in full, f"Forbidden import '{full}' in eligibility.py"


# ---------------------------------------------------------------------------
# Unit: orientation metadata unchanged
# ---------------------------------------------------------------------------


def test_orientation_policy_unchanged_by_eligibility() -> None:
    """signed_ic_v1 orientation policy is present on all rows after eligibility gate."""
    vf = _val_frame(
        names=["strong", "zero_obs_factor"],
        observations=[500, 0],
        ics=[0.05, None],
    )
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)
    assert (result["orientation_policy"] == "signed_ic_v1").all()


def test_selected_direction_sign_matches_ic_sign() -> None:
    """selected_direction +1/-1 matches IC sign regardless of eligibility."""
    vf = _val_frame(
        names=["pos_ic", "neg_ic"],
        observations=[500, 300],
        ics=[0.08, -0.06],
    )
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)
    pos_dir = result.filter(pl.col("factor_name") == "pos_ic")["selected_direction"][0]
    neg_dir = result.filter(pl.col("factor_name") == "neg_ic")["selected_direction"][0]
    assert pos_dir == 1
    assert neg_dir == -1


# ---------------------------------------------------------------------------
# Unit: abs(IC) ranking unchanged for eligible factors
# ---------------------------------------------------------------------------


def test_eligible_factors_ranked_by_abs_ic() -> None:
    """Eligible factors with higher abs(IC) receive lower (better) rank."""
    vf = _val_frame(
        names=["high_ic", "low_ic"],
        observations=[500, 400],
        ics=[0.12, 0.04],
    )
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)
    rank_high = result.filter(pl.col("factor_name") == "high_ic")["selection_rank"][0]
    rank_low = result.filter(pl.col("factor_name") == "low_ic")["selection_rank"][0]
    assert rank_high < rank_low


# ---------------------------------------------------------------------------
# Unit: ineligible factors cannot be selected
# ---------------------------------------------------------------------------


def test_ineligible_factors_never_selected() -> None:
    """Ineligible factors are always REJECTED, never SELECTED."""
    vf = _val_frame(
        names=["zero", "also_zero", "eligible"],
        observations=[0, 0, 200],
        ics=[None, None, 0.05],
    )
    engine = SimpleFactorSelectionEngine(top_n=10, eligibility_policy=_POLICY)
    result = engine.build(vf)
    selected = result.filter(pl.col("selected"))
    assert selected.height == 1
    assert selected["factor_name"][0] == "eligible"
    inelig = result.filter(pl.col("factor_name").is_in(["zero", "also_zero"]))
    assert not inelig["selected"].any()


def test_ineligible_selection_reason_is_hard_ineligible() -> None:
    """Ineligible factor selection_reason is 'hard_ineligible'."""
    vf = _val_frame(names=["zero"], observations=[0])
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)
    reason = result["selection_reason"][0]
    assert reason == "hard_ineligible"


def test_no_eligibility_policy_backward_compat() -> None:
    """Engine without eligibility_policy still produces eligibility metadata columns."""
    vf = _val_frame(names=["f1", "f2"], observations=[200, 0])
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=None)
    result = engine.build(vf)
    for col in ELIGIBILITY_COLUMNS:
        assert col in result.columns, f"Missing eligibility column: {col}"
    # Without policy, zero-obs factor can still be selected (backward compat)
    # This tests the backward-compatible mode explicitly.
    zero_row = result.filter(pl.col("factor_name") == "f2")
    # zero_obs factor still receives ELIGIBLE in compat mode (no policy gate)
    assert zero_row["eligibility_status"][0] == EligibilityStatus.ELIGIBLE.value


# ---------------------------------------------------------------------------
# Regression: real 1d fixture (companion-aligned 37-bar history)
# ---------------------------------------------------------------------------


# The 1d window: 2026-06-29 → 2026-07-15 = 17 days
_START_MS: int = 1_782_691_200_000  # 2026-06-29 UTC
_END_MS: int = 1_784_073_600_000    # 2026-07-15 UTC


def _make_1d_fixture() -> pl.DataFrame:
    """Build a factor validation fixture matching the real 1d degeneration scenario.

    - Long OHLCV history (216 bars from 2026-01-01)
    - Late companion start (2026-06-29)
    - Aligned history = 17 bars in validation window
    - Lookback-20 factors have 0 usable obs
    - Lookback-39 (atr_slope) has 0 usable obs
    - Short OOS window = 17 days
    """
    factors = [
        # name, lookback, observations_in_window
        ("atr_slope", 0),            # warmup=39, companion-aligned, 0 obs
        ("aggressive_buy_ratio", 0), # warmup=20, 0 obs
        ("atr_distance", 0),         # warmup=20, 0 obs
        ("bollinger_bandwidth", 0),  # warmup=20, 0 obs
        ("breakout_strength", 0),    # warmup=21, 0 obs
        ("accumulation_distribution", 3691),  # warmup=0, all obs
        ("buy_sell_imbalance", 3691),         # warmup=0, all obs
        ("open_interest_level", 3691),        # warmup=0, all obs
        ("stochastic_k", 492),               # warmup=14, partial obs
        ("money_flow_index", 369),           # warmup=14, partial obs
        ("rate_of_change", 615),             # warmup=12, partial obs
    ]
    names = [f[0] for f in factors]
    obs = [f[1] for f in factors]
    return _val_frame(
        names=names,
        observations=obs,
        timeframe="1d",
        validation_start_ms=_START_MS,
        validation_end_ms=_END_MS,
    )


def test_1d_zero_obs_factors_are_hard_ineligible() -> None:
    """Regression: zero-obs 1d factors are classified INELIGIBLE_ZERO_OBSERVATIONS."""
    vf = _make_1d_fixture()
    engine = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=_POLICY)
    result = engine.build(vf)
    zero_factors = [
        "atr_slope", "aggressive_buy_ratio", "atr_distance",
        "bollinger_bandwidth", "breakout_strength",
    ]
    for name in zero_factors:
        row = result.filter(pl.col("factor_name") == name)
        assert not row["selected"][0], f"{name} must not be selected"
        assert (
            row["eligibility_status"][0] == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS.value
        ), f"{name} eligibility_status wrong: {row['eligibility_status'][0]}"


def test_1d_accumulation_distribution_is_eligible() -> None:
    """Regression: zero-warmup factors with full obs are ELIGIBLE."""
    vf = _make_1d_fixture()
    engine = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=_POLICY)
    result = engine.build(vf)
    row = result.filter(pl.col("factor_name") == "accumulation_distribution")
    assert row["selected"][0]
    assert row["eligibility_status"][0] == EligibilityStatus.ELIGIBLE.value


def test_1d_selected_count_reduced_after_policy() -> None:
    """Regression: fewer factors selected after zero-obs gate applied."""
    vf = _make_1d_fixture()
    # Without policy
    engine_no_policy = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=None)
    result_no_policy = engine_no_policy.build(vf)
    selected_no_policy = result_no_policy.filter(pl.col("selected")).height

    # With policy
    engine_with_policy = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=_POLICY)
    result_with_policy = engine_with_policy.build(vf)
    selected_with_policy = result_with_policy.filter(pl.col("selected")).height

    assert selected_with_policy < selected_no_policy, (
        "Policy should remove zero-obs factors from selected set"
    )
    # Verify no zero-obs factors slip through
    assert result_with_policy.filter(
        pl.col("selected") & (pl.col("usable_observations") == 0)
    ).height == 0


def test_1d_atr_slope_classified_as_zero_obs() -> None:
    """Regression: atr_slope is INELIGIBLE_ZERO_OBSERVATIONS (0 usable obs in 1d window)."""
    vf = _make_1d_fixture()
    engine = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=_POLICY)
    result = engine.build(vf)
    row = result.filter(pl.col("factor_name") == "atr_slope")
    assert row["eligibility_status"][0] == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS.value
    assert not row["selected"][0]


def test_atr_slope_warmup_39_exceeds_37_bars_is_ineligible() -> None:
    """Regression: atr_slope warmup=39 with only 37-bar history is insufficient warmup."""
    # Simulate 37-bar available history with non-zero obs (hypothetical)
    decision = _POLICY.evaluate(
        factor_name="atr_slope",
        timeframe="1d",
        usable_observations=5,  # some obs, but warmup too large
        declared_lookback=20,
        available_history=37,
    )
    assert decision.status == EligibilityStatus.INELIGIBLE_INSUFFICIENT_WARMUP
    assert decision.effective_warmup == 39
    assert decision.warmup_sufficient is False


# ---------------------------------------------------------------------------
# Integration: Factor Validation → Eligibility → Factor Selection → orientation
# ---------------------------------------------------------------------------


def test_integration_eligibility_orientation_selection() -> None:
    """Integration: eligibility gate does not disturb orientation or IC ranking.

    Pipeline:
        Factor Validation frame
        → SimpleFactorSelectionEngine (with FactorEligibilityPolicy)
        → canonical Factor Selection output
            - ineligible factors: selection_reason=hard_ineligible, selected=False
            - eligible factors: ranked by abs(IC), orientation_policy=signed_ic_v1
    """
    vf = pl.DataFrame(
        {
            "factor_name": ["best", "medium", "zero_obs"],
            "factor_version": [_FACTOR_V] * 3,
            "factor_category": ["price"] * 3,
            "timeframe": [_TIMEFRAME] * 3,
            "validation_time": [1_750_000_000_000] * 3,
            "information_coefficient": [0.15, 0.05, None],
            "rank_information_coefficient": [0.14, 0.04, None],
            "ic_information_ratio": [1.2, 0.5, None],
            "ic_p_value": [0.005, 0.05, None],
            "ic_t_stat": [4.0, 2.5, None],
            "ic_decay": [0.7, 0.6, None],
            "turnover": [0.15, 0.25, None],
            "monotonicity_score": [0.8, 0.6, None],
            "quantile_spread": [0.04, 0.02, None],
            "observations": [1000, 500, 0],
            "status": ["PASS", "PASS", "FAIL"],
        }
    )
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)

    # Schema: eligibility columns present
    for col in ELIGIBILITY_COLUMNS:
        assert col in result.columns

    # zero_obs factor is rejected
    zero_row = result.filter(pl.col("factor_name") == "zero_obs")
    assert not zero_row["selected"][0]
    assert zero_row["eligibility_status"][0] == EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS.value

    # eligible factors are selected and ranked correctly
    selected = result.filter(pl.col("selected")).sort("selection_rank")
    assert selected.height == 2
    assert selected["factor_name"][0] == "best"   # higher abs(IC)
    assert selected["factor_name"][1] == "medium"

    # orientation policy intact
    assert (result["orientation_policy"] == "signed_ic_v1").all()

    # selection_strength = abs(IC) unchanged
    best_row = result.filter(pl.col("factor_name") == "best")
    assert best_row["selection_ic"][0] == pytest.approx(0.15)
    assert best_row["selected_direction"][0] == 1  # positive IC → +1


def test_integration_eligible_factors_orientation_with_negative_ic() -> None:
    """Integration: negative-IC eligible factors get direction=-1 unchanged."""
    vf = _val_frame(names=["neg"], observations=[300], ics=[-0.08])
    engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=_POLICY)
    result = engine.build(vf)
    row = result.filter(pl.col("factor_name") == "neg")
    assert row["selected_direction"][0] == -1
    assert row["eligibility_status"][0] == EligibilityStatus.ELIGIBLE.value


def test_integration_produces_stable_schema_with_and_without_policy() -> None:
    """Integration: ELIGIBILITY_COLUMNS present regardless of policy attachment."""
    vf = _val_frame(names=["f"], observations=[200])
    for policy in (None, _POLICY):
        engine = SimpleFactorSelectionEngine(top_n=5, eligibility_policy=policy)
        result = engine.build(vf)
        for col in ELIGIBILITY_COLUMNS:
            assert col in result.columns, f"col={col} missing with policy={policy}"
