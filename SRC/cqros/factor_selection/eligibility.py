"""CQROS Factor Eligibility Policy.

Purpose:
    Define a versioned, leakage-safe policy that classifies each factor
    candidate as ELIGIBLE or HARD INELIGIBLE before Factor Selection ranking.

Responsibilities:
    - Define ``FACTOR_ELIGIBILITY_POLICY`` version identifier
    - Define ``EligibilityStatus`` machine-readable status codes
    - Define ``EligibilityDecision`` as an immutable result per factor
    - Define ``FactorEligibilityPolicy`` that evaluates candidates from
      Factor Validation metrics (selection-window data only)
    - Hard-reject factors with zero usable observations
    - Hard-reject factors whose required warmup exceeds available aligned history
    - Hard-reject factors with missing eligibility metadata when fail-closed
      mode is active
    - Expose ``TIMEFRAME_BAR_MILLISECONDS`` for time-equivalent logging
    - Remain free of OOS data, Walk-Forward, Purged-CV, Alpha, Regime, and
      ML imports

Dependencies:
    ``polars``, ``cqros.factor_selection.exceptions``,
    ``cqros.factor_selection.schema``.

Public API:
    ``FACTOR_ELIGIBILITY_POLICY``, ``ELIGIBILITY_POLICY_VERSION``,
    ``LEGACY_ELIGIBILITY_ERROR_CODE``, ``EligibilityStatus``,
    ``EligibilityDecision``, ``FactorEligibilityPolicy``,
    ``TIMEFRAME_BAR_MILLISECONDS``, ``evaluate_eligibility``,
    ``require_eligibility_metadata``

Notes:
    Eligibility is evaluated exclusively from Factor Validation metrics
    produced inside the selection window. OOS rows must never be passed in.

    The policy is fail-closed for missing metadata: if an artifact is loaded
    that predates the eligibility schema, ``require_eligibility_metadata``
    raises ``FactorSelectionError`` with code
    ``FACTOR_SELECTION_LEGACY_ELIGIBILITY``.

    ``atr_slope`` (lookback=20, effective_warmup=39) with only 37 aligned 1d
    bars will produce ``INELIGIBLE_INSUFFICIENT_WARMUP`` because
    ``required_lookback > available_history``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

__all__ = [
    "ELIGIBILITY_METADATA_COLUMNS",
    "ELIGIBILITY_POLICY_VERSION",
    "FACTOR_ELIGIBILITY_POLICY",
    "LEGACY_ELIGIBILITY_ERROR_CODE",
    "TIMEFRAME_BAR_MILLISECONDS",
    "EligibilityDecision",
    "EligibilityStatus",
    "FactorEligibilityPolicy",
    "evaluate_eligibility",
    "require_eligibility_metadata",
]

# Versioned policy identifier persisted on every Factor Selection row.
FACTOR_ELIGIBILITY_POLICY: Final[str] = "coverage_v1"
ELIGIBILITY_POLICY_VERSION: Final[str] = FACTOR_ELIGIBILITY_POLICY

# Error code emitted when a pre-eligibility Factor Selection artifact is loaded.
LEGACY_ELIGIBILITY_ERROR_CODE: Final[str] = "FACTOR_SELECTION_LEGACY_ELIGIBILITY"

# Milliseconds per bar for each supported research timeframe.
# Used for time-equivalent logging only; not for eligibility gating.
TIMEFRAME_BAR_MILLISECONDS: Final[MappingProxyType[str, int]] = MappingProxyType(
    {
        "1s": 1_000,
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }
)

# Columns that must be present on a Factor Selection artifact produced by this policy.
ELIGIBILITY_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "eligibility_status",
    "eligibility_reason",
    "eligibility_policy",
)


class EligibilityStatus(str, Enum):  # noqa: UP042
    """Machine-readable factor eligibility classification.

    Attributes:
        ELIGIBLE: Factor passed all eligibility criteria and may be ranked.
        INELIGIBLE_ZERO_OBSERVATIONS: Factor has zero usable (factor, label)
            pairs in the selection window. Hard ineligible.
        INELIGIBLE_LOW_COVERAGE: Factor coverage ratio is below the policy
            minimum. Currently unused (thresholds are evidence-based); reserved
            for future policy versions.
        INELIGIBLE_INSUFFICIENT_WARMUP: Factor required warmup exceeds the
            available aligned history for its timeframe. Hard ineligible.
        INELIGIBLE_COMPANION_HISTORY: Factor input requires companion columns
            (OI, taker, funding, long-short) but companion history is too
            short. Hard ineligible.
        INELIGIBLE_MISSING_METADATA: Factor metadata is missing or incomplete;
            eligibility cannot be determined. Fail-closed treatment applies.
    """

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_ZERO_OBSERVATIONS = "INELIGIBLE_ZERO_OBSERVATIONS"
    INELIGIBLE_LOW_COVERAGE = "INELIGIBLE_LOW_COVERAGE"
    INELIGIBLE_INSUFFICIENT_WARMUP = "INELIGIBLE_INSUFFICIENT_WARMUP"
    INELIGIBLE_COMPANION_HISTORY = "INELIGIBLE_COMPANION_HISTORY"
    INELIGIBLE_MISSING_METADATA = "INELIGIBLE_MISSING_METADATA"


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Immutable eligibility decision for a single factor candidate.

    All fields are derived from selection-window data only. OOS observations
    must never appear in any of these fields.

    Attributes:
        factor_name: Factor identifier.
        timeframe: Bar interval of the selection window.
        status: Machine-readable eligibility classification.
        reason: Human-readable explanation of the decision.
        usable_observations: Count of valid (factor, label) pairs in the
            selection window.
        total_observations: Total rows in the selection-window factor panel
            (null + non-null). None when not available from validation data.
        coverage_ratio: ``usable_observations / total_observations`` or None
            when total is not available or zero.
        null_rate: ``1 - coverage_ratio`` or None.
        required_lookback: Declared factor lookback (bars).
        effective_warmup: Effective warmup bars before the first non-null
            value is produced (may exceed ``required_lookback`` for composite
            windows such as ATR slope). None when not known.
        available_history: Post-companion-alignment history available to the
            factor (bars) for this selection window. None when not available.
        warmup_sufficient: True when effective warmup <= available history.
            None when either is unknown.
        companion_dependencies: Tuple of companion input column names this
            factor requires (beyond OHLCV).
        companion_coverage_status: Short human-readable description of
            companion availability. None when OHLCV-only.
        policy_version: Policy identifier that produced this decision.
    """

    factor_name: str
    timeframe: str
    status: EligibilityStatus
    reason: str
    usable_observations: int
    total_observations: int | None
    coverage_ratio: float | None
    null_rate: float | None
    required_lookback: int
    effective_warmup: int | None
    available_history: int | None
    warmup_sufficient: bool | None
    companion_dependencies: tuple[str, ...]
    companion_coverage_status: str | None
    policy_version: str

    @property
    def is_eligible(self) -> bool:
        """Return True when the factor may enter Factor Selection ranking."""
        return self.status == EligibilityStatus.ELIGIBLE


# Companion columns that are not OHLCV baseline.
_COMPANION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "funding_rate",
        "mark_price",
        "open_interest",
        "taker_buy_volume",
        "taker_sell_volume",
        "long_short_ratio",
    }
)

# Known effective warmup overrides (factor_name → effective_warmup_bars).
# Mirrors the _FACTOR_SPECS catalog in factor_stability_1d_degeneration.py
# but is maintained here as the runtime authority for selection gating.
# Key: factor_name; Value: effective warmup in bars.
_EFFECTIVE_WARMUP_OVERRIDES: Final[MappingProxyType[str, int]] = MappingProxyType(
    {
        # ATR then rolling OLS: 2*lookback - 1
        "atr_slope": 39,
        # breakout_strength uses shift(lookback)+1
        "breakout_strength": 21,
        # on_balance_volume and price_volume_trend need 1 prior bar
        "on_balance_volume": 1,
        "price_volume_trend": 1,
    }
)


@dataclass(frozen=True, slots=True)
class FactorEligibilityPolicy:
    """Versioned, leakage-safe factor eligibility policy.

    Evaluates each factor candidate against the following criteria, in order:

    A. Zero-observation hard block: if ``usable_observations == 0`` the factor
       is immediately ``INELIGIBLE_ZERO_OBSERVATIONS``.

    B. Warmup check: if the factor's effective warmup bars exceed the
       available aligned history bars, the factor is
       ``INELIGIBLE_INSUFFICIENT_WARMUP``.  This correctly classifies
       ``atr_slope`` (warmup 39, available 1d history 37) as ineligible.

    C. All checks passed: factor is ``ELIGIBLE``.

    Coverage-ratio gating (INELIGIBLE_LOW_COVERAGE) is reserved for a future
    policy version once the distribution across all timeframes is stable. The
    current data shows 5m/15m/1h/4h candidates all have substantial
    observations (min obs ≥ 98 for 4h, ≥ 601 for 1h); only 1d has the
    degenerate distribution. A threshold chosen without evidence would be
    arbitrary. The zero-obs and warmup gates are both evidence-backed.

    Attributes:
        policy_version: Identifier of the policy. Defaults to
            ``FACTOR_ELIGIBILITY_POLICY``.
        effective_warmup_overrides: Optional mapping of factor_name to
            effective warmup bars, supplementing or overriding the built-in
            catalog. Used primarily for testing.
    """

    policy_version: str = FACTOR_ELIGIBILITY_POLICY
    effective_warmup_overrides: MappingProxyType[str, int] | None = None

    def effective_warmup_bars(self, factor_name: str, declared_lookback: int) -> int:
        """Return effective warmup bars for ``factor_name``.

        Checks caller-supplied overrides first, then built-in catalog, then
        falls back to ``declared_lookback``.

        Args:
            factor_name: Factor identifier.
            declared_lookback: Declared lookback from factor metadata.

        Returns:
            Effective warmup bars before the first non-null value is produced.
        """
        if self.effective_warmup_overrides is not None:
            override = self.effective_warmup_overrides.get(factor_name)
            if override is not None:
                return int(override)
        builtin = _EFFECTIVE_WARMUP_OVERRIDES.get(factor_name)
        if builtin is not None:
            return builtin
        return max(0, int(declared_lookback))

    def evaluate(
        self,
        *,
        factor_name: str,
        timeframe: str,
        usable_observations: int,
        total_observations: int | None = None,
        declared_lookback: int = 0,
        available_history: int | None = None,
        required_features: tuple[str, ...] = (),
    ) -> EligibilityDecision:
        """Evaluate one factor candidate for eligibility.

        All inputs must derive from selection-window data only.
        OOS observations must never be passed in.

        Args:
            factor_name: Factor identifier.
            timeframe: Bar interval.
            usable_observations: Valid (factor, label) pairs in the selection
                window. Must be >= 0.
            total_observations: Total observation rows (null + non-null).
                None when not available.
            declared_lookback: Lookback from factor metadata (bars). 0 when
                unknown.
            available_history: Post-companion-alignment history (bars).
                None when not available.
            required_features: Input column names from factor metadata.

        Returns:
            An ``EligibilityDecision`` describing eligibility and reason.
        """
        coverage_ratio, null_rate = _compute_coverage(usable_observations, total_observations)
        effective_warmup = self.effective_warmup_bars(factor_name, declared_lookback)
        warmup_sufficient = _compute_warmup_sufficient(effective_warmup, available_history)
        companion_deps = _companion_deps(required_features)
        companion_status = _companion_status(companion_deps)

        # A: Zero-observation hard block
        if usable_observations == 0:
            return EligibilityDecision(
                factor_name=factor_name,
                timeframe=timeframe,
                status=EligibilityStatus.INELIGIBLE_ZERO_OBSERVATIONS,
                reason=(
                    f"usable_observations=0; no valid (factor, label) pairs in "
                    f"the selection window for timeframe={timeframe}"
                ),
                usable_observations=usable_observations,
                total_observations=total_observations,
                coverage_ratio=coverage_ratio,
                null_rate=null_rate,
                required_lookback=declared_lookback,
                effective_warmup=effective_warmup,
                available_history=available_history,
                warmup_sufficient=warmup_sufficient,
                companion_dependencies=companion_deps,
                companion_coverage_status=companion_status,
                policy_version=self.policy_version,
            )

        # B: Warmup check (only when available_history is known)
        if warmup_sufficient is False:
            return EligibilityDecision(
                factor_name=factor_name,
                timeframe=timeframe,
                status=EligibilityStatus.INELIGIBLE_INSUFFICIENT_WARMUP,
                reason=(
                    f"effective_warmup={effective_warmup} bars exceeds "
                    f"available_history={available_history} bars for "
                    f"timeframe={timeframe}; factor cannot produce non-null "
                    f"values inside the selection window"
                ),
                usable_observations=usable_observations,
                total_observations=total_observations,
                coverage_ratio=coverage_ratio,
                null_rate=null_rate,
                required_lookback=declared_lookback,
                effective_warmup=effective_warmup,
                available_history=available_history,
                warmup_sufficient=False,
                companion_dependencies=companion_deps,
                companion_coverage_status=companion_status,
                policy_version=self.policy_version,
            )

        # Eligible
        return EligibilityDecision(
            factor_name=factor_name,
            timeframe=timeframe,
            status=EligibilityStatus.ELIGIBLE,
            reason="passed all eligibility criteria",
            usable_observations=usable_observations,
            total_observations=total_observations,
            coverage_ratio=coverage_ratio,
            null_rate=null_rate,
            required_lookback=declared_lookback,
            effective_warmup=effective_warmup,
            available_history=available_history,
            warmup_sufficient=warmup_sufficient,
            companion_dependencies=companion_deps,
            companion_coverage_status=companion_status,
            policy_version=self.policy_version,
        )


def evaluate_eligibility(
    *,
    factor_name: str,
    timeframe: str,
    usable_observations: int,
    total_observations: int | None = None,
    declared_lookback: int = 0,
    available_history: int | None = None,
    required_features: tuple[str, ...] = (),
    policy: FactorEligibilityPolicy | None = None,
) -> EligibilityDecision:
    """Convenience wrapper: evaluate eligibility with a default policy.

    Args:
        factor_name: Factor identifier.
        timeframe: Bar interval.
        usable_observations: Valid (factor, label) pairs.
        total_observations: Total rows (null + non-null).
        declared_lookback: Lookback from factor metadata.
        available_history: Post-companion-alignment history (bars).
        required_features: Factor input column names.
        policy: Optional policy instance. Defaults to ``FactorEligibilityPolicy()``.

    Returns:
        ``EligibilityDecision`` for the factor.
    """
    resolved = policy if policy is not None else FactorEligibilityPolicy()
    return resolved.evaluate(
        factor_name=factor_name,
        timeframe=timeframe,
        usable_observations=usable_observations,
        total_observations=total_observations,
        declared_lookback=declared_lookback,
        available_history=available_history,
        required_features=required_features,
    )


def require_eligibility_metadata(columns: list[str] | tuple[str, ...]) -> None:
    """Raise when a Factor Selection artifact predates the eligibility policy.

    Legacy artifacts without eligibility metadata must be regenerated. This
    function never silently assumes ``ELIGIBLE``.

    Args:
        columns: Column names present on the loaded Factor Selection frame.

    Raises:
        FactorSelectionError: If any eligibility metadata column is missing.
    """
    from cqros.factor_selection.exceptions import (
        FactorSelectionError,  # local import to avoid cycles
    )

    present = set(columns)
    missing = [col for col in ELIGIBILITY_METADATA_COLUMNS if col not in present]
    if missing:
        raise FactorSelectionError(
            "Factor Selection artifact predates eligibility policy; regenerate "
            "Factor Selection to attach eligibility metadata",
            error_code=LEGACY_ELIGIBILITY_ERROR_CODE,
            details={
                "missing_columns": tuple(missing),
                "required_columns": ELIGIBILITY_METADATA_COLUMNS,
                "policy_version": FACTOR_ELIGIBILITY_POLICY,
                "legacy_behavior": "regeneration_required",
            },
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_coverage(
    usable_observations: int,
    total_observations: int | None,
) -> tuple[float | None, float | None]:
    """Return (coverage_ratio, null_rate) or (None, None) when total is unavailable."""
    if total_observations is None or total_observations == 0:
        return None, None
    ratio = usable_observations / total_observations
    return ratio, 1.0 - ratio


def _compute_warmup_sufficient(
    effective_warmup: int,
    available_history: int | None,
) -> bool | None:
    """Return warmup sufficiency or None when available_history is unknown."""
    if available_history is None:
        return None
    return effective_warmup <= available_history


def _companion_deps(required_features: tuple[str, ...]) -> tuple[str, ...]:
    """Return companion column names required by the factor (non-OHLCV)."""
    return tuple(f for f in required_features if f in _COMPANION_COLUMNS)


def _companion_status(companion_deps: tuple[str, ...]) -> str | None:
    """Return a short companion coverage description or None for OHLCV-only."""
    if not companion_deps:
        return None
    return f"requires_companion:{','.join(sorted(companion_deps))}"
