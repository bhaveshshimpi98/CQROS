"""CQROS factor-specific input partitioning.

Purpose:
    Align factor inputs to each factor's declared ``required_features`` so
    OHLCV-only factors are not truncated by unrelated companion availability.

Responsibilities:
    - Classify factors from static ``required_features`` declarations
    - Identify required companion datasets for a factor
    - Align an already-joined factor-input frame to only the companions
      required by that factor
    - Fail loudly on invalid dependency declarations for raw factor-input
      features
    - Remain free of factor formulas, labels, OOS evaluation, predictions,
      signals, regime logic, forward-fill, and backfill

Dependencies:
    ``polars``, ``cqros.factors.exceptions``.

Public API:
    Dependency-class and dataset constants,
    ``FactorInputPartition``,
    ``classify_dependency_class``,
    ``required_companion_columns``,
    ``required_datasets``.

Notes:
    Companion joins remain causal ``join_asof(backward)`` at load time. This
    module only drops leading rows that lack the factor's required companion
    values. It never fills missing history and never consults evaluation
    outcomes to choose dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import polars as pl

from cqros.factors.exceptions import FactorValidationError

__all__ = [
    "CLASS_FUNDING_DEPENDENT",
    "CLASS_LONG_SHORT_DEPENDENT",
    "CLASS_MULTI_COMPANION_DEPENDENT",
    "CLASS_OHLCV_ONLY",
    "CLASS_OHLCV_PLUS_VOLUME",
    "CLASS_OI_DEPENDENT",
    "CLASS_TAKER_DEPENDENT",
    "CLASS_UNKNOWN",
    "DATASET_FUNDING",
    "DATASET_LONG_SHORT",
    "DATASET_OHLCV",
    "DATASET_OPEN_INTEREST",
    "DATASET_TAKER_VOLUME",
    "FactorInputPartition",
    "KNOWN_FACTOR_INPUT_FEATURES",
    "OHLCV_FEATURES",
    "VOLUME_FEATURES",
    "classify_dependency_class",
    "required_companion_columns",
    "required_datasets",
]

_logger = logging.getLogger(__name__)

CLASS_OHLCV_ONLY: Final[str] = "OHLCV_ONLY"
CLASS_OHLCV_PLUS_VOLUME: Final[str] = "OHLCV_PLUS_VOLUME"
CLASS_OI_DEPENDENT: Final[str] = "OI_DEPENDENT"
CLASS_TAKER_DEPENDENT: Final[str] = "TAKER_DEPENDENT"
CLASS_LONG_SHORT_DEPENDENT: Final[str] = "LONG_SHORT_DEPENDENT"
CLASS_FUNDING_DEPENDENT: Final[str] = "FUNDING_DEPENDENT"
CLASS_MULTI_COMPANION_DEPENDENT: Final[str] = "MULTI_COMPANION_DEPENDENT"
CLASS_UNKNOWN: Final[str] = "UNKNOWN"

DATASET_OHLCV: Final[str] = "ohlcv"
DATASET_FUNDING: Final[str] = "funding"
DATASET_OPEN_INTEREST: Final[str] = "open_interest"
DATASET_TAKER_VOLUME: Final[str] = "taker_volume"
DATASET_LONG_SHORT: Final[str] = "global_long_short_account_ratio"

OHLCV_FEATURES: Final[frozenset[str]] = frozenset({"open", "high", "low", "close", "trade_count"})
VOLUME_FEATURES: Final[frozenset[str]] = frozenset({"volume"})
_OI_FEATURES: Final[frozenset[str]] = frozenset({"open_interest"})
_TAKER_FEATURES: Final[frozenset[str]] = frozenset({"taker_buy_volume", "taker_sell_volume"})
_LONG_SHORT_FEATURES: Final[frozenset[str]] = frozenset({"long_short_ratio"})
_FUNDING_FEATURES: Final[frozenset[str]] = frozenset({"funding_rate", "mark_price"})
_COMPANION_FEATURES: Final[frozenset[str]] = (
    _OI_FEATURES | _TAKER_FEATURES | _LONG_SHORT_FEATURES | _FUNDING_FEATURES
)
KNOWN_FACTOR_INPUT_FEATURES: Final[frozenset[str]] = (
    OHLCV_FEATURES | VOLUME_FEATURES | _COMPANION_FEATURES
)

_FEATURE_TO_DATASET: Final[Mapping[str, str]] = MappingProxyType(
    {
        "open": DATASET_OHLCV,
        "high": DATASET_OHLCV,
        "low": DATASET_OHLCV,
        "close": DATASET_OHLCV,
        "volume": DATASET_OHLCV,
        "trade_count": DATASET_OHLCV,
        "funding_rate": DATASET_FUNDING,
        "mark_price": DATASET_FUNDING,
        "open_interest": DATASET_OPEN_INTEREST,
        "taker_buy_volume": DATASET_TAKER_VOLUME,
        "taker_sell_volume": DATASET_TAKER_VOLUME,
        "long_short_ratio": DATASET_LONG_SHORT,
    }
)

_ERROR_INVALID_FEATURE: Final[str] = "FACTOR-INPUT-PART-001"
_ERROR_UNKNOWN_FEATURE: Final[str] = "FACTOR-INPUT-PART-002"
_ERROR_MISSING_COLUMNS: Final[str] = "FACTOR-INPUT-PART-003"
_ERROR_NO_COMPLETE_ROWS: Final[str] = "FACTOR-INPUT-PART-004"

_PRIMARY_ALIGN_INDEX: Final[str] = "_cqros_factor_align_idx"


def classify_dependency_class(required_features: Sequence[str]) -> str:
    """Classify a factor from inspected ``required_features`` only.

    Args:
        required_features: Declared factor input feature names.

    Returns:
        One dependency-class constant (for example ``OHLCV_ONLY``).
    """
    required = set(required_features)
    has_ohlcv = bool(required & OHLCV_FEATURES)
    has_volume = bool(required & VOLUME_FEATURES)
    has_oi = bool(required & _OI_FEATURES)
    has_taker = bool(required & _TAKER_FEATURES)
    has_ls = bool(required & _LONG_SHORT_FEATURES)
    has_funding = bool(required & _FUNDING_FEATURES)
    other = sorted(required - KNOWN_FACTOR_INPUT_FEATURES)
    companion_classes = sum([has_oi, has_taker, has_ls, has_funding])
    if other:
        if companion_classes >= 1:
            return CLASS_MULTI_COMPANION_DEPENDENT
        return CLASS_UNKNOWN
    if companion_classes > 1:
        return CLASS_MULTI_COMPANION_DEPENDENT
    if has_oi:
        return CLASS_OI_DEPENDENT
    if has_taker:
        return CLASS_TAKER_DEPENDENT
    if has_ls:
        return CLASS_LONG_SHORT_DEPENDENT
    if has_funding:
        return CLASS_FUNDING_DEPENDENT
    if has_volume:
        return CLASS_OHLCV_PLUS_VOLUME
    if has_ohlcv or required:
        return CLASS_OHLCV_ONLY
    return CLASS_UNKNOWN


def required_companion_columns(required_features: Sequence[str]) -> tuple[str, ...]:
    """Return companion columns required by ``required_features`` in stable order."""
    return tuple(sorted(feature for feature in required_features if feature in _COMPANION_FEATURES))


def required_datasets(required_features: Sequence[str]) -> tuple[str, ...]:
    """Return distinct source datasets required by ``required_features``.

    OHLCV is always included as the bar timeline because factor generation is
    keyed by OHLCV ``open_time``.

    Raises:
        FactorValidationError: If any feature is not a known raw factor-input
            feature (no silent fallback to all datasets).
    """
    datasets: list[str] = []
    seen: set[str] = set()
    for feature in required_features:
        if feature not in _FEATURE_TO_DATASET:
            raise FactorValidationError(
                f"unknown factor input feature for dataset mapping: {feature}",
                error_code=_ERROR_UNKNOWN_FEATURE,
                details={
                    "feature": feature,
                    "required_features": tuple(required_features),
                    "known_features": tuple(sorted(KNOWN_FACTOR_INPUT_FEATURES)),
                },
            )
        dataset = _FEATURE_TO_DATASET[feature]
        if dataset not in seen:
            seen.add(dataset)
            datasets.append(dataset)
    if not datasets:
        return (DATASET_OHLCV,)
    if DATASET_OHLCV not in seen:
        return (DATASET_OHLCV, *datasets)
    remainder = [name for name in datasets if name != DATASET_OHLCV]
    return (DATASET_OHLCV, *remainder)


class FactorInputPartition:
    """Build factor-ready input frames from declared dependencies only.

    This abstraction does not compute factors, infer dependencies from OOS
    results, fill missing history, or inspect labels/predictions/signals.
    """

    __slots__ = ("_logger",)

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the partition helper.

        Args:
            logger: Optional logger. Defaults to the module logger.
        """
        self._logger = logger if logger is not None else _logger

    def dependency_class(self, required_features: Sequence[str]) -> str:
        """Return the dependency class for ``required_features``."""
        return classify_dependency_class(required_features)

    def required_companion_columns(self, required_features: Sequence[str]) -> tuple[str, ...]:
        """Return companion columns required by ``required_features``."""
        return required_companion_columns(required_features)

    def required_datasets(self, required_features: Sequence[str]) -> tuple[str, ...]:
        """Return source datasets required by ``required_features``."""
        return required_datasets(required_features)

    def validate_required_features(
        self,
        required_features: Sequence[str],
        *,
        allow_non_raw: bool = False,
    ) -> None:
        """Validate a factor dependency declaration.

        Args:
            required_features: Declared factor input feature names.
            allow_non_raw: When ``False`` (default for production raw-input
                factors), every feature must be in
                ``KNOWN_FACTOR_INPUT_FEATURES``. When ``True``, non-raw
                research features are permitted but still never expand
                alignment to unrelated companions.

        Raises:
            FactorValidationError: If any feature name is empty/blank or, when
                ``allow_non_raw`` is ``False``, unknown to the raw factor-input
                catalog.
        """
        normalized = tuple(required_features)
        invalid = tuple(feature for feature in normalized if feature.strip() == "")
        if invalid:
            raise FactorValidationError(
                "factor required_features contains invalid feature names",
                error_code=_ERROR_INVALID_FEATURE,
                details={
                    "invalid_features": invalid,
                    "required_features": normalized,
                },
            )
        if allow_non_raw:
            return
        unknown = tuple(
            sorted(feature for feature in normalized if feature not in KNOWN_FACTOR_INPUT_FEATURES)
        )
        if unknown:
            raise FactorValidationError(
                "factor required_features contains unknown raw-input features",
                error_code=_ERROR_UNKNOWN_FEATURE,
                details={
                    "unknown_features": unknown,
                    "required_features": normalized,
                    "known_features": tuple(sorted(KNOWN_FACTOR_INPUT_FEATURES)),
                },
            )

    def align_frame(
        self,
        frame: pl.DataFrame,
        required_features: Sequence[str],
        *,
        allow_non_raw: bool = False,
    ) -> pl.DataFrame:
        """Return ``frame`` sliced to the first bar with required inputs present.

        Leading rows are dropped only when a required companion (or, for
        allowed non-raw research features, a required non-OHLCV feature) is
        null. Unrelated companions never participate. Values are never filled.

        Args:
            frame: Joined factor-input frame sorted by open time.
            required_features: Declared factor dependencies.
            allow_non_raw: Forwarded to ``validate_required_features``.

        Returns:
            Factor-ready frame retaining the earliest timestamp permitted by
            the factor's true dependencies.

        Raises:
            FactorValidationError: If validation fails, required columns are
                missing, or no complete row exists for required companions.
        """
        self.validate_required_features(required_features, allow_non_raw=allow_non_raw)
        alignment_columns = self._alignment_columns(required_features, allow_non_raw=allow_non_raw)
        missing = tuple(column for column in alignment_columns if column not in frame.columns)
        if missing:
            raise FactorValidationError(
                "factor input frame missing required alignment columns",
                error_code=_ERROR_MISSING_COLUMNS,
                details={
                    "missing_columns": missing,
                    "alignment_columns": alignment_columns,
                    "available_columns": tuple(frame.columns),
                    "required_features": tuple(required_features),
                },
            )
        if not alignment_columns or frame.height == 0:
            return frame

        complete_mask = pl.all_horizontal(
            *(pl.col(name).is_not_null() for name in alignment_columns)
        )
        indexed = frame.with_row_index(_PRIMARY_ALIGN_INDEX)
        first_complete = indexed.filter(complete_mask).select(_PRIMARY_ALIGN_INDEX).head(1)
        if first_complete.height == 0:
            raise FactorValidationError(
                "no factor-input rows have complete coverage for required features",
                error_code=_ERROR_NO_COMPLETE_ROWS,
                details={
                    "alignment_columns": alignment_columns,
                    "required_features": tuple(required_features),
                    "row_count": frame.height,
                    "dependency_class": classify_dependency_class(required_features),
                },
            )
        start_index = int(first_complete.item())
        if start_index == 0:
            return frame
        self._logger.debug(
            "Factor-specific input alignment applied",
            extra={
                "required_features": tuple(required_features),
                "alignment_columns": alignment_columns,
                "start_index": start_index,
                "dependency_class": classify_dependency_class(required_features),
            },
        )
        return frame.slice(start_index)

    def _alignment_columns(
        self,
        required_features: Sequence[str],
        *,
        allow_non_raw: bool,
    ) -> tuple[str, ...]:
        """Return columns that gate leading-row alignment for a factor."""
        companions = required_companion_columns(required_features)
        if not allow_non_raw:
            return companions
        extras = tuple(
            sorted(
                feature
                for feature in required_features
                if feature not in OHLCV_FEATURES
                and feature not in VOLUME_FEATURES
                and feature not in _COMPANION_FEATURES
            )
        )
        return tuple(sorted({*companions, *extras}))
