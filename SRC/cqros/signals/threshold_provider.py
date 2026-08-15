"""CQROS regression signal threshold provider.

Purpose:
    Supply BUY/SELL regression thresholds for signal generation without
    estimating, calibrating, or reading prediction datasets.

Responsibilities:
    - Define ``RegressionThresholds`` as the immutable threshold contract
    - Define ``ThresholdProvider`` as the read-only lookup contract
    - Provide an in-memory provider with deterministic override resolution
    - Provide a repository-backed provider with global-default fallback
    - Remain free of signal generation, calibration, and trading

Dependencies:
    ``dataclasses``, ``typing``, ``cqros.signals.exceptions``, and
    ``cqros.storage.threshold_repository``.

Public API:
    ``RegressionThresholds``, ``ThresholdProvider``,
    ``InMemoryThresholdProvider``, ``RepositoryThresholdProvider``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from cqros.signals.exceptions import SignalValidationError
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.storage.threshold_repository import ThresholdRepository

__all__ = [
    "InMemoryThresholdProvider",
    "RegressionThresholds",
    "RepositoryThresholdProvider",
    "ThresholdProvider",
]

_ERROR_NAME_BLANK: Final[str] = "SIGNAL-THR-001"
_ERROR_THRESHOLDS_TYPE: Final[str] = "SIGNAL-THR-002"
_ERROR_REPOSITORY_TYPE: Final[str] = "SIGNAL-THR-003"
_ERROR_PROFILE_BLANK: Final[str] = "SIGNAL-THR-004"

_DEFAULT_PROFILE: Final[str] = "Balanced"


@dataclass(frozen=True, slots=True)
class RegressionThresholds:
    """Immutable BUY/SELL thresholds for regression signal generation.

    Attributes:
        buy_threshold: Inclusive lower bound for ``BUY`` signals.
        sell_threshold: Inclusive upper bound for ``SELL`` signals.
    """

    buy_threshold: float
    sell_threshold: float


@runtime_checkable
class ThresholdProvider(Protocol):
    """Read-only contract for supplying regression thresholds per partition.

    Implementations must not estimate thresholds, compute percentiles, or read
    prediction datasets. Callers validate threshold ordering and finiteness.
    """

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        """Return regression thresholds for one partition identity.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval.
            model_name: Stable model identifier.
            model_version: Model version identifier.

        Returns:
            Immutable ``RegressionThresholds`` for the requested partition.
        """
        ...


class InMemoryThresholdProvider:
    """In-memory ``ThresholdProvider`` with layered override resolution.

    Resolution order (first match wins):

    1. ``symbol + timeframe + model_name + model_version``
    2. ``symbol + timeframe``
    3. ``symbol``
    4. ``model_name + model_version``
    5. configured global defaults

    Args:
        global_thresholds: Default thresholds when no override matches.
        symbol_overrides: Optional per-symbol thresholds.
        symbol_timeframe_overrides: Optional per-symbol/timeframe thresholds.
        model_overrides: Optional per-model-name/version thresholds.
        partition_overrides: Optional full partition-key thresholds.

    Notes:
        Lookup is read-only. Overrides are fixed at construction and never
        mutated by ``get_thresholds``.
    """

    __slots__ = (
        "_global_thresholds",
        "_model_overrides",
        "_partition_overrides",
        "_symbol_overrides",
        "_symbol_timeframe_overrides",
    )

    _global_thresholds: RegressionThresholds
    _symbol_overrides: dict[str, RegressionThresholds]
    _symbol_timeframe_overrides: dict[tuple[str, str], RegressionThresholds]
    _model_overrides: dict[tuple[str, str], RegressionThresholds]
    _partition_overrides: dict[tuple[str, str, str, str], RegressionThresholds]

    def __init__(
        self,
        *,
        global_thresholds: RegressionThresholds,
        symbol_overrides: Mapping[str, RegressionThresholds] | None = None,
        symbol_timeframe_overrides: Mapping[tuple[str, str], RegressionThresholds] | None = None,
        model_overrides: Mapping[tuple[str, str], RegressionThresholds] | None = None,
        partition_overrides: Mapping[tuple[str, str, str, str], RegressionThresholds] | None = None,
    ) -> None:
        """Initialize global defaults and optional override tables.

        Args:
            global_thresholds: Default thresholds when no override matches.
            symbol_overrides: Optional per-symbol thresholds.
            symbol_timeframe_overrides: Optional per-symbol/timeframe
                thresholds.
            model_overrides: Optional per-model-name/version thresholds.
            partition_overrides: Optional full partition-key thresholds.

        Raises:
            SignalValidationError: If ``global_thresholds`` is not a
                ``RegressionThresholds`` instance or any override key is blank.
        """
        self._global_thresholds = _require_thresholds(global_thresholds)
        self._symbol_overrides = {
            _require_name(symbol, parameter="symbol"): _require_thresholds(thresholds)
            for symbol, thresholds in (symbol_overrides or {}).items()
        }
        self._symbol_timeframe_overrides = {
            (
                _require_name(symbol, parameter="symbol"),
                _require_name(timeframe, parameter="timeframe"),
            ): _require_thresholds(thresholds)
            for (symbol, timeframe), thresholds in (symbol_timeframe_overrides or {}).items()
        }
        self._model_overrides = {
            (
                _require_name(model_name, parameter="model_name"),
                _require_name(model_version, parameter="model_version"),
            ): _require_thresholds(thresholds)
            for (model_name, model_version), thresholds in (model_overrides or {}).items()
        }
        self._partition_overrides = {
            (
                _require_name(symbol, parameter="symbol"),
                _require_name(timeframe, parameter="timeframe"),
                _require_name(model_name, parameter="model_name"),
                _require_name(model_version, parameter="model_version"),
            ): _require_thresholds(thresholds)
            for (
                symbol,
                timeframe,
                model_name,
                model_version,
            ), thresholds in (partition_overrides or {}).items()
        }

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        """Resolve thresholds using the documented override precedence.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval.
            model_name: Stable model identifier.
            model_version: Model version identifier.

        Returns:
            Matching override thresholds, or configured global defaults.

        Raises:
            SignalValidationError: If any lookup key is blank.
        """
        validated_symbol = _require_name(symbol, parameter="symbol")
        validated_timeframe = _require_name(timeframe, parameter="timeframe")
        validated_model_name = _require_name(model_name, parameter="model_name")
        validated_model_version = _require_name(model_version, parameter="model_version")

        partition_key = (
            validated_symbol,
            validated_timeframe,
            validated_model_name,
            validated_model_version,
        )
        if partition_key in self._partition_overrides:
            return self._partition_overrides[partition_key]

        symbol_timeframe_key = (validated_symbol, validated_timeframe)
        if symbol_timeframe_key in self._symbol_timeframe_overrides:
            return self._symbol_timeframe_overrides[symbol_timeframe_key]

        if validated_symbol in self._symbol_overrides:
            return self._symbol_overrides[validated_symbol]

        model_key = (validated_model_name, validated_model_version)
        if model_key in self._model_overrides:
            return self._model_overrides[model_key]

        return self._global_thresholds


class RepositoryThresholdProvider:
    """``ThresholdProvider`` backed by ``ThresholdRepository`` with fallback.

    Looks up production-approved thresholds for the configured profile. When
    no calibrated partition or matching profile row exists, returns the
    configured global defaults. Never estimates or calibrates thresholds.

    Args:
        repository: Threshold repository used for read-only lookups.
        global_thresholds: Fallback thresholds when no approved row exists.
        profile: Profile name to select within a partition. Defaults to
            ``Balanced``.
    """

    __slots__ = ("_global_thresholds", "_profile", "_repository")

    _repository: ThresholdRepository
    _global_thresholds: RegressionThresholds
    _profile: str

    def __init__(
        self,
        repository: ThresholdRepository,
        *,
        global_thresholds: RegressionThresholds,
        profile: str = _DEFAULT_PROFILE,
    ) -> None:
        """Initialize repository lookup with global-default fallback.

        Args:
            repository: Threshold repository used for read-only lookups.
            global_thresholds: Fallback thresholds when no approved row exists.
            profile: Profile name to select within a partition.

        Raises:
            SignalValidationError: If ``repository`` or ``global_thresholds``
                are invalid, or ``profile`` is blank.
        """
        if not isinstance(repository, ThresholdRepository):
            raise SignalValidationError(
                "repository must be a ThresholdRepository",
                error_code=_ERROR_REPOSITORY_TYPE,
                details={"value_type": type(repository).__name__},
            )
        self._repository = repository
        self._global_thresholds = _require_thresholds(global_thresholds)
        self._profile = _require_name(profile, parameter="profile")

    def get_thresholds(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
    ) -> RegressionThresholds:
        """Load approved thresholds or fall back to configured globals.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval.
            model_name: Stable model identifier.
            model_version: Model version identifier.

        Returns:
            Stored thresholds for ``profile``, or global defaults when the
            partition or profile is absent.

        Raises:
            SignalValidationError: If any lookup key is blank.
        """
        validated_symbol = _require_name(symbol, parameter="symbol")
        validated_timeframe = _require_name(timeframe, parameter="timeframe")
        validated_model_name = _require_name(model_name, parameter="model_name")
        validated_model_version = _require_name(model_version, parameter="model_version")

        if not self._repository.exists(
            model_name=validated_model_name,
            model_version=validated_model_version,
            symbol=validated_symbol,
            timeframe=validated_timeframe,
        ):
            return self._global_thresholds

        try:
            frame = self._repository.load(
                model_name=validated_model_name,
                model_version=validated_model_version,
                symbol=validated_symbol,
                timeframe=validated_timeframe,
                profile=self._profile,
            )
        except DatasetNotFoundError:
            return self._global_thresholds

        if frame.height == 0:
            return self._global_thresholds

        row = frame.row(0, named=True)
        return RegressionThresholds(
            buy_threshold=float(row["buy_threshold"]),
            sell_threshold=float(row["sell_threshold"]),
        )


def _require_name(value: object, *, parameter: str) -> str:
    """Require a non-blank string lookup key.

    Args:
        value: Candidate key value.
        parameter: Parameter name used in error messages.

    Returns:
        The validated string.

    Raises:
        SignalValidationError: If ``value`` is not a non-blank string.
    """
    error_code = _ERROR_PROFILE_BLANK if parameter == "profile" else _ERROR_NAME_BLANK
    if not isinstance(value, str) or value.strip() == "":
        raise SignalValidationError(
            f"{parameter} must be a non-blank string",
            error_code=error_code,
            details={
                "parameter": parameter,
                "value": value,
            },
        )
    return value


def _require_thresholds(value: object) -> RegressionThresholds:
    """Require a ``RegressionThresholds`` instance.

    Args:
        value: Candidate thresholds object.

    Returns:
        The validated ``RegressionThresholds`` instance.

    Raises:
        SignalValidationError: If ``value`` is not ``RegressionThresholds``.
    """
    if not isinstance(value, RegressionThresholds):
        raise SignalValidationError(
            "thresholds must be a RegressionThresholds instance",
            error_code=_ERROR_THRESHOLDS_TYPE,
            details={"value_type": type(value).__name__},
        )
    return value
