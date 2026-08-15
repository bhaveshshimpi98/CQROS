"""CQROS Alpha Engine contracts and implementation.

Purpose:
    Convert a Factor Orthogonalization dataset into a deterministic alpha
    DataFrame conforming to ``ALPHA_SCHEMA``.

Responsibilities:
    - Define ``AlphaEngine`` as the shared alpha-generation contract
    - Provide ``SimpleAlphaEngine`` for combination-unit alpha assembly
    - Validate Factor Orthogonalization DataFrame structure
    - Restrict participation to rows where ``selected`` is ``True`` and
      ``status`` is ``PASS``
    - Load validation-window factor observations through an injected
      ``FactorObservationSource``
    - Emit one deterministic alpha row per accepted combination × symbol ×
      open_time with equal-weight combination scores
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.alpha.exceptions``, ``cqros.alpha.schema``, and
    ``cqros.factor_selection.redundancy.FactorObservationSource``.

Public API:
    ``AlphaEngine``, ``SimpleAlphaEngine``, ``ALPHA_INPUT_COLUMNS``,
    ``validate_factor_orthogonalization_frame``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol, cast, runtime_checkable

import polars as pl

from cqros.alpha.exceptions import AlphaError
from cqros.alpha.schema import ALPHA_SCHEMA, CANONICAL_COLUMN_ORDER, AlphaStatus
from cqros.factor_selection.redundancy import FactorObservationSource

__all__ = [
    "ALPHA_INPUT_COLUMNS",
    "AlphaEngine",
    "SimpleAlphaEngine",
    "validate_factor_orthogonalization_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "ALPHA_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "ALPHA_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "ALPHA_MISSING_COLUMNS"
_ERROR_NO_COMBINATIONS: Final[str] = "ALPHA_NO_COMBINATIONS"
_ERROR_OBSERVATION_SOURCE: Final[str] = "ALPHA_OBSERVATION_SOURCE_REQUIRED"
_ERROR_TIMEFRAME: Final[str] = "ALPHA_TIMEFRAME_INCONSISTENT"
_ERROR_VALIDATION_WINDOW: Final[str] = "ALPHA_VALIDATION_WINDOW_INCONSISTENT"
_ERROR_SYMBOL: Final[str] = "ALPHA_SYMBOL_INVALID"

_ALPHA_MODEL: Final[str] = "placeholder"
_ALPHA_VERSION: Final[str] = "1.0"
_PASS_STATUS: Final[str] = "PASS"
_PREDICTION_HORIZON: Final[int] = 1
_EQUAL_WEIGHT: Final[str] = "equal_weight"
_KEY_SEP: Final[str] = "\x1f"

# Factor Orthogonalization columns required to assemble alpha rows.
ALPHA_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "factor_names",
    "factor_versions",
    "factor_categories",
    "timeframe",
    "combination_method",
    "selected",
    "status",
    "orthogonalization_rank",
    "validation_start_time",
    "validation_end_time",
)


@runtime_checkable
class AlphaEngine(Protocol):
    """Structural contract for converting orthogonalization into alpha rows.

    Implementations own alpha-generation semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(
        self,
        factor_orthogonalization: pl.DataFrame,
        *,
        symbol: str,
    ) -> pl.DataFrame:
        """Convert a Factor Orthogonalization dataset into alpha rows.

        Args:
            factor_orthogonalization: Canonical Factor Orthogonalization
                dataset. Must not be mutated.
            symbol: Tradeable symbol for which alpha rows are generated.

        Returns:
            A new DataFrame containing the columns required by
            ``ALPHA_SCHEMA``.
        """
        ...


class SimpleAlphaEngine:
    """Generate combination-unit alpha rows from Factor Orthogonalization.

    Only rows with ``selected == True`` and ``status == PASS`` participate.
    Each accepted combination remains one factor set
    (``factor_set_id = combination_id``). Member factor observations within the
    FO validation window are combined with an equal-weight arithmetic mean at
    each ``(symbol, open_time)``, matching Factor Orthogonalization redundancy
    finite-member semantics.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical FO frames, observation
        panels, and symbol. Wall-clock time is never part of Alpha identity.
    """

    __slots__ = ("_observation_source",)

    _observation_source: FactorObservationSource

    def __init__(
        self,
        *,
        observation_source: FactorObservationSource | None,
    ) -> None:
        """Initialize the engine with an observation source.

        Args:
            observation_source: Validation-window factor observation loader.

        Raises:
            AlphaError: If ``observation_source`` is ``None``.
        """
        if observation_source is None:
            raise AlphaError(
                "observation_source is required for combination-unit alpha",
                error_code=_ERROR_OBSERVATION_SOURCE,
                details={"observation_source": None},
            )
        self._observation_source = observation_source

    def build(
        self,
        factor_orthogonalization: pl.DataFrame,
        *,
        symbol: str,
    ) -> pl.DataFrame:
        """Convert a Factor Orthogonalization dataset into alpha rows.

        Args:
            factor_orthogonalization: Canonical Factor Orthogonalization
                dataset. Must not be mutated.
            symbol: Tradeable symbol for which alpha rows are generated.

        Returns:
            A new DataFrame matching ``ALPHA_SCHEMA``.

        Raises:
            AlphaError: If the input fails structural validation, required
                columns are missing, the symbol is invalid, accepted rows are
                inconsistent, or no selected PASS combinations remain.
        """
        frame = validate_factor_orthogonalization_frame(factor_orthogonalization)
        _require_columns(frame, ALPHA_INPUT_COLUMNS, "factor_orthogonalization")
        resolved_symbol = _require_symbol(symbol)
        return _build_alpha_rows(
            frame,
            symbol=resolved_symbol,
            observation_source=self._observation_source,
        )


def validate_factor_orthogonalization_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Orthogonalization dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        AlphaError: If ``frame`` is not a Polars DataFrame or contains no
            rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise AlphaError(
            "factor_orthogonalization frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_orthogonalization",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise AlphaError(
            "factor_orthogonalization frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_orthogonalization", "rows": frame.height},
        )
    return frame


def _build_alpha_rows(
    factor_orthogonalization: pl.DataFrame,
    *,
    symbol: str,
    observation_source: FactorObservationSource,
) -> pl.DataFrame:
    """Assemble canonical alpha rows from Factor Orthogonalization."""
    surviving = factor_orthogonalization.filter(
        (pl.col("selected") == True) & (pl.col("status") == _PASS_STATUS)  # noqa: E712
    )
    if surviving.height == 0:
        raise AlphaError(
            "factor_orthogonalization frame contains no selected PASS combinations",
            error_code=_ERROR_NO_COMBINATIONS,
            details={
                "dataset": "factor_orthogonalization",
                "rows": factor_orthogonalization.height,
                "surviving_rows": 0,
            },
        )

    ordered = _order_surviving_combinations(surviving)
    timeframe = _require_single_timeframe(ordered)
    start_time, end_time = _require_single_validation_window(ordered)

    factor_names, factor_versions = _collect_member_identities(ordered)
    observations = observation_source.load_panel(
        timeframe=timeframe,
        factor_names=factor_names,
        factor_versions=factor_versions,
        start_time=start_time,
        end_time=end_time,
    )
    symbol_observations = _filter_symbol_observations(observations, symbol=symbol)
    wide = _pivot_observations(symbol_observations)

    row_frames: list[pl.DataFrame] = []
    for combination in ordered.to_dicts():
        method = str(combination.get("combination_method", _EQUAL_WEIGHT))
        if method != _EQUAL_WEIGHT:
            continue
        combination_rows = _build_combination_alpha_rows(
            wide=wide,
            combination=combination,
            symbol=symbol,
            timeframe=timeframe,
        )
        if combination_rows.height > 0:
            row_frames.append(combination_rows)

    if len(row_frames) == 0:
        return pl.DataFrame(schema=ALPHA_SCHEMA)

    return (
        pl.concat(row_frames, how="vertical")
        .sort(
            "factor_set_id",
            "prediction_time",
            descending=[False, False],
            nulls_last=True,
            maintain_order=True,
        )
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(ALPHA_SCHEMA)
    )


def _build_combination_alpha_rows(
    *,
    wide: pl.DataFrame,
    combination: dict[str, object],
    symbol: str,
    timeframe: str,
) -> pl.DataFrame:
    """Build alpha rows for one accepted equal-weight combination."""
    combination_id = str(combination["combination_id"])
    names = _as_string_list(combination["factor_names"])
    versions = _as_string_list(combination["factor_versions"])
    if len(names) == 0 or len(names) != len(versions):
        return pl.DataFrame(schema=ALPHA_SCHEMA)

    member_keys = [
        _factor_key(name, version) for name, version in zip(names, versions, strict=True)
    ]
    if any(key not in wide.columns for key in member_keys):
        # Match FO redundancy: absent member identity yields no signal.
        return pl.DataFrame(schema=ALPHA_SCHEMA)
    if wide.height == 0:
        return pl.DataFrame(schema=ALPHA_SCHEMA)

    finite_mask = pl.all_horizontal(
        [(pl.col(key).is_not_null() & pl.col(key).is_finite()) for key in member_keys]
    )
    scored = (
        wide.filter(finite_mask)
        .with_columns(pl.mean_horizontal([pl.col(key) for key in member_keys]).alias("alpha_score"))
        .select(
            pl.lit(combination_id).alias("factor_set_id"),
            pl.lit(_ALPHA_MODEL).alias("alpha_model"),
            pl.lit(_ALPHA_VERSION).alias("alpha_version"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(timeframe).alias("timeframe"),
            pl.col("open_time").cast(pl.Int64).alias("prediction_time"),
            pl.lit(None, dtype=pl.Float64).alias("expected_return"),
            pl.col("alpha_score").cast(pl.Float64),
            pl.lit(None, dtype=pl.Float64).alias("confidence"),
            pl.lit(None, dtype=pl.Float64).alias("uncertainty"),
            pl.lit(_PREDICTION_HORIZON).cast(pl.Int32).alias("prediction_horizon"),
            pl.lit(AlphaStatus.PASS.value).alias("status"),
        )
    )
    if scored.height == 0:
        return pl.DataFrame(schema=ALPHA_SCHEMA)
    return scored.select(list(CANONICAL_COLUMN_ORDER)).cast(ALPHA_SCHEMA)


def _order_surviving_combinations(surviving: pl.DataFrame) -> pl.DataFrame:
    """Order surviving combinations for deterministic alpha emission."""
    return surviving.sort(
        "orthogonalization_rank",
        "combination_id",
        descending=[False, False],
        nulls_last=True,
        maintain_order=True,
    )


def _require_symbol(symbol: object) -> str:
    """Validate and return a non-empty symbol identifier."""
    if not isinstance(symbol, str) or symbol.strip() == "":
        raise AlphaError(
            "symbol must be a non-empty string",
            error_code=_ERROR_SYMBOL,
            details={
                "parameter": "symbol",
                "value": symbol,
                "actual_type": type(symbol).__name__,
            },
        )
    return symbol


def _require_single_timeframe(frame: pl.DataFrame) -> str:
    """Require exactly one timeframe value within accepted rows."""
    values = frame.select("timeframe").unique().to_series().to_list()
    if len(values) != 1:
        raise AlphaError(
            "factor_orthogonalization accepted rows must contain exactly one timeframe",
            error_code=_ERROR_TIMEFRAME,
            details={"timeframes": tuple(sorted(str(item) for item in values))},
        )
    return str(values[0])


def _require_single_validation_window(frame: pl.DataFrame) -> tuple[int, int]:
    """Require one consistent inclusive validation window on accepted rows."""
    windows = (
        frame.select("validation_start_time", "validation_end_time")
        .unique()
        .sort("validation_start_time", "validation_end_time")
    )
    if windows.height != 1:
        raise AlphaError(
            "factor_orthogonalization accepted rows must share one validation window",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "windows": tuple(
                    (int(row["validation_start_time"]), int(row["validation_end_time"]))
                    for row in windows.to_dicts()
                )
            },
        )
    start = int(windows["validation_start_time"][0])
    end = int(windows["validation_end_time"][0])
    if start > end:
        raise AlphaError(
            "validation window start_time must be <= end_time",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "validation_start_time": start,
                "validation_end_time": end,
            },
        )
    return start, end


def _collect_member_identities(
    frame: pl.DataFrame,
) -> tuple[list[str], list[str]]:
    """Collect unique member factor name/version pairs for observation loading."""
    names: list[str] = []
    versions: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in frame.select("factor_names", "factor_versions").to_dicts():
        row_names = _as_string_list(row["factor_names"])
        row_versions = _as_string_list(row["factor_versions"])
        for name, version in zip(row_names, row_versions, strict=False):
            key = (name, version)
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
            versions.append(version)
    return names, versions


def _filter_symbol_observations(observations: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
    """Restrict an observation panel to the requested symbol."""
    required = ("symbol", "open_time", "factor_name", "factor_version", "factor_value")
    if observations.height == 0:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "factor_value": pl.Float64,
            }
        )
    missing = [column for column in required if column not in observations.columns]
    if missing:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                "factor_value": pl.Float64,
            }
        )
    return observations.filter(pl.col("symbol") == symbol).select(list(required))


def _pivot_observations(observations: pl.DataFrame) -> pl.DataFrame:
    """Pivot long observations to wide columns keyed by factor identity."""
    if observations.height == 0:
        return pl.DataFrame(schema={"open_time": pl.Int64})

    prepared = (
        observations.select(
            pl.col("open_time"),
            pl.col("factor_name"),
            pl.col("factor_version"),
            pl.col("factor_value"),
        )
        .with_columns(
            (pl.col("factor_name") + pl.lit(_KEY_SEP) + pl.col("factor_version")).alias(
                "_factor_key"
            )
        )
        .filter(pl.col("factor_value").is_not_null())
    )
    if prepared.height == 0:
        return pl.DataFrame(schema={"open_time": pl.Int64})
    return prepared.pivot(
        values="factor_value",
        index=["open_time"],
        on="_factor_key",
        aggregate_function="first",
    ).sort("open_time", maintain_order=True)


def _factor_key(name: str, version: str) -> str:
    """Compose a deterministic wide-column key for name+version identity."""
    return f"{name}{_KEY_SEP}{version}"


def _as_string_list(value: object) -> list[str]:
    """Normalize list-like member identity columns to ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in cast(Sequence[object], value)]
    return [str(value)]


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AlphaError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )
