"""CQROS Walk-Forward evaluation-input adapter.

Purpose:
    Assemble the evaluation-only input frame required by
    ``SimpleWalkForwardEngine`` by joining Factor Selection decisions onto
    Factors observations and Labels ``future_return_1`` without mutating the
    canonical Factor Selection schema.

Responsibilities:
    - Inner-join Factors and Labels on ``(symbol, timeframe, open_time)``
    - Attach Factor Selection ``selected`` flags by factor identity
    - Map ``open_time`` onto ``selection_time`` for the walk-forward engine
    - Reject duplicate label keys and duplicate selection join keys
    - Refuse to fabricate missing ``future_return_1`` values
    - Remain free of walk-forward fold math, Alpha/Regime/ML imports, and
      Factor Selection scoring

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.factors.repository``,
    ``cqros.labels.schema``, ``cqros.storage.label_repository``, and
    ``cqros.walk_forward.exceptions``.

Public API:
    ``OBSERVATION_JOIN_KEYS``, ``SELECTION_JOIN_KEYS``,
    ``WALK_FORWARD_EVALUATION_COLUMNS``, ``TARGET_COLUMN``,
    ``assemble_walk_forward_input``, ``WalkForwardInputBuilder``

Notes:
    ``future_return_1`` is retrospective evaluation data for Walk Forward /
    Purged CV only. This adapter must not write into Factor Selection,
    Alpha, Regime, Predictions, or Signals contracts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factor_selection.orientation import (
    FACTOR_ORIENTATION_POLICY,
    is_orientation_metadata_complete,
)
from cqros.factors.repository import FactorsRepository
from cqros.labels.schema import PRIMARY_KEY_COLUMNS as LABEL_PRIMARY_KEY_COLUMNS
from cqros.storage.label_repository import LabelRepository
from cqros.walk_forward.exceptions import WalkForwardError

__all__ = [
    "OBSERVATION_JOIN_KEYS",
    "SELECTION_JOIN_KEYS",
    "TARGET_COLUMN",
    "WALK_FORWARD_EVALUATION_COLUMNS",
    "WalkForwardInputBuilder",
    "assemble_walk_forward_input",
    "assemble_walk_forward_symbol_input",
    "require_orientation_metadata",
]

_logger = logging.getLogger(__name__)

_ERROR_FRAME_TYPE: Final[str] = "WF_EVAL_FRAME_TYPE"
_ERROR_MISSING_COLUMNS: Final[str] = "WF_EVAL_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "WF_EVAL_DUPLICATE_KEYS"
_ERROR_EMPTY_JOIN: Final[str] = "WF_EVAL_EMPTY_JOIN"
_ERROR_EMPTY_PANEL: Final[str] = "WF_EVAL_EMPTY_PANEL"
_ERROR_FRAME_EMPTY: Final[str] = "WF_EVAL_FRAME_EMPTY"
_ERROR_LEGACY_ORIENTATION: Final[str] = "WF_EVAL_LEGACY_ORIENTATION"
_ERROR_INVALID_ORIENTATION: Final[str] = "WF_EVAL_INVALID_ORIENTATION"

# Canonical observation identity shared by Factors and Labels.
OBSERVATION_JOIN_KEYS: Final[tuple[str, ...]] = LABEL_PRIMARY_KEY_COLUMNS
if OBSERVATION_JOIN_KEYS != ("symbol", "timeframe", "open_time"):
    raise RuntimeError(
        "Walk-Forward evaluation input requires Labels primary key "
        "(symbol, timeframe, open_time)."
    )

# Factor Selection identity used to broadcast selection decisions onto bars.
SELECTION_JOIN_KEYS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
)

TARGET_COLUMN: Final[str] = "future_return_1"

# Long-format Factors uniqueness: bar identity plus factor name.
_FACTOR_OBSERVATION_KEYS: Final[tuple[str, ...]] = (
    *OBSERVATION_JOIN_KEYS,
    "factor_name",
)

_FACTOR_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *OBSERVATION_JOIN_KEYS,
    "factor_name",
    "factor_version",
    "factor_value",
)

_SELECTION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *SELECTION_JOIN_KEYS,
    "selected",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
)

_LABEL_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *OBSERVATION_JOIN_KEYS,
    TARGET_COLUMN,
)

# Evaluation frame retained for engine input plus join-audit identity.
# ``factor_value`` remains the canonical raw factor observation.
# Orientation metadata is inherited from Factor Selection and must never be
# recomputed from OOS Labels.
WALK_FORWARD_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "factor_name",
    "factor_version",
    "factor_value",
    "selected",
    "selection_time",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
    TARGET_COLUMN,
)

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_time",
    "symbol",
    "factor_name",
    "factor_version",
)

_OBSERVATION_JOIN_KEY_LIST: Final[list[str]] = list(OBSERVATION_JOIN_KEYS)
_SELECTION_JOIN_KEY_LIST: Final[list[str]] = list(SELECTION_JOIN_KEYS)
_FACTOR_OBSERVATION_KEY_LIST: Final[list[str]] = list(_FACTOR_OBSERVATION_KEYS)
_LABEL_SELECT_COLUMNS: Final[list[str]] = list(_LABEL_REQUIRED_COLUMNS)


class WalkForwardInputBuilder:
    """Load Factors + Labels panels and assemble walk-forward evaluation input.

    The builder owns repository loads and panel concatenation. Join semantics
    and selection attachment remain in ``assemble_walk_forward_input``.
    Caller frames and lake partitions are never mutated in-place beyond
    repository reads.

    Args:
        factors_repository: Repository facade for canonical Factors partitions.
        label_repository: Repository facade for merged Labels partitions.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_factors_repository", "_label_repository", "_logger")

    _factors_repository: FactorsRepository
    _label_repository: LabelRepository
    _logger: logging.Logger

    def __init__(
        self,
        factors_repository: FactorsRepository,
        label_repository: LabelRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the builder with injected repositories.

        Args:
            factors_repository: Factors partition repository.
            label_repository: Labels partition repository.
            logger: Optional logger instance.
        """
        self._factors_repository = factors_repository
        self._label_repository = label_repository
        self._logger = logger if logger is not None else _logger

    def build(
        self,
        factor_selection: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> pl.DataFrame:
        """Load panel partitions and assemble walk-forward evaluation input.

        Args:
            factor_selection: Canonical Factor Selection frame. Never mutated.
            manager: Order manager identifier for Factors partitions.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Bar interval.
            year: Calendar year of the panel.
            symbols: Optional symbol allowlist. ``None`` includes every symbol
                that has both Factors and Labels for the panel key.

        Returns:
            Evaluation input containing Factor Selection decisions plus
            ``future_return_1`` aligned on ``(symbol, timeframe, open_time)``.

        Raises:
            WalkForwardError: If the panel cannot be assembled or the join
                contracts fail.
        """
        panel_symbols = self._resolve_panel_symbols(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
            symbols=symbols,
        )
        factors_parts: list[pl.DataFrame] = []
        labels_parts: list[pl.DataFrame] = []
        for symbol in panel_symbols:
            factors_parts.append(
                self._factors_repository.load(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            )
            labels_parts.append(
                self._label_repository.load(
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            )
        factors = pl.concat(factors_parts, how="vertical")
        labels = pl.concat(labels_parts, how="vertical")
        self._logger.info(
            "Assembling walk-forward evaluation input",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "symbol_count": len(panel_symbols),
                "factor_rows": factors.height,
                "label_rows": labels.height,
                "selection_rows": factor_selection.height,
            },
        )
        return assemble_walk_forward_input(factor_selection, factors, labels)

    def _resolve_panel_symbols(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols that have both Factors and Labels for the panel."""
        if symbols is not None:
            candidates = tuple(sorted({symbol for symbol in symbols if symbol.strip() != ""}))
        else:
            candidates = self._discover_factor_symbols(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            )

        panel_symbols = [
            symbol
            for symbol in candidates
            if self._factors_repository.exists(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            and self._label_repository.exists(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        ]
        if len(panel_symbols) == 0:
            raise WalkForwardError(
                "no symbols with both Factors and Labels for walk-forward panel",
                error_code=_ERROR_EMPTY_PANEL,
                details={
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "timeframe": timeframe,
                    "year": year,
                    "requested_symbols": None if symbols is None else tuple(symbols),
                    "candidate_symbols": candidates,
                },
            )
        return tuple(panel_symbols)

    def _discover_factor_symbols(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> tuple[Symbol, ...]:
        """Discover sorted Factors symbols present for the panel key."""
        factor_symbols: list[Symbol] = []
        for partition in self._factors_repository.discover_partitions(
            managers=(manager,),
            timeframes=(timeframe,),
            exchange=exchange,
            market=market,
        ):
            if partition.year != year:
                continue
            if partition.symbol not in factor_symbols:
                factor_symbols.append(partition.symbol)
        return tuple(sorted(factor_symbols))


def assemble_walk_forward_input(
    factor_selection: pl.DataFrame,
    factors: pl.DataFrame,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Join Labels ``future_return_1`` onto selected-factor observations.

    Join rules:
        - Factors INNER JOIN Labels on ``(symbol, timeframe, open_time)``
        - Observations INNER JOIN Factor Selection on
          ``(factor_name, factor_version, timeframe)`` to attach ``selected``
        - ``selection_time`` is set to ``open_time`` for the walk-forward engine
        - Missing label matches are dropped; values are never fabricated
        - Duplicate label keys and duplicate selection join keys raise

    Args:
        factor_selection: Canonical Factor Selection DataFrame. Never mutated.
        factors: Canonical Factors DataFrame. Never mutated.
        labels: Canonical Labels DataFrame. Never mutated.

    Returns:
        Deterministic evaluation input ordered by
        ``(timeframe, selection_time, symbol, factor_name, factor_version)``.

    Raises:
        WalkForwardError: If inputs fail structural checks, keys are
            duplicated, required columns are missing, or the join is empty.
    """
    return _assemble_walk_forward_input(
        factor_selection,
        factors,
        labels,
        allow_empty=False,
    )


def assemble_walk_forward_symbol_input(
    factor_selection: pl.DataFrame,
    factors: pl.DataFrame,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble one bounded symbol shard with canonical join semantics.

    Unlike the full-panel adapter, an unmatched symbol returns an empty,
    schema-preserving frame because another symbol may still produce the
    globally non-empty canonical panel. Duplicate and orientation checks are
    identical to :func:`assemble_walk_forward_input`.
    """
    return _assemble_walk_forward_input(
        factor_selection,
        factors,
        labels,
        allow_empty=True,
    )


def _assemble_walk_forward_input(
    factor_selection: pl.DataFrame,
    factors: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    allow_empty: bool,
) -> pl.DataFrame:
    """Implement canonical joins, optionally allowing an empty symbol shard."""
    selection_frame = _require_dataframe(factor_selection, side="factor_selection")
    factors_frame = _require_dataframe(factors, side="factors")
    labels_frame = _require_dataframe(labels, side="labels")

    if selection_frame.height == 0:
        raise WalkForwardError(
            "factor_selection frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_selection", "rows": selection_frame.height},
        )

    _require_columns(selection_frame, _SELECTION_REQUIRED_COLUMNS, side="factor_selection")
    require_orientation_metadata(selection_frame)
    _require_columns(factors_frame, _FACTOR_REQUIRED_COLUMNS, side="factors")
    _require_columns(labels_frame, _LABEL_REQUIRED_COLUMNS, side="labels")

    _require_unique_keys(
        labels_frame,
        _OBSERVATION_JOIN_KEY_LIST,
        side="labels",
        join_keys=OBSERVATION_JOIN_KEYS,
    )
    _require_unique_keys(
        factors_frame,
        _FACTOR_OBSERVATION_KEY_LIST,
        side="factors",
        join_keys=_FACTOR_OBSERVATION_KEYS,
    )
    selection_keys = _selection_decision_frame(selection_frame)
    _require_unique_keys(
        selection_keys,
        _SELECTION_JOIN_KEY_LIST,
        side="factor_selection",
        join_keys=SELECTION_JOIN_KEYS,
    )

    label_subset = labels_frame.select(_LABEL_SELECT_COLUMNS)
    observations = factors_frame.join(
        label_subset,
        on=_OBSERVATION_JOIN_KEY_LIST,
        how="inner",
    )
    if observations.height == 0 and not allow_empty:
        raise WalkForwardError(
            "Factors and Labels join produced no matching rows",
            error_code=_ERROR_EMPTY_JOIN,
            details={
                "join_keys": OBSERVATION_JOIN_KEYS,
                "factor_rows": factors_frame.height,
                "label_rows": labels_frame.height,
            },
        )

    enriched = observations.join(
        selection_keys,
        on=_SELECTION_JOIN_KEY_LIST,
        how="inner",
    )
    if enriched.height == 0 and not allow_empty:
        raise WalkForwardError(
            "Factor Selection join produced no matching observation rows",
            error_code=_ERROR_EMPTY_JOIN,
            details={
                "join_keys": SELECTION_JOIN_KEYS,
                "observation_rows": observations.height,
                "selection_rows": selection_keys.height,
            },
        )

    return (
        enriched.with_columns(pl.col("open_time").alias("selection_time"))
        .select(list(WALK_FORWARD_EVALUATION_COLUMNS))
        .sort(list(_SORT_COLUMNS))
    )


def _selection_decision_frame(factor_selection: pl.DataFrame) -> pl.DataFrame:
    """Return Factor Selection columns required to attach selection decisions."""
    return factor_selection.select(
        [
            *_SELECTION_JOIN_KEY_LIST,
            "selected",
            "selection_ic",
            "selected_direction",
            "orientation_policy",
        ]
    )


def require_orientation_metadata(factor_selection: pl.DataFrame) -> None:
    """Reject pre-orientation Factor Selection artifacts without silent defaults.

    Legacy artifacts lacking orientation metadata must be regenerated. This
    function never invents ``selected_direction = +1``.

    Raises:
        WalkForwardError: If orientation columns are missing, directions are
            outside ``{-1, +1}``, or the policy identifier is blank / unknown.
    """
    if not is_orientation_metadata_complete(tuple(factor_selection.columns)):
        missing = [
            column
            for column in ("selection_ic", "selected_direction", "orientation_policy")
            if column not in factor_selection.columns
        ]
        raise WalkForwardError(
            "Factor Selection artifact predates orientation policy; regenerate "
            "Factor Selection before Walk-Forward evaluation",
            error_code=_ERROR_LEGACY_ORIENTATION,
            details={
                "missing_columns": tuple(missing),
                "required_policy": FACTOR_ORIENTATION_POLICY,
                "legacy_behavior": "regeneration_required",
            },
        )

    invalid_direction = int(
        factor_selection.select((~pl.col("selected_direction").is_in([-1, 1])).sum()).item()
    )
    if invalid_direction > 0:
        raise WalkForwardError(
            "selected_direction must be -1 or +1",
            error_code=_ERROR_INVALID_ORIENTATION,
            details={"invalid_direction_rows": invalid_direction},
        )

    blank_policy = int(
        factor_selection.select(
            (pl.col("orientation_policy").is_null() | (pl.col("orientation_policy") == "")).sum()
        ).item()
    )
    if blank_policy > 0:
        raise WalkForwardError(
            "orientation_policy must be present for every Factor Selection row",
            error_code=_ERROR_INVALID_ORIENTATION,
            details={"blank_policy_rows": blank_policy},
        )

    unknown_policy = int(
        factor_selection.select(
            (pl.col("orientation_policy") != FACTOR_ORIENTATION_POLICY).sum()
        ).item()
    )
    if unknown_policy > 0:
        raise WalkForwardError(
            "unsupported orientation_policy on Factor Selection artifact",
            error_code=_ERROR_INVALID_ORIENTATION,
            details={
                "unsupported_rows": unknown_policy,
                "required_policy": FACTOR_ORIENTATION_POLICY,
            },
        )


def _require_dataframe(frame: object, *, side: str) -> pl.DataFrame:
    """Raise when ``frame`` is not a Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise WalkForwardError(
            f"{side} input must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"side": side, "actual_type": type(frame).__name__},
        )
    return frame


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    *,
    side: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise WalkForwardError(
            f"{side} input is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "side": side,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_keys(
    frame: pl.DataFrame,
    key_columns: list[str],
    *,
    side: str,
    join_keys: tuple[str, ...],
) -> None:
    """Raise when ``key_columns`` combinations are duplicated in ``frame``."""
    unique_keys = frame.select(key_columns).n_unique()
    if unique_keys != frame.height:
        raise WalkForwardError(
            f"{side} input contains duplicate join keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "side": side,
                "join_keys": join_keys,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
