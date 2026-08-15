"""CQROS factor-validation dataset builder.

Purpose:
    Assemble the in-memory cross-sectional validation dataset expected by
    ``SimpleFactorValidationEngine.build`` by loading all canonical Factors
    and Labels partitions for a ``(manager, timeframe, year)`` panel and
    joining them on the shared research primary key.

Responsibilities:
    - Discover and load Factors through ``FactorsRepository`` for every
      contributing symbol in a ``(manager, timeframe, year)`` panel
    - Load matching Labels through ``LabelRepository``
    - Concatenate symbol partitions into one cross-sectional panel with
      deterministic symbol ordering
    - Validate primary keys and uniqueness contracts on both inputs
    - Inner-join on ``PRIMARY_KEY_COLUMNS``
    - Preserve every canonical Factors column
    - Append required validation label columns (``future_return_1``)
    - Finalize column order and dtypes against ``VALIDATION_DATASET_SCHEMA``
    - Remain free of validation statistics, engine logic, CLI, and persistence
      of validation ledgers

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.factor_validation.exceptions``,
    ``cqros.factor_validation.validation_dataset_schema``,
    ``cqros.factors.repository``, and ``cqros.storage.label_repository``.

Public API:
    ``ValidationDatasetBuilder``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, Protocol

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factor_validation.exceptions import FactorValidationError
from cqros.factor_validation.validation_dataset_schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_COLUMNS,
    FACTOR_OBSERVATION_KEY_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    VALIDATION_DATASET_SCHEMA,
    VALIDATION_LABEL_COLUMNS,
)
from cqros.factors.repository import FactorsRepository
from cqros.storage.label_repository import LabelRepository

__all__ = ["ValidationDatasetBuilder"]

_ERROR_FRAME_TYPE: Final[str] = "FVAL_VDB_FRAME_TYPE"
_ERROR_MISSING_PRIMARY_KEY: Final[str] = "FVAL_VDB_MISSING_PRIMARY_KEY"
_ERROR_DUPLICATE_KEYS: Final[str] = "FVAL_VDB_DUPLICATE_KEYS"
_ERROR_MISSING_COLUMNS: Final[str] = "FVAL_VDB_MISSING_COLUMNS"
_ERROR_EMPTY_JOIN: Final[str] = "FVAL_VDB_EMPTY_JOIN"
_ERROR_SCHEMA_CAST: Final[str] = "FVAL_VDB_SCHEMA_CAST"
_ERROR_EMPTY_PANEL: Final[str] = "FVAL_VDB_EMPTY_PANEL"

_JOIN_KEYS: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)
_FACTOR_OBSERVATION_KEYS: Final[list[str]] = list(FACTOR_OBSERVATION_KEY_COLUMNS)
_LABEL_SELECT_COLUMNS: Final[list[str]] = [
    *PRIMARY_KEY_COLUMNS,
    *VALIDATION_LABEL_COLUMNS,
]

_logger = logging.getLogger(__name__)


class _ValidationPanelSpillWriter(Protocol):
    """Structural contract for spilling joined per-symbol validation partitions."""

    def write_symbol_partition(self, symbol: Symbol, frame: pl.DataFrame) -> object:
        """Persist one joined symbol partition."""
        ...


class ValidationDatasetBuilder:
    """Assemble a cross-sectional Factors+Labels panel for validation.

    The builder owns repository loads, panel concatenation, and the join. It
    never mutates caller frames, never persists artifacts, and never computes
    validation metrics.

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
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> pl.DataFrame:
        """Load all symbol partitions for one panel and assemble the dataset.

        Discovers Factors year partitions for ``(manager, timeframe, year)``,
        intersects with symbols that also have Labels for the same
        ``(timeframe, year)``, loads every contributing symbol in sorted order,
        concatenates into one cross-sectional panel, and joins Factors to
        Labels.

        Args:
            manager: Order manager identifier for the Factors partitions.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Bar interval.
            year: Calendar year of the panel.
            symbols: Optional symbol allowlist. ``None`` includes every symbol
                that has both Factors and Labels for the panel key.

        Returns:
            A new DataFrame matching ``VALIDATION_DATASET_SCHEMA``.

        Raises:
            FactorValidationError: If no contributing symbols remain, inputs
                fail structural checks, join keys are duplicated, required
                columns are missing, the join is empty, or casting fails.
        """
        panel_symbols = self._resolve_panel_symbols(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
            symbols=symbols,
        )
        self._logger.debug(
            "Loading Factors and Labels panel for validation dataset assembly",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "symbols": panel_symbols,
                "symbol_count": len(panel_symbols),
            },
        )
        # Join each symbol independently, then concatenate. This keeps peak
        # memory near one symbol partition instead of the full universe.
        assembled: pl.DataFrame | None = None
        factor_rows = 0
        label_rows = 0
        for symbol in panel_symbols:
            factors = self._factors_repository.load(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            labels = self._label_repository.load(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            factor_rows += factors.height
            label_rows += labels.height
            part = self.assemble(factors, labels)
            del factors
            del labels
            assembled = part if assembled is None else pl.concat([assembled, part], how="vertical")
            del part

        if assembled is None:
            raise FactorValidationError(
                "no symbols with both Factors and Labels for validation panel",
                error_code=_ERROR_EMPTY_PANEL,
                details={
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "timeframe": timeframe,
                    "year": year,
                    "requested_symbols": None if symbols is None else tuple(symbols),
                    "candidate_symbols": panel_symbols,
                },
            )
        self._logger.info(
            "Assembled validation dataset panel",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "symbol_count": len(panel_symbols),
                "factor_rows": factor_rows,
                "label_rows": label_rows,
                "assembled_rows": assembled.height,
                "columns": assembled.width,
            },
        )
        return assembled

    def spill_panel(
        self,
        *,
        spill: _ValidationPanelSpillWriter,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> int:
        """Join each symbol partition and spill it without full-panel concat.

        Loads Factors and Labels per symbol in sorted order, joins them with
        ``assemble``, writes each joined partition through ``spill``, and
        releases the in-memory frames before loading the next symbol. Peak
        memory stays near one symbol partition plus spill I/O buffers.

        Args:
            spill: Spill writer that persists each joined symbol partition.
            manager: Order manager identifier for the Factors partitions.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Bar interval.
            year: Calendar year of the panel.
            symbols: Optional symbol allowlist. ``None`` includes every symbol
                that has both Factors and Labels for the panel key.

        Returns:
            Number of symbol partitions spilled.

        Raises:
            FactorValidationError: If no contributing symbols remain or a
                per-symbol join fails structural checks.
        """
        panel_symbols = self._resolve_panel_symbols(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
            symbols=symbols,
        )
        self._logger.debug(
            "Spilling Factors and Labels panel for memory-efficient validation",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "symbols": panel_symbols,
                "symbol_count": len(panel_symbols),
            },
        )
        factor_rows = 0
        label_rows = 0
        assembled_rows = 0
        for symbol in panel_symbols:
            factors = self._factors_repository.load(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            labels = self._label_repository.load(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            factor_rows += factors.height
            label_rows += labels.height
            part = self.assemble(factors, labels)
            del factors
            del labels
            assembled_rows += part.height
            spill.write_symbol_partition(symbol, part)
            del part

        self._logger.info(
            "Spilled validation dataset panel",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "symbol_count": len(panel_symbols),
                "factor_rows": factor_rows,
                "label_rows": label_rows,
                "assembled_rows": assembled_rows,
            },
        )
        return len(panel_symbols)

    def assemble(
        self,
        factors: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join Factors and Labels into the validation-engine input frame.

        Args:
            factors: Canonical Factors DataFrame. Never mutated.
            labels: Canonical Labels DataFrame. Never mutated.

        Returns:
            A new DataFrame matching ``VALIDATION_DATASET_SCHEMA``.

        Raises:
            FactorValidationError: If inputs fail structural checks, join keys
                are duplicated, required columns are missing, the join is
                empty, or casting fails.
        """
        factors_frame = _require_dataframe(factors, side="factors")
        labels_frame = _require_dataframe(labels, side="labels")

        _require_primary_key_columns(factors_frame, side="factors")
        _require_primary_key_columns(labels_frame, side="labels")
        _require_factor_columns(factors_frame)
        _require_label_columns(labels_frame)
        _require_unique_factor_observation_keys(factors_frame)
        _require_unique_join_keys(labels_frame, side="labels")

        joined = _inner_join(factors_frame, labels_frame)
        if joined.height == 0:
            raise FactorValidationError(
                "Factors and Labels join produced no matching rows",
                error_code=_ERROR_EMPTY_JOIN,
                details={
                    "join_keys": PRIMARY_KEY_COLUMNS,
                    "factor_rows": factors_frame.height,
                    "label_rows": labels_frame.height,
                },
            )
        return _finalize(joined)

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
            raise FactorValidationError(
                "no symbols with both Factors and Labels for validation panel",
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


def _inner_join(factors: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Inner-join Factors to the required Labels columns on the shared key."""
    label_subset = labels.select(_LABEL_SELECT_COLUMNS)
    return factors.join(label_subset, on=_JOIN_KEYS, how="inner")


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply required-column checks, canonical ordering, and dtype casting."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "validation dataset is missing required columns after join",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    try:
        return ordered.cast(VALIDATION_DATASET_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorValidationError(
            "validation dataset failed VALIDATION_DATASET_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_dataframe(frame: object, *, side: str) -> pl.DataFrame:
    """Raise when ``frame`` is not a Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorValidationError(
            f"{side} input must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"side": side, "actual_type": type(frame).__name__},
        )
    return frame


def _require_primary_key_columns(frame: pl.DataFrame, *, side: str) -> None:
    """Raise when any join-key column is missing from ``frame``."""
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            f"{side} input is missing required primary-key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEY,
            details={
                "side": side,
                "missing_columns": tuple(missing),
                "required_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_factor_columns(frame: pl.DataFrame) -> None:
    """Raise when any canonical Factors column is missing."""
    missing = [column for column in FACTOR_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "factors input is missing required canonical columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "side": "factors",
                "missing_columns": tuple(missing),
                "required_columns": FACTOR_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_label_columns(frame: pl.DataFrame) -> None:
    """Raise when any required validation label column is missing."""
    missing = [column for column in VALIDATION_LABEL_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "labels input is missing required validation label columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "side": "labels",
                "missing_columns": tuple(missing),
                "required_columns": VALIDATION_LABEL_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_join_keys(frame: pl.DataFrame, *, side: str) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``."""
    unique_keys = frame.select(_JOIN_KEYS).n_unique()
    if unique_keys != frame.height:
        raise FactorValidationError(
            f"{side} input contains duplicate join keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "side": side,
                "join_keys": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )


def _require_unique_factor_observation_keys(frame: pl.DataFrame) -> None:
    """Raise when long-format Factors observation keys are duplicated."""
    unique_keys = frame.select(_FACTOR_OBSERVATION_KEYS).n_unique()
    if unique_keys != frame.height:
        raise FactorValidationError(
            "factors input contains duplicate observation keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "side": "factors",
                "join_keys": FACTOR_OBSERVATION_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )
