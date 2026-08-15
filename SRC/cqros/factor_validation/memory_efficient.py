"""CQROS memory-efficient Factor Validation execution.

Purpose:
    Execute the existing ``SimpleFactorValidationEngine`` semantics under a
    bounded-memory schedule by spilling the joined Factors+Labels panel to
    temporary Parquet partitions and validating one factor-identity batch at
    a time.

Responsibilities:
    - Define execution-mode configuration for full-panel vs memory-efficient
      Factor Validation
    - Spill per-symbol joined validation partitions without holding the full
      cross-sectional concat in RAM
    - Discover factor identities from the spill
    - Load deterministic factor-identity batches and delegate metric
      computation exclusively to an injected ``FactorValidationEngine``
    - Concatenate batch ledgers into the canonical validation output order
    - Remain free of alternative IC/quantile/turnover formulas, persistence of
      production artifacts, CLI argument parsing, and exchange I/O

Dependencies:
    ``logging``, ``pathlib``, ``shutil``, ``tempfile``, ``polars``,
    ``cqros.core.types``, ``cqros.factor_validation.dataset_builder``,
    ``cqros.factor_validation.engine``, ``cqros.factor_validation.exceptions``,
    and ``cqros.factor_validation.schema``.

Public API:
    ``FactorValidationExecutionConfig``,
    ``FactorValidationExecutionMode``,
    ``MemoryEfficientFactorValidationRunner``,
    ``ValidationPanelSpill``
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factor_validation.dataset_builder import ValidationDatasetBuilder
from cqros.factor_validation.engine import FactorValidationEngine
from cqros.factor_validation.exceptions import FactorValidationError
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_VALIDATION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
)

__all__ = [
    "FactorValidationExecutionConfig",
    "FactorValidationExecutionMode",
    "MemoryEfficientFactorValidationRunner",
    "ValidationPanelSpill",
]

_ERROR_BATCH_SIZE: Final[str] = "FVAL_MEM_BATCH_SIZE"
_ERROR_EMPTY_SPILL: Final[str] = "FVAL_MEM_EMPTY_SPILL"
_ERROR_SPILL_MISSING: Final[str] = "FVAL_MEM_SPILL_MISSING"

_FACTOR_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "timeframe",
)

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)
_IDENTITY_LIST: Final[list[str]] = list(_FACTOR_IDENTITY_COLUMNS)

_logger = logging.getLogger(__name__)


class FactorValidationExecutionMode(StrEnum):
    """Execution strategy for Factor Validation panel processing."""

    FULL_PANEL = "full_panel"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass(frozen=True, slots=True)
class FactorValidationExecutionConfig:
    """Immutable execution options for Factor Validation.

    Attributes:
        mode: ``full_panel`` materializes the entire cross-sectional panel in
            RAM (legacy path). ``memory_efficient`` spills symbol joins and
            validates factor-identity batches (production default).
        factor_batch_size: Number of distinct factor identities loaded into
            RAM per engine invocation when ``mode`` is ``memory_efficient``.
            Must be >= 1. Does not change metric semantics.
        spill_parent: Optional parent directory for temporary spill folders.
            When ``None``, the system temporary directory is used.
    """

    mode: FactorValidationExecutionMode = FactorValidationExecutionMode.MEMORY_EFFICIENT
    factor_batch_size: int = 1
    spill_parent: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.factor_batch_size < 1:
            raise FactorValidationError(
                "factor_batch_size must be >= 1",
                error_code=_ERROR_BATCH_SIZE,
                details={"factor_batch_size": self.factor_batch_size},
            )


@dataclass(frozen=True, slots=True)
class _FactorIdentity:
    """One factor identity key used for spill filtering."""

    factor_name: str
    factor_version: str
    timeframe: str


class ValidationPanelSpill:
    """Temporary on-disk store of joined validation partitions by factor.

    Each joined symbol partition is sliced by factor identity and written under
    ``root / "factors" / <identity_key> / <symbol>.parquet``. Discovery uses
    identities observed during writes. Batch loads read only the requested
    factor directories so peak memory stays near one factor identity.
    """

    __slots__ = ("_identities", "_logger", "_root", "_symbol_count")

    _root: Path
    _symbol_count: int
    _identities: dict[tuple[str, str, str], list[Path]]
    _logger: logging.Logger

    def __init__(self, root: Path, *, logger: logging.Logger | None = None) -> None:
        """Initialize an empty spill directory.

        Args:
            root: Directory that will own factor-identity Parquet trees.
            logger: Optional logger instance.
        """
        self._root = root
        self._symbol_count = 0
        self._identities = {}
        self._logger = logger if logger is not None else _logger
        (self._root / "factors").mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the spill root directory."""
        return self._root

    @property
    def symbol_file_count(self) -> int:
        """Return the number of symbols spilled."""
        return self._symbol_count

    def write_symbol_partition(self, symbol: Symbol, frame: pl.DataFrame) -> Path:
        """Persist factor-sliced partitions for one joined symbol frame.

        Args:
            symbol: Symbol identity used for deterministic filenames.
            frame: Joined validation dataset rows for ``symbol``.

        Returns:
            Directory path that received this symbol's factor slices.
        """
        symbol_marker = self._root / "factors" / f"_symbols_seen_{symbol}.marker"
        identity_frame = frame.select(_IDENTITY_LIST).unique(maintain_order=False)
        partitions = frame.partition_by(_IDENTITY_LIST, as_dict=True, maintain_order=True)
        for key_tuple, sliced in partitions.items():
            factor_name = str(key_tuple[0])
            factor_version = str(key_tuple[1])
            timeframe = str(key_tuple[2])
            key = (factor_name, factor_version, timeframe)
            factor_dir = self._root / "factors" / _identity_dirname(*key)
            factor_dir.mkdir(parents=True, exist_ok=True)
            target = factor_dir / f"{symbol}.parquet"
            sliced.write_parquet(target)
            self._identities.setdefault(key, []).append(target)
        symbol_marker.write_text(symbol, encoding="utf-8")
        self._symbol_count += 1
        self._logger.debug(
            "Spilled validation symbol partition by factor identity",
            extra={
                "symbol": symbol,
                "rows": frame.height,
                "factor_identities": identity_frame.height,
            },
        )
        return self._root / "factors"

    def discover_factor_identities(self) -> tuple[_FactorIdentity, ...]:
        """Return sorted unique factor identities observed during spill writes."""
        identities = tuple(
            _FactorIdentity(
                factor_name=factor_name,
                factor_version=factor_version,
                timeframe=timeframe,
            )
            for factor_name, factor_version, timeframe in sorted(self._identities)
        )
        self._logger.info(
            "Discovered factor identities from validation spill",
            extra={
                "root": str(self._root),
                "symbol_partitions": self._symbol_count,
                "factor_identities": len(identities),
            },
        )
        return identities

    def load_factor_batch(self, identities: Sequence[_FactorIdentity]) -> pl.DataFrame:
        """Collect spill rows for the supplied factor identities.

        Args:
            identities: Non-empty factor identity batch.

        Returns:
            Eager DataFrame containing only the requested identities.
        """
        if len(identities) == 0:
            return pl.DataFrame()
        paths: list[Path] = []
        for identity in identities:
            key = (identity.factor_name, identity.factor_version, identity.timeframe)
            paths.extend(self._identities.get(key, ()))
        if len(paths) == 0:
            return pl.DataFrame()
        existing = [str(path) for path in paths if path.is_file()]
        if len(existing) == 0:
            raise FactorValidationError(
                "validation panel spill is missing expected factor files",
                error_code=_ERROR_SPILL_MISSING,
                details={"requested": tuple(str(identity) for identity in identities)},
            )
        return pl.read_parquet(existing)

    def cleanup(self) -> None:
        """Remove the spill directory tree if it still exists."""
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
            self._logger.debug("Removed validation panel spill", extra={"root": str(self._root)})


def _identity_dirname(factor_name: str, factor_version: str, timeframe: str) -> str:
    """Return a filesystem-safe directory name for one factor identity."""
    safe_name = factor_name.replace("/", "_").replace("\\", "_")
    safe_version = factor_version.replace("/", "_").replace("\\", "_")
    safe_timeframe = timeframe.replace("/", "_").replace("\\", "_")
    return f"{safe_name}__{safe_version}__{safe_timeframe}"


class MemoryEfficientFactorValidationRunner:
    """Bounded-memory Factor Validation using spill + factor-identity batches.

    The runner never reimplements IC, Rank IC, quantile, or turnover math. It
    only changes how the validation dataset is materialized and partitioned
    before calling ``FactorValidationEngine.build``.
    """

    __slots__ = ("_builder", "_engine", "_config", "_logger")

    _builder: ValidationDatasetBuilder
    _engine: FactorValidationEngine
    _config: FactorValidationExecutionConfig
    _logger: logging.Logger

    def __init__(
        self,
        builder: ValidationDatasetBuilder,
        engine: FactorValidationEngine,
        config: FactorValidationExecutionConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the runner with injected collaborators.

        Args:
            builder: Loads and joins Factors/Labels per symbol.
            engine: Canonical validation engine (unchanged math).
            config: Memory-efficient execution configuration.
            logger: Optional logger instance.
        """
        self._builder = builder
        self._engine = engine
        self._config = config
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> pl.DataFrame:
        """Spill the panel, validate factor batches, and return the ledger.

        Args:
            manager: Order manager identifier for Factors partitions.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Bar interval.
            year: Calendar year of the panel.
            symbols: Optional symbol allowlist.

        Returns:
            Canonical factor-validation DataFrame (engine schema columns).

        Raises:
            FactorValidationError: If spill/assembly fails or no identities
                remain.
        """
        spill = _create_spill(self._config.spill_parent, logger=self._logger)
        try:
            symbol_count = self._builder.spill_panel(
                spill=spill,
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
                symbols=symbols,
            )
            identities = spill.discover_factor_identities()
            if len(identities) == 0:
                raise FactorValidationError(
                    "validation panel spill contains no factor identities",
                    error_code=_ERROR_EMPTY_SPILL,
                    details={
                        "manager": manager,
                        "timeframe": timeframe,
                        "year": year,
                        "symbol_partitions": spill.symbol_file_count,
                    },
                )
            self._logger.info(
                "Running memory-efficient factor validation",
                extra={
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "timeframe": timeframe,
                    "year": year,
                    "symbol_partitions": symbol_count,
                    "factor_identities": len(identities),
                    "factor_batch_size": self._config.factor_batch_size,
                },
            )
            ledgers: list[pl.DataFrame] = []
            for batch_index, batch in enumerate(
                _iter_identity_batches(identities, self._config.factor_batch_size)
            ):
                chunk = spill.load_factor_batch(batch)
                if chunk.height == 0:
                    continue
                ledger = self._engine.build(chunk)
                ledgers.append(ledger)
                self._logger.debug(
                    "Validated factor-identity batch",
                    extra={
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "chunk_rows": chunk.height,
                        "ledger_rows": ledger.height,
                    },
                )
                del chunk
            if len(ledgers) == 0:
                raise FactorValidationError(
                    "memory-efficient validation produced no ledger rows",
                    error_code=_ERROR_EMPTY_SPILL,
                    details={
                        "manager": manager,
                        "timeframe": timeframe,
                        "year": year,
                        "factor_identities": len(identities),
                    },
                )
            combined = pl.concat(ledgers, how="vertical")
            return (
                combined.select(list(CANONICAL_COLUMN_ORDER))
                .sort(_PRIMARY_KEY_LIST, maintain_order=True)
                .cast(FACTOR_VALIDATION_SCHEMA)
            )
        finally:
            spill.cleanup()


def _create_spill(
    spill_parent: Path | None,
    *,
    logger: logging.Logger,
) -> ValidationPanelSpill:
    """Create a unique temporary spill directory."""
    parent = spill_parent if spill_parent is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cqros_fval_spill_", dir=str(parent)))
    return ValidationPanelSpill(root, logger=logger)


def _iter_identity_batches(
    identities: Sequence[_FactorIdentity],
    batch_size: int,
) -> Iterator[tuple[_FactorIdentity, ...]]:
    """Yield contiguous factor-identity batches of ``batch_size``."""
    for start in range(0, len(identities), batch_size):
        yield tuple(identities[start : start + batch_size])
