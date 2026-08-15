"""CQROS memory-efficient Factor Selection execution.

Purpose:
    Execute existing ``SimpleFactorSelectionEngine`` semantics under bounded
    memory by spilling candidate-factor observations to temporary Parquet
    files and computing greedy redundancy pairwise without materializing the
    full multi-factor observation panel in RAM.

Responsibilities:
    - Define execution-mode configuration for full-panel vs memory-efficient
      Factor Selection
    - Spill validation-window factor observations per factor identity
    - Apply greedy redundancy filtering that is mathematically equivalent to
      ``apply_greedy_redundancy_filter`` using pairwise inner joins
    - Provide ``MemoryEfficientFactorsObservationLoader`` implementing
      ``FactorObservationSource`` with a ``spill_panel`` entry point
    - Remain free of alternative scoring, eligibility, orientation, Top-N,
      or correlation-threshold policy changes

Dependencies:
    ``logging``, ``pathlib``, ``shutil``, ``tempfile``, ``polars``,
    ``cqros.core.constants``, ``cqros.core.types``,
    ``cqros.factor_selection.exceptions``, ``cqros.factor_selection.redundancy``,
    and ``cqros.storage.layout``.

Public API:
    ``FactorSelectionExecutionConfig``,
    ``FactorSelectionExecutionMode``,
    ``FactorObservationSpill``,
    ``MemoryEfficientFactorsObservationLoader``,
    ``apply_greedy_redundancy_filter_from_spill``,
    ``DEFAULT_FACTOR_BATCH_SIZE``
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

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTORS,
)
from cqros.core.types import Exchange, Market
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.redundancy import (
    REASON_OUTSIDE_CANDIDATE_N,
    REASON_OUTSIDE_TOP_N,
    REASON_REDUNDANT,
    REASON_TOP_N,
    RedundancyConfig,
    pairwise_abs_pearson,
)
from cqros.storage.layout import StorageLayout

__all__ = [
    "DEFAULT_FACTOR_BATCH_SIZE",
    "FactorObservationSpill",
    "FactorSelectionExecutionConfig",
    "FactorSelectionExecutionMode",
    "MemoryEfficientFactorsObservationLoader",
    "apply_greedy_redundancy_filter_from_spill",
]

_ERROR_BATCH_SIZE: Final[str] = "FSEL_MEM_BATCH_SIZE"
_ERROR_SELECTION_RANK: Final[str] = "FSEL_SELECTION_RANK_INVALID"
_ERROR_SPILL_FAILED: Final[str] = "FSEL_MEM_SPILL_FAILED"

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_OBSERVATION_SCHEMA: Final[dict[str, pl.DataType]] = {
    "symbol": pl.String(),
    "open_time": pl.Int64(),
    "factor_name": pl.String(),
    "factor_version": pl.String(),
    "factor_value": pl.Float64(),
}

_SPILL_COLUMNS: Final[tuple[str, ...]] = ("symbol", "open_time", "factor_value")

DEFAULT_FACTOR_BATCH_SIZE: Final[int] = 1

_logger = logging.getLogger(__name__)


class FactorSelectionExecutionMode(StrEnum):
    """Execution strategy for Factor Selection observation materialization."""

    FULL_PANEL = "full_panel"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass(frozen=True, slots=True)
class FactorSelectionExecutionConfig:
    """Immutable execution options for Factor Selection.

    Attributes:
        mode: ``full_panel`` loads the full candidate observation panel in RAM
            (legacy path). ``memory_efficient`` spills per-factor observations
            and computes pairwise redundancy without a dense multi-factor panel.
        factor_batch_size: Number of distinct factor identities spilled per
            symbol-scan batch when ``mode`` is ``memory_efficient``. Must be
            >= 1. Does not change selection semantics.
        spill_parent: Optional parent directory for temporary spill folders.
            When ``None``, the system temporary directory is used.
    """

    mode: FactorSelectionExecutionMode = FactorSelectionExecutionMode.MEMORY_EFFICIENT
    factor_batch_size: int = DEFAULT_FACTOR_BATCH_SIZE
    spill_parent: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.factor_batch_size < 1:
            raise FactorSelectionError(
                "factor_batch_size must be >= 1",
                error_code=_ERROR_BATCH_SIZE,
                details={"factor_batch_size": self.factor_batch_size},
            )


class FactorObservationSpill:
    """Temporary on-disk store of validation-window observations by factor.

    Each factor identity is stored as one Parquet file with columns
    ``symbol``, ``open_time``, ``factor_value`` (non-null values only).
    """

    __slots__ = ("_files", "_logger", "_root")

    _root: Path
    _files: dict[tuple[str, str], Path]
    _logger: logging.Logger

    def __init__(self, root: Path, *, logger: logging.Logger | None = None) -> None:
        """Initialize an empty spill directory.

        Args:
            root: Directory that will own per-factor Parquet files.
            logger: Optional logger instance.
        """
        self._root = root
        self._files = {}
        self._logger = logger if logger is not None else _logger
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the spill root directory."""
        return self._root

    def append_factor_rows(
        self,
        *,
        factor_name: str,
        factor_version: str,
        frame: pl.DataFrame,
    ) -> None:
        """Append non-null observation rows for one factor identity.

        Args:
            factor_name: Factor name.
            factor_version: Factor version.
            frame: Rows containing at least ``symbol``, ``open_time``,
                ``factor_value``. Null ``factor_value`` rows are dropped.
        """
        if frame.height == 0:
            return
        prepared = (
            frame.select(_SPILL_COLUMNS)
            .filter(pl.col("factor_value").is_not_null())
            .unique(subset=["symbol", "open_time"], keep="first", maintain_order=True)
        )
        if prepared.height == 0:
            return
        key = (factor_name, factor_version)
        target = self._files.get(key)
        if target is None:
            target = self._root / f"{_identity_filename(factor_name, factor_version)}.parquet"
            self._files[key] = target
            prepared.write_parquet(target)
            return
        existing = pl.read_parquet(target)
        # Preserve first-seen row order so pairwise Pearson input order matches
        # the legacy collect + pivot(aggregate=first) path.
        combined = pl.concat([existing, prepared], how="vertical").unique(
            subset=["symbol", "open_time"],
            keep="first",
            maintain_order=True,
        )
        combined.write_parquet(target)

    def ensure_factor(self, factor_name: str, factor_version: str) -> None:
        """Ensure a spill entry exists for ``factor`` even when empty."""
        key = (factor_name, factor_version)
        if key in self._files:
            return
        target = self._root / f"{_identity_filename(factor_name, factor_version)}.parquet"
        pl.DataFrame(
            schema={"symbol": pl.String, "open_time": pl.Int64, "factor_value": pl.Float64}
        ).write_parquet(target)
        self._files[key] = target

    def load_factor(self, factor_name: str, factor_version: str) -> pl.DataFrame:
        """Load spilled observations for one factor identity.

        Returns:
            Frame with ``symbol``, ``open_time``, ``factor_value``.
        """
        key = (factor_name, factor_version)
        path = self._files.get(key)
        if path is None or not path.is_file():
            return pl.DataFrame(
                schema={"symbol": pl.String, "open_time": pl.Int64, "factor_value": pl.Float64}
            )
        return pl.read_parquet(path)

    def load_long_panel(
        self,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
    ) -> pl.DataFrame:
        """Reconstruct a long-format observation panel from spilled factors.

        Intended for small fixtures and equivalence tests. Production
        redundancy should use ``apply_greedy_redundancy_filter_from_spill``.
        """
        pieces: list[pl.DataFrame] = []
        for name, version in zip(factor_names, factor_versions, strict=False):
            frame = self.load_factor(str(name), str(version))
            if frame.height == 0:
                continue
            pieces.append(
                frame.with_columns(
                    pl.lit(str(name)).alias("factor_name"),
                    pl.lit(str(version)).alias("factor_version"),
                ).select(
                    [
                        "symbol",
                        "open_time",
                        "factor_name",
                        "factor_version",
                        "factor_value",
                    ]
                )
            )
        if len(pieces) == 0:
            return pl.DataFrame(schema=_OBSERVATION_SCHEMA)
        return pl.concat(pieces, how="vertical")

    def cleanup(self) -> None:
        """Remove the spill directory tree if it still exists."""
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
            self._logger.debug(
                "Removed factor-selection observation spill",
                extra={"root": str(self._root)},
            )


class MemoryEfficientFactorsObservationLoader:
    """Bounded-memory Factors observation loader for redundancy filtering.

    Implements ``FactorObservationSource.load_panel`` for Protocol
    compatibility and exposes ``spill_panel`` for the production
    memory-efficient redundancy path used by ``SimpleFactorSelectionEngine``.
    """

    __slots__ = (
        "_batch_size",
        "_exchange",
        "_layout",
        "_logger",
        "_manager",
        "_market",
        "_spill_parent",
        "_year",
    )

    def __init__(
        self,
        layout: StorageLayout,
        *,
        manager: str,
        year: int,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
        factor_batch_size: int = DEFAULT_FACTOR_BATCH_SIZE,
        spill_parent: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the loader for one Factors year scope.

        Args:
            layout: Canonical storage layout.
            manager: Order manager identifier.
            year: Calendar year partition.
            exchange: Exchange identifier.
            market: Market segment.
            factor_batch_size: Factor identities spilled per symbol-scan batch.
            spill_parent: Optional parent directory for spill folders.
            logger: Optional logger instance.
        """
        if factor_batch_size < 1:
            raise FactorSelectionError(
                "factor_batch_size must be >= 1",
                error_code=_ERROR_BATCH_SIZE,
                details={"factor_batch_size": factor_batch_size},
            )
        self._layout = layout
        self._manager = manager
        self._year = year
        self._exchange = exchange
        self._market = market
        self._batch_size = factor_batch_size
        self._spill_parent = spill_parent
        self._logger = logger if logger is not None else _logger

    @property
    def factor_batch_size(self) -> int:
        """Return the configured spill batch size."""
        return self._batch_size

    def spill_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> FactorObservationSpill:
        """Spill validation-window observations for requested factors.

        Symbol partitions are scanned one file at a time. Factor identities
        are processed in deterministic batches of ``factor_batch_size``.
        """
        spill = _create_spill(self._spill_parent, logger=self._logger)
        identities: tuple[tuple[str, str], ...] = ()
        try:
            identities = _paired_identities(factor_names, factor_versions)
            for name, version in identities:
                spill.ensure_factor(name, version)

            paths = self._partition_paths(timeframe)
            if len(paths) == 0 or len(identities) == 0:
                return spill

            self._logger.info(
                "Spilling factor-selection observation panel",
                extra={
                    "manager": self._manager,
                    "timeframe": timeframe,
                    "year": self._year,
                    "symbol_partitions": len(paths),
                    "factor_identities": len(identities),
                    "factor_batch_size": self._batch_size,
                },
            )

            for batch_index, batch in enumerate(
                _iter_identity_batches(identities, self._batch_size)
            ):
                identity_lazy = pl.DataFrame(
                    {
                        "factor_name": [name for name, _ in batch],
                        "factor_version": [version for _, version in batch],
                    }
                ).lazy()
                for path in paths:
                    chunk = (
                        pl.scan_parquet(str(path))
                        .join(
                            identity_lazy,
                            on=["factor_name", "factor_version"],
                            how="semi",
                        )
                        .filter(pl.col("open_time") >= start_time)
                        .filter(pl.col("open_time") <= end_time)
                        .select(
                            [
                                "symbol",
                                "open_time",
                                "factor_name",
                                "factor_version",
                                "factor_value",
                            ]
                        )
                        .collect()
                    )
                    if chunk.height == 0:
                        continue
                    partitions = chunk.partition_by(
                        ["factor_name", "factor_version"],
                        as_dict=True,
                        maintain_order=True,
                    )
                    for key_tuple, sliced in partitions.items():
                        factor_name = str(key_tuple[0])
                        factor_version = str(key_tuple[1])
                        spill.append_factor_rows(
                            factor_name=factor_name,
                            factor_version=factor_version,
                            frame=sliced,
                        )
                self._logger.debug(
                    "Spilled factor-identity observation batch",
                    extra={"batch_index": batch_index, "batch_size": len(batch)},
                )
            return spill
        except Exception as exc:
            self._logger.exception(
                "memory-efficient factor observation spill failed; "
                "not falling back to full_panel",
                extra={
                    "manager": self._manager,
                    "timeframe": timeframe,
                    "year": self._year,
                    "factor_identities": len(identities),
                    "factor_batch_size": self._batch_size,
                    "cause_type": type(exc).__name__,
                },
            )
            spill.cleanup()
            raise FactorSelectionError(
                "memory-efficient factor observation spill failed; "
                "not falling back to full_panel",
                error_code=_ERROR_SPILL_FAILED,
                details={
                    "manager": self._manager,
                    "timeframe": timeframe,
                    "year": self._year,
                    "factor_identities": len(identities),
                    "factor_batch_size": self._batch_size,
                    "cause_type": type(exc).__name__,
                    "cause": str(exc),
                },
            ) from exc

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        """Load long-format observations via spill then reconstruct.

        Used for Protocol compatibility and small-fixture tests. Production
        redundancy uses ``spill_panel`` directly to avoid reconstructing the
        full multi-factor panel.
        """
        spill = self.spill_panel(
            timeframe=timeframe,
            factor_names=factor_names,
            factor_versions=factor_versions,
            start_time=start_time,
            end_time=end_time,
        )
        try:
            return spill.load_long_panel(factor_names, factor_versions)
        finally:
            spill.cleanup()

    def _partition_paths(self, timeframe: str) -> list[Path]:
        """Return existing Factors parquet paths for ``timeframe``/year."""
        base = (
            self._layout.root / STORAGE_DIR_FACTORS / self._manager / self._exchange / self._market
        )
        if not base.is_dir():
            return []
        paths: list[Path] = []
        for symbol_dir in sorted(base.iterdir()):
            if not symbol_dir.is_dir():
                continue
            path = symbol_dir / timeframe / f"{self._year}{FILE_EXTENSION_PARQUET}"
            if path.is_file():
                paths.append(path)
        return paths


def apply_greedy_redundancy_filter_from_spill(
    ranked: pl.DataFrame,
    spill: FactorObservationSpill,
    config: RedundancyConfig,
) -> pl.DataFrame:
    """Apply greedy redundancy filtering using spilled per-factor observations.

    Semantically equivalent to ``apply_greedy_redundancy_filter``: pairwise
    complete observations are the inner join of two factors on
    ``(symbol, open_time)`` after null ``factor_value`` rows are excluded,
    matching the wide-pivot pairwise mask.
    """
    ordered = ranked.sort(
        "selection_rank",
        "factor_name",
        "factor_version",
        maintain_order=True,
    )
    rows = ordered.to_dicts()
    if len(rows) == 0:
        return ordered

    candidate_limit = min(config.candidate_n, len(rows))
    candidates = rows[:candidate_limit]
    outside = rows[candidate_limit:]

    accepted: list[dict[str, object]] = []
    accepted_frames: dict[tuple[str, str], pl.DataFrame] = {}
    decisions: list[dict[str, object]] = []

    for candidate in candidates:
        decision, loaded_frame = _evaluate_candidate_from_spill(
            candidate=candidate,
            accepted=accepted,
            accepted_frames=accepted_frames,
            spill=spill,
            config=config,
        )
        decisions.append(decision)
        if not bool(decision["redundancy_rejected"]):
            accepted.append(candidate)
            if loaded_frame is not None:
                accepted_frames[
                    (str(candidate["factor_name"]), str(candidate["factor_version"]))
                ] = loaded_frame

    selected_keys = {
        (row["factor_name"], row["factor_version"]) for row in accepted[: config.top_n]
    }

    finalized: list[dict[str, object]] = []
    for decision in decisions:
        key = (decision["factor_name"], decision["factor_version"])
        if bool(decision["redundancy_rejected"]):
            reason = REASON_REDUNDANT
            selected = False
        elif key in selected_keys:
            reason = REASON_TOP_N
            selected = True
        else:
            reason = REASON_OUTSIDE_TOP_N
            selected = False
        finalized.append(
            {
                **decision,
                "selected": selected,
                "selection_reason": reason,
            }
        )

    for row in outside:
        finalized.append(
            {
                "factor_name": row["factor_name"],
                "factor_version": row["factor_version"],
                "candidate_rank": None,
                "redundancy_checked": False,
                "redundancy_rejected": False,
                "redundancy_reference_factor": None,
                "redundancy_reference_factor_version": None,
                "redundancy_correlation": None,
                "redundancy_overlap": None,
                "selected": False,
                "selection_reason": REASON_OUTSIDE_CANDIDATE_N,
            }
        )

    audit = pl.DataFrame(finalized)
    return ordered.join(
        audit,
        on=["factor_name", "factor_version"],
        how="left",
    )


def _evaluate_candidate_from_spill(
    *,
    candidate: dict[str, object],
    accepted: Sequence[dict[str, object]],
    accepted_frames: dict[tuple[str, str], pl.DataFrame],
    spill: FactorObservationSpill,
    config: RedundancyConfig,
) -> tuple[dict[str, object], pl.DataFrame | None]:
    """Evaluate one candidate against accepted factors using spilled frames.

    Only accepted factor frames are cached. Rejected candidates are not retained
    after the pairwise comparisons complete.
    """
    selection_rank = candidate["selection_rank"]
    if not isinstance(selection_rank, int):
        raise FactorSelectionError(
            "selection_rank must be an integer",
            error_code=_ERROR_SELECTION_RANK,
            details={"selection_rank": selection_rank},
        )
    base: dict[str, object] = {
        "factor_name": candidate["factor_name"],
        "factor_version": candidate["factor_version"],
        "candidate_rank": selection_rank,
        "redundancy_checked": True,
        "redundancy_rejected": False,
        "redundancy_reference_factor": None,
        "redundancy_reference_factor_version": None,
        "redundancy_correlation": None,
        "redundancy_overlap": None,
    }
    if len(accepted) == 0:
        return base, None

    candidate_frame = spill.load_factor(
        str(candidate["factor_name"]),
        str(candidate["factor_version"]),
    )

    for reference in accepted:
        reference_key = (str(reference["factor_name"]), str(reference["factor_version"]))
        reference_frame = accepted_frames.get(reference_key)
        if reference_frame is None:
            reference_frame = spill.load_factor(reference_key[0], reference_key[1])
            accepted_frames[reference_key] = reference_frame
        abs_corr, overlap = _pairwise_abs_pearson_frames(candidate_frame, reference_frame)
        if overlap < config.min_overlap:
            continue
        if abs_corr is None:
            continue
        if abs_corr >= config.max_factor_correlation:
            return (
                {
                    **base,
                    "redundancy_rejected": True,
                    "redundancy_reference_factor": reference["factor_name"],
                    "redundancy_reference_factor_version": reference["factor_version"],
                    "redundancy_correlation": abs_corr,
                    "redundancy_overlap": overlap,
                },
                None,
            )
    return base, candidate_frame


def _pairwise_abs_pearson_frames(
    left: pl.DataFrame,
    right: pl.DataFrame,
) -> tuple[float | None, int]:
    """Return ``(|pearson|, overlap)`` via inner join on ``(symbol, open_time)``.

    Pairwise-complete observations from the join are mathematically equivalent
    to the legacy wide-pivot finite mask. Pearson / overlap use the shared
    ``pairwise_abs_pearson`` implementation.
    """
    if left.height == 0 or right.height == 0:
        return None, 0
    joined = left.join(
        right,
        on=["symbol", "open_time"],
        how="inner",
        suffix="_right",
    )
    if joined.height == 0:
        return None, 0
    left_vals = joined["factor_value"].to_numpy()
    right_vals = joined["factor_value_right"].to_numpy()
    return pairwise_abs_pearson(left_vals, right_vals)


def _create_spill(
    spill_parent: Path | None,
    *,
    logger: logging.Logger,
) -> FactorObservationSpill:
    """Create a unique temporary spill directory."""
    parent = spill_parent if spill_parent is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cqros_fsel_spill_", dir=str(parent)))
    return FactorObservationSpill(root, logger=logger)


def _paired_identities(
    factor_names: Sequence[str],
    factor_versions: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Pair factor names/versions deterministically, dropping incomplete pairs."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name, version in zip(factor_names, factor_versions, strict=False):
        key = (str(name), str(version))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return tuple(pairs)


def _iter_identity_batches(
    identities: Sequence[tuple[str, str]],
    batch_size: int,
) -> Iterator[tuple[tuple[str, str], ...]]:
    """Yield contiguous factor-identity batches of ``batch_size``."""
    for start in range(0, len(identities), batch_size):
        yield tuple(identities[start : start + batch_size])


def _identity_filename(factor_name: str, factor_version: str) -> str:
    """Return a filesystem-safe filename stem for one factor identity."""
    safe_name = factor_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_version = factor_version.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_name}__{safe_version}"
