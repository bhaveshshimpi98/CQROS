"""CQROS leakage-safe factor orientation policy.

Purpose:
    Derive an explicit, versioned factor direction from the signed Factor
    Validation information coefficient available at Factor Selection time.

Responsibilities:
    - Define the locked orientation policy identifier
    - Map signed selection IC onto ``selected_direction ∈ {-1, +1}``
    - Preserve ranking based on ``abs(IC)`` (selection strength)
    - Document the zero-IC deterministic convention
    - Remain free of Walk-Forward / Purged-CV OOS evaluation and upward
      Alpha / Regime / ML imports

Dependencies:
    The Python standard library only.

Public API:
    ``FACTOR_ORIENTATION_POLICY``, ``ORIENTATION_ZERO_IC_DIRECTION``,
    ``ORIENTATION_SOURCE_METRIC``, ``selected_direction_from_ic``,
    ``oriented_selection_ic``, ``is_orientation_metadata_complete``

Discovered selection path (inspection evidence):
    1. Factor Validation emits signed ``information_coefficient``.
    2. ``SimpleFactorSelectionEngine.attach_selection_score_components``
       ranks with ``abs(information_coefficient)`` (``WEIGHT_ABS_IC``).
    3. Canonical Factor Selection previously persisted only
       ``selected`` / ``selection_score`` / ``selection_rank`` — no direction.
    4. ``assemble_walk_forward_input`` joins ``selected`` onto raw
       ``factor_value``; Walk-Forward / Purged-CV OOS IC used raw values.
    Orientation therefore belongs on the Factor Selection artifact and must
    be inherited downstream without recomputing from OOS rows.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "FACTOR_ORIENTATION_POLICY",
    "ORIENTATION_METADATA_COLUMNS",
    "ORIENTATION_SOURCE_METRIC",
    "ORIENTATION_ZERO_IC_DIRECTION",
    "is_orientation_metadata_complete",
    "oriented_selection_ic",
    "selected_direction_from_ic",
]

# Versioned policy identifier persisted on every Factor Selection row.
FACTOR_ORIENTATION_POLICY: Final[str] = "signed_ic_v1"

# Source metric available to Factor Selection at selection time.
ORIENTATION_SOURCE_METRIC: Final[str] = "information_coefficient"

# Deterministic zero-IC convention (governance default when IC is exactly 0).
ORIENTATION_ZERO_IC_DIRECTION: Final[int] = 1

ORIENTATION_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "selection_ic",
    "selected_direction",
    "orientation_policy",
)


def selected_direction_from_ic(raw_ic: object) -> int:
    """Return leakage-safe factor direction from signed selection IC.

    Args:
        raw_ic: Signed information coefficient from Factor Validation /
            Factor Selection time. Must not be derived from OOS rows.
            ``None`` is treated as exact zero under the locked zero-IC
            convention so scoring and orientation remain aligned.

    Returns:
        ``+1`` when ``raw_ic >= 0`` (including exact zero / null), otherwise
        ``-1``.

    Raises:
        TypeError: If ``raw_ic`` is not a real number or ``None``.
        ValueError: If ``raw_ic`` is non-finite.
    """
    if raw_ic is None:
        return ORIENTATION_ZERO_IC_DIRECTION
    if isinstance(raw_ic, bool) or not isinstance(raw_ic, int | float):
        raise TypeError(
            "raw_ic must be a finite float for orientation",
        )
    value = float(raw_ic)
    if value != value or value in (float("inf"), float("-inf")):  # noqa: PLW0177
        raise ValueError("raw_ic must be finite for orientation")
    if value < 0.0:
        return -1
    return ORIENTATION_ZERO_IC_DIRECTION


def oriented_selection_ic(raw_ic: float, selected_direction: int) -> float:
    """Return selection IC after applying ``selected_direction``.

    Oriented selection IC equals ``raw_ic * selected_direction`` and is
    non-negative under ``signed_ic_v1`` (including the zero-IC case).
    """
    if selected_direction not in (-1, 1):
        raise ValueError(
            "selected_direction must be -1 or +1",
        )
    return float(raw_ic) * float(selected_direction)


def is_orientation_metadata_complete(columns: list[str] | tuple[str, ...]) -> bool:
    """Return whether a frame exposes the orientation metadata contract."""
    present = set(columns)
    return all(column in present for column in ORIENTATION_METADATA_COLUMNS)
