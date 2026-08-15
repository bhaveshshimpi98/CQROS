"""CQROS research-layer exception hierarchy.

Purpose:
    Provide research-specific exception types used by target generation,
    information-coefficient analysis, and related research workflows.

Responsibilities:
    - Re-export shared research exception roots from the core taxonomy
    - Expose ``TargetDefinitionError`` for invalid target definitions
    - Remain free of logging, validation pipelines, and business logic

Dependencies:
    ``cqros.core.exceptions``.

Public API:
    The exception types listed in ``__all__``.
"""

from __future__ import annotations

from cqros.core.exceptions import ResearchError, TargetError

__all__ = [
    "ResearchError",
    "TargetError",
    "TargetDefinitionError",
]


class TargetDefinitionError(TargetError):
    """Raised when a research target definition is invalid or incomplete."""

    __slots__ = ()
