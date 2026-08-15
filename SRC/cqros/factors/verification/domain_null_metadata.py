"""Metadata lookup for factors allowed to emit domain NULLs.

Purpose:
    Centralize the allowlist of factors whose post-warmup ``factor_value``
    NULLs are mathematically defined domain outcomes rather than data
    corruption.

Responsibilities:
    - Expose a single lookup for domain-NULL permission
    - Keep the allowlist out of verifier classification logic

Dependencies:
    Python standard library only.

Public API:
    ``factor_allows_domain_nulls``
"""

from __future__ import annotations

from typing import Final

__all__ = ["factor_allows_domain_nulls"]

# Factors whose formulas may emit NULL after warmup for defined domain
# conditions (for example zero denominators). Extend only here.
_DOMAIN_NULL_ALLOWED_FACTORS: Final[frozenset[str]] = frozenset(
    {
        "ease_of_movement",
        "volume_rate_of_change",
    }
)


def factor_allows_domain_nulls(factor_name: str) -> bool:
    """Return whether ``factor_name`` may emit post-warmup domain NULLs.

    Args:
        factor_name: Factor identity as stored in long-format frames.

    Returns:
        ``True`` when post-warmup ``factor_value`` NULLs for this factor
        must be classified as ``DOMAIN_NULLS`` rather than unexpected.
    """
    return factor_name in _DOMAIN_NULL_ALLOWED_FACTORS
