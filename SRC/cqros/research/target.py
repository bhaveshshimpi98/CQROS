"""CQROS forward-return target generation.

Purpose:
    Create deterministic forward-return prediction targets for factor
    research from price columns in a research DataFrame.

Responsibilities:
    - Define immutable ``TargetDefinition`` metadata
    - Compute forward returns via ``ForwardReturnTarget.transform``
    - Fail fast on invalid definitions and missing price columns
    - Remain independent of models, features, factors, training, and
      backtesting

Dependencies:
    ``polars``, the Python standard library, and
    ``cqros.research.exceptions``.

Public API:
    ``TargetDefinition``, ``ForwardReturnTarget``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.research.exceptions import TargetDefinitionError, TargetError

__all__ = [
    "TargetDefinition",
    "ForwardReturnTarget",
]

_DEFAULT_PRICE_COLUMN: Final[str] = "close"
_DEFAULT_OUTPUT_COLUMN: Final[str] = "forward_return"
_ERROR_NAME_BLANK: Final[str] = "RESEARCH-TARGET-001"
_ERROR_HORIZON_INVALID: Final[str] = "RESEARCH-TARGET-002"
_ERROR_PRICE_COLUMN_BLANK: Final[str] = "RESEARCH-TARGET-003"
_ERROR_OUTPUT_COLUMN_BLANK: Final[str] = "RESEARCH-TARGET-004"
_ERROR_MISSING_PRICE: Final[str] = "RESEARCH-TARGET-005"


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    """Immutable definition of a forward-return research target.

    Attributes:
        name: Stable target identifier.
        horizon: Number of rows ahead used for the forward return.
        price_column: Input price column used in the return calculation.
        output_column: Name of the column written by ``transform``.
    """

    name: str
    horizon: int
    price_column: str = _DEFAULT_PRICE_COLUMN
    output_column: str = _DEFAULT_OUTPUT_COLUMN

    def __post_init__(self) -> None:
        """Validate definition invariants.

        Raises:
            TargetDefinitionError: If any field violates definition rules.
        """
        _require_non_blank_str(
            cast(object, self.name),
            parameter="name",
            error_code=_ERROR_NAME_BLANK,
        )
        _require_positive_int(
            cast(object, self.horizon),
            parameter="horizon",
            error_code=_ERROR_HORIZON_INVALID,
        )
        _require_non_blank_str(
            cast(object, self.price_column),
            parameter="price_column",
            error_code=_ERROR_PRICE_COLUMN_BLANK,
        )
        _require_non_blank_str(
            cast(object, self.output_column),
            parameter="output_column",
            error_code=_ERROR_OUTPUT_COLUMN_BLANK,
        )


class ForwardReturnTarget:
    """Forward-return target generator for CQROS factor research.

    Computes ``(price.shift(-horizon) / price) - 1`` and appends the result
    under ``definition.output_column``. The final ``horizon`` rows are null.
    Missing values are never filled. The input DataFrame is never mutated.
    """

    __slots__ = ("_definition",)

    def __init__(self, definition: TargetDefinition) -> None:
        """Initialize with an immutable target definition.

        Args:
            definition: Validated forward-return target definition.
        """
        self._definition = definition

    @property
    def definition(self) -> TargetDefinition:
        """Return the immutable target definition."""
        return self._definition

    @property
    def name(self) -> str:
        """Return the target name from the definition."""
        return self._definition.name

    @property
    def horizon(self) -> int:
        """Return the forward horizon in rows."""
        return self._definition.horizon

    @property
    def price_column(self) -> str:
        """Return the input price column name."""
        return self._definition.price_column

    @property
    def output_column(self) -> str:
        """Return the output target column name."""
        return self._definition.output_column

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append forward returns without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``price_column``.

        Returns:
            A new DataFrame with all original columns plus ``output_column``.
            The final ``horizon`` rows of the target column are null.

        Raises:
            TargetError: If ``price_column`` is not present in ``frame``.
        """
        price_column = self._definition.price_column
        if price_column not in frame.columns:
            raise TargetError(
                f"required column missing: {price_column}",
                error_code=_ERROR_MISSING_PRICE,
                details={
                    "target": self._definition.name,
                    "required_column": price_column,
                    "available_columns": tuple(frame.columns),
                },
            )

        price = pl.col(price_column)
        horizon = self._definition.horizon
        shifted = price.shift(-horizon)  # pyright: ignore[reportUnknownMemberType]
        forward_return = (shifted / price) - 1
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            forward_return.alias(self._definition.output_column)
        )


def _require_non_blank_str(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``TargetDefinitionError`` when ``value`` is not a non-blank string."""
    if not isinstance(value, str) or value.strip() == "":
        raise TargetDefinitionError(
            f"{parameter} must be a non-blank string",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )


def _require_positive_int(value: object, *, parameter: str, error_code: str) -> None:
    """Raise ``TargetDefinitionError`` when ``value`` is not an int >= 1."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TargetDefinitionError(
            f"{parameter} must be an integer greater than or equal to 1",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )
