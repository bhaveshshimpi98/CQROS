"""CQROS shared type aliases, type variables, and protocols.

Purpose:
    Provide reusable typing definitions used across CQROS packages for
    domain identifiers, market quantities, research structures, and
    lightweight structural contracts.

Responsibilities:
    - Define domain type aliases for symbols, assets, exchanges, markets,
      timeframes, prices, quantities, and related values
    - Define research and infrastructure aliases for features, metadata,
      identifiers, timestamps, and file paths
    - Expose generic type variables for reusable generic APIs
    - Define lightweight protocols for dict conversion, validation, and
      serialization
    - Remain free of business logic, helpers, and side effects

Dependencies:
    Python standard library only.

Public API:
    The type aliases, type variables, and protocols listed in ``__all__``.

Notes:
    Open ``str`` aliases (for example ``Symbol``, ``Exchange``, ``Timeframe``)
    intentionally accept values beyond the current allowlists so the typing
    surface stays extensible. Closed ``Literal`` aliases constrain known
    structural sets without replacing the open domain aliases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, TypeVar

__all__ = [
    # Generic type variables
    "T",
    "KT",
    "VT",
    # Domain identifiers
    "Symbol",
    "Asset",
    "Exchange",
    "Market",
    "Timeframe",
    # Market quantities
    "Price",
    "Quantity",
    "Volume",
    "Percentage",
    "Leverage",
    # Time and identity
    "Timestamp",
    "UnixTimestamp",
    "UnixTimestampMs",
    "Id",
    # Research structures
    "FeatureValue",
    "FeatureVector",
    "FeatureMatrix",
    "Metadata",
    # Infrastructure
    "FilePath",
    "JSONPrimitive",
    "JSONValue",
    # Closed literal aliases
    "EnvironmentName",
    "SupportedTimeframe",
    "CompressionCodec",
    "FileFormat",
    "HashAlgorithm",
    "MissingDataPolicy",
    # Protocols
    "SupportsToDict",
    "SupportsValidate",
    "SupportsSerialize",
]

# ---------------------------------------------------------------------------
# Generic type variables
# ---------------------------------------------------------------------------

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")

# ---------------------------------------------------------------------------
# Domain identifiers
# ---------------------------------------------------------------------------

type Symbol = str
"""Tradeable instrument symbol (for example ``BTCUSDT``)."""

type Asset = str
"""Asset code (for example ``BTC`` or ``USDT``)."""

type Exchange = str
"""Exchange identifier (for example ``binance``)."""

type Market = str
"""Market category identifier (for example ``usdt_perpetual``)."""

type Timeframe = str
"""Bar or sampling interval identifier (for example ``1m`` or ``1h``)."""

# ---------------------------------------------------------------------------
# Market quantities
# ---------------------------------------------------------------------------

type Price = float
"""Price expressed in quote-asset units."""

type Quantity = float
"""Order or position quantity in base-asset units."""

type Volume = float
"""Traded or aggregated volume."""

type Percentage = float
"""Fractional percentage in the closed unit interval convention ``[0.0, 1.0]``."""

type Leverage = float
"""Position leverage multiplier (for example ``1.0`` for unlevered exposure)."""

# ---------------------------------------------------------------------------
# Time and identity
# ---------------------------------------------------------------------------

type Timestamp = datetime
"""Timezone-aware UTC timestamp.

Naive ``datetime`` values are not permitted by CQROS time rules. Callers must
ensure values are timezone-aware and normalized to UTC.
"""

type UnixTimestamp = int
"""Unix time in whole seconds since the Unix epoch."""

type UnixTimestampMs = int
"""Unix time in whole milliseconds since the Unix epoch."""

type Id = str
"""Opaque string identifier for entities, artifacts, and correlation IDs."""

# ---------------------------------------------------------------------------
# Research structures
# ---------------------------------------------------------------------------

type FeatureValue = float
"""Scalar feature observation."""

type FeatureVector = Sequence[FeatureValue]
"""One-dimensional sequence of feature values for a single observation."""

type FeatureMatrix = Sequence[FeatureVector]
"""Two-dimensional sequence of feature vectors (rows are observations)."""

type JSONPrimitive = str | int | float | bool | None
"""JSON-compatible scalar value."""

type JSONValue = JSONPrimitive | Sequence[JSONValue] | Mapping[str, JSONValue]
"""JSON-compatible value tree used by metadata and serialization contracts."""

type Metadata = Mapping[str, JSONValue]
"""Immutable-by-convention metadata mapping with JSON-compatible values."""

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

type FilePath = Path | str
"""Filesystem path accepted as ``pathlib.Path`` or path string."""

# ---------------------------------------------------------------------------
# Closed literal aliases
# ---------------------------------------------------------------------------

type EnvironmentName = Literal[
    "development",
    "testing",
    "paper",
    "production",
]
"""Supported runtime environment names."""

type SupportedTimeframe = Literal[
    "1s",
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
    "1w",
]
"""Known research timeframe identifiers from the project allowlist."""

type CompressionCodec = Literal[
    "zstd",
    "snappy",
    "gzip",
    "lz4",
    "brotli",
    "uncompressed",
    "none",
]
"""Supported dataset compression codec names."""

type FileFormat = Literal[
    "parquet",
    "json",
    "yaml",
    "toml",
    "csv",
]
"""Supported file format identifiers."""

type HashAlgorithm = Literal["sha256"]
"""Supported content-hash algorithm names."""

type MissingDataPolicy = Literal[
    "reject",
    "warn",
    "ignore",
]
"""Supported missing-data handling policies."""


# ---------------------------------------------------------------------------
# Lightweight protocols
# ---------------------------------------------------------------------------


class SupportsToDict(Protocol):
    """Object that can convert itself to a plain dictionary."""

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible dictionary representation.

        Returns:
            Mapping of string keys to JSON-compatible values.
        """
        ...


class SupportsValidate(Protocol):
    """Object that can validate its own state.

    Implementations should raise a CQROS validation error on failure and
    return ``None`` on success (fail-fast).
    """

    def validate(self) -> None:
        """Validate the object state.

        Raises:
            Exception: When the object state is invalid. Concrete CQROS
                components should raise a project validation error type.
        """
        ...


class SupportsSerialize(Protocol):
    """Object that can serialize itself to bytes."""

    def serialize(self) -> bytes:
        """Serialize the object to a byte payload.

        Returns:
            Encoded byte representation of the object.
        """
        ...
